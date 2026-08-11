"""模糊检测：基于 Laplacian 方差的经典清晰度指标。

原理：清晰照片边缘锐利，灰度梯度变化大；模糊照片梯度小。
对灰度图做 Laplacian 算子得到二阶梯度，方差越大 = 越清晰。
"""

import cv2
import numpy as np

# 建议阈值（对 1600px 长边、8MP 以上的照片经验值，可在配置中调整）
SHARP_VAR = 100.0   # >= 视为清晰
BLURRY_VAR = 40.0   # <= 视为明显模糊，中间为边界

# 缩小到统一长边再算，避免照片分辨率差异影响方差量级
TARGET_MAX_EDGE = 800


def laplacian_variance(bgr: np.ndarray) -> float:
    """计算 Laplacian 方差，越大越清晰。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 统一尺寸：把长边缩到 TARGET_MAX_EDGE，保证不同分辨率照片可比
    h, w = gray.shape
    scale = min(1.0, TARGET_MAX_EDGE / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def verdict(var: float) -> str:
    """返回 sharp / borderline / blurry。"""
    if var >= SHARP_VAR:
        return "sharp"
    if var <= BLURRY_VAR:
        return "blurry"
    return "borderline"


def assess(bgr: np.ndarray, face_boxes=None) -> dict:
    """整图清晰度为主判定，另附人脸区域清晰度供 LLM 复核（浅景深时用）。"""
    var = laplacian_variance(bgr)
    face_var = _face_region_variance(bgr, face_boxes) if face_boxes else None
    return {
        "laplacian_variance": round(var, 1),
        "face_laplacian_variance": round(face_var, 1) if face_var is not None else None,
        "has_face": bool(face_boxes),
        "verdict": verdict(var),
    }


def _face_region_variance(bgr: np.ndarray, face_boxes) -> float:
    """取所有人脸框中最低的 Laplacian 方差（最糊的那张脸）。

    注意：人脸区域（平滑皮肤）的方差天然比整图（背景纹理）低很多，
    不能直接套用整图阈值——这里只作为 LLM 复核的参考特征。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h_orig, w_orig = gray.shape
    min_var = None
    for (x, y, w, h) in face_boxes:
        if w <= 0 or h <= 0:
            continue
        x, y, w, h = int(x), int(y), int(w), int(h)
        x, y = max(0, x), max(0, y)
        w, h = min(w, w_orig - x), min(h, h_orig - y)
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        scale = min(1.0, TARGET_MAX_EDGE / max(h, w))
        if scale < 1.0:
            roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        v = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        if min_var is None or v < min_var:
            min_var = v
    return min_var
