"""曝光检测：亮度直方图 + 过曝/欠曝裁剪比例 + 平均亮度。

方法：
- 把图像转到 HSV，取 V（亮度）通道
- 平均亮度 mean_v：整体亮暗
- 过曝比例 clip_high：亮度 >= 250 的像素占比（高光死白）
- 欠曝比例 clip_low：亮度 <= 5 的像素占比（暗部死黑）
- 人脸区亮度（可选）：若有人脸框，只看人脸上的曝光，避免被背景误导
"""

import cv2
import numpy as np

CLIP_HIGH_V = 250   # 亮度 >= 此值视为死白
CLIP_LOW_V = 5      # 亮度 <= 此值视为死黑

# 人脸亮度区间：落在其中视为"人脸曝光正常"
FACE_MEAN_OK = (35.0, 240.0)

# 整图平均亮度的过曝/欠曝阈值（无人脸时的主要依据）
OVEREXPOSED_MEAN = 245.0   # 平均亮度 >= 此值：整体爆亮，过曝
UNDEREXPOSED_MEAN = 10.0   # 平均亮度 <= 此值：整体死黑，欠曝

# 轻微偏亮/偏暗（交给 LLM 判断场景的边界）
BORDER_MEAN_HIGH = 225.0
BORDER_MEAN_LOW = 50.0


def brightness_stats(bgr: np.ndarray, face_boxes=None) -> dict:
    """face_boxes: [(x, y, w, h), ...] 相对整图的人脸框。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)  # V 通道（亮度）
    total = v.size
    mean_v = float(v.mean())
    clip_high = float((v >= CLIP_HIGH_V).sum() / total)
    clip_low = float((v <= CLIP_LOW_V).sum() / total)

    face_mean = None
    if face_boxes:
        vals = []
        for (x, y, w, h) in face_boxes:
            if w <= 0 or h <= 0:
                continue
            roi = v[y:y + h, x:x + w]
            if roi.size:
                vals.append(float(roi.mean()))
        if vals:
            face_mean = float(np.mean(vals))

    return {
        "mean_brightness": round(mean_v, 1),
        "overexposed_ratio": round(clip_high, 4),
        "underexposed_ratio": round(clip_low, 4),
        "face_mean_brightness": face_mean and round(face_mean, 1),
    }


def verdict(stats: dict, has_face: bool = False) -> str:
    """返回 over / under / normal / borderline。

    人像优先原则（对应知识库）：
      有人脸时以人脸亮度为准，背景高光/暗部不影响判定；
      只有人脸亮度爆表/死黑才判曝光错误。
    无人脸（大景别/风景）时以整图平均亮度为准：
      整体爆亮/死黑才是曝光错误；局部高光（蓝天/白衣）不算。
    轻微偏亮/偏暗返回 borderline，交给 LLM Agent 结合场景判断。
    """
    mean = stats["mean_brightness"]
    face_mean = stats.get("face_mean_brightness")

    if has_face and face_mean is not None:
        if face_mean >= FACE_MEAN_OK[1]:
            return "overexposed"
        if face_mean <= FACE_MEAN_OK[0]:
            return "underexposed"
        return "normal"  # 人脸曝光正常，背景高光不算数

    if mean >= OVEREXPOSED_MEAN:
        return "overexposed"
    if mean <= UNDEREXPOSED_MEAN:
        return "underexposed"
    if mean >= BORDER_MEAN_HIGH or mean <= BORDER_MEAN_LOW:
        return "borderline"
    return "normal"


def assess(bgr: np.ndarray, face_boxes=None) -> dict:
    stats = brightness_stats(bgr, face_boxes)
    return {**stats, "verdict": verdict(stats, has_face=bool(face_boxes))}
