"""阶段1：CV 特征提取 + 硬性归类。

对每张照片提取结构化特征（模糊/曝光/人脸闭眼），并按硬规则决定：
  - keep   → 保留（交给 keep 目录，或进 LLM 复核）
  - reject → 直接废片（模糊 / 闭眼 / 曝光不对）
  - llm    → 信号矛盾的边界照片，交给阶段2 的 Agent 判断
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .screen import blur as blur_mod
from .screen import exposure as exposure_mod
from .screen.eyes import EyeDetector


@dataclass
class PhotoFeatures:
    path: Path
    filename: str
    blur: dict = field(default_factory=dict)
    exposure: dict = field(default_factory=dict)
    eyes: dict = field(default_factory=dict)
    verdict: str = "llm"       # keep / reject / llm
    category: str = ""         # keep / reject_blur / reject_closed_eyes / reject_exposure
    reason: str = ""           # 中文理由

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "path": str(self.path),
            "blur": self.blur,
            "exposure": self.exposure,
            "eyes": self.eyes,
            "verdict": self.verdict,
            "category": self.category,
            "reason": self.reason,
        }


def load_image(path: Path) -> np.ndarray:
    """读取图片，兼容中文/Unicode 路径（cv2.imread 在 Windows 上不支持中文路径）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法解码图片: {path}")
    return img


def decide(f: PhotoFeatures) -> PhotoFeatures:
    """按硬规则判定。优先级：模糊 > 闭眼 > 曝光。"""
    blur_verdict = f.blur.get("verdict", "sharp")
    eyes = f.eyes
    exp_verdict = f.exposure.get("verdict", "normal")

    # 1) 模糊 —— 整图模糊且无人脸可确认主体，直接废
    if blur_verdict == "blurry":
        if f.blur.get("has_face"):
            # 可能是浅景深虚化（背景糊但人脸清晰），交给 Agent 复核人脸清晰度
            f.verdict = "llm"
            f.category = ""
            f.reason = "整图偏糊但有脸，可能是浅景深，待复核"
            return f
        f.verdict = "reject"
        f.category = "reject_blur"
        f.reason = "照片模糊（清晰度不足）"
        return f

    # 2) 眼神偏低 —— 有人脸但眼没全睁开。
    #    真闭眼和低眉笑/眯眼在 EAR 上高度重叠，无法用阈值可靠区分，
    #    因此一律降级为人工复核（宁可不误删低眉笑这类好照片，由用户眼睛定夺）。
    if eyes.get("faces", 0) > 0 and not eyes.get("all_eyes_open", True):
        f.verdict = "llm"
        f.category = ""
        f.reason = "眼神偏低（可能眯眼/低眉笑/闭眼），需人工复核"
        return f

    # 3) 曝光硬伤
    if exp_verdict in ("overexposed", "underexposed"):
        f.verdict = "reject"
        f.category = "reject_exposure"
        f.reason = "曝光过度/不足" if exp_verdict == "overexposed" else "曝光不足/过暗"
        return f

    # 4) 明确的正常照片
    if blur_verdict == "sharp" and exp_verdict == "normal":
        f.verdict = "keep"
        f.category = "keep"
        f.reason = "清晰、曝光正常"
        return f

    # 5) 其余：信号矛盾的边界照片，交给 LLM Agent 复核
    f.verdict = "llm"
    f.category = ""
    f.reason = "信号矛盾，需 Agent 复核"
    return f


def extract_features(path: Path, detector: EyeDetector) -> PhotoFeatures:
    img = load_image(path)
    f = PhotoFeatures(path=path, filename=path.name)
    f.eyes = detector.assess(img)
    face_boxes = f.eyes.get("face_boxes")
    f.blur = blur_mod.assess(img, face_boxes=face_boxes)
    f.exposure = exposure_mod.assess(img, face_boxes=face_boxes)
    return f
