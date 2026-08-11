"""闭眼检测。

主方案：MediaPipe Face Landmarker（深度学习模型，输出人脸关键点）+ 眼纵横比 EAR。
  EAR = (上下眼睑关键点距离之和) / (2 * 水平眼角距离)
  睁眼 EAR 约 0.25+，闭眼显著下降。EAR 低于阈值 → 闭眼。
  （EAR 方法出自 Soukupová & Čech 2016 闭眼检测论文，MediaPipe 负责提供关键点）

模型：首次运行自动从 Google 下载 face_landmarker.task（约 3.5MB）到 models/（已 gitignore）。
兜底：OpenCV Haar 眼睛级联（零额外依赖），MediaPipe 不可用时自动回退。
"""

from pathlib import Path

import cv2
import numpy as np

# MediaPipe Face Mesh 左右眼关键点索引（468 点方案，478 点输出前 468 相同）
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR 阈值：睁眼约 0.25+，闭眼 < 0.15（眯眼/低眉笑/弱光会偏低，可在配置调）
EAR_EYE_CLOSED = 0.19   # max(EAR) >= 此值 → 眼睁开
EAR_CLOSED = 0.15       # max(EAR) < 此值 → 明显闭眼（低眉笑多落在 0.15~0.19 之间）

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


def _download_model(url: str, dest: Path) -> Path:
    """下载模型文件（std-lib urllib，无需额外依赖）。"""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    print(f"  下载人脸关键点模型 {url} -> {dest}")
    tmp = dest.with_suffix(".task.tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


class EyeDetector:
    """闭眼检测器，优先 MediaPipe，装不上自动回退 Haar。"""

    def __init__(self, try_mediapipe: bool = True, download: bool = True):
        self._mp = None
        self._landmarker = None
        self._haar = None
        if try_mediapipe:
            self._init_mediapipe(download)
        if self._landmarker is None:
            self._init_haar()

    # ---------- MediaPipe ----------
    def _init_mediapipe(self, download: bool):
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            if not MODEL_PATH.exists():
                if not download:
                    raise FileNotFoundError(MODEL_PATH)
                _download_model(MODEL_URL, MODEL_PATH)
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.IMAGE,
                min_face_detection_confidence=0.3,
                num_faces=8,
            )
            self._mp = vision
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:  # noqa: BLE001 —— 装不上就回退 Haar
            self._landmarker = None
            self._mediapipe_error = str(e)

    # ---------- Haar 兜底 ----------
    def _init_haar(self):
        try:
            root = cv2.data.haarcascades
            self._haar = cv2.CascadeClassifier(root + "haarcascade_frontalface_default.xml")
            self._haar_eye = cv2.CascadeClassifier(root + "haarcascade_eye.xml")
        except Exception:  # noqa: BLE001
            self._haar = None

    def backends(self) -> list:
        names = []
        if self._landmarker is not None:
            names.append("mediapipe")
        if self._haar is not None:
            names.append("haar")
        return names

    # ---------- 主入口 ----------
    def assess(self, bgr: np.ndarray) -> dict:
        if self._landmarker is not None:
            return self._assess_mediapipe(bgr)
        if self._haar is not None:
            return self._assess_haar(bgr)
        return {"faces": 0, "all_eyes_open": True, "detail": "no_detector", "face_boxes": []}

    # ---------- MediaPipe 实现 ----------
    def _assess_mediapipe(self, bgr: np.ndarray) -> dict:
        import mediapipe as mp

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(image)

        faces = 0
        eyes_ear = []   # 每张脸：(左眼EAR, 右眼EAR)
        face_boxes = []
        if results.face_landmarks:
            for lm in results.face_landmarks:
                faces += 1
                pts = np.array([(p.x * w, p.y * h) for p in lm])
                x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
                x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
                face_boxes.append((x, y, x2 - x, y2 - y))
                left_ear = self._ear(pts, LEFT_EYE)
                right_ear = self._ear(pts, RIGHT_EYE)
                eyes_ear.append((round(left_ear, 3), round(right_ear, 3)))

        if faces == 0:
            return {"faces": 0, "all_eyes_open": True, "detail": "no_face", "face_boxes": []}

        # 每张脸的状态分级：
        #   open  —— max(EAR) 正常，眼睁开
        #   squint —— EAR 偏低但未到"明显闭眼"，可能是眯眼/低眉笑（古风常见，不能武断删）
        #   closed—— EAR 极低，明显闭眼
        open_ = int(sum(1 for e in eyes_ear if max(e) >= EAR_EYE_CLOSED))
        squint = int(sum(1 for e in eyes_ear if max(e) < EAR_EYE_CLOSED and max(e) >= EAR_CLOSED))
        closed = int(sum(1 for e in eyes_ear if max(e) < EAR_CLOSED))
        return {
            "faces": faces,
            "all_eyes_open": open_ == faces,
            "closed_faces": closed,
            "eyes_ear": eyes_ear,
            "detail": "mediapipe",
            "face_boxes": face_boxes,
        }

    @staticmethod
    def _ear(pts: np.ndarray, eye_idx) -> float:
        """眼纵横比：上下关键点距离 / 水平眼角距离。"""
        p = pts[eye_idx]
        vertical = (
            np.linalg.norm(p[1] - p[5]) +   # p160-p144
            np.linalg.norm(p[2] - p[4])     # p158-p153
        )
        horizontal = np.linalg.norm(p[0] - p[3])  # p33-p133
        if horizontal < 1e-6:
            return 0.0
        return float(vertical / (2.0 * horizontal))

    # ---------- Haar 兜底实现 ----------
    def _assess_haar(self, bgr: np.ndarray) -> dict:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        face_boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        if len(faces) == 0:
            return {"faces": 0, "all_eyes_open": True, "detail": "no_face", "face_boxes": []}

        open_eyes_total = 0
        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            eyes = self._haar_eye.detectMultiScale(roi, 1.1, 5, minSize=(15, 15))
            open_eyes_total += len(eyes)
        expected = len(faces) * 2
        ratio = open_eyes_total / max(expected, 1)
        return {
            "faces": len(faces),
            "all_eyes_open": bool(ratio >= 0.6),
            "closed_faces": max(0, len(faces) - int(ratio)),
            "detail": "haar",
            "face_boxes": face_boxes,
        }


def assess(bgr: np.ndarray, detector: EyeDetector) -> dict:
    return detector.assess(bgr)
