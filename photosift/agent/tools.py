"""Function Calling 工具定义 + 实现。

大模型可以自主调用这些工具来"复核"一张照片的某个指标，
而不是盲目相信预检特征。这是 Agent 的关键特征：模型决定调什么、不调什么。
"""

from pathlib import Path

from ..features import load_image
from ..screen import blur as blur_mod
from ..screen import exposure as exposure_mod

# 给大模型的工具 schema（OpenAI Function Calling 格式）
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "assess_blur",
            "description": "复核照片的清晰度。返回整图 Laplacian 方差（越大越清晰），若给 face_boxes 还返回人脸区域方差（浅景深时人脸清晰而背景模糊）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "照片文件路径"},
                    "face_boxes": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "[x, y, w, h] 人脸框，可省略",
                        },
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_exposure",
            "description": "复核照片曝光。返回平均亮度、过曝/欠曝比例。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "照片文件路径"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_eyes",
            "description": "复核照片中的人脸与闭眼情况。返回人脸数、每张脸的睁眼状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "照片文件路径"}
                },
                "required": ["path"],
            },
        },
    },
]


class ToolBox:
    """工具实现。路径已含于 tool 参数中。"""

    def __init__(self, detector=None):
        self.detector = detector  # EyeDetector，延迟导入避免循环依赖

    def call(self, name: str, arguments: dict) -> str:
        path = arguments.get("path", "")
        if not path:
            return '{"error": "缺少 path 参数"}'
        if name == "assess_blur":
            boxes = arguments.get("face_boxes")
            return str(blur_mod.assess(load_image(Path(path)), face_boxes=boxes))
        if name == "assess_exposure":
            return str(exposure_mod.assess(load_image(Path(path))))
        if name == "detect_eyes":
            from ..screen.eyes import EyeDetector
            det = self.detector or EyeDetector()
            return str(det.assess(load_image(Path(path))))
        return f'{{"error": "未知工具 {name}"}}'


def get_toolbox(detector=None) -> ToolBox:
    return ToolBox(detector=detector)
