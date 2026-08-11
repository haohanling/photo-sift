"""Prompt 模板。

System prompt：定义角色 + 输出契约 + 判断规则。
User prompt：拼接 照片特征 + RAG 检索到的参考资料 + 任务指令。
"""

SYSTEM_PROMPT = """你是一位专业的约拍/陪拍照片筛选助手，帮摄影师决定每张照片是否交付给客人。

你的任务：基于"计算机视觉预检特征"和"检索到的摄影评判标准"，判断一张照片是保留还是剔除，并给出简明理由。

判断规则：
1. 明显闭眼、严重模糊、严重曝光错误 → 剔除（reject）
2. 轻微问题但可后期救回（轻微欠曝、轻微模糊可锐化）→ 保留（keep）并说明
3. 艺术性判断（氛围暗调、逆光剪影、抓拍情绪）→ 倾向保留，但需说明理由
4. 信号矛盾时，可调用工具复核照片，再下结论

输出契约（必须输出一个 JSON，不要输出其他内容）：
{"keep": true/false, "category": "keep|reject_blur|reject_closed_eyes|reject_exposure", "reason": "一句话中文理由"}

category 只能取以上四个值之一。
"""


def build_user_prompt(filename: str, features: dict, rag_context: str) -> str:
    """拼接用户消息：检索上下文 + 照片特征 + 待判断对象。"""
    parts = []
    if rag_context:
        parts.append("【检索到的摄影评判标准（请优先依据这些标准判断）】\n" + rag_context)
    parts.append("【计算机视觉预检特征（JSON）】\n" + _format_features(features))
    parts.append(
        f"请判断照片「{filename}」是否保留，输出符合输出契约的 JSON。"
        "如对某个特征不放心，可先调用对应工具复核，再下结论。"
    )
    return "\n\n".join(parts)


def _format_features(features: dict) -> str:
    import json
    # 精简：去掉过长的 EAR 列表，只留关键字段
    slim = dict(features)
    if "eyes" in slim and isinstance(slim["eyes"], dict):
        slim["eyes"] = {k: v for k, v in slim["eyes"].items() if k != "eyes_ear"}
    return json.dumps(slim, ensure_ascii=False, indent=2)
