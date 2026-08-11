"""Agent 主体：DeepSeek 客户端 + Function Calling 循环。

流程（对每张边界照片）：
1. 拼 prompt：RAG 检索上下文 + CV 预检特征
2. 调用 DeepSeek；若它要求调用工具 → 执行工具、把结果回填、继续循环（最多 3 轮）
3. 解析最终 JSON 判定 {keep, category, reason}
"""

import json
import os
import re
from pathlib import Path

from ..rag.retriever import Retriever
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .tools import TOOL_SCHEMAS, ToolBox

MAX_TOOL_ROUNDS = 3


class PhotoJudge:
    """对单张照片做 Agent 判定。"""

    def __init__(self, client, retriever: Retriever, toolbox: ToolBox):
        self.client = client
        self.retriever = retriever
        self.toolbox = toolbox

    def judge(self, filename: str, features: dict, model: str, photo_path: str = "") -> dict:
        rag_context = self.retriever.context(_rag_query(features), k=3)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(filename, features, rag_context)},
        ]
        self._photo_dir = str(Path(photo_path).parent) if photo_path else "."
        self._photo_path = photo_path

        for _ in range(MAX_TOOL_ROUNDS):
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                # 模型要求调工具 → 先把 assistant 消息（含 tool_calls）加回对话
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
                # 执行并把结果回填
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    args["path"] = self._resolve(args.get("path", ""))
                    result = self.toolbox.call(tc.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                messages.append({"role": "user", "content": "工具结果已给出，请基于复核结果给出最终 JSON 判定。"})
                continue

            if msg.content:
                return parse_verdict(msg.content)

        # 工具循环用尽仍无结论：保守保留，交给人工
        return {"keep": True, "category": "keep", "reason": "Agent 未给出结论，转人工复核"}

    def _resolve(self, path: str) -> str:
        """模型可能回传奇怪路径，逐个候选解析，最后兜底到当前照片。"""
        candidates = [path, str(Path(self._photo_dir) / path)] if path else []
        for c in candidates:
            if c and Path(c).exists():
                return c
        return self._photo_path or path


def _rag_query(features: dict) -> str:
    """从特征推导检索 query：把照片情况翻译成知识库查询。"""
    b = features.get("blur", {})
    e = features.get("exposure", {})
    y = features.get("eyes", {})
    parts = ["判断这张约拍照片是否保留"]
    if b.get("verdict"):
        parts.append(f"清晰度判定{b.get('verdict')}")
    if e.get("verdict"):
        parts.append(f"曝光判定{e.get('verdict')}")
    if y.get("faces"):
        parts.append(f"人脸{y.get('faces')}张")
    return " ".join(parts)


def parse_verdict(content: str) -> dict:
    """从模型输出中稳健地解析 JSON 判定。"""
    text = content.strip()
    # 去掉 markdown 代码围栏
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    # 找到第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {"keep": True, "category": "keep", "reason": "解析失败，转人工复核"}
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        # 容错：reason 里可能有不合法字符，尝试按行恢复
        return {"keep": True, "category": "keep", "reason": "JSON 解析失败，转人工复核"}
    keep = bool(data.get("keep"))
    category = data.get("category", "keep" if keep else "reject_exposure")
    reason = data.get("reason", "")
    return {"keep": keep, "category": category, "reason": reason}
