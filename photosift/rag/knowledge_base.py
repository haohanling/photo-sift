"""知识库加载与分块。

把 markdown 知识库按 `##` 分节切成 chunk，每个 chunk 带标题和正文。
这是 RAG 的"检索侧"数据源——模型的回答基于这些被检索到的段落。
"""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_KB = Path(__file__).resolve().parents[2] / "knowledge" / "photography_criteria.md"


@dataclass
class Chunk:
    title: str
    text: str

    def full_text(self) -> str:
        return f"【{self.title}】{self.text}"


def load_knowledge(path: Path = DEFAULT_KB) -> list[Chunk]:
    """读取 md，按 ## 分节成 chunk。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    chunks: list[Chunk] = []
    current_title = "概述"
    current_lines: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if current_lines:
                chunks.append(Chunk(current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        elif line.strip() and not line.startswith("#"):
            current_lines.append(line)
    if current_lines:
        chunks.append(Chunk(current_title, "\n".join(current_lines).strip()))
    return [c for c in chunks if c.text]
