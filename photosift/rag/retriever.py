"""手写 BM25 检索器（无第三方依赖）。

中文不按空格分词，这里用**字符二元组（bigram）+ 单字**作为 token，
配合英文/数字词 token。对小语料（几十个 chunk）足够鲁棒，且完全可解释。

BM25 公式（标准 Okapi BM25，k1=1.5, b=0.75）：
  score(d,q) = Σ IDF(t) * [ f(t,d)*(k1+1) / (f(t,d) + k1*(1-b+b*|d|/avgdl)) ]
"""

import math
import re
from collections import Counter

from .knowledge_base import Chunk

_CJK_RE = re.compile(r"[一-鿿]+")
_ASCII_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """中英文混合 tokenizer：CJK 用 bigram+单字，ASCII 用整词。"""
    text = text.lower()
    toks: list[str] = []
    for m in _ASCII_RE.finditer(text):
        toks.append(m.group())
    for seg in _CJK_RE.findall(text):
        if not seg:
            continue
        for ch in seg:
            toks.append(ch)
        for i in range(len(seg) - 1):
            toks.append(seg[i:i + 2])
    return toks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(len(docs), 1)
        self.doc_count = len(docs)
        # 文档频率 df（出现在多少篇文档里）
        self.df: Counter = Counter()
        for d in docs:
            self.df.update(set(d))
        self.idf = {t: self._idf(t) for t in self.df}

    def _idf(self, term: str) -> float:
        n = self.df[term]
        # +1 平滑，避免低频词被过度放大；负数归零
        return math.log(1 + (self.doc_count - n + 0.5) / (n + 0.5))

    def score(self, query_toks: list[str], doc_idx: int) -> float:
        tf = Counter(self.docs[doc_idx])
        dl = self.doc_len[doc_idx]
        total = 0.0
        for t in query_toks:
            if t not in self.idf:
                continue
            f = tf[t]
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += self.idf[t] * (f * (self.k1 + 1)) / denom
        return total

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        q_toks = tokenize(query)
        if not q_toks:
            return []
        scores = [(i, self.score(q_toks, i)) for i in range(self.doc_count)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class Retriever:
    """加载知识库并检索，产出可供 prompt 拼接的上下文。"""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25([tokenize(c.full_text()) for c in chunks])

    @classmethod
    def from_knowledge(cls, path=None):
        from .knowledge_base import load_knowledge, DEFAULT_KB
        return cls(load_knowledge(path or DEFAULT_KB))

    def retrieve(self, query: str, k: int = 3) -> list[Chunk]:
        hits = self.bm25.search(query, k=k)
        return [self.chunks[i] for i, _ in hits if _ > 0]

    def context(self, query: str, k: int = 3) -> str:
        """返回可直接拼进 prompt 的检索上下文。"""
        hits = self.retrieve(query, k)
        if not hits:
            return ""
        blocks = []
        for i, c in enumerate(hits, 1):
            blocks.append(f"[参考资料{i}] {c.full_text()}")
        return "\n\n".join(blocks)
