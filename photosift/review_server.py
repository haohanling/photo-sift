"""本地复核服务器：让 HTML 报告里的"保留/删除"按钮真正生效。

浏览器里的 HTML 是静态的，无法直接读写文件。这个小服务器解决两件事：
1. 以 http:// 方式提供报告页 + 原图（lightbox 不依赖 file:// 相对路径）
2. 提供 /api/vote 接口：点击"保留/删除"时移动文件、更新 verdicts.json，
   刷新页面状态保持

用法：
    python -m photosift.cli <照片目录> --serve [--port 8765]
"""

import http.server
import json
import os
import shutil
import socketserver
import urllib.parse
from pathlib import Path

from .cli import CATEGORY_DIRS

CATEGORY_DIRS = dict(CATEGORY_DIRS)
CATEGORY_DIRS["reject_manual"] = "reject_manual"


def _load_verdicts(out_dir: Path) -> list[dict]:
    p = out_dir / "verdicts.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _save_verdicts(out_dir: Path, data: list[dict]) -> None:
    (out_dir / "verdicts.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _find_file_in(out_dir: Path, filename: str):
    """在任一分类目录里找该文件，返回 (dir_path, file_path)。"""
    for d in CATEGORY_DIRS.values():
        f = out_dir / d / filename
        if f.exists():
            return out_dir / d, f
    return None, None


def apply_vote(out_dir: Path, photo_dir: Path, filename: str, action: str) -> str:
    """把照片移动到目标分类，更新 verdicts.json，返回新 category。"""
    target_cat = "keep" if action == "keep" else "reject_manual"
    target_dir = out_dir / CATEGORY_DIRS[target_cat]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename

    # 确保目标有文件：优先从别的分类目录挪，没有就从原目录复制
    if not target.exists():
        src_dir, src = _find_file_in(out_dir, filename)
        if src and src.parent != target_dir:
            shutil.move(str(src), str(target))
        else:
            shutil.copy2(str(photo_dir / filename), str(target))

    # 从其他分类目录清理副本
    for d in CATEGORY_DIRS.values():
        if d == CATEGORY_DIRS[target_cat]:
            continue
        dup = out_dir / d / filename
        if dup.exists() and dup.resolve() != target.resolve():
            try:
                dup.unlink()
            except OSError:
                pass

    # 更新 verdicts.json
    data = _load_verdicts(out_dir)
    for entry in data:
        if entry["filename"] == filename:
            entry["category"] = target_cat
            entry["verdict"] = target_cat
            entry["reason"] = "人工复核：" + ("保留" if action == "keep" else "标记删除")
    _save_verdicts(out_dir, data)
    return target_cat


class _Handler(http.server.BaseHTTPRequestHandler):
    OUT_DIR: Path = None
    PHOTO_DIR: Path = None

    def log_message(self, fmt, *args):
        pass  # 安静点，不打访问日志

    # ---------- 工具 ----------
    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    # ---------- 路由 ----------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/report.html":
            return self._serve_report()
        if parsed.path == "/api/state":
            return self._serve_state()
        if parsed.path.startswith("/photo/"):
            return self._serve_photo(parsed.path)
        self._send(404, b"not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/vote":
            return self._serve_vote()
        self._send(404, b"not found")

    # ---------- 实现 ----------
    def _serve_report(self):
        p = self.OUT_DIR / "report.html"
        if not p.exists():
            return self._send(404, "report.html not found — 请先运行筛选".encode("utf-8"))
        body = p.read_bytes()
        self._send(200, body, "text/html; charset=utf-8")

    def _serve_state(self):
        data = _load_verdicts(self.OUT_DIR)
        state = {e["filename"]: (e.get("category") or e.get("verdict")) for e in data}
        self._send_json(200, state)

    def _serve_photo(self, path):
        name = urllib.parse.unquote(path[len("/photo/"):])
        # 防目录穿越：只允许照片目录里的文件
        safe = (self.PHOTO_DIR / name).resolve()
        if not safe.is_relative_to(self.PHOTO_DIR.resolve()):
            return self._send(403, b"forbidden")
        if not safe.exists():
            return self._send(404, b"not found")
        body = safe.read_bytes()
        ext = safe.suffix.lower()
        ctype = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png", ".webp": "image/webp"}.get(ext, "application/octet-stream")
        self._send(200, body, ctype)

    def _serve_vote(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            req = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return self._send_json(400, {"ok": False, "error": "bad json"})
        filename = req.get("filename", "")
        action = req.get("action", "")
        if not filename or action not in ("keep", "reject"):
            return self._send_json(400, {"ok": False, "error": "bad params"})
        try:
            cat = apply_vote(self.OUT_DIR, self.PHOTO_DIR, filename, action)
            return self._send_json(200, {"ok": True, "category": cat})
        except Exception as e:  # noqa: BLE001
            return self._send_json(500, {"ok": False, "error": str(e)})


class _Server(socketserver.TCPServer):
    allow_reuse_address = True


def serve(out_dir: Path, photo_dir: Path, port: int = 8765) -> None:
    _Handler.OUT_DIR = out_dir
    _Handler.PHOTO_DIR = photo_dir
    url = f"http://127.0.0.1:{port}"
    print(f"\n复核服务器已启动：{url}")
    print("在浏览器打开上面地址。点卡片上的【保留/删除】按钮即可调整判定，Ctrl+C 停止。")
    with _Server(("127.0.0.1", port), _Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止。")
