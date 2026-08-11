"""photo-sift 命令行入口。

用法：
  python -m photosift.cli <照片目录> [--out 输出目录] [--no-llm] [--limit N]

阶段1（CV，总是运行）：模糊 / 曝光 / 闭眼硬性筛查。
阶段2（Agent，默认开启，--no-llm 关闭）：边界照片交给 DeepSeek Agent + RAG 判定。
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from .features import PhotoFeatures, decide, extract_features
from .report import write_json, write_report
from .screen.eyes import EyeDetector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CATEGORY_DIRS = {
    "keep": "keep",
    "reject_blur": "reject_blur",
    "reject_closed_eyes": "reject_closed_eyes",
    "reject_exposure": "reject_exposure",
}


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _get_deepseek():
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def _list_images(photo_dir: Path) -> list[Path]:
    return sorted(
        p for p in photo_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="自动筛选约拍照片")
    parser.add_argument("photo_dir", help="存放待筛选照片的文件夹")
    parser.add_argument("--out", default=None, help="输出目录（默认 <照片目录>_sifted）")
    parser.add_argument("--no-llm", action="store_true", help="只用 CV，不用大模型")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张（测试用）")
    parser.add_argument("--serve", action="store_true", help="筛选完启动本地复核服务器（支持保留/删除按钮）")
    parser.add_argument("--port", type=int, default=8765, help="复核服务器端口（默认 8765）")
    args = parser.parse_args(argv)

    # Windows 控制台默认 GBK，切到 UTF-8 避免中文乱码
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    _load_env()
    photo_dir = Path(args.photo_dir)
    if not photo_dir.is_dir():
        print(f"错误：目录不存在 {photo_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else Path(str(photo_dir) + "_sifted")
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in CATEGORY_DIRS.values():
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    images = _list_images(photo_dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print("没有找到可筛选的照片。", file=sys.stderr)
        return 1
    print(f"共 {len(images)} 张照片，开始 CV 筛查…")

    detector = EyeDetector()
    print(f"  人脸检测后端：{', '.join(detector.backends())}")

    results: list[PhotoFeatures] = []
    for i, img_path in enumerate(images, 1):
        try:
            f = extract_features(img_path, detector)
            f = decide(f)
            results.append(f)
            print(f"  [{i}/{len(images)}] {f.filename}: {f.verdict}/{f.category} ({f.reason})")
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(images)}] {img_path.name}: 处理失败 {e}")

    # 阶段2：Agent + RAG
    llm_photos = [p for p in results if p.verdict == "llm"]
    if llm_photos and not args.no_llm:
        client = _get_deepseek()
        if client is None:
            print("\n提示：未设置 DEEPSEEK_API_KEY，跳过 Agent 复核，边界照片保留。")
        else:
            from .agent.agent import PhotoJudge
            from .agent.tools import get_toolbox
            from .rag.retriever import Retriever

            model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
            retriever = Retriever.from_knowledge()
            toolbox = get_toolbox(detector=detector)
            judge = PhotoJudge(client, retriever, toolbox)
            print(f"\nAgent 复核 {len(llm_photos)} 张边界照片（RAG + Function Calling）…")
            for f in llm_photos:
                v = judge.judge(f.filename, f.to_dict(), model, photo_path=str(f.path))
                keep = v.get("keep", True)
                f.verdict = "keep" if keep else "reject"
                f.category = v.get("category", "keep" if keep else "reject_exposure")
                f.reason = v.get("reason", f.reason)
                print(f"  {f.filename} -> {f.category}: {f.reason}")

    # 移动文件到分类目录
    for f in results:
        if f.category in CATEGORY_DIRS:
            dest = out_dir / CATEGORY_DIRS[f.category] / f.filename
            shutil.copy2(f.path, dest)

    write_report(out_dir, results)
    write_json(out_dir, results)

    # 交互式 HTML 验证报告（点击放大原图 + 筛选）
    try:
        from .visual_report import generate as generate_html
        generate_html(results, photo_dir, out_dir / "report.html")
        html_path = out_dir / "report.html"
    except Exception as e:  # noqa: BLE001
        html_path = None
        print(f"  (HTML 报告生成失败: {e})")

    counts = {}
    for f in results:
        counts[f.category or f.verdict] = counts.get(f.category or f.verdict, 0) + 1
    print("\n完成。结果已输出到", out_dir)
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"报告：{out_dir / 'report.md'}")
    if html_path:
        print(f"可视化报告：{html_path}（点击缩略图可放大原图）")

    if args.serve:
        from .review_server import serve as serve_review
        serve_review(out_dir, photo_dir, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
