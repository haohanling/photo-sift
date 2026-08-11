"""生成 markdown 筛选报告。"""

import json
from pathlib import Path

from .features import PhotoFeatures


def write_report(out_dir: Path, photos: list[PhotoFeatures]) -> Path:
    """生成 report.md，含每张照片的判定与特征。"""
    lines = ["# 照片筛选报告", ""]
    lines.append(f"- 照片总数：{len(photos)}")
    counts = {}
    for p in photos:
        counts[p.category] = counts.get(p.category, 0) + 1
    lines.append("- 分布：" + "；".join(f"{k} {v}" for k, v in counts.items()))
    lines.append("")

    lines.append("| 文件名 | 判定 | 原因 | 清晰度 | 曝光 | 人脸 |")
    lines.append("|---|---|---|---|---|---|")
    for p in photos:
        cat = p.category or p.verdict
        lines.append(
            f"| {p.filename} | {cat} | {p.reason} | "
            f"{p.blur.get('verdict','-')}({p.blur.get('laplacian_variance','-')}) | "
            f"{p.exposure.get('verdict','-')} | {p.eyes.get('faces',0)}张"
            f"{'' if p.eyes.get('all_eyes_open', True) else '/闭眼'} |"
        )
    lines.append("")
    lines.append("> 由 photo-sift 自动生成。判定仅供参考，最终以人工复核为准。")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_json(out_dir: Path, photos: list[PhotoFeatures]) -> Path:
    path = out_dir / "verdicts.json"
    path.write_text(
        json.dumps([p.to_dict() for p in photos], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
