#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 test_m6_scenarios.py 生成的 summary_*.json 为一个 full summary（JSON + Markdown）。

用法示例：
  python tools/merge_m6_summaries.py \
    --inputs outputs/m6_high_risk_nonquick_part1/summary_*.json outputs/m6_high_risk_nonquick_part2/summary_*.json \
    --output-dir outputs/m6_high_risk_nonquick_full
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List


def _load_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"不是 list: {path}")
    return data


def _write_md(out_path: Path, items: List[Dict[str, Any]]):
    lines = []
    lines.append("# M6 场景批量测试汇总（合并）")
    lines.append("")
    lines.append(f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 条目数: {len(items)}")
    lines.append("")
    lines.append("| 场景 | 镜头 | 运动 | 通过 | 平均相似度 | 最低相似度 | 漂移% | 人脸检测% | 视频 | 报告 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|---|")
    for it in items:
        lines.append(
            "| {scenario} | {shot_type} | {motion_intensity} | {passed} | {avg:.3f} | {minv:.3f} | {drift:.1f} | {face:.1f} | {video} | {report} |".format(
                scenario=it.get("scenario", ""),
                shot_type=it.get("shot_type", ""),
                motion_intensity=it.get("motion_intensity", ""),
                passed="✅" if it.get("passed") else "❌",
                avg=float(it.get("avg_similarity", 0.0)),
                minv=float(it.get("min_similarity", 0.0)),
                drift=float(it.get("drift_ratio", 0.0)) * 100.0,
                face=float(it.get("face_detect_ratio", 0.0)) * 100.0,
                video=it.get("video_path", ""),
                report=it.get("report_path", ""),
            )
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="合并 M6 summary_*.json")
    parser.add_argument("--inputs", nargs="+", required=True, help="输入 summary_*.json 列表（支持多个）")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--dedup", action="store_true", help="按 scenario 去重（保留最后一次出现）")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged: List[Dict[str, Any]] = []
    for p in args.inputs:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(path)
        merged.extend(_load_list(path))

    if args.dedup:
        by_key: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for it in merged:
            k = it.get("scenario") or ""
            if k not in by_key:
                order.append(k)
            by_key[k] = it
        merged = [by_key[k] for k in order]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"summary_full_{ts}.json"
    out_md = out_dir / f"summary_full_{ts}.md"
    out_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(out_md, merged)

    print(f"✅ merged={len(merged)}")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()


