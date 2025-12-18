#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成一次“全套回归”总览报告（基于各子目录 summary_*.json）。

用法：
  python tools/regression_report.py \
    --base-dir outputs/m6_regression_full_20251218_185420 \
    --out-md  outputs/m6_regression_full_20251218_185420/REGRESSION_REPORT.md
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime


@dataclass
class SuiteStats:
    name: str
    summary_json: Path
    summary_md: Path
    total: int
    passed: int
    failed: int
    min_avg_similarity: float
    min_min_similarity: float
    max_drift_ratio: float
    min_face_detect_ratio: float
    failures: List[Dict[str, Any]]


def _load_list(p: Path) -> List[Dict[str, Any]]:
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"summary 不是 list: {p}")
    return data


def _find_summary_pairs(base_dir: Path) -> List[Tuple[str, Path, Path]]:
    pairs: List[Tuple[str, Path, Path]] = []
    for sub in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
        sj = next(iter(sorted(sub.glob("summary_*.json"))), None)
        sm = next(iter(sorted(sub.glob("summary_*.md"))), None)
        if sj and sm:
            pairs.append((sub.name, sj, sm))
    return pairs


def _calc_suite(name: str, sj: Path, sm: Path) -> SuiteStats:
    items = _load_list(sj)
    total = len(items)
    passed = sum(1 for it in items if bool(it.get("passed")))
    failed = total - passed

    min_avg = min(float(it.get("avg_similarity", 0.0)) for it in items) if items else 0.0
    min_min = min(float(it.get("min_similarity", 0.0)) for it in items) if items else 0.0
    max_drift = max(float(it.get("drift_ratio", 0.0)) for it in items) if items else 0.0
    min_face = min(float(it.get("face_detect_ratio", 0.0)) for it in items) if items else 0.0

    failures: List[Dict[str, Any]] = []
    for it in items:
        if it.get("passed"):
            continue
        failures.append(
            {
                "scenario": it.get("scenario", ""),
                "shot_type": it.get("shot_type", ""),
                "motion_intensity": it.get("motion_intensity", ""),
                "avg": float(it.get("avg_similarity", 0.0)),
                "min": float(it.get("min_similarity", 0.0)),
                "drift": float(it.get("drift_ratio", 0.0)),
                "face": float(it.get("face_detect_ratio", 0.0)),
                "issues": it.get("issues", []),
                "video": it.get("video_path", ""),
                "report": it.get("report_path", ""),
            }
        )

    return SuiteStats(
        name=name,
        summary_json=sj,
        summary_md=sm,
        total=total,
        passed=passed,
        failed=failed,
        min_avg_similarity=min_avg,
        min_min_similarity=min_min,
        max_drift_ratio=max_drift,
        min_face_detect_ratio=min_face,
        failures=failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 M6 全套回归总览报告")
    parser.add_argument("--base-dir", required=True, help="一次回归的输出根目录（包含 low_quick/medium_quick/... 子目录）")
    parser.add_argument("--out-md", default=None, help="输出 Markdown 路径（默认 base-dir/REGRESSION_REPORT.md）")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise SystemExit(f"base-dir 不存在: {base_dir}")

    out_md = Path(args.out_md) if args.out_md else (base_dir / "REGRESSION_REPORT.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    pairs = _find_summary_pairs(base_dir)
    if not pairs:
        raise SystemExit(f"未找到任何 summary_*.json/.md: {base_dir}")

    suites = [_calc_suite(name, sj, sm) for name, sj, sm in pairs]

    total_all = sum(s.total for s in suites)
    passed_all = sum(s.passed for s in suites)
    failed_all = sum(s.failed for s in suites)

    lines: List[str] = []
    lines.append("# M6 全套回归总览")
    lines.append("")
    lines.append(f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- base_dir: {base_dir}")
    lines.append(f"- 总计: {passed_all}/{total_all} 通过（失败 {failed_all}）")
    lines.append("")

    lines.append("## 套件汇总")
    lines.append("")
    lines.append("| suite | 通过 | min(avg) | min(min) | max(drift) | min(face) | summary_md | summary_json |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for s in suites:
        lines.append(
            "| {name} | {p}/{t} | {minavg:.3f} | {minmin:.3f} | {maxdrift:.3f} | {minface:.3f} | {md} | {js} |".format(
                name=s.name,
                p=s.passed,
                t=s.total,
                minavg=s.min_avg_similarity,
                minmin=s.min_min_similarity,
                maxdrift=s.max_drift_ratio,
                minface=s.min_face_detect_ratio,
                md=str(s.summary_md.relative_to(base_dir)),
                js=str(s.summary_json.relative_to(base_dir)),
            )
        )

    lines.append("")
    lines.append("## 失败项（如有）")
    lines.append("")
    if failed_all == 0:
        lines.append("- ✅ 本次回归全绿")
    else:
        for s in suites:
            if not s.failures:
                continue
            lines.append(f"### {s.name}")
            lines.append("")
            lines.append("| scenario | shot | motion | avg | min | drift | face | issues | video | report |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|---|")
            for f in s.failures:
                issues = "; ".join([str(x) for x in (f.get("issues") or [])])
                lines.append(
                    "| {scenario} | {shot} | {motion} | {avg:.3f} | {minv:.3f} | {drift:.3f} | {face:.3f} | {issues} | {video} | {report} |".format(
                        scenario=f.get("scenario", ""),
                        shot=f.get("shot_type", ""),
                        motion=f.get("motion_intensity", ""),
                        avg=float(f.get("avg", 0.0)),
                        minv=float(f.get("min", 0.0)),
                        drift=float(f.get("drift", 0.0)),
                        face=float(f.get("face", 0.0)),
                        issues=issues,
                        video=f.get("video", ""),
                        report=f.get("report", ""),
                    )
                )
            lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ report: {out_md}")


if __name__ == "__main__":
    main()


