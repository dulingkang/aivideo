#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行“严格阈值回归（非 quick）”：
- 复用既有 high_risk 非 quick 汇总（默认：outputs/m6_high_risk_nonquick_full 下最新 summary_full_*）
- 额外跑一套 battle_occlusion（战斗/遮挡强化）非 quick
- 生成 base_dir/REGRESSION_REPORT.md 总览

用法示例（推荐在 gen_video 目录执行）：
  python tools/run_m6_strict_threshold_regression.py

也可指定：
  python tools/run_m6_strict_threshold_regression.py \
    --config config.yaml \
    --reuse-high-risk-dir outputs/m6_high_risk_nonquick_full \
    --out-base outputs/m6_strict_threshold_nonquick_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def _latest_pair(dir_path: Path) -> Tuple[Path, Path]:
    """
    在某个目录中寻找最新的 (summary_*.json, summary_*.md)。
    兼容 summary_full_*.json / summary_full_*.md。
    """
    js = sorted(dir_path.glob("summary_*.json"))
    md = sorted(dir_path.glob("summary_*.md"))
    if not js or not md:
        raise FileNotFoundError(f"未找到 summary_*.json/.md: {dir_path}")
    # 以文件名排序的最后一个通常是最新时间戳
    return js[-1], md[-1]


def _copy_pair(src_json: Path, src_md: Path, dst_dir: Path) -> Tuple[Path, Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_json = dst_dir / src_json.name
    dst_md = dst_dir / src_md.name
    shutil.copy2(src_json, dst_json)
    shutil.copy2(src_md, dst_md)
    return dst_json, dst_md


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ (cd {cwd} && {' '.join(cmd)})")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="M6 严格阈值回归（非 quick）一键脚本")
    parser.add_argument("--config", default="config.yaml", help="gen_video/config.yaml 路径（相对 gen_video）")
    parser.add_argument(
        "--reuse-high-risk-dir",
        default="outputs/m6_high_risk_nonquick_full",
        help="复用 high_risk 非 quick 的 summary 目录（相对 gen_video）",
    )
    parser.add_argument(
        "--out-base",
        default=None,
        help="输出根目录（相对 gen_video）。默认 outputs/m6_strict_threshold_nonquick_<ts>",
    )
    parser.add_argument(
        "--run-high-risk",
        action="store_true",
        help="不复用，直接重新跑 high_risk 非 quick（耗时更长）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="覆盖最大重试次数（不传则使用 config.yaml）",
    )
    args = parser.parse_args()

    gen_video_dir = Path(__file__).resolve().parents[1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.out_base) if args.out_base else Path(f"outputs/m6_strict_threshold_nonquick_{ts}")
    out_base = (gen_video_dir / out_base).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    # 固化本次使用的配置快照，便于追溯阈值（尤其是 max_drift_ratio=0.05）
    cfg_path = (gen_video_dir / args.config).resolve()
    if not cfg_path.exists():
        raise SystemExit(f"config 不存在: {cfg_path}")
    shutil.copy2(cfg_path, out_base / "config_snapshot.yaml")

    # 1) high_risk：复用或重跑
    high_dir = out_base / "high_nonquick"
    if args.run_high_risk:
        cmd = [
            sys.executable,
            "test_m6_scenarios.py",
            "--config",
            str(cfg_path),
            "--suite",
            "high_risk",
            "--output-dir",
            str(high_dir),
        ]
        if args.max_retries is not None:
            cmd += ["--max-retries", str(args.max_retries)]
        _run(cmd, cwd=gen_video_dir)
    else:
        reuse_dir = (gen_video_dir / args.reuse_high_risk_dir).resolve()
        src_json, src_md = _latest_pair(reuse_dir)
        _copy_pair(src_json, src_md, high_dir)
        print(f"✅ 复用 high_risk summary: {src_json.name} / {src_md.name}")

    # 2) battle_occlusion：非 quick 生成 + 验证
    battle_dir = out_base / "battle_occlusion_nonquick"
    cmd2 = [
        sys.executable,
        "test_m6_scenarios.py",
        "--config",
        str(cfg_path),
        "--suite",
        "battle_occlusion",
        "--output-dir",
        str(battle_dir),
    ]
    if args.max_retries is not None:
        cmd2 += ["--max-retries", str(args.max_retries)]
    _run(cmd2, cwd=gen_video_dir)

    # 3) 生成总览报告
    report_path = out_base / "REGRESSION_REPORT.md"
    cmd3 = [
        sys.executable,
        "tools/regression_report.py",
        "--base-dir",
        str(out_base),
        "--out-md",
        str(report_path),
    ]
    _run(cmd3, cwd=gen_video_dir)

    print("\n✅ strict-threshold regression done")
    print(f"- base_dir: {out_base}")
    print(f"- report:   {report_path}")


if __name__ == "__main__":
    main()


