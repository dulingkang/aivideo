#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M6 “全套回归”一键脚本：
- 自动创建时间戳输出目录（默认 outputs/m6_regression_full_<ts>）
- 依次运行：
  - low_quick / low_semiquick
  - medium_quick / medium_semiquick
  - high_nonquick
- 自动生成 REGRESSION_REPORT.md
- 若有失败，自动将失败样本（mp4/json）拷贝到 _failures/<suite>/ 下，便于定点攻坚

注意：
- 建议先激活项目 venv（例如 /vepfs-dev/shawn/venv/py312），否则可能缺少 torch/diffusers 等依赖。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SuiteRun:
    name: str
    suite: str
    quick: bool
    num_frames: Optional[int]
    num_inference_steps: Optional[int]
    max_retries: Optional[int]


def _run(cmd: List[str], cwd: Path) -> None:
    print(f"\n$ (cd {cwd} && {' '.join(cmd)})")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _latest_summary_pair(dir_path: Path) -> Tuple[Path, Path]:
    js = sorted(dir_path.glob("summary_*.json"))
    md = sorted(dir_path.glob("summary_*.md"))
    if not js or not md:
        raise FileNotFoundError(f"未找到 summary_*.json/.md: {dir_path}")
    return js[-1], md[-1]


def _load_list(p: Path) -> List[Dict[str, Any]]:
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"summary 不是 list: {p}")
    return data


def _extract_failures(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        if it.get("passed"):
            continue
        out.append(it)
    return out


def _copy_failure_artifacts(base_dir: Path, suite_name: str, failures: List[Dict[str, Any]]) -> Optional[Path]:
    if not failures:
        return None
    out_dir = base_dir / "_failures" / suite_name
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for f in failures:
        vp = f.get("video_path") or ""
        rp = f.get("report_path") or ""
        for src in [vp, rp]:
            if not src:
                continue
            sp = Path(str(src))
            if not sp.exists():
                continue
            dst = out_dir / sp.name
            try:
                shutil.copy2(sp, dst)
                copied.append(dst.name)
            except Exception:
                pass

    # 写一个简短索引
    index = out_dir / "FAILURES.md"
    lines = []
    lines.append(f"# Failures: {suite_name}")
    lines.append("")
    lines.append(f"- count: {len(failures)}")
    lines.append("")
    lines.append("| scenario | shot | motion | avg | min | drift | face | issues | video | report |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for it in failures:
        issues = "; ".join([str(x) for x in (it.get("issues") or [])])
        lines.append(
            "| {scenario} | {shot} | {motion} | {avg:.3f} | {minv:.3f} | {drift:.3f} | {face:.3f} | {issues} | {video} | {report} |".format(
                scenario=it.get("scenario", ""),
                shot=it.get("shot_type", ""),
                motion=it.get("motion_intensity", ""),
                avg=float(it.get("avg_similarity", 0.0)),
                minv=float(it.get("min_similarity", 0.0)),
                drift=float(it.get("drift_ratio", 0.0)),
                face=float(it.get("face_detect_ratio", 0.0)),
                issues=issues,
                video=Path(str(it.get("video_path", ""))).name,
                report=Path(str(it.get("report_path", ""))).name,
            )
        )
    lines.append("")
    lines.append("## Copied artifacts")
    lines.append("")
    for name in sorted(set(copied)):
        lines.append(f"- {name}")
    index.write_text("\n".join(lines), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="M6 全套回归一键脚本（生成+汇总）")
    parser.add_argument("--config", default="config.yaml", help="gen_video/config.yaml 路径（相对 gen_video）")
    parser.add_argument("--out-base", default=None, help="输出根目录（相对 gen_video）")
    parser.add_argument("--input-image", default=None, help="Anchor 输入图（默认脚本内置 hanli_mid）")
    parser.add_argument("--reference-image", default=None, help="参考图（默认同 input-image）")

    # 套件参数（默认复刻之前目录命名：quick / semiquick / nonquick）
    parser.add_argument("--quick-frames", type=int, default=24)
    parser.add_argument("--quick-steps", type=int, default=8)
    parser.add_argument("--semi-frames", type=int, default=24)
    parser.add_argument("--semi-steps", type=int, default=15)
    parser.add_argument("--nonquick-frames", type=int, default=24)
    parser.add_argument("--nonquick-steps", type=int, default=25)

    # 重试：quick 默认 0（冒烟），其它默认 None（使用 config.yaml）
    parser.add_argument("--quick-max-retries", type=int, default=0)
    parser.add_argument("--semi-max-retries", type=int, default=None)
    parser.add_argument("--nonquick-max-retries", type=int, default=None)

    # 可选：加入 battle_occlusion（严格阈值回归里的战斗/遮挡集）
    parser.add_argument("--include-battle-occlusion", action="store_true")
    args = parser.parse_args()

    gen_video_dir = Path(__file__).resolve().parents[1]
    cfg_path = (gen_video_dir / args.config).resolve()
    if not cfg_path.exists():
        raise SystemExit(f"config 不存在: {cfg_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.out_base) if args.out_base else Path(f"outputs/m6_regression_full_{ts}")
    out_base = (gen_video_dir / out_base).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    # config 快照，确保阈值/重试策略可追溯
    shutil.copy2(cfg_path, out_base / "config_snapshot.yaml")

    runs: List[SuiteRun] = [
        SuiteRun(
            name="low_quick",
            suite="low_risk",
            quick=True,
            num_frames=args.quick_frames,
            num_inference_steps=args.quick_steps,
            max_retries=args.quick_max_retries,
        ),
        SuiteRun(
            name="medium_quick",
            suite="medium_risk",
            quick=True,
            num_frames=args.quick_frames,
            num_inference_steps=args.quick_steps,
            max_retries=args.quick_max_retries,
        ),
        SuiteRun(
            name="low_semiquick",
            suite="low_risk",
            quick=False,
            num_frames=args.semi_frames,
            num_inference_steps=args.semi_steps,
            max_retries=args.semi_max_retries,
        ),
        SuiteRun(
            name="medium_semiquick",
            suite="medium_risk",
            quick=False,
            num_frames=args.semi_frames,
            num_inference_steps=args.semi_steps,
            max_retries=args.semi_max_retries,
        ),
        SuiteRun(
            name="high_nonquick",
            suite="high_risk",
            quick=False,
            num_frames=args.nonquick_frames,
            num_inference_steps=args.nonquick_steps,
            max_retries=args.nonquick_max_retries,
        ),
    ]

    if args.include_battle_occlusion:
        runs.append(
            SuiteRun(
                name="battle_occlusion_nonquick",
                suite="battle_occlusion",
                quick=False,
                num_frames=args.nonquick_frames,
                num_inference_steps=args.nonquick_steps,
                max_retries=args.nonquick_max_retries,
            )
        )

    # 逐套件运行
    for r in runs:
        out_dir = out_base / r.name
        cmd = [
            sys.executable,
            "test_m6_scenarios.py",
            "--config",
            str(cfg_path),
            "--suite",
            r.suite,
            "--output-dir",
            str(out_dir),
        ]

        if args.input_image:
            cmd += ["--input-image", args.input_image]
        if args.reference_image:
            cmd += ["--reference-image", args.reference_image]

        if r.quick:
            cmd += ["--quick"]
            # quick 参数可覆盖（脚本 quick 默认 24/8，但这里显式传入更明确）
            if r.num_frames is not None:
                cmd += ["--num-frames", str(r.num_frames)]
            if r.num_inference_steps is not None:
                cmd += ["--num-inference-steps", str(r.num_inference_steps)]
        else:
            if r.num_frames is not None:
                cmd += ["--num-frames", str(r.num_frames)]
            if r.num_inference_steps is not None:
                cmd += ["--num-inference-steps", str(r.num_inference_steps)]

        if r.max_retries is not None:
            cmd += ["--max-retries", str(r.max_retries)]

        _run(cmd, cwd=gen_video_dir)

        # 失败样本拎出
        sj, _ = _latest_summary_pair(out_dir)
        items = _load_list(sj)
        failures = _extract_failures(items)
        fdir = _copy_failure_artifacts(out_base, r.name, failures)
        if fdir:
            print(f"⚠️  failures extracted: {fdir}")

    # 总览报告
    report = out_base / "REGRESSION_REPORT.md"
    _run(
        [
            sys.executable,
            "tools/regression_report.py",
            "--base-dir",
            str(out_base),
            "--out-md",
            str(report),
        ],
        cwd=gen_video_dir,
    )

    # 记录 latest 指针（文本 + 软链接）
    (out_base.parent / "m6_regression_full_latest.txt").write_text(str(out_base), encoding="utf-8")
    latest_link = out_base.parent / "m6_regression_full_latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(out_base.name)
    except Exception:
        pass

    print("\n✅ full regression done")
    print(f"- base_dir: {out_base}")
    print(f"- report:   {report}")


if __name__ == "__main__":
    main()


