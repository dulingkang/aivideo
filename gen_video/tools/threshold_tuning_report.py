#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阈值微调统计（基于 summary_*.json 聚合）

输入：test_m6_scenarios.py 输出的 summary_*.json（每个条目含 avg/min/drift/face_detect 等指标）
输出：Markdown 报告 + 推荐 config.yaml 片段

说明：
- 这里只能基于已记录的指标做“离线阈值评估”，无法重算 drift_ratio（其依赖 drift_threshold）
- 默认使用 config.yaml 的 shot_type_tolerance 计算 adjusted_threshold = similarity_threshold - tolerance(shot_type)
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class ThresholdConfig:
    similarity_threshold: float
    max_drift_ratio: float
    min_face_detect_ratio: float
    min_similarity_floor: float


def _load_json_list(p: Path) -> List[Dict[str, Any]]:
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    raise ValueError(f"summary 不是 list: {p}")


def _infer_risk(scenario: str) -> str:
    if scenario.startswith("low_"):
        return "low"
    if scenario.startswith("mid_") or scenario.startswith("medium_"):
        return "medium"
    if scenario.startswith("high_"):
        return "high"
    return "unknown"


def _risk_weight(risk: str) -> float:
    return {"low": 1.0, "medium": 2.0, "high": 3.0}.get(risk, 1.0)


def _load_shot_tolerance(config_path: Optional[str]) -> Dict[str, float]:
    default = {"wide": 0.10, "medium": 0.05, "medium_close": 0.04, "close": 0.03, "extreme_close": 0.02}
    if not config_path or yaml is None:
        return default
    p = Path(config_path)
    if not p.exists():
        return default
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    tol = (((cfg.get("video") or {}).get("identity_verification") or {}).get("shot_type_tolerance") or {})
    if isinstance(tol, dict) and tol:
        merged = default.copy()
        for k, v in tol.items():
            try:
                merged[str(k)] = float(v)
            except Exception:
                pass
        return merged
    return default


def _evaluate_item(it: Dict[str, Any], tcfg: ThresholdConfig, shot_tol: Dict[str, float]) -> bool:
    shot_type = str(it.get("shot_type", "medium"))
    tol = float(shot_tol.get(shot_type, 0.05))
    adjusted = tcfg.similarity_threshold - tol
    avg_sim = float(it.get("avg_similarity", 0.0))
    min_sim = float(it.get("min_similarity", 0.0))
    drift = float(it.get("drift_ratio", 1.0))
    face = float(it.get("face_detect_ratio", 0.0))
    return (
        avg_sim >= adjusted
        and drift <= tcfg.max_drift_ratio
        and face >= tcfg.min_face_detect_ratio
        and min_sim >= tcfg.min_similarity_floor
    )


def _fail_reasons(it: Dict[str, Any], tcfg: ThresholdConfig, shot_tol: Dict[str, float]) -> List[str]:
    """给出该 item 在某个阈值组合下失败的原因（便于定点攻坚）。"""
    shot_type = str(it.get("shot_type", "medium"))
    tol = float(shot_tol.get(shot_type, 0.05))
    adjusted = tcfg.similarity_threshold - tol
    avg_sim = float(it.get("avg_similarity", 0.0))
    min_sim = float(it.get("min_similarity", 0.0))
    drift = float(it.get("drift_ratio", 1.0))
    face = float(it.get("face_detect_ratio", 0.0))

    reasons: List[str] = []
    if avg_sim < adjusted:
        reasons.append(f"avg<{adjusted:.2f}({avg_sim:.3f})")
    if drift > tcfg.max_drift_ratio:
        reasons.append(f"drift>{tcfg.max_drift_ratio:.2f}({drift:.3f})")
    if face < tcfg.min_face_detect_ratio:
        reasons.append(f"face<{tcfg.min_face_detect_ratio:.2f}({face:.3f})")
    if min_sim < tcfg.min_similarity_floor:
        reasons.append(f"min<{tcfg.min_similarity_floor:.2f}({min_sim:.3f})")
    return reasons


def _grid() -> List[ThresholdConfig]:
    sims = [round(x, 2) for x in [0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75]]
    drifts = [0.05, 0.10, 0.15]
    faces = [0.75, 0.80, 0.90]
    mins = [0.10, 0.20, 0.30]
    out: List[ThresholdConfig] = []
    for s in sims:
        for d in drifts:
            for f in faces:
                for m in mins:
                    out.append(ThresholdConfig(s, d, f, m))
    return out


def _format_cfg_snippet(tcfg: ThresholdConfig) -> str:
    return (
        "video:\n"
        "  identity_verification:\n"
        f"    similarity_threshold: {tcfg.similarity_threshold:.2f}\n"
        f"    max_drift_ratio: {tcfg.max_drift_ratio:.2f}\n"
        f"    min_face_detect_ratio: {tcfg.min_face_detect_ratio:.2f}\n"
        f"    min_similarity_floor: {tcfg.min_similarity_floor:.2f}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="M6 阈值微调统计（基于 summary JSON）")
    parser.add_argument("--inputs", nargs="*", default=None, help="summary_*.json 文件列表；不传则自动扫描 outputs/**/summary*.json")
    parser.add_argument("--outputs-dir", default="outputs", help="自动扫描的 outputs 目录（相对 gen_video）")
    parser.add_argument("--config", default="config.yaml", help="读取 shot_type_tolerance 的配置文件路径")
    parser.add_argument("--out-md", default="outputs/THRESHOLD_TUNING_REPORT.md", help="输出 Markdown 报告路径（相对 gen_video）")
    args = parser.parse_args()

    base_dir = Path.cwd()
    inputs: List[Path] = []
    if args.inputs:
        inputs = [Path(p) for p in args.inputs]
    else:
        inputs = sorted(Path(args.outputs_dir).glob("**/summary*.json"))

    inputs = [p for p in inputs if p.exists()]
    if not inputs:
        raise SystemExit("未找到 summary*.json，请先运行 test_m6_scenarios.py 生成回归数据")

    shot_tol = _load_shot_tolerance(args.config)

    # 收集样本
    items: List[Dict[str, Any]] = []
    for p in inputs:
        for it in _load_json_list(p):
            it = dict(it)
            it["_source"] = str(p)
            it["_risk"] = _infer_risk(str(it.get("scenario", "")))
            items.append(it)

    # 约束：极端崩坏样本不允许通过（防止阈值过松）
    catastrophic = [
        it for it in items
        if float(it.get("min_similarity", 0.0)) < 0.15 or float(it.get("drift_ratio", 0.0)) > 0.20
    ]

    best: List[Tuple[float, ThresholdConfig, Dict[str, Any]]] = []
    for tcfg in _grid():
        weighted_total = 0.0
        weighted_pass = 0.0
        by_risk = {"low": [0, 0], "medium": [0, 0], "high": [0, 0], "unknown": [0, 0]}

        ok_catastrophic = True
        for it in catastrophic:
            if _evaluate_item(it, tcfg, shot_tol):
                ok_catastrophic = False
                break
        if not ok_catastrophic:
            continue

        for it in items:
            risk = it["_risk"]
            w = _risk_weight(risk)
            weighted_total += w
            passed = _evaluate_item(it, tcfg, shot_tol)
            if passed:
                weighted_pass += w
                by_risk[risk][0] += 1
            by_risk[risk][1] += 1

        score = weighted_pass / max(weighted_total, 1e-9)
        meta = {"by_risk": by_risk, "inputs": len(inputs), "items": len(items), "catastrophic": len(catastrophic)}
        best.append((score, tcfg, meta))

    best.sort(key=lambda x: x[0], reverse=True)
    top = best[:5]
    if not top:
        raise SystemExit("没有找到满足约束的阈值组合（请放宽网格或约束）")

    # 输出报告
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# M6 阈值微调统计报告（离线评估）")
    lines.append("")
    lines.append(f"- summary 文件数: {len(inputs)}")
    lines.append(f"- 样本条目数: {len(items)}")
    lines.append(f"- 极端崩坏样本数(约束 must-fail): {len(catastrophic)}")
    lines.append("")
    lines.append("## Top 配置（按加权通过率）")
    lines.append("")
    lines.append("| rank | score | sim_th | max_drift | min_face | min_sim_floor | low | medium | high |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, (score, tcfg, meta) in enumerate(top, 1):
        by_risk = meta["by_risk"]
        def fmt(r: str) -> str:
            a, b = by_risk[r]
            return f"{a}/{b}"
        lines.append(
            f"| {i} | {score:.3f} | {tcfg.similarity_threshold:.2f} | {tcfg.max_drift_ratio:.2f} | {tcfg.min_face_detect_ratio:.2f} | {tcfg.min_similarity_floor:.2f} | {fmt('low')} | {fmt('medium')} | {fmt('high')} |"
        )

    best_cfg = top[0][1]
    lines.append("")
    lines.append("## 推荐配置片段（写入 config.yaml）")
    lines.append("")
    lines.append("```yaml")
    lines.append(_format_cfg_snippet(best_cfg).rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## Best 配置下失败样本（便于定点攻坚）")
    lines.append("")
    failures: List[Dict[str, Any]] = []
    for it in items:
        if not _evaluate_item(it, best_cfg, shot_tol):
            failures.append(
                {
                    "scenario": str(it.get("scenario", "")),
                    "risk": str(it.get("_risk", "unknown")),
                    "shot_type": str(it.get("shot_type", "medium")),
                    "avg": float(it.get("avg_similarity", 0.0)),
                    "min": float(it.get("min_similarity", 0.0)),
                    "drift": float(it.get("drift_ratio", 1.0)),
                    "face": float(it.get("face_detect_ratio", 0.0)),
                    "reasons": ", ".join(_fail_reasons(it, best_cfg, shot_tol)),
                    "source": Path(str(it.get("_source", ""))).name,
                }
            )
    if not failures:
        lines.append("- ✅ 全部通过（在离线阈值判定下）")
    else:
        # 失败按风险分层 + 场景名排序
        failures.sort(key=lambda x: ({"high": 0, "medium": 1, "low": 2}.get(x["risk"], 9), x["scenario"]))
        lines.append("| scenario | risk | shot | avg | min | drift | face | reasons | source |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
        for f in failures[:50]:
            lines.append(
                f"| {f['scenario']} | {f['risk']} | {f['shot_type']} | {f['avg']:.3f} | {f['min']:.3f} | {f['drift']:.3f} | {f['face']:.3f} | {f['reasons']} | {f['source']} |"
            )
        if len(failures) > 50:
            lines.append("")
            lines.append(f"- … 省略其余 {len(failures) - 50} 条失败样本（可按需提高上限）")
    lines.append("")
    lines.append("## 说明")
    lines.append("- 本报告基于 summary 中记录的 avg/min/drift/face_detect 指标做离线阈值评估。")
    lines.append("- drift_ratio 的计算依赖 drift_threshold；若你调整 drift_threshold，需要重新跑回归生成新的 summary 才能准确评估。")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ report: {out_md}")


if __name__ == "__main__":
    main()


