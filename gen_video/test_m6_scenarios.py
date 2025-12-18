#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M6 批量场景测试脚本

目标：
1) 在同一张 Anchor 图 / 不同场景 prompt 下，验证 HunyuanVideo 1.5 I2V 的身份保持稳定性
2) 覆盖不同 shot_type / motion_intensity 的组合，快速发现漂移边界
3) 产出视频 + 身份报告 + 汇总 JSON/Markdown，便于后续阈值微调与回归
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# 确保可从 gen_video 目录导入
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_video_generator_m6 import EnhancedVideoGeneratorM6


logger = logging.getLogger(__name__)


def _default_assets() -> Dict[str, str]:
    """返回默认素材路径（相对 gen_video 目录）。"""
    return {
        "hanli_mid": "reference_image/hanli_mid.jpg",
        "hanli_front_neutral": "character_profiles/hanli/front/neutral.jpg",
        "hanli_front_happy": "character_profiles/hanli/front/happy.jpg",
        "hanli_three_quarter_neutral": "character_profiles/hanli/three_quarter/neutral.jpg",
    }


def _default_scenarios() -> List[Dict[str, Any]]:
    """默认场景集合（可按需扩展）。"""
    return [
        {
            "name": "garden_static",
            "risk": "low",
            "prompt": "Han Li standing in a mystical garden, minimal slow movement, cinematic lighting, high quality",
            "motion_intensity": "gentle",
            "shot_type": "medium",
        },
        {
            "name": "bamboo_walk",
            "risk": "low",
            "prompt": "Han Li walking slowly through a bamboo forest, smooth natural movement, soft wind moving hair, cinematic, high quality",
            "motion_intensity": "moderate",
            "shot_type": "medium",
        },
        {
            "name": "training_qigong",
            "risk": "low",
            "prompt": "Han Li practicing qigong cultivation, subtle aura flow around him, calm breathing, stable camera, cinematic, high quality",
            "motion_intensity": "gentle",
            "shot_type": "medium_close",
        },
        {
            "name": "battle_wide",
            "risk": "high",
            "prompt": "Han Li performing sword technique, dynamic action, flowing robes, sparks and dust in the air, cinematic, high quality",
            "motion_intensity": "dynamic",
            "shot_type": "wide",
        },
    ]


def _low_risk_suite() -> List[Dict[str, Any]]:
    """
    低风险回归集：
    - 镜头：wide/medium/medium_close 为主，避免 close/extreme_close
    - 运动：gentle/moderate，强调 steady/static camera
    - 目的：高通过率、稳定回归、快速发现系统性回归问题
    """
    return [
        {
            "name": "low_portrait_intro",
            "risk": "low",
            "prompt": "Han Li introducing himself calmly, medium shot, clear face, steady camera, cinematic lighting, high quality",
            "motion_intensity": "gentle",
            "shot_type": "medium",
        },
        {
            "name": "low_dialogue_teahouse",
            "risk": "low",
            "prompt": "Han Li talking quietly in an ancient teahouse, subtle head movement, stable camera, cinematic, high quality",
            "motion_intensity": "gentle",
            "shot_type": "medium",
        },
        {
            "name": "low_training_meditation",
            "risk": "low",
            "prompt": "Han Li meditating in a quiet cave, subtle breathing motion, calm atmosphere, stable camera, high quality",
            "motion_intensity": "gentle",
            "shot_type": "medium",
        },
        {
            "name": "low_walk_corridor",
            "risk": "low",
            "prompt": "Han Li walking slowly along an ancient corridor, smooth natural movement, steady camera, cinematic, high quality",
            "motion_intensity": "moderate",
            "shot_type": "medium",
        },
        {
            "name": "low_establishing_mountain",
            "risk": "low",
            "prompt": "Wide establishing shot of Han Li standing on a mountain ridge, gentle wind, stable camera, cinematic, high quality",
            "motion_intensity": "gentle",
            "shot_type": "wide",
        },
        {
            "name": "low_bamboo_garden",
            "risk": "low",
            "prompt": "Han Li standing in a bamboo garden, gentle breeze, minimal slow movement, stable camera, cinematic, high quality",
            "motion_intensity": "gentle",
            "shot_type": "medium",
        },
    ]

def _medium_risk_suite() -> List[Dict[str, Any]]:
    """
    中风险回归集（走路/转身）：
    - 运动更明显：walking / turning / looking back
    - 仍保持安全镜头与稳定相机：medium/wide 为主
    - 目的：覆盖“人体姿态变化/转头转身”导致的身份漂移边界
    """
    return [
        {
            "name": "mid_walk_street_turn_head",
            "risk": "medium",
            "prompt": "Han Li walking slowly on an ancient street, then turning his head slightly to look back, smooth motion, steady camera, cinematic, high quality",
            "motion_intensity": "moderate",
            "shot_type": "medium",
        },
        {
            "name": "mid_walk_garden_half_turn",
            "risk": "medium",
            "prompt": "Han Li walking in a garden, then making a gentle half turn of his body, natural movement, steady camera, cinematic lighting, high quality",
            "motion_intensity": "moderate",
            "shot_type": "medium",
        },
        {
            "name": "mid_corridor_turn_around",
            "risk": "medium",
            "prompt": "Han Li walking along an ancient corridor, then slowly turning around to face the camera, smooth natural movement, stable camera, cinematic, high quality",
            "motion_intensity": "moderate",
            "shot_type": "medium",
        },
        {
            "name": "mid_bamboo_walk_look_back",
            "risk": "medium",
            "prompt": "Han Li walking through a bamboo forest, gentle wind, then looking back over his shoulder briefly, smooth motion, steady camera, high quality",
            "motion_intensity": "moderate",
            "shot_type": "medium",
        },
        {
            "name": "mid_market_walk_stop_turn",
            "risk": "medium",
            "prompt": "Han Li walking in an ancient market scene, then stopping and turning slightly, subtle facial expression, stable camera, cinematic, high quality",
            "motion_intensity": "moderate",
            "shot_type": "medium_close",
        },
        {
            "name": "mid_wide_walk_turn_body",
            "risk": "medium",
            "prompt": "Wide shot of Han Li walking across an open courtyard, then turning his body slowly, smooth motion, steady camera, cinematic, high quality",
            "motion_intensity": "moderate",
            "shot_type": "wide",
        },
    ]

def _high_risk_suite() -> List[Dict[str, Any]]:
    """
    高风险回归集（非 quick 建议）：
    - 更强的运动/姿态变化/遮挡/转身幅度：最容易触发尾段崩脸
    - 包含 close / extreme_close（高风险镜头）
    - 目的：验证分层调参+重试策略是否能稳定“救回来”
    """
    return [
        {
            "name": "high_turn_over_shoulder_close",
            "risk": "high",
            "prompt": "Close-up of Han Li walking forward, then turning over his shoulder to look back, noticeable head turn, cinematic, high quality",
            "motion_intensity": "dynamic",
            "shot_type": "close",
        },
        {
            "name": "high_turn_around_face_camera",
            "risk": "high",
            "prompt": "Han Li walking, then turning around to face the camera, clear face, noticeable body turn, cinematic lighting, high quality",
            "motion_intensity": "dynamic",
            "shot_type": "medium_close",
        },
        {
            "name": "high_occlusion_hand_adjust_hair",
            "risk": "high",
            "prompt": "Han Li adjusting his hair with his hand, brief face occlusion, then face clearly visible again, stable camera, cinematic, high quality",
            "motion_intensity": "moderate",
            "shot_type": "medium_close",
        },
        {
            "name": "high_fast_walk_wide",
            "risk": "high",
            "prompt": "Wide shot of Han Li walking faster across an open courtyard, strong motion, flowing robes, dust in the air, cinematic, high quality",
            "motion_intensity": "dynamic",
            "shot_type": "wide",
        },
    ]


def _resolve_scenarios(suite: str) -> List[Dict[str, Any]]:
    if suite == "low_risk":
        return _low_risk_suite()
    if suite == "medium_risk":
        return _medium_risk_suite()
    if suite == "high_risk":
        return _high_risk_suite()
    # default：保持向后兼容
    return _default_scenarios()


def _ensure_exists(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path_str}")
    return str(p)


def _write_markdown_summary(summary_path: Path, items: List[Dict[str, Any]]):
    lines = []
    lines.append("# M6 场景批量测试汇总")
    lines.append("")
    lines.append(f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 条目数: {len(items)}")
    lines.append("")
    lines.append("| 场景 | 输入图 | 镜头 | 运动 | 通过 | 平均相似度 | 漂移% | 人脸检测% | 视频 | 报告 |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---|---|")
    for it in items:
        lines.append(
            "| {name} | {input_image} | {shot_type} | {motion_intensity} | {passed} | {avg_similarity:.3f} | {drift_pct:.1f} | {face_pct:.1f} | {video_rel} | {report_rel} |".format(
                name=it.get("scenario"),
                input_image=Path(it.get("input_image", "")).name,
                shot_type=it.get("shot_type"),
                motion_intensity=it.get("motion_intensity"),
                passed="✅" if it.get("passed") else "❌",
                avg_similarity=float(it.get("avg_similarity", 0.0)),
                drift_pct=float(it.get("drift_ratio", 0.0)) * 100.0,
                face_pct=float(it.get("face_detect_ratio", 0.0)) * 100.0,
                video_rel=it.get("video_path", ""),
                report_rel=it.get("report_path", ""),
            )
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="M6 批量场景测试（HunyuanVideo + 身份验证）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--output-dir", default="outputs/m6_scenarios", help="输出目录")
    parser.add_argument(
        "--suite",
        default="default",
        choices=["default", "low_risk", "medium_risk", "high_risk"],
        help="场景套件：default=原有集合，low_risk=低风险回归集，medium_risk=走路/转身中风险回归集，high_risk=高风险（遮挡/大幅转身/close）",
    )
    parser.add_argument("--scenarios", nargs="*", default=None, help="仅运行指定场景名（不传则跑默认全部）")

    # 输入/参考图
    parser.add_argument("--input-image", default=None, help="Anchor 输入图路径（默认 hanli_mid）")
    parser.add_argument("--reference-image", default=None, help="参考图路径（用于身份验证，默认同 input-image）")

    # 快速模式：尽快筛查能否跑通 + 是否漂移
    parser.add_argument("--quick", action="store_true", help="快速模式：减少步数/帧数（适合批量冒烟测试）")
    parser.add_argument("--num-frames", type=int, default=None, help="覆盖 HunyuanVideo 帧数")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="覆盖 HunyuanVideo 推理步数")
    parser.add_argument("--max-retries", type=int, default=None, help="覆盖最大重试次数（验证失败重试）")

    # 其它覆盖参数
    parser.add_argument("--model-path", default=None, help="覆盖 HunyuanVideo 模型路径")
    parser.add_argument("--width", type=int, default=None, help="覆盖分辨率宽度")
    parser.add_argument("--height", type=int, default=None, help="覆盖分辨率高度")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 选择输入/参考图
    assets = _default_assets()
    input_image = args.input_image or assets["hanli_mid"]
    reference_image = args.reference_image or input_image
    _ensure_exists(input_image)
    _ensure_exists(reference_image)

    # 选择场景集合
    scenarios = _resolve_scenarios(args.suite)
    if args.scenarios:
        wanted = set(args.scenarios)
        scenarios = [s for s in scenarios if s["name"] in wanted]
        missing = wanted - set([s["name"] for s in scenarios])
        if missing:
            raise ValueError(f"未找到场景: {sorted(missing)}")

    if not scenarios:
        raise ValueError("没有可运行的场景")

    logger.info("输入图: %s", input_image)
    logger.info("参考图: %s", reference_image)
    logger.info("输出目录: %s", str(out_dir))
    logger.info("场景数: %d", len(scenarios))

    # 初始化生成器（模型会在第一次 generate 时加载）
    generator = EnhancedVideoGeneratorM6(args.config)

    # 统一覆盖 HunyuanVideo 参数（用于批量测试）
    generator.video_config.setdefault("hunyuanvideo", {})
    hv = generator.video_config["hunyuanvideo"]
    if args.model_path:
        hv["model_path"] = args.model_path

    # quick 默认值（可被显式参数覆盖）
    if args.quick:
        hv["num_frames"] = int(args.num_frames or 24)
        hv["num_inference_steps"] = int(args.num_inference_steps or 8)
    else:
        if args.num_frames is not None:
            hv["num_frames"] = int(args.num_frames)
        if args.num_inference_steps is not None:
            hv["num_inference_steps"] = int(args.num_inference_steps)

    if args.width is not None:
        hv["width"] = int(args.width)
    if args.height is not None:
        hv["height"] = int(args.height)

    # 覆盖验证重试次数
    max_retries = args.max_retries
    if max_retries is None and args.quick:
        max_retries = 0

    items: List[Dict[str, Any]] = []

    try:
        for s in scenarios:
            name = s["name"]
            shot_type = s["shot_type"]
            motion_intensity = s["motion_intensity"]
            prompt = s["prompt"]
            negative_prompt = s.get("negative_prompt", "")

            video_path = out_dir / f"{name}_{timestamp}.mp4"
            logger.info("运行场景: %s (shot=%s, motion=%s)", name, shot_type, motion_intensity)

            scene_config = {
                "prompt": prompt,
                "description": name,
                "motion_intensity": motion_intensity,
                "negative_prompt": negative_prompt,
            }

            vp, result = generator.generate_video_with_identity_check(
                image_path=input_image,
                output_path=str(video_path),
                reference_image=reference_image,
                scene=scene_config,
                shot_type=shot_type,
                enable_verification=True,
                max_retries=max_retries,
            )

            report_path = out_dir / f"{name}_{timestamp}.json"
            payload: Dict[str, Any] = {
                "scenario": name,
                "shot_type": shot_type,
                "motion_intensity": motion_intensity,
                "input_image": input_image,
                "reference_image": reference_image,
                "video_path": str(vp) if vp else "",
                "report_path": str(report_path),
                "passed": False,
                "avg_similarity": 0.0,
                "min_similarity": 0.0,
                "drift_ratio": 1.0,
                "face_detect_ratio": 0.0,
                "issues": [],
            }

            if result is not None:
                payload.update(
                    {
                        "passed": bool(result.passed),
                        "avg_similarity": float(result.avg_similarity),
                        "min_similarity": float(result.min_similarity),
                        "drift_ratio": float(result.drift_ratio),
                        "face_detect_ratio": float(result.face_detect_ratio),
                        "issues": list(result.issues or []),
                    }
                )
            else:
                payload["issues"] = ["无验证结果（result=None）"]

            report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            items.append(payload)

        summary_json = out_dir / f"summary_{timestamp}.json"
        summary_md = out_dir / f"summary_{timestamp}.md"
        summary_json.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_markdown_summary(summary_md, items)

        passed_cnt = sum(1 for it in items if it.get("passed"))
        logger.info("完成：通过 %d/%d", passed_cnt, len(items))
        logger.info("汇总 JSON: %s", str(summary_json))
        logger.info("汇总 MD: %s", str(summary_md))

    finally:
        generator.unload_all()


if __name__ == "__main__":
    main()


