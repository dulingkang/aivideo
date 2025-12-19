#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 Scene JSON -> pipeline 兼容层

目标：
- 保持 v2 的结构化字段不丢失（intent / visual_constraints / character / camera / quality_target）
- 同时补齐现有流水线（main.py / tts / video_generator）依赖的 v1 扁平字段：
  - scene_id -> id
  - narration.text -> narration（string）
  - duration_sec -> duration
  - target_fps -> fps
  - notes / visual_constraints.environment -> description / prompt（fallback）
  - camera.movement -> camera_motion.type（v1 风格）
  - intent.emotion / quality_target.lighting_style -> mood / lighting（v1 风格）
  - quality_target.motion_intensity -> motion_intensity（v1 风格）

该模块尽量设计为幂等：多次 normalize 不会破坏数据。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional


def _is_v2_scene(scene: Dict[str, Any]) -> bool:
    if not isinstance(scene, dict):
        return False
    if scene.get("version") == "v2":
        return True
    # v2 特征字段
    return any(
        k in scene for k in ("scene_id", "intent", "visual_constraints", "generation_policy", "quality_target")
    )


def is_v2_script(script_data: Dict[str, Any]) -> bool:
    scenes = (script_data or {}).get("scenes", [])
    if not scenes or not isinstance(scenes, list):
        return False
    first = scenes[0] if isinstance(scenes[0], dict) else {}
    return _is_v2_scene(first)


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except Exception:
            return None
    return None


def _map_motion_intensity(value: Any) -> str:
    v = (value or "").strip().lower() if isinstance(value, str) else ""
    # VideoGenerator 主要识别 gentle/moderate/dynamic；static 也能跑但意义不大
    if v in ("static", "still", "none"):
        return "gentle"
    if v in ("gentle", "low", "slow"):
        return "gentle"
    if v in ("moderate", "medium", "normal"):
        return "moderate"
    if v in ("dynamic", "high", "fast", "intense"):
        return "dynamic"
    return v or "moderate"


def _map_camera_motion_type(value: Any) -> str:
    v = (value or "").strip().lower() if isinstance(value, str) else ""
    if v in ("static", "still", ""):
        return "static"
    if v in ("pan",):
        return "pan"
    if v in ("zoom", "zoom_in", "zoom_out"):
        return "zoom"
    if v in ("dolly", "dolly_in", "dolly_out", "push_in", "pull_out"):
        return "dolly"
    # fallback：保持原样
    return v


def normalize_scene_for_pipeline(scene: Dict[str, Any], *, idx0: int) -> Tuple[Dict[str, Any], bool]:
    """
    将单个场景归一化为流水线可消费字段。

    Returns:
        (normalized_scene, changed)
    """
    if not isinstance(scene, dict):
        return scene, False

    changed = False
    out: Dict[str, Any] = dict(scene)  # shallow copy

    # --- id / scene_id ---
    if out.get("id") is None:
        sid = out.get("scene_id")
        if sid is not None:
            out["id"] = sid
            changed = True

    # --- fps ---
    if out.get("fps") is None and out.get("target_fps") is not None:
        out["fps"] = out.get("target_fps")
        changed = True

    # --- duration ---
    if out.get("duration") is None and out.get("duration_sec") is not None:
        out["duration"] = out.get("duration_sec")
        changed = True

    # --- narration: dict -> string ---
    narration = out.get("narration")
    if isinstance(narration, dict):
        # 保留元信息
        if out.get("narration_meta") is None:
            out["narration_meta"] = narration
            changed = True
        text = narration.get("text", "") if isinstance(narration.get("text"), str) else ""
        out["narration"] = text
        changed = True

    # --- v1 风格字段：mood / lighting / action / motion_intensity ---
    intent = out.get("intent") or {}
    quality = out.get("quality_target") or {}
    camera = out.get("camera") or {}
    vconstraints = out.get("visual_constraints") or {}

    if out.get("mood") is None and isinstance(intent, dict) and intent.get("emotion"):
        out["mood"] = intent.get("emotion")
        changed = True

    if out.get("lighting") is None and isinstance(quality, dict) and quality.get("lighting_style"):
        out["lighting"] = quality.get("lighting_style")
        changed = True

    if out.get("action") is None and isinstance(intent, dict) and intent.get("type"):
        out["action"] = intent.get("type")
        changed = True

    if out.get("motion_intensity") is None and isinstance(quality, dict) and quality.get("motion_intensity"):
        out["motion_intensity"] = _map_motion_intensity(quality.get("motion_intensity"))
        changed = True
    elif isinstance(out.get("motion_intensity"), str):
        mapped = _map_motion_intensity(out.get("motion_intensity"))
        if mapped != out.get("motion_intensity"):
            out["motion_intensity"] = mapped
            changed = True

    # --- camera_motion: dict ---
    if out.get("camera_motion") is None and isinstance(camera, dict) and camera.get("movement"):
        out["camera_motion"] = {"type": _map_camera_motion_type(camera.get("movement"))}
        changed = True
    elif isinstance(out.get("camera_motion"), dict):
        if out["camera_motion"].get("type") and isinstance(out["camera_motion"]["type"], str):
            mapped = _map_camera_motion_type(out["camera_motion"]["type"])
            if mapped != out["camera_motion"]["type"]:
                out["camera_motion"]["type"] = mapped
                changed = True

    # --- description / prompt fallback ---
    if not out.get("description"):
        notes = out.get("notes") if isinstance(out.get("notes"), str) else ""
        env = ""
        if isinstance(vconstraints, dict) and isinstance(vconstraints.get("environment"), str):
            env = vconstraints.get("environment") or ""
        desc = notes.strip() or env.strip()
        if desc:
            out["description"] = desc
            changed = True

    if not out.get("prompt"):
        # VideoGenerator 会用 prompt 或 description。这里提供一个更稳的 fallback。
        pieces = []
        if isinstance(vconstraints, dict):
            env = vconstraints.get("environment")
            if isinstance(env, str) and env.strip():
                pieces.append(env.strip())
        desc = out.get("description")
        if isinstance(desc, str) and desc.strip():
            pieces.append(desc.strip())
        if pieces:
            out["prompt"] = ", ".join(pieces)
            changed = True

    # --- scene_number：如果两边都没有，补一个（用于部分旧逻辑打印） ---
    if out.get("scene_number") is None:
        sid_int = _safe_int(out.get("id"))
        out["scene_number"] = (sid_int + 1) if sid_int is not None else (idx0 + 1)
        changed = True

    # --- visual：补齐 lighting/mood/environment（VideoGenerator prompt builder 会读 visual.*） ---
    visual = out.get("visual")
    if not isinstance(visual, dict):
        visual = {}
    visual_changed = False
    if isinstance(vconstraints, dict) and vconstraints.get("environment") and not visual.get("environment"):
        visual["environment"] = vconstraints.get("environment")
        visual_changed = True
    if out.get("lighting") and not visual.get("lighting"):
        visual["lighting"] = out.get("lighting")
        visual_changed = True
    if out.get("mood") and not visual.get("mood"):
        visual["mood"] = out.get("mood")
        visual_changed = True
    if visual_changed:
        out["visual"] = visual
        changed = True

    return out, changed


def normalize_script_for_pipeline(script_data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    归一化整个脚本（主要处理 scenes）。

    Returns:
        (normalized_script, changed)
    """
    if not isinstance(script_data, dict):
        return script_data, False

    scenes = script_data.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        return script_data, False

    if not any(_is_v2_scene(s) for s in scenes if isinstance(s, dict)):
        return script_data, False

    changed_any = False
    new_scenes = []
    for idx0, s in enumerate(scenes):
        if isinstance(s, dict):
            ns, changed = normalize_scene_for_pipeline(s, idx0=idx0)
            new_scenes.append(ns)
            changed_any = changed_any or changed
        else:
            new_scenes.append(s)

    if not changed_any:
        return script_data, False

    out = dict(script_data)
    out["scenes"] = new_scenes
    out["_pipeline_normalized_from_v2"] = True
    return out, True


