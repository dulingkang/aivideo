#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V1 到 v2.2-final 格式转换器

直接从 V1 格式转换为 v2.2-final 格式，无需中间步骤
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# 复用 V1→V2 的映射函数
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lingjie"))
from convert_scene_v1_to_v2 import (
    map_scene_role, map_scene_type, map_emotion, map_tension_level,
    map_intent_type, map_time_of_day, map_lighting_style,
    map_camera_shot, map_camera_angle, map_camera_movement,
    has_hanli, map_visibility, map_body_coverage, infer_face_visible,
    map_motion_intensity
)

# 延迟导入，避免循环依赖
def _get_execution_rules():
    """延迟导入执行规则"""
    sys.path.insert(0, str(Path(__file__).parent))
    from execution_rules_v2_1 import ShotType, PoseType, get_execution_rules
    return ShotType, PoseType, get_execution_rules()


def _convert_pose_type(v1_action: str, character_pose: str) -> str:
    """从 V1 的 action 和 character_pose 转换为 v2.2-final 的 pose 类型"""
    # 优先使用 character_pose
    pose_text = (character_pose or "").lower()
    action_text = (v1_action or "").lower()
    
    # 映射表
    pose_map = {
        "lying": "lying",
        "lying_motionless": "lying",
        "motionless": "lying",
        "sitting": "sit",
        "sit": "sit",
        "standing": "stand",
        "stand": "stand",
        "walking": "walk",
        "walk": "walk",
        "kneeling": "kneel",
        "kneel": "kneel",
        "recalling": "stand",
        "enduring_pain": "stand",
    }
    
    # 检查 character_pose
    for key, value in pose_map.items():
        if key in pose_text:
            return value
    
    # 检查 action
    for key, value in pose_map.items():
        if key in action_text:
            return value
    
    # 默认
    return "stand"


def _get_pose_description(pose: str) -> str:
    """获取 pose 描述"""
    descriptions = {
        "lying": "lying on the ground, motionless",
        "sit": "sitting, seated",
        "stand": "standing pose, upright posture",
        "walk": "walking, in motion",
        "kneel": "kneeling, on knees",
        "face_only": "close-up face, portrait, face only",
    }
    return descriptions.get(pose.lower(), "standing pose")


def _build_character_config(v1_scene: Dict[str, Any], present: bool) -> Dict[str, Any]:
    """从 V1 场景构建 character 配置"""
    if not present:
        return {
            "present": False
        }
    
    # 韩立的配置
    return {
        "id": "hanli",
        "name": "韩立",
        "present": True,
        "role": "main",
        
        "identity": {
            "gender": "male",
            "age_range": "young_adult",
            "physique": "lean",
            "face_shape": "sharp"
        },
        
        "lora_config": {
            "type": "single",
            "lora_path": "models/lora/hanli/pytorch_lora_weights.safetensors",
            "weight": 0.9,
            "trigger": "hanli"
        },
        
        "anchor_patches": {
            "temperament_anchor": "calm and restrained temperament, sharp but composed eyes, determined expression",
            "explicit_lock_words": "wearing his iconic mid-late-stage green daoist robe, traditional Chinese cultivation attire",
            "face_detailer": {
                "enable": True,
                "trigger": "shot_scale >= medium",
                "denoise": 0.35,
                "steps": 12
            }
        },
        
        "reference_image": "character_references/hanli_reference.jpg",
        "negative_gender_lock": [
            "female", "woman", "girl",
            "soft facial features", "delicate face",
            "long eyelashes", "narrow shoulders",
            "slim waist", "feminine body"
        ]
    }


def _build_environment(v1_scene: Dict[str, Any], visual: Dict[str, Any], mood: str, lighting: str) -> Dict[str, Any]:
    """从 V1 场景构建 environment 配置"""
    # 提取 location
    environment = visual.get("environment", "").strip()
    composition = visual.get("composition", "").strip()
    description = v1_scene.get("description", "").strip()
    
    # 如果 environment 为空，从其他字段提取
    if not environment:
        # 从 composition 提取
        if composition:
            environment = composition
        # 从 description 提取
        elif description:
            # 简单提取：查找常见的位置关键词
            if "沙地" in description or "沙漠" in description:
                environment = "desert wasteland, gray-green sand"
            elif "天空" in description or "仙域" in description:
                environment = "Sky of the immortal realm, mist-wreathed"
            else:
                environment = description
    
    # 构建 atmosphere
    atmosphere_parts = []
    
    # 从 mood 提取情绪
    emotion = map_emotion(mood)
    if emotion:
        atmosphere_parts.append(f"{emotion} mood")
    
    # 从 lighting 提取风格
    lighting_style = map_lighting_style(lighting)
    if lighting_style:
        atmosphere_parts.append(f"{lighting_style} lighting")
    
    # 添加风格
    atmosphere_parts.append("xianxia_anime style")
    
    atmosphere = ", ".join(atmosphere_parts) if atmosphere_parts else "serene atmosphere"
    
    # 提取背景元素
    fx = visual.get("fx", "")
    elements = []
    lower_all = (composition + " " + fx).lower()
    if "scroll" in lower_all:
        elements.append("golden_scroll")
    if "spirit" in lower_all or "particle" in lower_all:
        elements.append("spirit_particles")
    
    return {
        "location": environment,
        "time": map_time_of_day(lighting),
        "weather": "mist" if "mist" in environment.lower() else "clear",
        "atmosphere": atmosphere,
        "background_elements": elements
    }


def _build_generation_params(camera_shot: str) -> Dict[str, Any]:
    """构建 generation_params 配置"""
    params = {
        "width": 1536,
        "height": 1536,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "seed": -1
    }
    
    # 根据 shot_type 调整
    if camera_shot == "close_up":
        params["num_inference_steps"] = 50
    elif camera_shot == "wide":
        params["num_inference_steps"] = 40
    
    return params


def _build_prompt_config(v1_scene: Dict[str, Any], character: Dict[str, Any], environment: Dict[str, Any], pose_type: str) -> Dict[str, Any]:
    """构建 prompt 配置"""
    # 构建基础 prompt
    prompt_parts = []
    
    # 角色
    if character.get("present", False):
        prompt_parts.append("hanli")
        prompt_parts.append("calm and restrained temperament, sharp but composed eyes, determined expression")
        prompt_parts.append("wearing his iconic mid-late-stage green daoist robe, traditional Chinese cultivation attire")
        
        # Pose
        pose_desc = _get_pose_description(pose_type)
        prompt_parts.append(pose_desc)
    
    # 环境
    if environment.get("location"):
        prompt_parts.append(f"in {environment['location']}")
    
    # 风格和质量
    prompt_parts.append("cinematic lighting, high detail, epic atmosphere, Chinese fantasy illustration style")
    
    final_prompt = ", ".join(prompt_parts)
    
    return {
        "base_template": "{{character.name}}, {{character.anchor_patches.temperament_anchor}}, {{character.anchor_patches.explicit_lock_words}}, {{pose.description}}, in {{environment.location}}, {{environment.atmosphere}}, {{environment.background_elements}}, cinematic lighting, high detail, epic atmosphere, Chinese fantasy illustration style",
        
        "llm_enhancement": {
            "enable": False,
            "role": "copywriter",
            "tasks": [
                "enhance_scene_description",
                "add_atmosphere_details"
            ],
            "forbidden_tasks": [
                "decide_shot_type",
                "decide_pose_type",
                "decide_model_route"
            ]
        },
        
        "final": final_prompt
    }


def convert_v1_to_v22(v1_scene: Dict[str, Any], episode_id: str, scene_id: int, total_scenes: int) -> Dict[str, Any]:
    """
    将单个 V1 场景转换为 v2.2-final 格式
    
    Args:
        v1_scene: V1 格式的场景字典
        episode_id: episode ID（如 "lingjie_ep1"）
        scene_id: 场景 ID
        total_scenes: 总场景数
        
    Returns:
        v2.2-final 格式的场景字典
    """
    # 提取 V1 字段
    duration = float(v1_scene.get("duration", 3))
    mood = v1_scene.get("mood", "")
    lighting = v1_scene.get("lighting", "")
    action = v1_scene.get("action", "")
    camera = v1_scene.get("camera", "")
    visual = v1_scene.get("visual") or {}
    motion = visual.get("motion") or {}
    narration_text = v1_scene.get("narration", "")
    description = v1_scene.get("description", "")
    
    # 映射到 v2.2-final 格式
    scene_role = map_scene_role(scene_id, total_scenes)
    intent_type = map_intent_type(action, scene_id)
    emotion = map_emotion(mood)
    tension_level = map_tension_level(mood)
    
    # Shot 和 Pose
    camera_shot = map_camera_shot(camera)
    # 修复：extreme_close 不是有效类型，转换为 close_up
    if camera_shot in ["extreme_close", "extreme_close"]:
        camera_shot = "close_up"
    camera_angle = map_camera_angle(camera)
    camera_movement = map_camera_movement(camera, motion)
    character_pose = visual.get("character_pose", "")
    pose_type = _convert_pose_type(action, character_pose)
    pose_description = _get_pose_description(pose_type)
    
    # Character
    present = has_hanli(v1_scene)
    character = _build_character_config(v1_scene, present)
    
    # Environment
    environment = _build_environment(v1_scene, visual, mood, lighting)
    
    # Model Route
    if present:
        visibility = map_visibility(camera)
        if visibility in ["high", "mid"]:
            base_model = "flux"
            identity_engine = "pulid"
        else:
            base_model = "flux"
            identity_engine = "pulid"
        character_role = "main"
    else:
        base_model = "flux"
        identity_engine = "none"
        character_role = "supporting"
    
    # 验证 shot+pose 兼容性
    ShotType, PoseType, rules = _get_execution_rules()
    shot_mapping = {
        "wide": ShotType.WIDE,
        "medium": ShotType.MEDIUM,
        "close_up": ShotType.CLOSE_UP,
        "aerial": ShotType.AERIAL,
    }
    pose_mapping = {
        "stand": PoseType.STAND,
        "walk": PoseType.WALK,
        "sit": PoseType.SIT,
        "lying": PoseType.LYING,
        "kneel": PoseType.KNEEL,
        "face_only": PoseType.FACE_ONLY,
    }
    
    shot_type_enum = shot_mapping.get(camera_shot.lower(), ShotType.MEDIUM)
    pose_type_enum = pose_mapping.get(pose_type.lower(), PoseType.STAND)
    
    # ⚡ 自动修正不合法组合
    pose_decision = rules.validate_pose(shot_type_enum, pose_type)
    if pose_decision.auto_corrected:
        pose_type = pose_decision.pose_type.value
        pose_description = _get_pose_description(pose_type)
        print(f"  ⚠ 自动修正: {camera_shot} + {pose_type_enum.value} → {camera_shot} + {pose_type} (原因: {pose_decision.correction_reason})")
    
    shot_pose_rules = rules.SHOT_POSE_RULES.get(shot_type_enum, {})
    allowed_poses = shot_pose_rules.get("allow", [])
    forbidden_poses = shot_pose_rules.get("forbid", [])
    shot_pose_compatible = pose_decision.pose_type not in forbidden_poses and (not allowed_poses or pose_decision.pose_type in allowed_poses)
    
    # 构建 v2.2-final 格式
    v22_scene = {
        "version": "v2.2-final",
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "description": f"从 V1 格式转换的场景 {scene_id}",
            "episode_id": episode_id,
            "scene_id": scene_id
        },
        "scene": {
            "version": "v2.2-final",  # 验证器需要 scene 内部也有 version 字段
            "id": f"scene_{scene_id:03d}",
            "scene_id": scene_id,
            "duration_sec": duration,
            
            # Intent
            "intent": {
                "type": intent_type,
                "description": description
            },
            
            # Shot
            "shot": {
                "type": camera_shot,
                "locked": True,
                "source": "v1_conversion",
                "description": f"{camera_shot} shot"
            },
            
            # Pose
            "pose": {
                "type": pose_type,
                "locked": True,
                "validated_by": "shot_pose_rules" if pose_decision.auto_corrected else "v1_conversion",
                "auto_corrected": pose_decision.auto_corrected,
                "description": pose_description
            },
            
            # Model Route
            "model_route": {
                "base_model": base_model,
                "identity_engine": identity_engine,
                "locked": True,
                "decision_reason": "v1_conversion",
                "character_role": character_role
            },
            
            # Character
            "character": character,
            
            # Environment
            "environment": environment,
            
            # Prompt
            "prompt": _build_prompt_config(v1_scene, character, environment, pose_type),
            
            # Generation Params
            "generation_params": _build_generation_params(camera_shot),
            
            # Camera
            "camera": {
                "shot": camera_shot,
                "angle": camera_angle,
                "movement": camera_movement,
                "focal_length_mm": 35,
                "depth_of_field": "shallow"
            },
            
            # Narration
            "narration": {
                "text": narration_text,
                "voice_id": "yunjuan_xianyin",
                "emotion_hint": emotion,
                "timing_policy": "fit_scene"
            },
            
            # Validation
            "validation": {
                "shot_pose_compatible": shot_pose_compatible,
                "model_route_valid": True,
                "character_anchor_complete": present,
                "prompt_template_valid": True
            }
        }
    }
    
    return v22_scene


def convert_file(input_path: str, output_dir: str, max_scenes: int = None) -> List[str]:
    """
    转换整个 V1 JSON 文件
    
    Args:
        input_path: 输入的 V1 JSON 文件路径
        output_dir: 输出目录
        max_scenes: 最大转换场景数（None表示全部）
        
    Returns:
        生成的 v2.2-final JSON 文件路径列表
    """
    # 读取 V1 JSON
    with open(input_path, "r", encoding="utf-8") as f:
        v1_data = json.load(f)
    
    episode = v1_data.get("episode", 1)
    title = v1_data.get("title", "")
    episode_id = f"lingjie_ep{episode}"
    
    scenes = v1_data.get("scenes", [])
    if max_scenes:
        scenes = scenes[:max_scenes]
    
    total_scenes = len(v1_data.get("scenes", []))
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 转换每个场景
    output_files = []
    for scene in scenes:
        scene_id = scene.get("id", 0)
        v22_scene = convert_v1_to_v22(scene, episode_id, scene_id, total_scenes)
        
        # 保存为单独的 JSON 文件
        output_file = output_path / f"scene_{scene_id:03d}_v22.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(v22_scene, f, ensure_ascii=False, indent=2)
        
        output_files.append(str(output_file))
        print(f"✓ 转换场景 {scene_id}: {output_file.name}")
    
    # 也保存一个包含所有场景的文件
    all_scenes_file = output_path / "all_scenes_v22.json"
    all_scenes = [convert_v1_to_v22(s, episode_id, s.get("id", i), total_scenes) for i, s in enumerate(scenes)]
    with open(all_scenes_file, "w", encoding="utf-8") as f:
        json.dump(all_scenes, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 转换完成: {len(scenes)} 个场景")
    print(f"✓ 单独文件: {len(output_files)} 个")
    print(f"✓ 合并文件: {all_scenes_file}")
    
    return output_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将 V1 JSON 转换为 v2.2-final 格式")
    parser.add_argument("input", help="输入的 V1 JSON 文件路径")
    parser.add_argument("--output-dir", default="lingjie/v22", help="输出目录（默认: lingjie/v22）")
    parser.add_argument("--max-scenes", type=int, default=None, help="最大转换场景数（默认: 全部）")
    
    args = parser.parse_args()
    
    convert_file(args.input, args.output_dir, args.max_scenes)

