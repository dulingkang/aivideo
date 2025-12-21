#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 到 v2.2-final 格式转换器

将旧的 v2 JSON 格式转换为新的 v2.2-final 格式
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# 延迟导入，避免循环依赖
def _get_execution_rules():
    """延迟导入执行规则"""
    from execution_rules_v2_1 import ShotType, PoseType, get_execution_rules
    return ShotType, PoseType, get_execution_rules()


def convert_v2_to_v22(v2_scene: Dict[str, Any]) -> Dict[str, Any]:
    """
    将单个 v2 场景转换为 v2.2-final 格式
    
    Args:
        v2_scene: v2 格式的场景字典
        
    Returns:
        v2.2-final 格式的场景字典
    """
    scene_id = v2_scene.get("scene_id", 0)
    episode_id = v2_scene.get("episode_id", "lingjie_ep1")
    
    # 提取基本信息
    intent = v2_scene.get("intent", {})
    visual_constraints = v2_scene.get("visual_constraints", {})
    camera = v2_scene.get("camera", {})
    character = v2_scene.get("character", {})
    generation_policy = v2_scene.get("generation_policy", {})
    narration = v2_scene.get("narration", {})
    quality_target = v2_scene.get("quality_target", {})
    
    # 预先计算一些值
    pose_type = _convert_pose_type(character.get("pose", "stand"))
    pose_description = _get_pose_description(pose_type)
    shot_type = camera.get("shot", "medium")
    
    # 验证shot+pose兼容性（在构建字典之前）
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
    
    shot_type_enum = shot_mapping.get(shot_type.lower(), ShotType.MEDIUM)
    pose_type_enum = pose_mapping.get(pose_type.lower(), PoseType.STAND)
    
    # 检查shot+pose兼容性
    shot_pose_rules = rules.SHOT_POSE_RULES.get(shot_type_enum, {})
    allowed_poses = shot_pose_rules.get("allow", [])
    forbidden_poses = shot_pose_rules.get("forbid", [])
    shot_pose_compatible = pose_type_enum not in forbidden_poses and (not allowed_poses or pose_type_enum in allowed_poses)
    
    # 构建 v2.2-final 格式
    v22_scene = {
        "version": "v2.2-final",
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "description": f"从 v2 格式转换的场景 {scene_id}",
            "episode_id": episode_id,
            "scene_id": scene_id
        },
        "scene": {
            "id": f"scene_{scene_id:03d}",
            "scene_id": scene_id,
            "duration_sec": v2_scene.get("duration_sec", 4.0),
            
            # Intent
            "intent": {
                "type": intent.get("type", "general"),
                "description": v2_scene.get("notes", "")
            },
            
            # Shot (从 camera.shot 转换)
            "shot": {
                "type": shot_type,
                "locked": True,
                "source": "v2_conversion",
                "description": f"{shot_type} shot"
            },
            
            # Pose (从 character.pose 转换)
            "pose": {
                "type": pose_type,
                "locked": True,
                "validated_by": "v2_conversion",
                "auto_corrected": False,
                "description": pose_description
            },
            
            # Model Route (从 generation_policy 转换)
            "model_route": {
                "base_model": generation_policy.get("image_model", "flux"),
                "identity_engine": "pulid" if character.get("present", False) and character.get("id") == "hanli" else "none",
                "locked": True,
                "decision_reason": "v2_conversion",
                "character_role": "main" if character.get("importance") == "primary" else "supporting"
            },
            
            # Character
            "character": _build_character_config(character),
            
            # Environment
            "environment": {
                "location": visual_constraints.get("environment", ""),
                "time": visual_constraints.get("time_of_day", "day"),
                "weather": visual_constraints.get("weather", "clear"),
                "atmosphere": _build_atmosphere(visual_constraints, quality_target),
                "background_elements": visual_constraints.get("elements", [])
            },
            
            # Prompt
            "prompt": _build_prompt_config(v2_scene, character, visual_constraints),
            
            # Generation Params (可选，如果所有场景使用相同参数，可以从config.yaml读取)
            # 如果某个场景需要特殊参数，可以在这里覆盖
            # "generation_params": {
            #     "width": 1536,
            #     "height": 1536,
            #     "num_inference_steps": 50,
            #     "guidance_scale": 7.5,
            #     "seed": -1
            # },
            
            # Camera
            "camera": {
                "shot": camera.get("shot", "medium"),
                "angle": camera.get("angle", "eye_level"),
                "movement": camera.get("movement", "static"),
                "focal_length_mm": camera.get("focal_length_mm", 50),
                "depth_of_field": camera.get("depth_of_field", "medium")
            },
            
            # Narration
            "narration": {
                "text": narration.get("text", ""),
                "voice_id": narration.get("voice_id", "yunjuan_xianyin"),
                "emotion_hint": narration.get("emotion_hint", "neutral"),
                "timing_policy": narration.get("timing_policy", "fit_scene")
            },
            
            # Validation
            "validation": {
                "shot_pose_compatible": shot_pose_compatible,
                "model_route_valid": True,
                "character_anchor_complete": character.get("present", False),
                "prompt_template_valid": True
            }
        }
    }
    
    return v22_scene


def _convert_pose_type(v2_pose: str) -> str:
    """转换 v2 pose 类型到 v2.2-final 格式"""
    pose_map = {
        "lying_motionless": "lying",
        "lying": "lying",
        "sitting": "sit",
        "sit": "sit",
        "standing": "stand",
        "stand": "stand",
        "walking": "walk",
        "walk": "walk",
        "kneeling": "kneel",
        "kneel": "kneel",
    }
    return pose_map.get(v2_pose.lower(), "stand")


def _get_pose_description(pose: str) -> str:
    """获取 pose 描述"""
    descriptions = {
        "lying": "lying on the ground, motionless",
        "sit": "sitting, seated",
        "stand": "standing pose, upright posture",
        "walk": "walking, in motion",
        "kneel": "kneeling, on knees",
    }
    return descriptions.get(pose.lower(), "standing pose")


def _build_character_config(character: Dict[str, Any]) -> Dict[str, Any]:
    """构建 character 配置"""
    if not character.get("present", False):
        return {
            "present": False
        }
    
    character_id = character.get("id", "hanli")
    
    # 韩立的默认配置
    if character_id == "hanli":
        return {
            "id": "hanli",
            "name": "韩立",
            "present": True,
            "role": "main" if character.get("importance") == "primary" else "supporting",
            
            "identity": {
                "gender": "male",
                "age_range": "young_adult",
                "physique": "lean",
                "face_shape": "sharp"
            },
            
            "lora_config": {
                "type": "single",
                "lora_path": "",
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
            
            "reference_image": "",
            "negative_gender_lock": [
                "female", "woman", "girl",
                "soft facial features", "delicate face",
                "long eyelashes", "narrow shoulders",
                "slim waist", "feminine body"
            ]
        }
    else:
        # 其他角色的简化配置
        return {
            "id": character_id,
            "name": character_id,
            "present": True,
            "role": "supporting",
            "identity": {
                "gender": "unknown",
                "age_range": "adult",
                "physique": "normal",
                "face_shape": "normal"
            },
            "lora_config": {
                "type": "single",
                "lora_path": "",
                "weight": 0.8,
                "trigger": character_id
            },
            "anchor_patches": {},
            "reference_image": "",
            "negative_gender_lock": []
        }


def _build_atmosphere(visual_constraints: Dict[str, Any], quality_target: Dict[str, Any]) -> str:
    """构建 atmosphere 描述"""
    parts = []
    
    # 从 visual_constraints 提取
    if visual_constraints.get("environment"):
        parts.append(visual_constraints["environment"])
    
    # 从 quality_target 提取风格
    style = quality_target.get("style", "")
    if style:
        parts.append(f"{style} style")
    
    lighting = quality_target.get("lighting_style", "")
    if lighting:
        parts.append(f"{lighting} lighting")
    
    return ", ".join(parts) if parts else "serene atmosphere"


def _build_prompt_config(v2_scene: Dict[str, Any], character: Dict[str, Any], visual_constraints: Dict[str, Any]) -> Dict[str, Any]:
    """构建 prompt 配置"""
    # 构建基础 prompt
    prompt_parts = []
    
    # 角色
    if character.get("present", False):
        char_id = character.get("id", "hanli")
        if char_id == "hanli":
            prompt_parts.append("hanli")
            prompt_parts.append("calm and restrained temperament, sharp but composed eyes, determined expression")
            prompt_parts.append("wearing his iconic mid-late-stage green daoist robe, traditional Chinese cultivation attire")
        
        # Pose - 使用转换后的pose类型
        pose_type = _convert_pose_type(character.get("pose", "stand"))
        pose_desc = _get_pose_description(pose_type)
        prompt_parts.append(pose_desc)
    
    # 环境
    if visual_constraints.get("environment"):
        prompt_parts.append(f"in {visual_constraints['environment']}")
    
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


def convert_file(input_path: str, output_dir: str, max_scenes: int = None) -> List[str]:
    """
    转换整个 v2 JSON 文件
    
    Args:
        input_path: 输入的 v2 JSON 文件路径
        output_dir: 输出目录
        max_scenes: 最大转换场景数（None表示全部）
        
    Returns:
        生成的 v2.2-final JSON 文件路径列表
    """
    # 读取 v2 JSON
    with open(input_path, "r", encoding="utf-8") as f:
        v2_data = json.load(f)
    
    scenes = v2_data.get("scenes", [])
    if max_scenes:
        scenes = scenes[:max_scenes]
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 转换每个场景
    output_files = []
    for scene in scenes:
        scene_id = scene.get("scene_id", 0)
        v22_scene = convert_v2_to_v22(scene)
        
        # 保存为单独的 JSON 文件
        output_file = output_path / f"scene_{scene_id:03d}_v22.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(v22_scene, f, ensure_ascii=False, indent=2)
        
        output_files.append(str(output_file))
        print(f"✓ 转换场景 {scene_id}: {output_file.name}")
    
    # 也保存一个包含所有场景的文件
    all_scenes_file = output_path / "all_scenes_v22.json"
    all_scenes = [convert_v2_to_v22(s) for s in scenes]
    with open(all_scenes_file, "w", encoding="utf-8") as f:
        json.dump(all_scenes, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 转换完成: {len(scenes)} 个场景")
    print(f"✓ 单独文件: {len(output_files)} 个")
    print(f"✓ 合并文件: {all_scenes_file}")
    
    return output_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将 v2 JSON 转换为 v2.2-final 格式")
    parser.add_argument("input", help="输入的 v2 JSON 文件路径")
    parser.add_argument("--output-dir", default="lingjie/v22", help="输出目录（默认: lingjie/v22）")
    parser.add_argument("--max-scenes", type=int, default=None, help="最大转换场景数（默认: 全部）")
    
    args = parser.parse_args()
    
    convert_file(args.input, args.output_dir, args.max_scenes)

