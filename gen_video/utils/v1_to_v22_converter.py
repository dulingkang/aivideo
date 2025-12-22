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


def _build_prompt_config(v1_scene: Dict[str, Any], character: Dict[str, Any], environment: Dict[str, Any], pose_type: str, pose_auto_corrected: bool = False, original_pose: str = "") -> Dict[str, Any]:
    """
    构建 prompt 配置（V1 增强版）
    
    策略：
    1. 角色锚定词放最前面（确保人物一致性）
    2. 直接使用 V1 原始 prompt（保留场景信息）
    3. 补充情绪、环境、特效
    4. 为 close_up 添加背景暗示
    5. 当 pose 被自动修正时，替换冲突关键词
    """
    prompt_parts = []
    
    # 1. 角色锚定词（如果角色存在）- 放在最前面确保身份
    if character.get("present", False):
        # 触发词 + 简短身份描述（不要太长，避免压制场景描述）
        prompt_parts.append("hanli")
        prompt_parts.append("young male cultivator with sharp composed eyes")
        prompt_parts.append("wearing green daoist robe")
    
    # 2. ⭐ 核心：使用 V1 原始 prompt 或 composition（最有价值的信息！）
    v1_prompt = v1_scene.get("prompt", "").strip()
    v1_composition = v1_scene.get("visual", {}).get("composition", "").strip()
    
    if v1_prompt:
        # 清理 prompt，避免重复角色名（因为已经在前面添加了 hanli）
        cleaned_prompt = v1_prompt
        # ⚡ 修复：直接删除 "Han Li" 而不是替换为代词，避免 "hanli" + "he" 被理解为两个人
        cleaned_prompt = cleaned_prompt.replace("Han Li's", "")  # 删除 "Han Li's"
        cleaned_prompt = cleaned_prompt.replace("Han Li ", "")   # 删除 "Han Li " (带空格)
        cleaned_prompt = cleaned_prompt.replace("Han Li,", "")   # 删除 "Han Li," 
        cleaned_prompt = cleaned_prompt.replace("Han Li", "")    # 删除剩余的 "Han Li"
        # 清理多余的空格和逗号
        cleaned_prompt = cleaned_prompt.replace("  ", " ").replace(" ,", ",").replace(",,", ",")
        cleaned_prompt = cleaned_prompt.strip().strip(",").strip()
        prompt_parts.append(cleaned_prompt)
    elif v1_composition:
        # 使用 composition 作为备选
        cleaned_composition = v1_composition
        # ⚡ 同样的修复
        cleaned_composition = cleaned_composition.replace("Han Li's", "")
        cleaned_composition = cleaned_composition.replace("Han Li ", "")
        cleaned_composition = cleaned_composition.replace("Han Li,", "")
        cleaned_composition = cleaned_composition.replace("Han Li", "")
        cleaned_composition = cleaned_composition.replace("  ", " ").replace(" ,", ",").replace(",,", ",")
        cleaned_composition = cleaned_composition.strip().strip(",").strip()
        prompt_parts.append(cleaned_composition)
    else:
        # 回退：使用 pose 描述
        pose_desc = _get_pose_description(pose_type)
        prompt_parts.append(pose_desc)
    
    # 3. 添加情绪描述（从 mood 字段）
    mood = v1_scene.get("mood", "").strip().lower()
    mood_map = {
        "tense": "tense expression",
        "solemn": "solemn contemplative expression",
        "alert": "alert vigilant expression",
        "calm": "calm serene expression",
        "agony": "expression showing pain",
        "fierce": "fierce determined expression",
        "mysterious": "mysterious contemplative expression",
        "resolute": "resolute determined expression",
        "focused": "focused concentrated expression",
        "perilous": "wary cautious expression",
        "brutal": "intense grim expression",
        "contemplative": "thoughtful expression",
        "serene": "peaceful serene expression"
    }
    if mood in mood_map:
        prompt_parts.append(mood_map[mood])
    
    # 4. 添加环境（如果 V1 prompt 中没有明确的环境描述）
    v1_environment = v1_scene.get("visual", {}).get("environment", "").strip()
    # 检查是否已有环境描述
    current_prompt = ", ".join(prompt_parts).lower()
    has_environment = any(kw in current_prompt for kw in ["in ", "on ", "at ", "under ", "above "])
    
    if v1_environment and not has_environment:
        prompt_parts.append(f"in {v1_environment}")
    
    # 5. 添加特效（如果有）
    fx = v1_scene.get("visual", {}).get("fx", "").strip()
    if fx:
        prompt_parts.append(fx)
    
    # 6. 为 close_up 镜头添加背景暗示（避免纯人像，增加场景感）
    camera = v1_scene.get("camera", "").lower()
    if "close" in camera:
        # 检查是否已有背景描述
        current_prompt = ", ".join(prompt_parts).lower()
        if "background" not in current_prompt and "in " not in current_prompt[-50:]:
            # 从 description 或 environment 推断背景
            desc = v1_scene.get("description", "")
            env = v1_environment.lower()
            
            if "沙" in desc or "desert" in env or "sand" in env or "gravel" in env:
                prompt_parts.append("with soft blurred desert background")
            elif "天" in desc or "sky" in env:
                prompt_parts.append("with blurred celestial sky background")
            elif "回想" in desc or "回忆" in desc or "recall" in v1_prompt.lower():
                prompt_parts.append("with soft bokeh background suggesting memories")
            else:
                prompt_parts.append("with soft bokeh background")
    
    # 7. 风格锚定词（放在最后）
    prompt_parts.append("xianxia anime style, Chinese fantasy, cinematic lighting, high detail, 4K")
    
    # 过滤空字符串并拼接
    final_prompt = ", ".join([p for p in prompt_parts if p])
    
    # 8. 如果 pose 被自动修正，替换冲突关键词
    if pose_auto_corrected:
        # 定义 pose 冲突关键词映射
        pose_conflict_replacements = {
            "lying": {
                # 当 lying 被修正为 stand 时的替换
                "lying": "resting",
                "motionless": "still",
                "on the ground": "contemplating",
                "lying on": "surveying",
            },
            "sit": {
                "sitting": "standing",
                "seated": "upright",
            }
        }
        
        # 获取原始 pose 的替换规则
        if original_pose.lower() in pose_conflict_replacements:
            replacements = pose_conflict_replacements[original_pose.lower()]
            for old_word, new_word in replacements.items():
                final_prompt = final_prompt.replace(old_word, new_word)
    
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
        
        "v1_original": v1_prompt,  # 保留原始 prompt 供调试
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
            "prompt": _build_prompt_config(
                v1_scene, character, environment, pose_type,
                pose_auto_corrected=pose_decision.auto_corrected,
                original_pose=pose_type_enum.value
            ),
            
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

