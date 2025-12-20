#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON v2 → v2.1-exec 自动转换器

功能：
1. 将v2格式的JSON转换为v2.1-exec执行型格式
2. 应用硬规则（Shot/Pose/Model锁定）
3. 添加角色锚配置
4. 添加性别负锁
5. 记录决策来源（decision trace）
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from .execution_rules_v2_1 import get_execution_rules, SceneIntent, ShotType, PoseType
    from .character_anchor_v2_1 import get_character_anchor_manager
except (ImportError, ValueError):
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    utils_path = Path(__file__).parent
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    from execution_rules_v2_1 import get_execution_rules, SceneIntent, ShotType, PoseType
    from character_anchor_v2_1 import get_character_anchor_manager

logger = logging.getLogger(__name__)


class JSONV2ToV21Converter:
    """JSON v2 → v2.1-exec 转换器"""
    
    def __init__(self, character_profiles: Optional[Dict[str, Any]] = None):
        """
        初始化转换器
        
        Args:
            character_profiles: 角色档案字典（可选）
        """
        self.rules = get_execution_rules()
        self.anchor_manager = get_character_anchor_manager(character_profiles)
        logger.info("JSON v2 → v2.1-exec 转换器初始化完成")
    
    def convert_scene(self, scene_v2: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换单个场景
        
        Args:
            scene_v2: v2格式的场景JSON
            
        Returns:
            v2.1-exec格式的场景JSON
        """
        # 1. 提取intent（如果没有，使用默认值）
        intent_type = self._extract_intent(scene_v2)
        
        # 2. 应用硬规则：Intent → Shot
        shot_decision = self.rules.get_shot_from_intent(intent_type)
        
        # 3. 验证并修正Pose
        character = scene_v2.get("character", {})
        pose_str = character.get("pose", "standing")
        pose_decision = self.rules.validate_pose(shot_decision.shot_type, pose_str)
        
        # 4. 获取Model路由
        has_character = character.get("present", False)
        model, identity_engine = self.rules.get_model_route(
            has_character=has_character,
            shot_type=shot_decision.shot_type
        )
        
        # 5. 构建v2.1-exec格式
        scene_v21 = {
            "version": "v2.1-exec",
            "scene_id": scene_v2.get("scene_id", 0),
            "episode_id": scene_v2.get("episode_id", ""),
            "duration_sec": scene_v2.get("duration_sec", 4.0),
            "target_fps": scene_v2.get("target_fps", 24),
            "scene_role": scene_v2.get("scene_role", "story"),
            "scene_type": scene_v2.get("scene_type", "novel"),
            
            # Shot（已锁定）
            "shot": {
                "type": shot_decision.shot_type.value,
                "locked": True,
                "source": shot_decision.source,
                "original_intent": intent_type
            },
            
            # Pose（已锁定，记录修正）
            "pose": {
                "type": pose_decision.pose_type.value,
                "locked": True,
                "validated_by": pose_decision.validated_by,
                "auto_corrected": pose_decision.auto_corrected,
                "original_pose": pose_str if pose_decision.auto_corrected else None
            },
            
            # Camera（保留原有配置）
            "camera": scene_v2.get("camera", {
                "angle": "eye_level",
                "motion": "static"
            }),
            
            # Character（添加锁定信息）
            "character": self._build_character_v21(character, model, identity_engine),
            
            # Model路由（已锁定，带决策原因）
            "model_route": {
                "base_model": model.value,
                "identity_engine": identity_engine,
                "allow_fallback": False,
                "decision_reason": self._get_model_decision_reason(
                    has_character, shot_decision.shot_type, model, identity_engine
                ),
                "confidence": 0.95  # 硬规则，高置信度
            },
            
            # Prompt（重构为执行型）
            "prompt": self._build_prompt_v21(scene_v2, character),
            
            # 性别负锁
            "negative_lock": self._build_negative_lock(character),
            
            # Narration（保留）
            "narration": scene_v2.get("narration", {}),
            
            # Assets（保留）
            "assets": scene_v2.get("assets", {}),
            
            # Notes
            "notes": scene_v2.get("notes", "") + " [已转换为v2.1-exec执行型]"
        }
        
        return scene_v21
    
    def _extract_intent(self, scene_v2: Dict[str, Any]) -> str:
        """提取intent类型"""
        intent = scene_v2.get("intent", {})
        if isinstance(intent, dict):
            intent_type = intent.get("type", "")
        else:
            intent_type = str(intent)
        
        # 映射到SceneIntent枚举
        intent_mapping = {
            "title_reveal": "opening",
            "opening": "opening",
            "establishing": "establishing",
            "character_intro": "character_intro",
            "introduce_world": "establishing",
            "dialogue": "dialogue",
            "emotional_beat": "dialogue",
            "action_light": "action_light",
            "action_heavy": "action_heavy",
            "conflict": "action_heavy",
            "transition": "transition",
            "ending": "ending",
            "flashback": "dialogue"
        }
        
        return intent_mapping.get(intent_type, "character_intro")
    
    def _build_character_v21(
        self,
        character: Dict[str, Any],
        model: Any,
        identity_engine: Optional[str]
    ) -> Dict[str, Any]:
        """构建v2.1格式的character配置"""
        character_id = character.get("id")
        gender = character.get("gender", "male")
        
        # 注册角色锚（如果还没有）
        if character_id:
            self.anchor_manager.register_character(
                character_id=character_id,
                gender=gender,
                lora_path=None,  # 从character_profiles中查找
                instantid_enabled=(identity_engine == "instantid"),
                instantid_strength=0.75
            )
        
        return {
            "present": character.get("present", False),
            "id": character_id,
            "gender": gender,
            "face_visible": character.get("face_visible", True),
            "visibility": character.get("visibility", "mid"),
            "body_coverage": character.get("body_coverage", "half_body"),
            "locks": {
                "gender": True,
                "face": True,
                "body_type": True
            }
        }
    
    def _build_prompt_v21(
        self,
        scene_v2: Dict[str, Any],
        character: Dict[str, Any]
    ) -> Dict[str, Any]:
        """构建v2.1格式的prompt"""
        # 提取原有prompt信息
        visual_constraints = scene_v2.get("visual_constraints", {})
        environment = visual_constraints.get("environment", "")
        
        # 构建positive_core（角色描述）
        character_id = character.get("id", "")
        if character_id:
            positive_core = f"{character_id}, " + character.get("description", "")
        else:
            positive_core = ""
        
        return {
            "positive_core": positive_core.strip(),
            "scene_description": environment,
            "style": scene_v2.get("quality_target", {}).get("style", "xianxia_anime")
        }
    
    def _build_negative_lock(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """构建性别负锁"""
        character_id = character.get("id")
        gender = character.get("gender", "male")
        
        if not character_id:
            return {"gender": False, "extra": []}
        
        negative_list = self.anchor_manager.get_negative_prompt_with_gender_lock(
            character_id=character_id,
            base_negative=[]
        )
        
        return {
            "gender": True,
            "extra": negative_list
        }
    
    def _get_model_decision_reason(
        self,
        has_character: bool,
        shot_type: ShotType,
        model: Any,
        identity_engine: Optional[str]
    ) -> str:
        """生成模型决策原因（可解释性）"""
        if not has_character:
            return "no_character -> flux"
        
        if shot_type == ShotType.WIDE:
            return f"character_present + wide_shot -> sdxl + {identity_engine}"
        elif shot_type in [ShotType.MEDIUM, ShotType.CLOSE_UP]:
            return f"character_present + {shot_type.value}_shot -> flux + {identity_engine}"
        else:
            return f"character_present + {shot_type.value} -> {model.value} + {identity_engine}"
    
    def convert_episode(
        self,
        episode_v2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        转换整集JSON
        
        Args:
            episode_v2: v2格式的episode JSON
            
        Returns:
            v2.1-exec格式的episode JSON
        """
        scenes_v2 = episode_v2.get("scenes", [])
        scenes_v21 = []
        
        for scene_v2 in scenes_v2:
            try:
                scene_v21 = self.convert_scene(scene_v2)
                scenes_v21.append(scene_v21)
            except Exception as e:
                logger.error(f"转换场景 {scene_v2.get('scene_id')} 失败: {e}")
                # 保留原场景，但标记为转换失败
                scene_v2["conversion_error"] = str(e)
                scenes_v21.append(scene_v2)
        
        return {
            "episode": episode_v2.get("episode", 0),
            "title": episode_v2.get("title", ""),
            "version": "v2.1-exec",
            "scenes": scenes_v21
        }


def convert_json_file(
    input_path: str,
    output_path: str,
    character_profiles: Optional[Dict[str, Any]] = None
) -> bool:
    """
    转换JSON文件
    
    Args:
        input_path: 输入文件路径（v2格式）
        output_path: 输出文件路径（v2.1-exec格式）
        character_profiles: 角色档案字典（可选）
        
    Returns:
        是否成功
    """
    try:
        # 读取v2 JSON
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建转换器
        converter = JSONV2ToV21Converter(character_profiles)
        
        # 转换
        if "scenes" in data:
            # 整集JSON
            result = converter.convert_episode(data)
        else:
            # 单个场景JSON
            result = converter.convert_scene(data)
        
        # 写入v2.1-exec JSON
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ 转换完成: {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"✗ 转换失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python json_v2_to_v21_converter.py <input.json> <output.json>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    success = convert_json_file(input_path, output_path)
    sys.exit(0 if success else 1)

