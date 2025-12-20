#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行型规则引擎 v2.1 - 工业级稳定决策表

核心原则：
1. Shot/Pose/Gender/Model 全部"锁死"，不可被LLM覆盖
2. 角色身份来自LoRA，不来自prompt
3. 任何不合法组合 → 自动修正，而不是fallback
4. LLM只能补充描述，不能决策
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class SceneIntent(Enum):
    """场景意图（有限枚举）"""
    OPENING = "opening"
    ESTABLISHING = "establishing"
    CHARACTER_INTRO = "character_intro"
    DIALOGUE = "dialogue"
    ACTION_LIGHT = "action_light"
    ACTION_HEAVY = "action_heavy"
    TRANSITION = "transition"
    ENDING = "ending"


class ShotType(Enum):
    """镜头类型（唯一来源）"""
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    AERIAL = "aerial"


class PoseType(Enum):
    """姿态类型"""
    STAND = "stand"
    WALK = "walk"
    SIT = "sit"
    LYING = "lying"
    KNEEL = "kneel"
    FACE_ONLY = "face_only"


class ModelType(Enum):
    """模型类型"""
    FLUX = "flux"
    SDXL = "sdxl"
    HUNYUAN_VIDEO = "hunyuan_video"


@dataclass
class ShotDecision:
    """Shot决策结果"""
    shot_type: ShotType
    source: str  # "intent_mapping"
    allow_override: bool = False


@dataclass
class PoseDecision:
    """Pose决策结果"""
    pose_type: PoseType
    validated_by: str  # "shot_pose_rules"
    auto_corrected: bool = False
    correction_level: Optional[str] = None  # "none", "level1", "level2"
    correction_reason: Optional[str] = None


class ExecutionRulesV21:
    """
    执行型规则引擎 v2.1
    
    核心功能：
    1. SceneIntent → Shot 硬映射（冻结）
    2. Shot → Pose 允许表（同步冻结）
    3. Model 路由表（硬切）
    4. 自动修正不合法组合
    """
    
    def __init__(self):
        """初始化规则引擎"""
        # 1. SceneIntent → Shot 硬映射（冻结）
        self.SCENE_TO_SHOT = {
            SceneIntent.OPENING: ShotType.WIDE,
            SceneIntent.ESTABLISHING: ShotType.WIDE,
            SceneIntent.CHARACTER_INTRO: ShotType.MEDIUM,
            SceneIntent.DIALOGUE: ShotType.MEDIUM,
            SceneIntent.ACTION_LIGHT: ShotType.MEDIUM,
            SceneIntent.ACTION_HEAVY: ShotType.MEDIUM,
            SceneIntent.TRANSITION: ShotType.WIDE,
            SceneIntent.ENDING: ShotType.MEDIUM,
        }
        
        # 2. Shot → Pose 允许表（同步冻结）
        self.SHOT_POSE_RULES = {
            ShotType.WIDE: {
                "allow": [PoseType.STAND, PoseType.WALK],
                "forbid": [PoseType.LYING, PoseType.KNEEL, PoseType.SIT]
            },
            ShotType.MEDIUM: {
                "allow": [PoseType.STAND, PoseType.WALK, PoseType.LYING, PoseType.SIT, PoseType.KNEEL],
                "forbid": []
            },
            ShotType.CLOSE_UP: {
                "allow": [PoseType.FACE_ONLY],
                "forbid": [PoseType.LYING]
            },
            ShotType.AERIAL: {
                "allow": [PoseType.STAND, PoseType.WALK],
                "forbid": [PoseType.LYING, PoseType.KNEEL, PoseType.SIT]
            }
        }
        
        # 3. Model 路由表（硬切）
        self.MODEL_ROUTING = {
            # 条件: (有人物, shot_type) -> (scene_model, identity_engine)
            (True, ShotType.MEDIUM): (ModelType.FLUX, "pulid"),
            (True, ShotType.CLOSE_UP): (ModelType.FLUX, "pulid"),
            (True, ShotType.WIDE): (ModelType.SDXL, "instantid"),  # 远景用SDXL
            (False, None): (ModelType.FLUX, None),  # 无人物场景
        }
        
        # 4. 性别负锁（工业级标配）
        self.NEG_GENDER_LOCK_MALE = [
            "female", "woman", "girl",
            "soft facial features", "delicate face",
            "long eyelashes", "narrow shoulders",
            "slim waist", "feminine body"
        ]
        
        self.NEG_GENDER_LOCK_FEMALE = [
            "male", "man", "boy",
            "rough facial features", "strong jawline",
            "broad shoulders", "masculine body"
        ]
        
        logger.info("执行型规则引擎 v2.1 初始化完成")
    
    def get_shot_from_intent(self, intent: str) -> ShotDecision:
        """
        从SceneIntent获取Shot（硬映射，不可覆盖）
        
        Args:
            intent: 场景意图字符串
            
        Returns:
            ShotDecision: Shot决策结果
        """
        try:
            scene_intent = SceneIntent(intent)
        except ValueError:
            # 如果intent不在枚举中，使用默认值
            logger.warning(f"未知的intent: {intent}，使用默认值 MEDIUM")
            scene_intent = SceneIntent.CHARACTER_INTRO
        
        shot_type = self.SCENE_TO_SHOT.get(scene_intent, ShotType.MEDIUM)
        
        return ShotDecision(
            shot_type=shot_type,
            source="intent_mapping",
            allow_override=False
        )
    
    def validate_pose(
        self,
        shot_type: ShotType,
        pose_str: str,
        story_context: Optional[Dict[str, Any]] = None
    ) -> PoseDecision:
        """
        验证Pose是否合法，如果不合法则自动修正（两级修正策略）
        
        Args:
            shot_type: Shot类型
            pose_str: 姿态字符串
            story_context: 剧情上下文（可选，用于Level 2语义修正）
            
        Returns:
            PoseDecision: Pose决策结果
        """
        # 归一化pose字符串
        pose_str_lower = pose_str.lower().replace("_", "").replace("-", "")
        
        # 映射到PoseType
        pose_mapping = {
            "stand": PoseType.STAND,
            "walk": PoseType.WALK,
            "sit": PoseType.SIT,
            "lying": PoseType.LYING,
            "lyingmotionless": PoseType.LYING,
            "kneel": PoseType.KNEEL,
            "faceonly": PoseType.FACE_ONLY,
        }
        
        original_pose_type = None
        for key, value in pose_mapping.items():
            if key in pose_str_lower:
                original_pose_type = value
                break
        
        if original_pose_type is None:
            # 如果无法识别，默认使用STAND
            original_pose_type = PoseType.STAND
            logger.warning(f"无法识别的pose: {pose_str}，使用默认值 STAND")
        
        pose_type = original_pose_type
        
        # 检查是否合法
        rules = self.SHOT_POSE_RULES.get(shot_type, {})
        allowed = rules.get("allow", [])
        forbidden = rules.get("forbid", [])
        
        auto_corrected = False
        correction_level = None
        correction_reason = None
        
        # Level 1: 无感修正（硬规则冲突）
        if pose_type in forbidden:
            if allowed:
                pose_type = allowed[0]
                auto_corrected = True
                correction_level = "level1"
                correction_reason = f"shot_pose_conflict: {shot_type.value}禁止{pose_str}"
                logger.warning(f"⚠ [Level 1] Pose {pose_str} 在 Shot {shot_type.value} 中不合法，自动修正为 {pose_type.value}")
            else:
                pose_type = PoseType.STAND
                auto_corrected = True
                correction_level = "level1"
                correction_reason = f"no_allowed_pose: {shot_type.value}没有允许的pose"
                logger.warning(f"⚠ [Level 1] Shot {shot_type.value} 没有允许的pose，使用默认值 STAND")
        elif pose_type not in allowed and allowed:
            # 不在允许列表中，但不在禁止列表中
            pose_type = allowed[0]
            auto_corrected = True
            correction_level = "level1"
            correction_reason = f"not_in_allowed_list: {pose_str}不在允许列表中"
            logger.info(f"ℹ [Level 1] Pose {pose_str} 不在允许列表中，修正为 {pose_type.value}")
        
        # Level 2: 语义修正（剧情冲突，需要story_context）
        if story_context and auto_corrected and correction_level == "level1":
            # 检查剧情是否需要特定pose
            story_requires_walking = story_context.get("requires_walking", False)
            story_requires_rest = story_context.get("requires_rest", False)
            
            if story_requires_walking and pose_type == PoseType.STAND:
                pose_type = PoseType.WALK
                correction_level = "level2"
                correction_reason = "story_flow_conflict: 剧情需要行走"
                logger.info(f"ℹ [Level 2] 剧情需要行走，修正为 WALK")
            elif story_requires_rest and pose_type == PoseType.LYING:
                # lying在非受伤剧情中可能不合适
                if not story_context.get("is_injured", False):
                    pose_type = PoseType.SIT
                    correction_level = "level2"
                    correction_reason = "story_flow_conflict: 非受伤剧情，lying改为sitting"
                    logger.info(f"ℹ [Level 2] 非受伤剧情，lying改为sitting")
        
        return PoseDecision(
            pose_type=pose_type,
            validated_by="shot_pose_rules",
            auto_corrected=auto_corrected,
            correction_level=correction_level,
            correction_reason=correction_reason
        )
    
    def get_model_route(
        self,
        has_character: bool,
        shot_type: ShotType
    ) -> Tuple[ModelType, Optional[str]]:
        """
        获取模型路由（硬规则，禁止动态切换）
        
        Args:
            has_character: 是否有人物
            shot_type: Shot类型
            
        Returns:
            Tuple[ModelType, Optional[str]]: (场景模型, 身份引擎)
        """
        # 查找匹配的路由规则
        for (has_char, shot), (model, identity) in self.MODEL_ROUTING.items():
            if has_char == has_character:
                if shot is None or shot == shot_type:
                    return model, identity
        
        # 默认路由
        if has_character:
            return ModelType.FLUX, "pulid"
        else:
            return ModelType.FLUX, None
    
    def get_gender_negative_lock(self, gender: str) -> List[str]:
        """
        获取性别负锁（工业级标配）
        
        Args:
            gender: 性别 ("male" 或 "female")
            
        Returns:
            List[str]: 负面提示词列表
        """
        if gender.lower() == "male":
            return self.NEG_GENDER_LOCK_MALE.copy()
        elif gender.lower() == "female":
            return self.NEG_GENDER_LOCK_FEMALE.copy()
        else:
            logger.warning(f"未知的gender: {gender}，返回空列表")
            return []
    
    def check_forbidden_combinations(self, shot_type: ShotType, pose_type: PoseType) -> bool:
        """
        检查是否为"死刑组合"（wide + lying）
        
        Args:
            shot_type: Shot类型
            pose_type: Pose类型
            
        Returns:
            bool: True表示是死刑组合
        """
        if shot_type == ShotType.WIDE and pose_type == PoseType.LYING:
            return True
        return False


# 全局实例
_execution_rules = None


def get_execution_rules() -> ExecutionRulesV21:
    """获取全局规则引擎实例"""
    global _execution_rules
    if _execution_rules is None:
        _execution_rules = ExecutionRulesV21()
    return _execution_rules

