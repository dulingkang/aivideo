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
    SDXL_TURBO = "sdxl_turbo"  # SDXL Turbo（快速生成）
    HUNYUAN_VIDEO = "hunyuan_video"


class CharacterRole(Enum):
    """角色类型"""
    MAIN_CHARACTER = "main"  # 主角（如韩立）
    IMPORTANT_SUPPORTING = "important_supporting"  # 重要配角（如南宫婉）
    NPC = "npc"  # 路人/一次性角色
    NO_CHARACTER = "no_character"  # 无角色（纯场景）


class TaskType(Enum):
    """任务类型"""
    CHARACTER_GENERATION = "character_generation"  # 角色生成
    SCENE_GENERATION = "scene_generation"  # 场景生成
    OUTPAINTING = "outpainting"  # 扩图
    INPAINTING = "inpainting"  # 修补
    CONTROLNET_LAYOUT = "controlnet_layout"  # 构图控制


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
        
        # 3. Model 路由表（硬切，向后兼容）
        self.MODEL_ROUTING = {
            # 条件: (有人物, shot_type) -> (scene_model, identity_engine)
            (True, ShotType.MEDIUM): (ModelType.FLUX, "pulid"),
            (True, ShotType.CLOSE_UP): (ModelType.FLUX, "pulid"),
            (True, ShotType.WIDE): (ModelType.SDXL, "instantid"),  # 远景用SDXL
            (False, None): (ModelType.FLUX, None),  # 无人物场景
        }
        
        # 智能分流路由表（v2.2新增）
        # 格式: (character_role, task_type, shot_type) -> (model, identity_engine)
        # None表示"任意匹配"
        self.SMART_ROUTING = {
            # 主角（韩立）-> 必须 Flux
            (CharacterRole.MAIN_CHARACTER, TaskType.CHARACTER_GENERATION, None): (ModelType.FLUX, "pulid"),
            
            # 重要配角（如有LoRA）-> Flux + LoRA
            (CharacterRole.IMPORTANT_SUPPORTING, TaskType.CHARACTER_GENERATION, None): (ModelType.FLUX, "pulid"),
            
            # NPC/路人 -> SDXL + InstantID（零成本换脸）
            (CharacterRole.NPC, TaskType.CHARACTER_GENERATION, None): (ModelType.SDXL, "instantid"),
            
            # 扩图任务 -> SDXL Inpainting（性价比之王）
            (None, TaskType.OUTPAINTING, None): (ModelType.SDXL, None),
            
            # 修补任务 -> SDXL Inpainting
            (None, TaskType.INPAINTING, None): (ModelType.SDXL, None),
            
            # 构图控制 -> SDXL ControlNet（生态成熟）
            (None, TaskType.CONTROLNET_LAYOUT, None): (ModelType.SDXL, None),
            
            # 纯场景 -> Flux（画质优先）
            (CharacterRole.NO_CHARACTER, TaskType.SCENE_GENERATION, None): (ModelType.FLUX, None),
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
        shot_type: ShotType,
        character_role: Optional[CharacterRole] = None,
        task_type: Optional[TaskType] = None,
        character_id: Optional[str] = None
    ) -> Tuple[ModelType, Optional[str]]:
        """
        获取模型路由（硬规则，支持智能分流）
        
        Args:
            has_character: 是否有人物
            shot_type: Shot类型
            character_role: 角色类型（可选，用于智能分流）
            task_type: 任务类型（可选，用于智能分流）
            character_id: 角色ID（可选，用于判断主角/NPC）
            
        Returns:
            Tuple[ModelType, Optional[str]]: (场景模型, 身份引擎)
        """
        # 智能分流（v2.2新增）
        if character_role is not None or task_type is not None:
            # 自动判断角色类型
            if character_role is None and character_id:
                if character_id == "hanli":
                    character_role = CharacterRole.MAIN_CHARACTER
                elif character_id.startswith("npc_") or character_id.startswith("random_"):
                    character_role = CharacterRole.NPC
                else:
                    character_role = CharacterRole.IMPORTANT_SUPPORTING
            
            # 自动判断任务类型
            if task_type is None:
                task_type = TaskType.CHARACTER_GENERATION if has_character else TaskType.SCENE_GENERATION
            
            # 查找智能分流路由
            for (role, task, shot), (model, identity) in self.SMART_ROUTING.items():
                if (role is None or role == character_role) and \
                   (task is None or task == task_type) and \
                   (shot is None or shot == shot_type):
                    logger.info(f"✓ 智能分流: {character_role.value if character_role else 'N/A'} + {task_type.value} -> {model.value} + {identity}")
                    return model, identity
        
        # 传统路由（向后兼容）
        for (has_char, shot), (model, identity) in self.MODEL_ROUTING.items():
            if has_char == has_character:
                if shot is None or shot == shot_type:
                    return model, identity
        
        # 默认路由
        if has_character:
            return ModelType.FLUX, "pulid"
        else:
            return ModelType.FLUX, None
    
    def get_model_route_for_npc(
        self,
        npc_description: str,
        reference_image: Optional[str] = None
    ) -> Tuple[ModelType, Optional[str]]:
        """
        获取NPC生成路由（SDXL + InstantID）
        
        Args:
            npc_description: NPC描述（如"满脸横肉的黑煞教弟子"）
            reference_image: 参考图像路径（可选）
            
        Returns:
            Tuple[ModelType, Optional[str]]: (SDXL, instantid)
        """
        logger.info(f"✓ NPC生成路由: {npc_description} -> SDXL + InstantID")
        return ModelType.SDXL, "instantid"
    
    def get_model_route_for_outpainting(
        self,
        base_image_path: str,
        target_aspect_ratio: str = "9:16"
    ) -> Tuple[ModelType, Optional[str]]:
        """
        获取扩图路由（SDXL Inpainting）
        
        Args:
            base_image_path: 基础图像路径
            target_aspect_ratio: 目标宽高比（如"9:16"）
            
        Returns:
            Tuple[ModelType, Optional[str]]: (SDXL, None)
        """
        logger.info(f"✓ 扩图路由: {target_aspect_ratio} -> SDXL Inpainting")
        return ModelType.SDXL, None
    
    def get_model_route_for_controlnet_layout(
        self,
        controlnet_type: str = "openpose"
    ) -> Tuple[ModelType, Optional[str]]:
        """
        获取构图控制路由（SDXL ControlNet）
        
        Args:
            controlnet_type: ControlNet类型（openpose/depth/canny/lineart）
            
        Returns:
            Tuple[ModelType, Optional[str]]: (SDXL, None)
        """
        logger.info(f"✓ 构图控制路由: {controlnet_type} -> SDXL ControlNet")
        return ModelType.SDXL, None
    
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

