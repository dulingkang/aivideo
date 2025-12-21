#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDXL智能分流集成

将SDXL智能分流规则集成到主流程中，支持：
1. NPC生成路由（SDXL + InstantID）
2. 扩图路由（SDXL Inpainting）
3. 构图控制路由（SDXL ControlNet）
"""

import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum

try:
    from .execution_rules_v2_1 import (
        ExecutionRulesV21,
        CharacterRole,
        TaskType,
        ModelType
    )
except (ImportError, ValueError):
    import sys
    from pathlib import Path
    utils_path = Path(__file__).parent
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    from execution_rules_v2_1 import (
        ExecutionRulesV21,
        CharacterRole,
        TaskType,
        ModelType
    )

logger = logging.getLogger(__name__)


class SDXLRoutingIntegration:
    """SDXL智能分流集成"""
    
    def __init__(self):
        """初始化SDXL路由集成"""
        self.rules = ExecutionRulesV21()
        logger.info("SDXL智能分流集成初始化完成")
    
    def get_model_route_for_scene(
        self,
        scene: Dict[str, Any]
    ) -> Tuple[str, Optional[str], str]:
        """
        获取场景的模型路由
        
        Args:
            scene: 场景配置（v2.2-final格式）
            
        Returns:
            Tuple[base_model, identity_engine, reason]: 模型路由信息
        """
        # 从scene中提取信息
        model_route = scene.get("model_route", {})
        character = scene.get("character", {})
        
        # 如果model_route已经锁定，直接返回
        if model_route.get("locked", False):
            base_model = model_route.get("base_model", "flux")
            identity_engine = model_route.get("identity_engine", "pulid")
            reason = model_route.get("decision_reason", "direct_specification")
            return base_model, identity_engine, reason
        
        # 使用智能分流规则
        character_id = character.get("id", "")
        character_role_str = character.get("role", "main")
        has_character = character.get("present", False)
        
        # 转换角色类型
        if character_role_str == "main":
            character_role = CharacterRole.MAIN_CHARACTER
        elif character_role_str == "npc" or character_id.startswith("npc_"):
            character_role = CharacterRole.NPC
        elif character_role_str == "important_supporting":
            character_role = CharacterRole.IMPORTANT_SUPPORTING
        else:
            character_role = CharacterRole.MAIN_CHARACTER if has_character else CharacterRole.NO_CHARACTER
        
        # 获取Shot类型
        shot_type_str = scene.get("shot", {}).get("type", "medium")
        from execution_rules_v2_1 import ShotType
        try:
            shot_type = ShotType(shot_type_str)
        except ValueError:
            shot_type = None
        
        # 使用智能分流
        model, identity = self.rules.get_model_route(
            has_character=has_character,
            shot_type=shot_type,
            character_role=character_role,
            task_type=TaskType.CHARACTER_GENERATION,
            character_id=character_id
        )
        
        # 生成决策原因
        if character_role == CharacterRole.NPC:
            reason = f"npc_role -> {model.value} + {identity}"
        elif character_role == CharacterRole.MAIN_CHARACTER:
            reason = f"main_character -> {model.value} + {identity}"
        else:
            reason = f"character_role_{character_role.value} -> {model.value} + {identity}"
        
        return model.value, identity, reason
    
    def get_model_route_for_npc(
        self,
        npc_description: str,
        reference_image: Optional[str] = None
    ) -> Tuple[str, Optional[str], str]:
        """
        获取NPC生成的模型路由
        
        Args:
            npc_description: NPC描述
            reference_image: 参考图像路径（可选）
            
        Returns:
            Tuple[base_model, identity_engine, reason]: 模型路由信息
        """
        model, identity = self.rules.get_model_route_for_npc(npc_description, reference_image)
        return model.value, identity, f"npc_generation -> {model.value} + {identity}"
    
    def get_model_route_for_outpainting(
        self,
        base_image_path: str,
        target_aspect_ratio: str = "9:16"
    ) -> Tuple[str, Optional[str], str]:
        """
        获取扩图的模型路由
        
        Args:
            base_image_path: 基础图像路径
            target_aspect_ratio: 目标宽高比
            
        Returns:
            Tuple[base_model, identity_engine, reason]: 模型路由信息
        """
        model, identity = self.rules.get_model_route_for_outpainting(base_image_path, target_aspect_ratio)
        return model.value, identity, f"outpainting -> {model.value}"
    
    def get_model_route_for_controlnet_layout(
        self,
        controlnet_type: str = "openpose"
    ) -> Tuple[str, Optional[str], str]:
        """
        获取构图控制的模型路由
        
        Args:
            controlnet_type: ControlNet类型
            
        Returns:
            Tuple[base_model, identity_engine, reason]: 模型路由信息
        """
        model, identity = self.rules.get_model_route_for_controlnet_layout(controlnet_type)
        return model.value, identity, f"controlnet_layout_{controlnet_type} -> {model.value}"


# 便捷函数
def get_sdxl_route_for_scene(scene: Dict[str, Any]) -> Tuple[str, Optional[str], str]:
    """
    获取场景的SDXL路由（便捷函数）
    
    Args:
        scene: 场景配置
        
    Returns:
        Tuple[base_model, identity_engine, reason]: 模型路由信息
    """
    integration = SDXLRoutingIntegration()
    return integration.get_model_route_for_scene(scene)

