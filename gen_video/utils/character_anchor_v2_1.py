#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色锚系统 v2.1 - 角色永不丢失

核心原则：
1. 角色身份 = LoRA（唯一来源，永不失效）
2. InstantID只是增强器，不是角色本体
3. 性别负锁是工业级标配
4. 角色锚优先级：LoRA > InstantID > 风格LoRA
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class CharacterAnchor:
    """角色锚配置"""
    character_id: str
    gender: str  # "male" 或 "female"
    
    # Layer 0: 角色LoRA（永远存在）
    lora_path: Optional[str] = None
    lora_weight: float = 0.6
    
    # Layer 1: 身份增强（可选）
    instantid_enabled: bool = True
    instantid_condition: str = "face_visible"  # "face_visible" 或 "always"
    instantid_strength: float = 0.75
    
    # Layer 2: 性别负锁（工业级标配）
    gender_negative_lock: List[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.gender_negative_lock is None:
            self.gender_negative_lock = self._get_default_gender_lock()
    
    def _get_default_gender_lock(self) -> List[str]:
        """获取默认性别负锁"""
        if self.gender.lower() == "male":
            return [
                "female", "woman", "girl",
                "soft facial features", "delicate face",
                "long eyelashes", "narrow shoulders",
                "slim waist", "feminine body"
            ]
        elif self.gender.lower() == "female":
            return [
                "male", "man", "boy",
                "rough facial features", "strong jawline",
                "broad shoulders", "masculine body"
            ]
        else:
            logger.warning(f"未知的gender: {self.gender}，返回空列表")
            return []


class CharacterAnchorManager:
    """
    角色锚管理器
    
    核心功能：
    1. 管理角色LoRA配置
    2. 管理InstantID配置（条件启用）
    3. 管理性别负锁
    4. 确保角色锚永不丢失
    """
    
    def __init__(self, character_profiles: Optional[Dict[str, Any]] = None):
        """
        初始化角色锚管理器
        
        Args:
            character_profiles: 角色档案字典（可选）
        """
        self.character_profiles = character_profiles or {}
        self._anchors: Dict[str, CharacterAnchor] = {}
        
        logger.info("角色锚管理器初始化完成")
    
    def register_character(
        self,
        character_id: str,
        gender: str,
        lora_path: Optional[str] = None,
        lora_weight: float = 0.6,
        instantid_enabled: bool = True,
        instantid_strength: float = 0.75
    ) -> CharacterAnchor:
        """
        注册角色锚
        
        Args:
            character_id: 角色ID
            gender: 性别
            lora_path: LoRA路径（如果为None，会从character_profiles中查找）
            lora_weight: LoRA权重
            instantid_enabled: 是否启用InstantID
            instantid_strength: InstantID强度
            
        Returns:
            CharacterAnchor: 角色锚配置
        """
        # 如果lora_path为None，尝试从character_profiles中查找
        if lora_path is None:
            profile = self.character_profiles.get(character_id)
            if profile:
                lora_path = profile.get("lora_path") or profile.get("lora")
        
        anchor = CharacterAnchor(
            character_id=character_id,
            gender=gender,
            lora_path=lora_path,
            lora_weight=lora_weight,
            instantid_enabled=instantid_enabled,
            instantid_strength=instantid_strength
        )
        
        self._anchors[character_id] = anchor
        logger.info(f"✓ 注册角色锚: {character_id} (LoRA: {lora_path}, 性别: {gender})")
        
        return anchor
    
    def get_anchor(self, character_id: str) -> Optional[CharacterAnchor]:
        """
        获取角色锚
        
        Args:
            character_id: 角色ID
            
        Returns:
            CharacterAnchor: 角色锚配置，如果不存在则返回None
        """
        return self._anchors.get(character_id)
    
    def should_use_instantid(
        self,
        character_id: str,
        face_visible: bool = True
    ) -> bool:
        """
        判断是否应该使用InstantID
        
        Args:
            character_id: 角色ID
            face_visible: 人脸是否可见
            
        Returns:
            bool: 是否应该使用InstantID
        """
        anchor = self.get_anchor(character_id)
        if anchor is None:
            return False
        
        if not anchor.instantid_enabled:
            return False
        
        if anchor.instantid_condition == "face_visible":
            return face_visible
        elif anchor.instantid_condition == "always":
            return True
        else:
            return False
    
    def get_negative_prompt_with_gender_lock(
        self,
        character_id: str,
        base_negative: Optional[List[str]] = None
    ) -> List[str]:
        """
        获取带性别负锁的负面提示词
        
        Args:
            character_id: 角色ID
            base_negative: 基础负面提示词列表
            
        Returns:
            List[str]: 合并后的负面提示词列表
        """
        anchor = self.get_anchor(character_id)
        if anchor is None:
            return base_negative or []
        
        result = (base_negative or []).copy()
        result.extend(anchor.gender_negative_lock)
        
        return result


# 全局实例
_character_anchor_manager = None


def get_character_anchor_manager(
    character_profiles: Optional[Dict[str, Any]] = None
) -> CharacterAnchorManager:
    """获取全局角色锚管理器实例"""
    global _character_anchor_manager
    if _character_anchor_manager is None:
        _character_anchor_manager = CharacterAnchorManager(character_profiles)
    return _character_anchor_manager

