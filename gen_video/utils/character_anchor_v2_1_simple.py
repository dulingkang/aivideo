#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色锚系统 v2.1 简化版 - 单LoRA + 运行时补丁

适用于：
- 只有一个人脸LoRA的情况
- 不需要LoRA Stack的简单场景
- 快速部署和测试

核心功能：
1. 单LoRA支持（兼容现有训练）
2. 运行时补丁（3个MVP Fix）
   - 气质锚点
   - 显式锁词（服饰描述）
   - FaceDetailer支持（可选）
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class CharacterAnchorSimple:
    """角色锚配置（简化版，单LoRA）"""
    character_id: str
    gender: str  # "male" 或 "female"
    
    # 单LoRA（兼容现有训练）
    lora_path: Optional[str] = None
    lora_weight: float = 0.85  # 推荐0.85-0.95（根据第三方分析）
    
    # 运行时补丁1: 气质锚点
    temperament_anchor: Optional[str] = None
    # 示例: "calm and restrained temperament, sharp but composed eyes"
    
    # 运行时补丁3: 显式锁词（服饰描述）
    costume_lock_words: Optional[List[str]] = None
    # 示例: ["wearing his iconic mid-late-stage green daoist robe"]
    
    # Layer 1: 身份增强（可选）
    instantid_enabled: bool = True
    instantid_condition: str = "face_visible"
    instantid_strength: float = 0.75
    
    # Layer 2: 性别负锁（工业级标配）
    gender_negative_lock: List[str] = field(default_factory=list)
    
    # 运行时补丁2: FaceDetailer配置（可选）
    face_refine: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.gender_negative_lock:
            self.gender_negative_lock = self._get_default_gender_lock()
        
        if self.costume_lock_words is None:
            self.costume_lock_words = []
        
        # 如果没有设置气质锚点，使用默认值
        if self.temperament_anchor is None and self.character_id == "hanli":
            self.temperament_anchor = "calm and restrained temperament, sharp but composed eyes"
    
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
    
    def get_temperament_prompt(self) -> str:
        """获取气质锚点Prompt"""
        return self.temperament_anchor or ""
    
    def get_costume_lock_prompt(self) -> str:
        """获取服饰显式锁词Prompt"""
        if self.costume_lock_words:
            return ", ".join(self.costume_lock_words)
        return ""
    
    def get_enhanced_prompt(self, base_prompt: str) -> str:
        """
        获取增强后的Prompt（包含气质锚点和显式锁词）
        
        这是运行时补丁1和3的组合
        
        Args:
            base_prompt: 基础Prompt
            
        Returns:
            str: 增强后的Prompt
        """
        parts = []
        
        # 1. 气质锚点（最前面，最高优先级）
        temperament = self.get_temperament_prompt()
        if temperament:
            parts.append(temperament)
        
        # 2. 基础Prompt
        parts.append(base_prompt)
        
        # 3. 显式锁词（服饰描述，确保不被忽略）
        costume_lock = self.get_costume_lock_prompt()
        if costume_lock:
            parts.append(costume_lock)
        
        return ", ".join(filter(None, parts))


class CharacterAnchorManagerSimple:
    """
    角色锚管理器（简化版，单LoRA）
    
    适用于：
    - 只有一个人脸LoRA的情况
    - 不需要LoRA Stack的简单场景
    - 快速部署和测试
    """
    
    def __init__(self, character_profiles: Optional[Dict[str, Any]] = None):
        """
        初始化角色锚管理器
        
        Args:
            character_profiles: 角色档案字典（可选）
        """
        self.character_profiles = character_profiles or {}
        self._anchors: Dict[str, CharacterAnchorSimple] = {}
        
        logger.info("角色锚管理器（简化版）初始化完成（单LoRA + 运行时补丁）")
    
    def register_character_simple(
        self,
        character_id: str,
        gender: str,
        lora_path: Optional[str] = None,
        lora_weight: float = 0.85,
        temperament_anchor: Optional[str] = None,
        costume_lock_words: Optional[List[str]] = None,
        instantid_enabled: bool = True,
        instantid_strength: float = 0.75,
        face_refine: Optional[Dict[str, Any]] = None
    ) -> CharacterAnchorSimple:
        """
        注册角色锚（简化版，单LoRA）
        
        Args:
            character_id: 角色ID
            gender: 性别
            lora_path: LoRA路径
            lora_weight: LoRA权重（推荐0.85-0.95）
            temperament_anchor: 气质锚点（Prompt）
            costume_lock_words: 服饰显式锁词列表
            instantid_enabled: 是否启用InstantID
            instantid_strength: InstantID强度
            face_refine: FaceDetailer配置（可选）
            
        Returns:
            CharacterAnchorSimple: 角色锚配置
        """
        # 如果lora_path为None，尝试从character_profiles中查找
        if lora_path is None:
            profile = self.character_profiles.get(character_id)
            if profile:
                lora_path = profile.get("lora_path") or profile.get("lora")
        
        anchor = CharacterAnchorSimple(
            character_id=character_id,
            gender=gender,
            lora_path=lora_path,
            lora_weight=lora_weight,
            temperament_anchor=temperament_anchor,
            costume_lock_words=costume_lock_words,
            instantid_enabled=instantid_enabled,
            instantid_strength=instantid_strength,
            face_refine=face_refine
        )
        
        self._anchors[character_id] = anchor
        
        logger.info(f"✓ 注册角色锚（简化版）: {character_id}")
        logger.info(f"  LoRA: {Path(lora_path).name if lora_path else 'None'} (权重: {lora_weight})")
        if anchor.temperament_anchor:
            logger.info(f"  气质锚点: {anchor.temperament_anchor[:50]}...")
        if anchor.costume_lock_words:
            logger.info(f"  服饰锁词: {len(anchor.costume_lock_words)} 项")
        
        return anchor
    
    def register_character_from_config(
        self,
        character_config: Dict[str, Any]
    ) -> CharacterAnchorSimple:
        """
        从配置注册角色（兼容v2.1格式）
        
        Args:
            character_config: 角色配置字典
            
        Returns:
            CharacterAnchorSimple: 角色锚配置
        """
        character_id = character_config.get("id")
        gender = character_config.get("gender", "male")
        
        # 从配置中提取
        lora_path = character_config.get("lora_path")
        lora_weight = character_config.get("lora_weight", 0.85)
        temperament_anchor = character_config.get("temperament_anchor")
        costume_lock_words = character_config.get("costume_lock_words", [])
        
        instantid_config = character_config.get("identity_engine", {})
        face_refine = character_config.get("face_refine")
        
        return self.register_character_simple(
            character_id=character_id,
            gender=gender,
            lora_path=lora_path,
            lora_weight=lora_weight,
            temperament_anchor=temperament_anchor,
            costume_lock_words=costume_lock_words,
            instantid_enabled=instantid_config.get("enabled", True),
            instantid_strength=instantid_config.get("strength", 0.75),
            face_refine=face_refine
        )
    
    def get_anchor(self, character_id: str) -> Optional[CharacterAnchorSimple]:
        """获取角色锚"""
        return self._anchors.get(character_id)
    
    def should_use_instantid(
        self,
        character_id: str,
        face_visible: bool = True
    ) -> bool:
        """判断是否应该使用InstantID"""
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
    
    def should_use_face_refine(
        self,
        character_id: str,
        shot_type: str
    ) -> bool:
        """判断是否应该使用FaceDetailer（运行时补丁2）"""
        anchor = self.get_anchor(character_id)
        if anchor is None or anchor.face_refine is None:
            return False
        
        if not anchor.face_refine.get("enable", False):
            return False
        
        # 检查触发条件
        trigger = anchor.face_refine.get("trigger", "shot_scale >= medium")
        if "medium" in trigger or "wide" in trigger:
            if shot_type in ["medium", "wide"]:
                return True
        
        return False
    
    def get_enhanced_prompt(
        self,
        character_id: str,
        base_prompt: str
    ) -> str:
        """
        获取增强后的Prompt（包含气质锚点和显式锁词）
        
        Args:
            character_id: 角色ID
            base_prompt: 基础Prompt
            
        Returns:
            str: 增强后的Prompt
        """
        anchor = self.get_anchor(character_id)
        if anchor is None:
            return base_prompt
        
        return anchor.get_enhanced_prompt(base_prompt)
    
    def get_negative_prompt_with_gender_lock(
        self,
        character_id: str,
        base_negative: Optional[List[str]] = None
    ) -> List[str]:
        """获取带性别负锁的负面提示词"""
        anchor = self.get_anchor(character_id)
        if anchor is None:
            return base_negative or []
        
        result = (base_negative or []).copy()
        result.extend(anchor.gender_negative_lock)
        
        return result


# 全局实例
_character_anchor_manager_simple = None


def get_character_anchor_manager_simple(
    character_profiles: Optional[Dict[str, Any]] = None
) -> CharacterAnchorManagerSimple:
    """获取全局角色锚管理器（简化版）实例"""
    global _character_anchor_manager_simple
    if _character_anchor_manager_simple is None:
        _character_anchor_manager_simple = CharacterAnchorManagerSimple(character_profiles)
    return _character_anchor_manager_simple

