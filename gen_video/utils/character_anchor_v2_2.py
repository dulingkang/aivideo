#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色锚系统 v2.2 - LoRA Stack（分层）支持

核心升级：
1. 支持LoRA Stack（分层LoRA）
   - 脸部LoRA（核心，永远不变）
   - 气质LoRA（阶段性，随剧情变）
   - 服饰LoRA（动态，随装备变）
   - 画风LoRA（可选）

2. 解耦训练策略支持
   - 脸部与服饰解耦
   - 气质与表情解耦

3. 运行时补丁
   - 气质锚点（Prompt中）
   - 显式锁词（服饰描述）
   - FaceDetailer支持（远景增强）

基于第三方分析报告（2025-12-20）的建议
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoRAItem:
    """单个LoRA项"""
    path: str
    weight: float
    trigger: Optional[str] = None  # 触发词（可选）
    adapter_name: Optional[str] = None  # 适配器名称（可选）
    layer: str = "unknown"  # 层级：face/costume/age/style


@dataclass
class LoRAStack:
    """LoRA堆栈配置"""
    # Layer 0: 核心脸部LoRA（永远不变，权重0.85-0.95）
    face_lora: Optional[LoRAItem] = None
    
    # Layer 1: 阶段性气质LoRA（随剧情变，权重0.6）
    age_lora: Optional[LoRAItem] = None
    
    # Layer 2: 服饰LoRA（动态，随装备变，权重0.8-0.9）
    costume_lora: Optional[LoRAItem] = None
    
    # Layer 3: 画风LoRA（可选，权重0.4）
    style_lora: Optional[LoRAItem] = None
    
    def get_active_loras(self) -> List[LoRAItem]:
        """获取激活的LoRA列表"""
        result = []
        if self.face_lora:
            result.append(self.face_lora)
        if self.age_lora:
            result.append(self.age_lora)
        if self.costume_lora:
            result.append(self.costume_lora)
        if self.style_lora:
            result.append(self.style_lora)
        return result
    
    def get_total_weight(self) -> float:
        """计算总权重（用于验证）"""
        total = 0.0
        for lora in self.get_active_loras():
            total += lora.weight
        return total


@dataclass
class CharacterAnchorV22:
    """角色锚配置 v2.2（支持LoRA Stack）"""
    character_id: str
    gender: str  # "male" 或 "female"
    
    # LoRA Stack（分层）
    lora_stack: LoRAStack = field(default_factory=LoRAStack)
    
    # 气质锚点（运行时补丁1）
    temperament_anchor: Optional[str] = None
    # 示例: "calm and restrained temperament, sharp but composed eyes"
    
    # 显式锁词（运行时补丁3）
    costume_lock_words: Optional[List[str]] = None
    # 示例: ["wearing his iconic mid-late-stage green daoist robe"]
    
    # Layer 1: 身份增强（可选）
    instantid_enabled: bool = True
    instantid_condition: str = "face_visible"
    instantid_strength: float = 0.75
    
    # Layer 2: 性别负锁（工业级标配）
    gender_negative_lock: List[str] = field(default_factory=list)
    
    # FaceDetailer配置（运行时补丁2）
    face_refine: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.gender_negative_lock:
            self.gender_negative_lock = self._get_default_gender_lock()
        
        if self.costume_lock_words is None:
            self.costume_lock_words = []
    
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
        if self.temperament_anchor:
            return self.temperament_anchor
        # 默认气质（根据角色）
        if self.character_id == "hanli":
            return "calm and restrained temperament, sharp but composed eyes"
        return ""
    
    def get_costume_lock_prompt(self) -> str:
        """获取服饰显式锁词Prompt"""
        if self.costume_lock_words:
            return ", ".join(self.costume_lock_words)
        return ""


class CharacterAnchorManagerV22:
    """
    角色锚管理器 v2.2（支持LoRA Stack）
    
    核心功能：
    1. 管理LoRA Stack（分层LoRA）
    2. 管理气质锚点
    3. 管理显式锁词
    4. 管理FaceDetailer配置
    5. 确保角色锚永不丢失
    """
    
    def __init__(self, character_profiles: Optional[Dict[str, Any]] = None):
        """
        初始化角色锚管理器
        
        Args:
            character_profiles: 角色档案字典（可选）
        """
        self.character_profiles = character_profiles or {}
        self._anchors: Dict[str, CharacterAnchorV22] = {}
        
        logger.info("角色锚管理器 v2.2 初始化完成（支持LoRA Stack）")
    
    def register_character_v22(
        self,
        character_id: str,
        gender: str,
        lora_stack: Optional[LoRAStack] = None,
        temperament_anchor: Optional[str] = None,
        costume_lock_words: Optional[List[str]] = None,
        instantid_enabled: bool = True,
        instantid_strength: float = 0.75,
        face_refine: Optional[Dict[str, Any]] = None
    ) -> CharacterAnchorV22:
        """
        注册角色锚 v2.2（支持LoRA Stack）
        
        Args:
            character_id: 角色ID
            gender: 性别
            lora_stack: LoRA堆栈配置
            temperament_anchor: 气质锚点（Prompt）
            costume_lock_words: 服饰显式锁词列表
            instantid_enabled: 是否启用InstantID
            instantid_strength: InstantID强度
            face_refine: FaceDetailer配置
            
        Returns:
            CharacterAnchorV22: 角色锚配置
        """
        anchor = CharacterAnchorV22(
            character_id=character_id,
            gender=gender,
            lora_stack=lora_stack or LoRAStack(),
            temperament_anchor=temperament_anchor,
            costume_lock_words=costume_lock_words,
            instantid_enabled=instantid_enabled,
            instantid_strength=instantid_strength,
            face_refine=face_refine
        )
        
        self._anchors[character_id] = anchor
        
        # 记录LoRA Stack信息
        active_loras = anchor.lora_stack.get_active_loras()
        logger.info(f"✓ 注册角色锚 v2.2: {character_id}")
        logger.info(f"  LoRA Stack: {len(active_loras)} 个LoRA")
        for lora in active_loras:
            logger.info(f"    - {lora.layer}: {Path(lora.path).name} (权重: {lora.weight})")
        if anchor.temperament_anchor:
            logger.info(f"  气质锚点: {anchor.temperament_anchor[:50]}...")
        if anchor.costume_lock_words:
            logger.info(f"  服饰锁词: {len(anchor.costume_lock_words)} 项")
        
        return anchor
    
    def register_character_from_json(
        self,
        character_config: Dict[str, Any]
    ) -> CharacterAnchorV22:
        """
        从JSON配置注册角色（v2.2格式）
        
        Args:
            character_config: 角色配置字典（v2.2格式）
            
        Returns:
            CharacterAnchorV22: 角色锚配置
        """
        character_id = character_config.get("id")
        gender = character_config.get("gender", "male")
        
        # 解析LoRA Stack
        lora_stack_config = character_config.get("lora_stack", [])
        lora_stack = self._parse_lora_stack(lora_stack_config)
        
        # 解析其他配置
        temperament_anchor = character_config.get("temperament_anchor")
        costume_lock_words = character_config.get("costume_lock_words", [])
        instantid_config = character_config.get("identity_engine", {})
        face_refine = character_config.get("face_refine")
        
        return self.register_character_v22(
            character_id=character_id,
            gender=gender,
            lora_stack=lora_stack,
            temperament_anchor=temperament_anchor,
            costume_lock_words=costume_lock_words,
            instantid_enabled=instantid_config.get("enabled", True),
            instantid_strength=instantid_config.get("strength", 0.75),
            face_refine=face_refine
        )
    
    def _parse_lora_stack(self, lora_stack_config: List[Dict[str, Any]]) -> LoRAStack:
        """解析LoRA Stack配置"""
        stack = LoRAStack()
        
        for item_config in lora_stack_config:
            layer = item_config.get("layer", "unknown")
            lora_item = LoRAItem(
                path=item_config["path"],
                weight=item_config.get("weight", 0.6),
                trigger=item_config.get("trigger"),
                adapter_name=item_config.get("adapter_name"),
                layer=layer
            )
            
            if layer == "face":
                stack.face_lora = lora_item
            elif layer == "age":
                stack.age_lora = lora_item
            elif layer == "costume":
                stack.costume_lora = lora_item
            elif layer == "style":
                stack.style_lora = lora_item
            else:
                logger.warning(f"未知的LoRA层级: {layer}")
        
        return stack
    
    def get_anchor(self, character_id: str) -> Optional[CharacterAnchorV22]:
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
        # 简单解析：shot_scale >= medium
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
        
        这是运行时补丁1和3的组合
        
        Args:
            character_id: 角色ID
            base_prompt: 基础Prompt
            
        Returns:
            str: 增强后的Prompt
        """
        anchor = self.get_anchor(character_id)
        if anchor is None:
            return base_prompt
        
        parts = []
        
        # 1. 气质锚点（最前面，最高优先级）
        temperament = anchor.get_temperament_prompt()
        if temperament:
            parts.append(temperament)
        
        # 2. 基础Prompt
        parts.append(base_prompt)
        
        # 3. 显式锁词（服饰描述，确保不被忽略）
        costume_lock = anchor.get_costume_lock_prompt()
        if costume_lock:
            parts.append(costume_lock)
        
        return ", ".join(filter(None, parts))
    
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
_character_anchor_manager_v22 = None


def get_character_anchor_manager_v22(
    character_profiles: Optional[Dict[str, Any]] = None
) -> CharacterAnchorManagerV22:
    """获取全局角色锚管理器 v2.2 实例"""
    global _character_anchor_manager_v22
    if _character_anchor_manager_v22 is None:
        _character_anchor_manager_v22 = CharacterAnchorManagerV22(character_profiles)
    return _character_anchor_manager_v22

