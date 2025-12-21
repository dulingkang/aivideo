#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行时补丁应用器

应用v2.2格式的运行时补丁：
1. 气质锚点（temperament_anchor）
2. 显式锁词（explicit_lock_words）
3. FaceDetailer（face_detailer）

这些补丁已经在Prompt构建时应用，这里提供工具函数用于ImageGenerator集成
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AnchorPatchesApplier:
    """运行时补丁应用器"""
    
    @staticmethod
    def apply_temperament_anchor(prompt: str, anchor: str) -> str:
        """
        应用气质锚点
        
        Args:
            prompt: 原始prompt
            anchor: 气质锚点文本
            
        Returns:
            应用后的prompt
        """
        if not anchor:
            return prompt
        
        # 如果prompt中已经包含anchor，不重复添加
        if anchor in prompt:
            return prompt
        
        # 在角色名称后添加气质锚点
        # 假设角色名称在prompt开头
        parts = prompt.split(",", 1)
        if len(parts) > 1:
            return f"{parts[0]}, {anchor}, {parts[1]}"
        else:
            return f"{prompt}, {anchor}"
    
    @staticmethod
    def apply_explicit_lock_words(prompt: str, lock_words: str) -> str:
        """
        应用显式锁词
        
        Args:
            prompt: 原始prompt
            lock_words: 显式锁词文本
            
        Returns:
            应用后的prompt
        """
        if not lock_words:
            return prompt
        
        # 如果prompt中已经包含lock_words，不重复添加
        if lock_words in prompt:
            return prompt
        
        # 在气质锚点后添加显式锁词
        return f"{prompt}, {lock_words}"
    
    @staticmethod
    def should_apply_face_detailer(
        scene: Dict[str, Any],
        face_detailer_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        判断是否应该应用FaceDetailer
        
        Args:
            scene: 场景配置
            face_detailer_config: FaceDetailer配置
            
        Returns:
            是否应该应用FaceDetailer
        """
        if not face_detailer_config or not face_detailer_config.get("enable", False):
            return False
        
        # 检查触发条件
        trigger = face_detailer_config.get("trigger", "shot_scale >= medium")
        
        # 解析触发条件
        shot_type = scene.get("shot", {}).get("type", "medium")
        
        # 简单的触发条件判断
        if "medium" in trigger or "wide" in trigger:
            # medium和wide镜头需要FaceDetailer
            return shot_type in ["medium", "wide"]
        elif "close_up" in trigger:
            # close_up通常不需要FaceDetailer
            return shot_type == "close_up"
        
        # 默认：medium及以上需要FaceDetailer
        return shot_type in ["medium", "wide"]
    
    @staticmethod
    def get_face_detailer_params(
        face_detailer_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        获取FaceDetailer参数
        
        Args:
            face_detailer_config: FaceDetailer配置
            
        Returns:
            FaceDetailer参数字典
        """
        if not face_detailer_config:
            return {}
        
        return {
            "enable": face_detailer_config.get("enable", False),
            "denoise": face_detailer_config.get("denoise", 0.35),
            "steps": face_detailer_config.get("steps", 12),
            "strength": face_detailer_config.get("strength", 0.8)
        }
    
    @staticmethod
    def apply_all_patches(
        prompt: str,
        scene: Dict[str, Any],
        character_info: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """
        应用所有运行时补丁
        
        Args:
            prompt: 原始prompt
            scene: 场景配置
            character_info: 角色信息
            
        Returns:
            (应用后的prompt, FaceDetailer参数)
        """
        anchor_patches = character_info.get("anchor_patches", {})
        
        # 应用气质锚点
        temperament_anchor = anchor_patches.get("temperament_anchor", "")
        if temperament_anchor:
            prompt = AnchorPatchesApplier.apply_temperament_anchor(prompt, temperament_anchor)
        
        # 应用显式锁词
        explicit_lock_words = anchor_patches.get("explicit_lock_words", "")
        if explicit_lock_words:
            prompt = AnchorPatchesApplier.apply_explicit_lock_words(prompt, explicit_lock_words)
        
        # 获取FaceDetailer参数
        face_detailer_config = anchor_patches.get("face_detailer", {})
        face_detailer_params = {}
        if AnchorPatchesApplier.should_apply_face_detailer(scene, face_detailer_config):
            face_detailer_params = AnchorPatchesApplier.get_face_detailer_params(face_detailer_config)
            logger.info(f"✓ FaceDetailer已启用: denoise={face_detailer_params.get('denoise')}")
        
        return prompt, face_detailer_params


# 便捷函数
def apply_anchor_patches(
    prompt: str,
    scene: Dict[str, Any],
    character_info: Dict[str, Any]
) -> tuple[str, Dict[str, Any]]:
    """
    应用所有运行时补丁（便捷函数）
    
    Args:
        prompt: 原始prompt
        scene: 场景配置
        character_info: 角色信息
        
    Returns:
        (应用后的prompt, FaceDetailer参数)
    """
    return AnchorPatchesApplier.apply_all_patches(prompt, scene, character_info)

