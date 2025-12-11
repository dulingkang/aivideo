#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型路由器
根据场景类型、用户等级、显存情况自动选择最优模型
"""

from typing import Dict, Any, Optional
import torch


class ModelRouter:
    """模型路由器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化路由器"""
        self.config = config or {}
        self.routing_config = self.config.get('model_routing', {})
        
        # 场景类型到模型的映射
        self.scene_type_mapping = self.routing_config.get('scene_type_mapping', {
            'government': 'hunyuanvideo',
            'enterprise': 'hunyuanvideo',
            'scientific': 'hunyuanvideo',
            'novel': 'cogvideox',
            'drama': 'cogvideox',
            'daily': 'cogvideox',
            'social': 'cogvideox',
            'general': 'cogvideox',
        })
        
        # 用户等级到模型的映射
        self.user_tier_mapping = self.routing_config.get('user_tier_mapping', {
            'free': 'cogvideox',
            'basic': 'cogvideox',
            'professional': 'cogvideox',
            'enterprise': 'hunyuanvideo',
            'premium': 'hunyuanvideo',
        })
        
        # 显存阈值
        self.memory_threshold = self.routing_config.get('memory_threshold', {
            'hunyuanvideo': 20,  # GB
            'cogvideox': 12,     # GB
        })
    
    def select_model(
        self,
        scene: Optional[Dict[str, Any]] = None,
        user_tier: str = 'basic',
        force_model: Optional[str] = None,
        available_memory: Optional[float] = None
    ) -> str:
        """
        选择最优模型
        
        Args:
            scene: 场景配置
            user_tier: 用户等级（free, basic, professional, enterprise, premium）
            force_model: 强制使用指定模型（可选）
            available_memory: 可用显存（GB），如果为None则自动检测
        
        Returns:
            选择的模型类型：'hunyuanvideo' 或 'cogvideox'
        """
        # 1. 如果强制指定模型，直接返回
        if force_model:
            return force_model
        
        # 2. 检测可用显存
        if available_memory is None:
            available_memory = self._get_available_memory()
        
        # 3. 根据用户等级初步选择
        model_by_tier = self.user_tier_mapping.get(user_tier, 'cogvideox')
        
        # 4. 根据场景类型选择
        model_by_scene = 'cogvideox'  # 默认
        if scene:
            scene_type = scene.get('type') or scene.get('scene_type') or 'general'
            model_by_scene = self.scene_type_mapping.get(scene_type, 'cogvideox')
        
        # 5. 决策逻辑
        # 优先考虑场景类型
        selected_model = model_by_scene
        
        # 如果场景类型指向HunyuanVideo，检查显存和用户等级
        if selected_model == 'hunyuanvideo':
            # 检查显存是否足够
            if available_memory < self.memory_threshold['hunyuanvideo']:
                print(f"  ⚠ 显存不足（{available_memory:.1f}GB < {self.memory_threshold['hunyuanvideo']}GB），降级到CogVideoX")
                selected_model = 'cogvideox'
            # 检查用户等级是否允许
            elif model_by_tier == 'cogvideox' and user_tier not in ['enterprise', 'premium']:
                print(f"  ⚠ 用户等级（{user_tier}）不允许使用HunyuanVideo，使用CogVideoX")
                selected_model = 'cogvideox'
        
        # 6. 如果用户等级指向HunyuanVideo，但场景类型是量产场景，使用CogVideoX
        if model_by_tier == 'hunyuanvideo' and model_by_scene == 'cogvideox':
            # 量产场景优先使用CogVideoX
            selected_model = 'cogvideox'
        
        return selected_model
    
    def _get_available_memory(self) -> float:
        """获取可用显存（GB）"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            reserved_memory = torch.cuda.memory_reserved(0) / 1024**3
            available = total_memory - reserved_memory
            return available
        except Exception:
            return 0.0
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'hunyuanvideo': {
                'name': 'HunyuanVideo 1.5',
                'description': '高质量视频生成，适合高端场景',
                'speed': '慢（20-30分钟）',
                'quality': '⭐⭐⭐⭐⭐',
                'memory_required': self.memory_threshold['hunyuanvideo'],
                'suitable_for': ['government', 'enterprise', 'scientific', 'premium']
            },
            'cogvideox': {
                'name': 'CogVideoX-5B',
                'description': '快速视频生成，适合量产场景',
                'speed': '快（2-5分钟）',
                'quality': '⭐⭐⭐⭐',
                'memory_required': self.memory_threshold['cogvideox'],
                'suitable_for': ['novel', 'drama', 'daily', 'social', 'general']
            }
        }
        return info.get(model_type, {})


if __name__ == "__main__":
    """测试模型路由器"""
    router = ModelRouter()
    
    # 测试场景1：小说推文（应该选择CogVideoX）
    scene1 = {"type": "novel", "description": "a story scene"}
    model1 = router.select_model(scene=scene1, user_tier="basic")
    print(f"场景1（小说推文）: {model1}")
    
    # 测试场景2：政府科普（应该选择HunyuanVideo，如果显存足够）
    scene2 = {"type": "government", "description": "a government scene"}
    model2 = router.select_model(scene=scene2, user_tier="enterprise", available_memory=25.0)
    print(f"场景2（政府科普，显存充足）: {model2}")
    
    # 测试场景3：政府科普（显存不足，降级到CogVideoX）
    model3 = router.select_model(scene=scene2, user_tier="enterprise", available_memory=15.0)
    print(f"场景3（政府科普，显存不足）: {model3}")
    
    # 测试场景4：基础用户尝试使用高端场景（应该降级）
    model4 = router.select_model(scene=scene2, user_tier="basic", available_memory=25.0)
    print(f"场景4（政府科普，基础用户）: {model4}")
    
    # 获取模型信息
    print(f"\n模型信息:")
    for model_type in ['hunyuanvideo', 'cogvideox']:
        info = router.get_model_info(model_type)
        print(f"  {model_type}: {info}")

