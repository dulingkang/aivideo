#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型选择器 - 根据任务类型自动选择最适合的模型
"""

from typing import Dict, Any, Optional, Literal
from enum import Enum


class TaskType(Enum):
    """任务类型"""
    CHARACTER = "character"  # 人物生成（主持人）
    SCENE = "scene"         # 场景生成（科普背景）
    BATCH = "batch"          # 批量生成（备选图）


class ModelSelector:
    """模型选择器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_config = config.get("image", {})
        self.model_selection = self.image_config.get("model_selection", {})
    
    def select_engine(
        self,
        task_type: Optional[TaskType] = None,
        prompt: Optional[str] = None,
        scene_context: Optional[Dict[str, Any]] = None,
        manual_engine: Optional[str] = None
    ) -> str:
        """
        根据任务类型和提示词选择最适合的模型引擎
        
        Args:
            task_type: 任务类型（人物/场景/批量）
            prompt: 生成提示词
            scene_context: 场景上下文信息
            manual_engine: 手动指定的引擎（优先级最高）
        
        Returns:
            模型引擎名称
        """
        # 如果手动指定了引擎，直接使用
        if manual_engine and manual_engine != "auto":
            return manual_engine
        
        # 如果没有指定任务类型，尝试自动检测
        if task_type is None:
            task_type = self._detect_task_type(prompt, scene_context)
        
        if task_type == TaskType.CHARACTER:
            # 人物生成：使用 Flux + InstantID
            return "flux-instantid"
        
        elif task_type == TaskType.SCENE:
            # 场景生成：根据提示词内容选择
            return self._select_scene_engine(prompt, scene_context)
        
        elif task_type == TaskType.BATCH:
            # 批量生成：使用 SD3 Turbo
            return "sd3-turbo"
        
        else:
            # 默认使用当前方案（InstantID）
            return self.image_config.get("engine", "instantid")
    
    def _detect_task_type(
        self,
        prompt: Optional[str],
        scene_context: Optional[Dict[str, Any]]
    ) -> TaskType:
        """自动检测任务类型"""
        # 如果场景上下文中有 face_reference_image_path，判断为人物生成
        if scene_context:
            if scene_context.get("face_reference_image_path"):
                return TaskType.CHARACTER
            if scene_context.get("character_lora"):
                return TaskType.CHARACTER
        
        # 如果提示词中包含人物相关关键词，判断为人物生成
        if prompt:
            character_keywords = [
                "主持人", "讲解员", "人物", "角色", "人像", "肖像",
                "presenter", "host", "character", "portrait", "person"
            ]
            prompt_lower = prompt.lower()
            if any(keyword in prompt_lower for keyword in character_keywords):
                return TaskType.CHARACTER
        
        # 默认判断为场景生成
        return TaskType.SCENE
    
    def _select_scene_engine(
        self,
        prompt: Optional[str],
        scene_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """根据提示词选择场景生成引擎"""
        if not prompt:
            # 默认使用 Flux.2（科学背景图，冲击力强）
            return "flux2"
        
        prompt_lower = prompt.lower()
        
        # Flux.2 场景关键词（科学背景图、太空/粒子/量子类，冲击力强）
        flux2_keywords = [
            "太空", "宇宙", "粒子", "量子", "科学背景", "科技背景",
            "冲击", "震撼", "爆炸", "能量", "光束", "粒子效果",
            "space", "particle", "quantum", "scientific background",
            "impact", "explosive", "energy", "beam"
        ]
        
        # Flux.1 场景关键词（实验室/医学，更干净自然）
        flux1_keywords = [
            "实验室", "医学", "医疗", "医院", "手术", "实验设备",
            "干净", "自然", "清晰", "专业",
            "laboratory", "medical", "hospital", "surgery", "clean", "natural"
        ]
        
        # 中文场景关键词（优先使用 Hunyuan-DiT）
        chinese_scene_keywords = [
            "科技馆", "博物馆", "天文台",
            "中国", "中文", "科教", "科普", "教育",
            "研究", "学术", "教学"
        ]
        
        # 真实感场景关键词（优先使用 Kolors）
        realism_keywords = [
            "真实", "照片", "摄影", "手部", "光影", "细节",
            "realistic", "photorealistic", "photo", "photography"
        ]
        
        # 批量生成关键词（使用 SD3 Turbo）
        batch_keywords = [
            "批量", "备选", "多个", "variations", "batch", "multiple"
        ]
        
        # 检查是否包含批量生成关键词
        if any(keyword in prompt_lower for keyword in batch_keywords):
            return "sd3-turbo"
        
        # 检查是否包含 Flux.2 关键词（优先）
        if any(keyword in prompt_lower for keyword in flux2_keywords):
            return "flux2"  # 科学背景图、太空/粒子/量子类，冲击力强
        
        # 检查是否包含 Flux.1 关键词
        if any(keyword in prompt_lower for keyword in flux1_keywords):
            return "flux1"  # 实验室/医学，更干净自然
        
        # 检查是否包含中文场景关键词
        if any(keyword in prompt for keyword in chinese_scene_keywords):
            return "hunyuan-dit"
        
        # 检查是否包含真实感关键词
        if any(keyword in prompt_lower for keyword in realism_keywords):
            return "kolors"  # 使用 Kolors（真人质感强，中文 prompt 理解优秀）
        
        # 默认使用 Flux.2（科学背景图，冲击力强）
        return "flux2"
    
    def get_engine_config(self, engine: str) -> Dict[str, Any]:
        """获取指定引擎的配置"""
        model_selection = self.image_config.get("model_selection", {})
        
        if engine == "flux-instantid":
            return model_selection.get("character", {})
        elif engine == "hunyuan-dit":
            scene_config = model_selection.get("scene", {})
            return scene_config.get("hunyuan_dit", {})
        elif engine == "kolors":
            scene_config = model_selection.get("scene", {})
            return scene_config.get("kolors", {})
        elif engine == "realistic-vision":
            scene_config = model_selection.get("scene", {})
            return scene_config.get("realistic_vision", {})
        elif engine == "sd3-turbo":
            scene_config = model_selection.get("scene", {})
            return scene_config.get("sd3_turbo", {})
        else:
            # 默认配置（InstantID 或 SDXL）
            if engine == "instantid":
                return self.image_config.get("instantid", {})
            else:
                return self.image_config.get("sdxl", {})

