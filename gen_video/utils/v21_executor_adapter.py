#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2.1 Executor适配器 - 集成Execution Executor到现有系统

功能：
1. 将v2.1-exec格式转换为现有系统可用的格式
2. 桥接Execution Executor和ImageGenerator/VideoGenerator
3. 保持向后兼容
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from utils.execution_executor_v21 import (
    ExecutionExecutorV21,
    ExecutionConfig,
    ExecutionMode
)
from utils.execution_validator import ExecutionValidator
from utils.execution_rules_v2_1 import get_execution_rules

logger = logging.getLogger(__name__)


class V21ExecutorAdapter:
    """
    v2.1 Executor适配器
    
    将v2.1-exec格式的执行结果转换为现有系统可用的格式
    """
    
    def __init__(
        self,
        image_generator=None,
        video_generator=None,
        tts_generator=None,
        config: ExecutionConfig = None
    ):
        """
        初始化适配器
        
        Args:
            image_generator: ImageGenerator实例
            video_generator: VideoGenerator实例
            tts_generator: TTSGenerator实例
            config: Execution配置
        """
        self.image_generator = image_generator
        self.video_generator = video_generator
        self.tts_generator = tts_generator
        
        # 创建Execution Executor
        self.executor = ExecutionExecutorV21(
            config=config or ExecutionConfig(mode=ExecutionMode.STRICT),
            image_generator=image_generator,
            video_generator=video_generator,
            tts_generator=tts_generator
        )
        
        self.validator = ExecutionValidator()
        self.rules = get_execution_rules()
        
        logger.info("V21 Executor适配器初始化完成")
    
    def prepare_scene_for_generation(
        self,
        scene: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        准备场景用于生成（转换为现有系统格式）
        
        Args:
            scene: v2.1-exec格式的场景JSON
            
        Returns:
            现有系统可用的场景字典
        """
        # 1. 校验JSON
        validation_result = self.validator.validate_scene(scene)
        if not validation_result.is_valid:
            logger.error(f"场景 {scene.get('scene_id')} 校验失败")
            raise ValueError(f"场景校验失败: {validation_result.errors_count} 个错误")
        
        # 2. 构建生成参数
        model_route = scene["model_route"]
        character = scene.get("character", {})
        character_id = character.get("id")
        
        # 3. 获取角色锚配置
        from utils.character_anchor_v2_1 import get_character_anchor_manager
        anchor_manager = get_character_anchor_manager()
        anchor = anchor_manager.get_anchor(character_id) if character_id else None
        
        # 4. 构建现有系统格式的scene字典
        legacy_scene = {
            "scene_id": scene.get("scene_id", 0),
            "episode_id": scene.get("episode_id", ""),
            
            # Camera（保持原有格式）
            "camera": scene.get("camera", {}),
            
            # Character（添加v2.1信息）
            "character": {
                "present": character.get("present", False),
                "id": character_id,
                "gender": character.get("gender", "male"),
                "pose": scene["pose"]["type"],
                "face_visible": character.get("face_visible", True),
                "visibility": character.get("visibility", "mid"),
                "body_coverage": character.get("body_coverage", "half_body")
            },
            
            # Generation Policy（从model_route转换）
            "generation_policy": {
                "image_model": model_route["base_model"],
                "video_model": "hunyuan_i2v",
                "identity_engine": model_route["identity_engine"],
                "allow_face_lock": anchor_manager.should_use_instantid(
                    character_id, character.get("face_visible", True)
                ) if character_id else False,
                "allow_upscale": True
            },
            
            # Visual Constraints（从prompt转换）
            "visual_constraints": {
                "environment": scene.get("prompt", {}).get("scene_description", "")
            },
            
            # Quality Target
            "quality_target": {
                "style": scene.get("prompt", {}).get("style", "xianxia_anime"),
                "detail_level": "high",
                "lighting_style": "soft_cinematic",
                "motion_intensity": "gentle"
            },
            
            # Narration（保留）
            "narration": scene.get("narration", {}),
            
            # Duration
            "duration_sec": scene.get("duration_sec", 4.0),
            "target_fps": scene.get("target_fps", 24),
            
            # v2.1扩展信息（保留）
            "_v21_metadata": {
                "shot_locked": scene["shot"]["locked"],
                "pose_locked": scene["pose"]["locked"],
                "model_route": scene["model_route"],
                "decision_trace": self.executor._build_decision_trace(scene)
            }
        }
        
        return legacy_scene
    
    def generate_from_v21_scene(
        self,
        scene: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, Any]:
        """
        从v2.1-exec场景生成图像/视频
        
        Args:
            scene: v2.1-exec格式的场景JSON
            output_dir: 输出目录
            
        Returns:
            生成结果字典
        """
        # 1. 准备场景
        legacy_scene = self.prepare_scene_for_generation(scene)
        
        # 2. 构建Prompt（使用Executor的方法）
        prompt = self.executor._build_prompt(scene)
        negative_prompt = self.executor._build_negative_prompt(scene)
        
        # 3. 获取生成参数
        model_route = scene["model_route"]
        character = scene.get("character", {})
        character_id = character.get("id")
        
        # 4. 调用ImageGenerator（如果可用）
        image_path = None
        if self.image_generator:
            try:
                # 使用现有ImageGenerator生成
                # 这里需要根据实际ImageGenerator接口调整
                logger.info(f"使用ImageGenerator生成场景 {scene.get('scene_id')}")
                # image = self.image_generator.generate_scene(legacy_scene, prompt, negative_prompt)
                # image_path = str(Path(output_dir) / f"scene_{scene['scene_id']:03d}_image.png")
                # image.save(image_path)
                logger.warning("ImageGenerator集成待完成")
            except Exception as e:
                logger.error(f"图像生成失败: {e}")
                raise
        
        # 5. 调用VideoGenerator（如果可用）
        video_path = None
        if image_path and self.video_generator:
            try:
                logger.info(f"使用VideoGenerator生成视频场景 {scene.get('scene_id')}")
                # video_path = self.video_generator.generate_video(image_path, ...)
                logger.warning("VideoGenerator集成待完成")
            except Exception as e:
                logger.error(f"视频生成失败: {e}")
        
        return {
            "success": image_path is not None,
            "image_path": image_path,
            "video_path": video_path,
            "prompt": prompt,
            "negative_prompt": ", ".join(negative_prompt),
            "decision_trace": self.executor._build_decision_trace(scene)
        }


def create_adapter_from_config(
    config_path: str,
    execution_mode: ExecutionMode = ExecutionMode.STRICT
) -> V21ExecutorAdapter:
    """
    从配置文件创建适配器
    
    Args:
        config_path: 配置文件路径
        execution_mode: 执行模式
        
    Returns:
        V21ExecutorAdapter实例
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 懒加载生成器（可选）
    image_generator = None
    video_generator = None
    tts_generator = None
    
    # 如果需要，可以在这里初始化生成器
    # from image_generator import ImageGenerator
    # image_generator = ImageGenerator(config_path)
    
    executor_config = ExecutionConfig(mode=execution_mode)
    
    return V21ExecutorAdapter(
        image_generator=image_generator,
        video_generator=video_generator,
        tts_generator=tts_generator,
        config=executor_config
    )

