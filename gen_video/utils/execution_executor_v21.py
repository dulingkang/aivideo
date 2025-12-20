#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execution Executor V2.1 - 瘦身版执行器

核心原则：
1. 不"计划"，只"执行已锁定结构"
2. 完全确定性路径，无LLM参与
3. 只做模板填充和参数传递
4. 失败重试策略（同模型低风险重试）

这是 Execution Planner V3 的瘦身版，专门用于执行v2.1-exec格式的JSON。
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

try:
    from .execution_rules_v2_1 import get_execution_rules, ShotType, PoseType, ModelType
    from .character_anchor_v2_1 import get_character_anchor_manager
    from .execution_validator import ExecutionValidator, ValidationResult
except (ImportError, ValueError):
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    utils_path = Path(__file__).parent
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    from execution_rules_v2_1 import get_execution_rules, ShotType, PoseType, ModelType
    from character_anchor_v2_1 import get_character_anchor_manager
    from execution_validator import ExecutionValidator, ValidationResult

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """执行模式"""
    STRICT = "strict"      # 严格模式：完全不用LLM
    LLM_ASSIST = "llm_assist"  # LLM辅助模式：仅用于描述润色


@dataclass
class ExecutionConfig:
    """执行配置"""
    mode: ExecutionMode = ExecutionMode.STRICT
    enable_retry: bool = True
    max_retry: int = 1
    retry_on_artifact: List[str] = None  # ["gender_mismatch", "composition_error"]
    
    def __post_init__(self):
        if self.retry_on_artifact is None:
            self.retry_on_artifact = ["gender_mismatch", "composition_error"]


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    decision_trace: Dict[str, Any] = None


class ExecutionExecutorV21:
    """
    Execution Executor V2.1 - 瘦身版执行器
    
    核心功能：
    1. 校验JSON可执行性
    2. 构建Prompt（模板填充）
    3. 执行图像生成
    4. 执行视频生成
    5. 失败重试（同模型低风险）
    """
    
    def __init__(
        self,
        config: ExecutionConfig = None,
        image_generator=None,
        video_generator=None,
        tts_generator=None
    ):
        """
        初始化执行器
        
        Args:
            config: 执行配置
            image_generator: 图像生成器（可选，懒加载）
            video_generator: 视频生成器（可选，懒加载）
            tts_generator: TTS生成器（可选，懒加载）
        """
        self.config = config or ExecutionConfig()
        self.rules = get_execution_rules()
        self.anchor_manager = get_character_anchor_manager()
        self.validator = ExecutionValidator()
        
        # 生成器（懒加载）
        self._image_generator = image_generator
        self._video_generator = video_generator
        self._tts_generator = tts_generator
        
        logger.info(f"Execution Executor V2.1 初始化完成 (mode: {self.config.mode.value})")
    
    def execute_scene(
        self,
        scene: Dict[str, Any],
        output_dir: str
    ) -> ExecutionResult:
        """
        执行单个场景
        
        Args:
            scene: v2.1-exec格式的场景JSON
            output_dir: 输出目录
            
        Returns:
            ExecutionResult: 执行结果
        """
        # 1. 校验JSON
        validation_result = self.validator.validate_scene(scene)
        if not validation_result.is_valid:
            error_msg = f"JSON校验失败: {validation_result.errors_count} 个错误"
            logger.error(error_msg)
            return ExecutionResult(
                success=False,
                error_message=error_msg,
                decision_trace={"validation": validation_result.to_dict()}
            )
        
        # 2. 构建决策trace
        decision_trace = self._build_decision_trace(scene)
        
        # 3. 构建Prompt（模板填充，无LLM）
        prompt = self._build_prompt(scene)
        negative_prompt = self._build_negative_prompt(scene)
        
        # 4. 执行图像生成
        image_result = self._execute_image_generation(
            scene=scene,
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_dir=output_dir
        )
        
        if not image_result["success"]:
            # 尝试重试
            if self.config.enable_retry and image_result.get("retryable"):
                return self._retry_execution(scene, output_dir, decision_trace)
            return ExecutionResult(
                success=False,
                error_message=image_result.get("error"),
                decision_trace=decision_trace
            )
        
        # 5. 执行视频生成
        video_result = self._execute_video_generation(
            scene=scene,
            image_path=image_result["path"],
            prompt=prompt,
            output_dir=output_dir
        )
        
        # 6. 执行音频生成
        audio_result = self._execute_audio_generation(
            scene=scene,
            output_dir=output_dir
        )
        
        return ExecutionResult(
            success=True,
            image_path=image_result["path"],
            video_path=video_result.get("path"),
            audio_path=audio_result.get("path"),
            decision_trace=decision_trace
        )
    
    def _build_decision_trace(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """构建决策trace（可解释性）"""
        return {
            "shot": {
                "type": scene["shot"]["type"],
                "source": scene["shot"].get("source", "unknown"),
                "locked": scene["shot"]["locked"]
            },
            "pose": {
                "type": scene["pose"]["type"],
                "auto_corrected": scene["pose"].get("auto_corrected", False),
                "original": scene["pose"].get("original_pose")
            },
            "model_route": {
                "base_model": scene["model_route"]["base_model"],
                "identity_engine": scene["model_route"]["identity_engine"],
                "reason": scene["model_route"].get("decision_reason", ""),
                "confidence": scene["model_route"].get("confidence", 0.95)
            },
            "character_anchor": {
                "character_id": scene["character"].get("id"),
                "lora_enabled": True,  # LoRA总是启用
                "instantid_enabled": scene.get("identity_engine", {}).get("enabled", False)
            }
        }
    
    def _build_prompt(self, scene: Dict[str, Any]) -> str:
        """
        构建Prompt（模板填充，无LLM）
        
        这是Prompt Builder的瘦身版，只做字符串拼接
        """
        prompt_config = scene.get("prompt", {})
        shot_type = scene["shot"]["type"]
        pose_type = scene["pose"]["type"]
        
        # Shot描述模板
        shot_descriptions = {
            "wide": "远景，人物在画面中可见，显示广阔环境，风景构图",
            "medium": "中景，上半身，人物中等大小",
            "close_up": "近景，头部和肩膀，脸部突出",
            "aerial": "鸟瞰视角，俯视场景"
        }
        shot_desc = shot_descriptions.get(shot_type, "中景")
        
        # Pose描述模板
        pose_descriptions = {
            "stand": "standing pose, upright posture",
            "walk": "walking, in motion",
            "sit": "sitting, seated",
            "lying": "lying on the ground, reclined",
            "kneel": "kneeling, on knees",
            "face_only": "face only, close-up"
        }
        pose_desc = pose_descriptions.get(pose_type, "standing pose")
        
        # 合并Prompt
        parts = []
        
        # 1. Shot描述
        parts.append(shot_desc)
        
        # 2. Pose描述
        if pose_type != "face_only":
            parts.append(pose_desc)
        
        # 3. 场景描述
        scene_desc = prompt_config.get("scene_description", "")
        if scene_desc:
            parts.append(scene_desc)
        
        # 4. 角色描述
        positive_core = prompt_config.get("positive_core", "")
        if positive_core:
            parts.append(positive_core)
        
        # 5. 风格
        style = prompt_config.get("style", "")
        if style:
            parts.append(style)
        
        return ", ".join(filter(None, parts))
    
    def _build_negative_prompt(self, scene: Dict[str, Any]) -> List[str]:
        """构建负面Prompt（包含性别负锁）"""
        negative_base = [
            "low quality", "blurry", "distorted",
            "bad anatomy", "bad proportions"
        ]
        
        # 添加性别负锁
        negative_lock = scene.get("negative_lock", {})
        if negative_lock.get("gender"):
            negative_base.extend(negative_lock.get("extra", []))
        
        return negative_base
    
    def _execute_image_generation(
        self,
        scene: Dict[str, Any],
        prompt: str,
        negative_prompt: List[str],
        output_dir: str
    ) -> Dict[str, Any]:
        """执行图像生成"""
        # 这里应该调用实际的ImageGenerator
        # 为了演示，返回模拟结果
        
        if self._image_generator is None:
            logger.warning("ImageGenerator未初始化，返回模拟结果")
            return {
                "success": False,
                "error": "ImageGenerator未初始化",
                "retryable": False
            }
        
        try:
            # 获取模型路由
            model_route = scene["model_route"]
            base_model = model_route["base_model"]
            identity_engine = model_route["identity_engine"]
            
            # 获取角色锚
            character = scene["character"]
            character_id = character.get("id")
            
            # 构建生成参数
            generation_params = {
                "prompt": prompt,
                "negative_prompt": ", ".join(negative_prompt),
                "model": base_model,
                "identity_engine": identity_engine,
                "character_id": character_id,
                "output_dir": output_dir
            }
            
            # 调用ImageGenerator（这里需要实际实现）
            # image_path = self._image_generator.generate(**generation_params)
            
            # 模拟成功
            return {
                "success": True,
                "path": f"{output_dir}/scene_{scene['scene_id']:03d}_image.png"
            }
            
        except Exception as e:
            logger.error(f"图像生成失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "retryable": True
            }
    
    def _execute_video_generation(
        self,
        scene: Dict[str, Any],
        image_path: str,
        prompt: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """执行视频生成"""
        if self._video_generator is None:
            logger.warning("VideoGenerator未初始化，返回模拟结果")
            return {"success": False}
        
        try:
            # 调用VideoGenerator（这里需要实际实现）
            # video_path = self._video_generator.generate(...)
            
            # 模拟成功
            return {
                "success": True,
                "path": f"{output_dir}/scene_{scene['scene_id']:03d}_video.mp4"
            }
        except Exception as e:
            logger.error(f"视频生成失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_audio_generation(
        self,
        scene: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, Any]:
        """执行音频生成"""
        if self._tts_generator is None:
            logger.warning("TTSGenerator未初始化，返回模拟结果")
            return {"success": False}
        
        try:
            narration = scene.get("narration", {})
            text = narration.get("text", "")
            voice_id = narration.get("voice_id", "default")
            
            # 调用TTSGenerator（这里需要实际实现）
            # audio_path = self._tts_generator.generate(text, voice_id, output_dir)
            
            # 模拟成功
            return {
                "success": True,
                "path": f"{output_dir}/scene_{scene['scene_id']:03d}_audio.wav"
            }
        except Exception as e:
            logger.error(f"音频生成失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _retry_execution(
        self,
        scene: Dict[str, Any],
        output_dir: str,
        decision_trace: Dict[str, Any]
    ) -> ExecutionResult:
        """
        失败重试（同模型低风险重试）
        
        策略：
        - 不切换模型
        - 降低CFG scale
        - 降低InstantID强度
        """
        logger.info(f"开始重试执行场景 {scene['scene_id']}")
        
        # 修改参数（低风险调整）
        retry_scene = scene.copy()
        model_route = retry_scene["model_route"].copy()
        
        # 这里应该调整生成参数，但保持模型不变
        # 实际实现时需要修改ImageGenerator的调用参数
        
        # 模拟重试
        return ExecutionResult(
            success=False,
            error_message="重试失败（模拟）",
            retry_count=1,
            decision_trace=decision_trace
        )


# 便捷函数
def execute_scene_from_json(
    json_path: str,
    output_dir: str,
    config: ExecutionConfig = None
) -> ExecutionResult:
    """
    从JSON文件执行场景
    
    Args:
        json_path: JSON文件路径
        output_dir: 输出目录
        config: 执行配置
        
    Returns:
        ExecutionResult: 执行结果
    """
    import json
    
    with open(json_path, 'r', encoding='utf-8') as f:
        scene = json.load(f)
    
    executor = ExecutionExecutorV21(config=config)
    return executor.execute_scene(scene, output_dir)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python execution_executor_v21.py <scene.json> <output_dir>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    config = ExecutionConfig(mode=ExecutionMode.STRICT)
    result = execute_scene_from_json(json_path, output_dir, config)
    
    if result.success:
        print(f"✓ 执行成功")
        print(f"  图像: {result.image_path}")
        print(f"  视频: {result.video_path}")
        print(f"  音频: {result.audio_path}")
    else:
        print(f"✗ 执行失败: {result.error_message}")
        sys.exit(1)

