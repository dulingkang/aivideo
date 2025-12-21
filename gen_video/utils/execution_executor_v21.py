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
    
    def _normalize_scene_format(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """
        规范化场景格式，支持v2.1-exec和v2.2-final
        
        Args:
            scene: 原始场景JSON
            
        Returns:
            规范化后的场景JSON
        """
        version = scene.get("version", "")
        
        # v2.2-final格式：scene在顶层
        if version == "v2.2-final" and "scene" in scene:
            scene_data = scene["scene"].copy()  # 使用copy避免修改原数据
            # 保留version字段，验证器需要它
            scene_data["version"] = "v2.2-final"
            # 添加scene_id（如果缺失）
            if "scene_id" not in scene_data:
                scene_id_str = scene_data.get("id", "scene_001")
                try:
                    if scene_id_str.startswith("scene_"):
                        # 尝试提取数字部分
                        parts = scene_id_str.split("_")
                        if len(parts) > 1:
                            # 尝试解析数字
                            try:
                                scene_id = int(parts[1])
                            except ValueError:
                                # 如果不是数字，使用索引
                                scene_id = 1
                        else:
                            scene_id = 1
                    else:
                        # 尝试直接解析为数字
                        try:
                            scene_id = int(scene_id_str)
                        except ValueError:
                            scene_id = 1
                except Exception:
                    scene_id = 1
                scene_data["scene_id"] = scene_id
            return scene_data
        
        # v2.1-exec格式：直接返回
        return scene
    
    def execute_scene(
        self,
        scene: Dict[str, Any],
        output_dir: str
    ) -> ExecutionResult:
        """
        执行单个场景
        
        支持格式:
        - v2.1-exec: 旧格式（向后兼容）
        - v2.2-final: 新格式（推荐）
        
        Args:
            scene: v2.1-exec或v2.2-final格式的场景JSON
            output_dir: 输出目录
            
        Returns:
            ExecutionResult: 执行结果
        """
        # 0. 检测并规范化JSON格式
        scene = self._normalize_scene_format(scene)
        
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
        shot_info = scene.get("shot", {})
        pose_info = scene.get("pose", {})
        model_info = scene.get("model_route", {})
        character_info = scene.get("character", {})
        
        return {
            "shot": {
                "type": shot_info.get("type", "medium"),
                "source": shot_info.get("source", "direct_specification"),
                "locked": shot_info.get("locked", True)
            },
            "pose": {
                "type": pose_info.get("type", "stand"),
                "auto_corrected": pose_info.get("auto_corrected", False),
                "original": pose_info.get("original_pose")
            },
            "model_route": {
                "base_model": model_info.get("base_model", "flux"),
                "identity_engine": model_info.get("identity_engine", "pulid"),
                "reason": model_info.get("decision_reason", "direct_specification"),
                "character_role": model_info.get("character_role", "main")
            },
            "character_anchor": {
                "character_id": character_info.get("id"),
                "lora_enabled": True,  # LoRA总是启用
                "lora_config": character_info.get("lora_config", {}),
                "anchor_patches": character_info.get("anchor_patches", {})
            }
        }
    
    def _build_prompt(self, scene: Dict[str, Any]) -> str:
        """
        构建Prompt（模板填充，无LLM）
        
        支持v2.2-final格式的prompt配置
        """
        prompt_config = scene.get("prompt", {})
        character_info = scene.get("character", {})
        environment_info = scene.get("environment", {})
        
        # v2.2-final格式：直接使用final字段（如果存在）
        if "final" in prompt_config:
            return prompt_config["final"]
        
        # v2.2-final格式：使用base_template（如果存在）
        if "base_template" in prompt_config:
            template = prompt_config["base_template"]
            # 简单的模板替换
            template = template.replace("{{character.name}}", character_info.get("name", ""))
            template = template.replace("{{character.anchor_patches.temperament_anchor}}", 
                                       character_info.get("anchor_patches", {}).get("temperament_anchor", ""))
            template = template.replace("{{character.anchor_patches.explicit_lock_words}}", 
                                       character_info.get("anchor_patches", {}).get("explicit_lock_words", ""))
            template = template.replace("{{environment.location}}", environment_info.get("location", ""))
            template = template.replace("{{environment.atmosphere}}", environment_info.get("atmosphere", ""))
            return template
        
        # v2.1-exec格式：使用旧逻辑
        shot_type = scene.get("shot", {}).get("type", "medium")
        pose_type = scene.get("pose", {}).get("type", "stand")
        
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
        
        # 1. 角色名称和锚点（v2.2格式）
        if character_info:
            name = character_info.get("name", "")
            if name:
                parts.append(name)
            
            # 气质锚点
            temperament = character_info.get("anchor_patches", {}).get("temperament_anchor", "")
            if temperament:
                parts.append(temperament)
            
            # 显式锁词
            lock_words = character_info.get("anchor_patches", {}).get("explicit_lock_words", "")
            if lock_words:
                parts.append(lock_words)
        
        # 2. Shot描述
        parts.append(shot_desc)
        
        # 3. Pose描述
        if pose_type != "face_only":
            parts.append(pose_desc)
        
        # 4. 环境描述
        if environment_info:
            location = environment_info.get("location", "")
            if location:
                parts.append(f"in {location}")
            atmosphere = environment_info.get("atmosphere", "")
            if atmosphere:
                parts.append(f"{atmosphere} atmosphere")
        
        # 5. 场景描述（v2.1格式兼容）
        scene_desc = prompt_config.get("scene_description", "")
        if scene_desc:
            parts.append(scene_desc)
        
        # 6. 风格
        style = prompt_config.get("style", "")
        if style:
            parts.append(style)
        
        # 7. 通用质量标签
        parts.append("cinematic lighting, high detail, epic atmosphere")
        
        return ", ".join(filter(None, parts))
    
    def _build_negative_prompt(self, scene: Dict[str, Any]) -> List[str]:
        """构建负面Prompt（包含性别负锁）"""
        negative_base = [
            "low quality", "blurry", "distorted",
            "bad anatomy", "bad proportions"
        ]
        
        # v2.2-final格式：从character.negative_gender_lock读取
        character_info = scene.get("character", {})
        negative_gender_lock = character_info.get("negative_gender_lock", [])
        if negative_gender_lock:
            negative_base.extend(negative_gender_lock)
        
        # v2.1-exec格式：从negative_lock读取
        negative_lock = scene.get("negative_lock", {})
        if negative_lock.get("gender"):
            gender_lock_list = negative_lock.get("gender_male", []) or negative_lock.get("gender_female", [])
            if gender_lock_list:
                negative_base.extend(gender_lock_list)
            extra = negative_lock.get("extra", [])
            if extra:
                negative_base.extend(extra)
        
        return negative_base
    
    def _execute_image_generation(
        self,
        scene: Dict[str, Any],
        prompt: str,
        negative_prompt: List[str],
        output_dir: str
    ) -> Dict[str, Any]:
        """执行图像生成"""
        if self._image_generator is None:
            logger.warning("ImageGenerator未初始化")
            return {
                "success": False,
                "error": "ImageGenerator未初始化",
                "retryable": False
            }
        
        try:
            from pathlib import Path
            
            # 构建输出路径
            # ⚡ 修复：如果output_dir已经包含scene_XXX，不再重复添加
            scene_id = scene.get("scene_id", 0)
            output_dir_path = Path(output_dir)
            
            # 检查output_dir是否已经包含scene_XXX
            if output_dir_path.name.startswith("scene_"):
                # 已经包含scene_XXX，直接使用
                output_path = output_dir_path / "novel_image.png"
            else:
                # 需要添加scene_XXX
                output_path = output_dir_path / f"scene_{scene_id:03d}" / "novel_image.png"
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 构建scene字典（兼容ImageGenerator的格式）
            image_scene = scene.copy()
            
            # 添加width和height（如果scene中没有）
            if "width" not in image_scene:
                image_scene["width"] = 768  # 默认值
            if "height" not in image_scene:
                image_scene["height"] = 1152  # 默认值
            
            # 调用ImageGenerator生成图像
            logger.info(f"开始生成图像: scene_id={scene_id}, prompt={prompt[:50]}...")
            image_path = self._image_generator.generate_image(
                prompt=prompt,
                output_path=str(output_path),
                scene=image_scene,
                negative_prompt=", ".join(negative_prompt) if negative_prompt else None
            )
            
            # 检查文件是否生成成功
            if image_path and Path(image_path).exists():
                logger.info(f"图像生成成功: {image_path}")
                return {
                    "success": True,
                    "path": str(image_path)
                }
            else:
                logger.error(f"图像文件不存在: {image_path}")
                return {
                    "success": False,
                    "error": f"图像文件未生成: {image_path}",
                    "retryable": True
                }
            
        except Exception as e:
            logger.error(f"图像生成失败: {e}", exc_info=True)
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

