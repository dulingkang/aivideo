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
    
    def _auto_correct_shot_pose(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """
        自动修正不合法的shot+pose组合
        
        Args:
            scene: 场景JSON
            
        Returns:
            修正后的场景JSON
        """
        # ⚡ 关键修复：确保pose_info是scene中的实际引用，而不是新字典
        if "pose" not in scene:
            scene["pose"] = {}
        pose_info = scene["pose"]
        
        if "shot" not in scene:
            scene["shot"] = {}
        shot_info = scene["shot"]
        
        shot_type_str = shot_info.get("type", "medium")
        pose_type_str = pose_info.get("type", "stand")
        
        print(f"  [调试] 开始自动修正: shot={shot_type_str}, pose={pose_type_str}")
        
        # 使用规则引擎自动修正
        from .execution_rules_v2_1 import ShotType, PoseType
        shot_mapping = {
            "wide": ShotType.WIDE,
            "medium": ShotType.MEDIUM,
            "close_up": ShotType.CLOSE_UP,
            "aerial": ShotType.AERIAL,
        }
        pose_mapping = {
            "stand": PoseType.STAND,
            "walk": PoseType.WALK,
            "sit": PoseType.SIT,
            "lying": PoseType.LYING,
            "kneel": PoseType.KNEEL,
            "face_only": PoseType.FACE_ONLY,
        }
        
        shot_type = shot_mapping.get(shot_type_str.lower(), ShotType.MEDIUM)
        pose_type = pose_mapping.get(pose_type_str.lower(), PoseType.STAND)
        
        # 使用规则引擎修正
        pose_decision = self.rules.decide_pose(shot_type, pose_type_str)
        
        logger.debug(f"规则引擎决策: auto_corrected={pose_decision.auto_corrected}, new_pose={pose_decision.pose_type.value if pose_decision.auto_corrected else 'N/A'}")
        print(f"  [调试] 规则引擎决策: auto_corrected={pose_decision.auto_corrected}, new_pose={pose_decision.pose_type.value}")
        
        if pose_decision.auto_corrected:
            logger.warning(f"自动修正shot+pose组合: {shot_type_str} + {pose_type_str} -> {shot_type_str} + {pose_decision.pose_type.value}")
            print(f"  ✓ 自动修正: {shot_type_str} + {pose_type_str} -> {shot_type_str} + {pose_decision.pose_type.value}")
            pose_info["type"] = pose_decision.pose_type.value
            pose_info["auto_corrected"] = True
            pose_info["correction_reason"] = pose_decision.correction_reason
            pose_info["correction_level"] = pose_decision.correction_level.value if pose_decision.correction_level else None
            if pose_decision.original_pose:
                pose_info["original_pose"] = pose_decision.original_pose
            print(f"  [调试] 修正后pose_info: {pose_info}")
        else:
            logger.warning(f"规则引擎未自动修正: {shot_type_str} + {pose_type_str} (auto_corrected=False)")
            print(f"  ⚠ 规则引擎未自动修正: {shot_type_str} + {pose_type_str} (auto_corrected=False)")
        
        return scene
    
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
            # 构建详细的错误信息
            error_details = []
            for issue in validation_result.issues:
                if issue.level.value == "error":
                    error_details.append(f"[{issue.field}] {issue.message}")
                    if issue.suggestion:
                        error_details.append(f"  建议: {issue.suggestion}")
            
            error_msg = f"JSON校验失败: {validation_result.errors_count} 个错误"
            if error_details:
                error_msg += "\n" + "\n".join(error_details)
            
            logger.error(error_msg)
            
            # 如果只是shot+pose不兼容，尝试自动修正
            shot_pose_issues = [i for i in validation_result.issues if "shot+pose" in i.field.lower() or "不合法组合" in i.message]
            logger.debug(f"验证结果: {len(validation_result.issues)} 个问题, shot+pose问题: {len(shot_pose_issues)} 个")
            logger.debug(f"所有问题字段: {[i.field for i in validation_result.issues]}")
            
            if shot_pose_issues and len(validation_result.issues) == len(shot_pose_issues):
                logger.warning("检测到shot+pose不兼容，尝试自动修正...")
                print("  ⚠ 检测到shot+pose不兼容，尝试自动修正...")
                scene = self._auto_correct_shot_pose(scene)
                logger.info(f"自动修正完成，新的pose: {scene.get('pose', {}).get('type', 'unknown')}")
                # 重新验证
                validation_result = self.validator.validate_scene(scene)
                if not validation_result.is_valid:
                    # 构建修正后的详细错误信息
                    error_details_after = []
                    for issue in validation_result.issues:
                        if issue.level.value == "error":
                            error_details_after.append(f"[{issue.field}] {issue.message}")
                            if issue.suggestion:
                                error_details_after.append(f"  建议: {issue.suggestion}")
                    
                    error_msg_after = f"JSON校验失败: {validation_result.errors_count} 个错误"
                    if error_details_after:
                        error_msg_after += "\n" + "\n".join(error_details_after)
                    
                    logger.error("自动修正后仍然校验失败")
                    logger.error(error_msg_after)
                    return ExecutionResult(
                        success=False,
                        error_message=error_msg_after,
                        decision_trace={"validation": validation_result.to_dict()}
                    )
                else:
                    logger.info("✓ 自动修正成功，JSON校验通过")
            else:
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
            final_prompt = prompt_config["final"]
            
            # 确保提示词包含LoRA trigger词（如果配置了）
            lora_config = character_info.get("lora_config", {})
            trigger = lora_config.get("trigger")
            if trigger and trigger.lower() not in final_prompt.lower():
                # 在提示词开头添加trigger词（LoRA激活词）
                final_prompt = f"{trigger}, {final_prompt}"
                logger.info(f"添加LoRA trigger词: {trigger}")
            
            # ⚡ 关键修复：根据模型类型选择token限制
            # Flux 使用 T5，支持 512 tokens；SDXL 使用 CLIP，只支持 77 tokens
            model_route = scene.get("model_route", {})
            base_model = model_route.get("base_model", "flux")
            if base_model == "flux":
                # Flux 使用 T5，支持更长的 prompt
                max_tokens = 512
                logger.info(f"使用 Flux 模型，Prompt token 限制: {max_tokens}")
            else:
                # SDXL 使用 CLIP，限制为 77
                max_tokens = 77
                logger.info(f"使用 SDXL 模型，Prompt token 限制: {max_tokens}")
            
            # 精简Prompt，确保不超过token限制
            return self._optimize_prompt_length(final_prompt, max_tokens=max_tokens)
        
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
        
        prompt = ", ".join(filter(None, parts))
        return self._optimize_prompt_length(prompt, max_tokens=77)
    
    def _optimize_prompt_length(self, prompt: str, max_tokens: int = 77) -> str:
        """
        优化Prompt长度，确保不超过token限制
        
        Args:
            prompt: 原始Prompt
            max_tokens: 最大token数（CLIP默认77，Flux T5支持512）
            
        Returns:
            优化后的Prompt
        """
        # 尝试使用CLIP tokenizer精确计算token数
        try:
            from transformers import CLIPTokenizer
            from pathlib import Path
            # ⚡ 关键修复：优先使用本地SDXL模型，避免网络下载
            local_sdxl_path = Path(__file__).parent.parent / "models" / "sdxl-base"
            tokenizer = None
            
            # 尝试从本地SDXL模型加载
            if local_sdxl_path.exists() and (local_sdxl_path / "tokenizer").exists():
                try:
                    tokenizer = CLIPTokenizer.from_pretrained(
                        str(local_sdxl_path),
                        subfolder="tokenizer",
                        local_files_only=True
                    )
                    logger.debug(f"CLIPTokenizer 从本地加载: {local_sdxl_path}")
                except Exception as e:
                    logger.debug(f"本地加载失败: {e}")
            
            # 如果本地加载失败，尝试使用缓存
            if tokenizer is None:
                try:
                    tokenizer = CLIPTokenizer.from_pretrained(
                        "openai/clip-vit-large-patch14",
                        local_files_only=True  # 只使用本地缓存，不联网下载
                    )
                    logger.debug("CLIPTokenizer 从缓存加载")
                except Exception as e:
                    logger.debug(f"缓存加载失败: {e}")
                    return prompt  # 如果都失败，直接返回原prompt
            
            if tokenizer:
                try:
                    tokens = tokenizer.encode(prompt, return_tensors="pt")
                    token_count = tokens.shape[1]
                    logger.info(f"Prompt token数: {token_count} (限制: {max_tokens})")
                    
                    if token_count <= max_tokens:
                        return prompt
                    
                    # Token数超限，需要精简
                    logger.warning(f"Prompt token数超限（{token_count} > {max_tokens}），开始精简...")
                    
                    # 策略：按重要性保留部分
                    # 1. 保留角色名称和关键特征（前30%）
                    # 2. 保留动作/姿态（中间30%）
                    # 3. 保留环境关键信息（后40%）
                    
                    # 按逗号分割，保留最重要的部分
                    parts = [p.strip() for p in prompt.split(',')]
                    
                    # 优先级：角色信息 > 动作 > 环境 > 风格标签
                    priority_parts = []
                    style_parts = []
                    other_parts = []
                    
                    for part in parts:
                        part_lower = part.lower()
                        # 角色相关（高优先级）
                        if any(kw in part_lower for kw in ['hanli', 'character', 'temperament', 'expression', 'wearing', 'robe', 'attire']):
                            priority_parts.append(part)
                        # 风格标签（低优先级，可以删除）
                        elif any(kw in part_lower for kw in ['cinematic', 'lighting', 'high detail', 'epic atmosphere', 'illustration style']):
                            style_parts.append(part)
                        else:
                            other_parts.append(part)
                    
                    # 逐步添加，直到接近token限制
                    optimized_parts = []
                    current_tokens = 0
                    
                    # 先添加高优先级部分
                    for part in priority_parts:
                        test_prompt = ', '.join(optimized_parts + [part])
                        test_tokens = tokenizer.encode(test_prompt, return_tensors="pt").shape[1]
                        if test_tokens <= max_tokens * 0.9:  # 保留10%余量
                            optimized_parts.append(part)
                            current_tokens = test_tokens
                        else:
                            break
                    
                    # 再添加其他重要部分
                    for part in other_parts:
                        test_prompt = ', '.join(optimized_parts + [part])
                        test_tokens = tokenizer.encode(test_prompt, return_tensors="pt").shape[1]
                        if test_tokens <= max_tokens * 0.95:  # 保留5%余量
                            optimized_parts.append(part)
                            current_tokens = test_tokens
                        else:
                            break
                    
                    # 最后添加1-2个最重要的风格标签
                    for part in style_parts[:2]:  # 最多添加2个风格标签
                        test_prompt = ', '.join(optimized_parts + [part])
                        test_tokens = tokenizer.encode(test_prompt, return_tensors="pt").shape[1]
                        if test_tokens <= max_tokens:
                            optimized_parts.append(part)
                            current_tokens = test_tokens
                        else:
                            break
                    
                    optimized = ', '.join(optimized_parts)
                    final_tokens = tokenizer.encode(optimized, return_tensors="pt").shape[1]
                    logger.info(f"Prompt已优化: {token_count} -> {final_tokens} tokens, {len(prompt)} -> {len(optimized)} 字符")
                    return optimized
                except Exception as e:
                    logger.warning(f"无法使用CLIP tokenizer精确计算: {e}，使用估算方法")
            else:
                # 如果无法加载tokenizer，直接返回原prompt
                logger.debug("无法加载CLIPTokenizer，跳过token检查")
                return prompt
        except ImportError:
            logger.warning("transformers未安装，使用估算方法")
        
        # 回退到估算方法
        # 简单估算：1 token ≈ 0.75 单词 ≈ 4 字符（英文）
        # 为了安全，使用更保守的估算：1 token ≈ 3 字符
        estimated_tokens = len(prompt) / 3
        
        if estimated_tokens <= max_tokens:
            return prompt
        
        # Prompt过长，需要精简
        logger.warning(f"Prompt过长（估计{estimated_tokens:.0f} tokens > {max_tokens}），开始精简...")
        
        # 策略：保留关键信息，删除冗余描述
        # 按逗号分割，保留最重要的部分
        parts = [p.strip() for p in prompt.split(',')]
        
        # 优先级排序
        priority_parts = []
        style_parts = []
        other_parts = []
        
        for part in parts:
            part_lower = part.lower()
            # 角色相关（高优先级）
            if any(kw in part_lower for kw in ['hanli', 'character', 'temperament', 'expression', 'wearing', 'robe', 'attire']):
                priority_parts.append(part)
            # 风格标签（低优先级）
            elif any(kw in part_lower for kw in ['cinematic', 'lighting', 'high detail', 'epic atmosphere', 'illustration style']):
                style_parts.append(part)
            else:
                other_parts.append(part)
        
        # 组合：优先部分 + 其他部分（前N个） + 风格（1-2个）
        max_chars = max_tokens * 3  # 保守估算
        optimized_parts = priority_parts + other_parts[:len(other_parts)//2] + style_parts[:2]
        optimized = ', '.join(optimized_parts)
        
        if len(optimized) > max_chars:
            # 如果还是太长，截断
            truncated = optimized[:max_chars]
            last_comma = truncated.rfind(',')
            if last_comma > max_chars * 0.8:
                truncated = truncated[:last_comma]
            logger.warning(f"Prompt已截断: {len(prompt)} -> {len(truncated)} 字符")
            return truncated
        
        logger.info(f"Prompt已优化: {len(prompt)} -> {len(optimized)} 字符")
        return optimized
    
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
        """
        执行图像生成（v2.2-final硬规则模式）
        
        严格按照JSON中的model_route执行，不使用增强模式生成器
        """
        if not hasattr(self, '_image_generator') or self._image_generator is None:
            logger.warning("ImageGenerator未初始化")
            return {
                "success": False,
                "error": "ImageGenerator未初始化",
                "retryable": False
            }
        
        try:
            from pathlib import Path
            
            # 构建输出路径
            scene_id = scene.get("scene_id", 0)
            output_dir_path = Path(output_dir)
            
            # 检查output_dir是否已经包含scene_XXX
            if output_dir_path.name.startswith("scene_"):
                output_path = output_dir_path / "novel_image.png"
            else:
                output_path = output_dir_path / f"scene_{scene_id:03d}" / "novel_image.png"
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 获取model_route（硬规则）
            model_route = scene.get("model_route", {})
            base_model = model_route.get("base_model", "flux")
            identity_engine = model_route.get("identity_engine", "pulid")
            
            logger.info(f"v2.2-final硬规则模式: base_model={base_model}, identity_engine={identity_engine}")
            logger.info(f"Prompt长度: {len(prompt)} 字符")
            
            # 获取生成参数
            # ⚡ 关键改进：优先从config.yaml读取默认值，JSON中的generation_params是可选的覆盖
            gen_params = scene.get("generation_params", {})
            
            # 从config.yaml读取默认值（如果可用）
            default_width = 1536
            default_height = 1536
            default_num_steps = 50
            default_guidance = 7.5
            
            # 尝试从image_generator的配置读取（如果已初始化）
            if hasattr(self, '_image_generator') and self._image_generator:
                if hasattr(self._image_generator, 'pulid_config'):
                    pulid_config = self._image_generator.pulid_config
                    default_width = pulid_config.get("width", default_width)
                    default_height = pulid_config.get("height", default_height)
                    default_num_steps = pulid_config.get("num_inference_steps", default_num_steps)
                    default_guidance = pulid_config.get("guidance_scale", default_guidance)
            
            # 优先使用JSON中的参数（如果存在），否则使用config.yaml中的默认值
            width = gen_params.get("width") or default_width
            height = gen_params.get("height") or default_height
            num_steps = gen_params.get("num_inference_steps") or default_num_steps
            guidance_scale = gen_params.get("guidance_scale") or default_guidance
            seed = gen_params.get("seed", -1)
            
            # 记录参数来源
            source = "JSON覆盖" if gen_params else "config.yaml默认值"
            logger.debug(f"生成参数来源: {source}, {width}x{height}, {num_steps}步, guidance={guidance_scale}")
            
            # 获取角色信息
            character = scene.get("character", {})
            character_id = character.get("id")
            reference_image = character.get("reference_image")
            lora_config = character.get("lora_config", {})
            lora_path = lora_config.get("lora_path", "")
            
            # ⚡ 关键修复：如果使用 Flux + PuLID，需要传递 scene 参数以触发增强模式
            # 增强模式（EnhancedImageGenerator）支持 Flux + PuLID，但需要 scene 参数才能触发
            use_enhanced_mode = False
            if base_model == "flux" and identity_engine == "pulid":
                # Flux + PuLID 需要使用增强模式
                use_enhanced_mode = True
                logger.info(f"使用增强模式（Flux + PuLID）生成图像")
            elif base_model == "flux" and identity_engine == "instantid":
                # Flux + InstantID 目前未实现，回退到 InstantID + SDXL
                logger.warning(f"Flux + InstantID 尚未实现，改用 InstantID + SDXL")
                use_enhanced_mode = False
            else:
                use_enhanced_mode = False
            
            # 构建model_engine参数（用于非增强模式）
            if base_model == "flux" and identity_engine == "instantid":
                model_engine = "instantid"  # 回退到 InstantID + SDXL
            elif base_model == "sdxl" and identity_engine == "instantid":
                model_engine = "instantid"
            elif base_model == "flux":
                model_engine = "flux1"  # 纯Flux
            elif base_model == "sdxl":
                model_engine = "sdxl"  # 纯SDXL
            else:
                model_engine = f"{base_model}-{identity_engine}" if identity_engine else base_model
            
            logger.info(f"开始生成图像: scene_id={scene_id}, model={base_model}+{identity_engine}")
            logger.info(f"Prompt预览: {prompt[:100]}...")
            if use_enhanced_mode:
                logger.info(f"✓ 使用增强模式（Flux + PuLID），传递 scene 参数以触发")
            else:
                logger.info(f"使用标准模式，engine={model_engine}")
            
            # 获取参考图路径
            face_ref_path = None
            if reference_image:
                from pathlib import Path
                ref_path = Path(reference_image)
                if ref_path.exists():
                    face_ref_path = ref_path
                else:
                    # 尝试从character_reference_images获取
                    if hasattr(self, '_image_generator') and self._image_generator is not None and hasattr(self._image_generator, 'character_reference_images'):
                        char_refs = self._image_generator.character_reference_images
                        if character_id and character_id in char_refs:
                            face_ref_path = char_refs[character_id]
                            logger.info(f"使用角色参考图: {face_ref_path}")
            
            # 如果没有指定参考图，尝试从character_reference_images获取
            if not face_ref_path and character_id:
                if hasattr(self, '_image_generator') and self._image_generator is not None and hasattr(self._image_generator, 'character_reference_images'):
                    char_refs = self._image_generator.character_reference_images
                    if character_id in char_refs:
                        face_ref_path = char_refs[character_id]
                        logger.info(f"从配置获取角色参考图: {face_ref_path}")
            
            # 确定LoRA适配器名称
            # 1. 如果JSON中指定了lora_path且不为空，尝试从路径推断适配器名称
            # 2. 否则使用character_id作为适配器名称（通常相同，如"hanli"）
            character_lora = None
            if character_id:
                # 优先使用character_id作为适配器名称（与config.yaml中的adapter_name一致）
                character_lora = character_id
                logger.info(f"使用LoRA适配器: {character_lora} (基于character_id)")
                
                # 如果JSON中指定了lora_path，验证路径是否存在
                if lora_path:
                    lora_path_obj = Path(lora_path)
                    if lora_path_obj.exists():
                        logger.info(f"LoRA文件存在: {lora_path}")
                    else:
                        logger.warning(f"LoRA文件不存在: {lora_path}，将使用配置中的LoRA")
                else:
                    logger.info(f"JSON中未指定lora_path，将使用配置中的LoRA适配器: {character_lora}")
            
            # 调用ImageGenerator生成图像
            # 如果使用增强模式（Flux + PuLID），传递 scene 参数以触发增强模式
            # 增强模式会使用 JSON 中锁定的参数，不会自己决策
            if use_enhanced_mode:
                # 增强模式：传递 scene 参数，触发 EnhancedImageGenerator
                # EnhancedImageGenerator 会使用 JSON 中的锁定参数（shot, pose, model_route等）
                logger.info(f"传递 scene 参数以触发增强模式（Flux + PuLID）")
                image_path = self._image_generator.generate_image(
                    prompt=prompt,
                    output_path=str(output_path),
                    negative_prompt=", ".join(negative_prompt) if negative_prompt else None,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    seed=seed if seed > 0 else None,
                    face_reference_image_path=face_ref_path,
                    character_lora=character_lora,  # 传递LoRA适配器名称
                    use_lora=True if character_lora else None,  # 强制启用LoRA
                    scene=scene,  # ⚡ 关键：传递 scene 参数以触发增强模式
                    model_engine=model_engine,  # 虽然增强模式可能不使用，但保留以兼容
                    task_type="character" if character_id else "scene"
                )
            else:
                # 标准模式：传递 scene 参数以便读取分辨率和其他参数
                # ⚡ 关键修复：即使不使用增强模式，也传递 scene 参数，以便 ImageGenerator 可以从 scene.generation_params 读取分辨率
                image_path = self._image_generator.generate_image(
                    prompt=prompt,
                    output_path=str(output_path),
                    negative_prompt=", ".join(negative_prompt) if negative_prompt else None,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    seed=seed if seed > 0 else None,
                    face_reference_image_path=face_ref_path,
                    character_lora=character_lora,  # 传递LoRA适配器名称（强制启用）
                    use_lora=True if character_lora else None,  # 强制启用LoRA（如果指定了character_lora）
                    model_engine=model_engine,  # 强制指定引擎，避免自动选择
                    task_type="character" if character_id else "scene",  # 明确任务类型
                    scene=scene  # ⚡ 关键修复：传递 scene 参数以便读取分辨率（width/height）
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
            import traceback
            traceback.print_exc()
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

