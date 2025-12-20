#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型图像生成器 - 整合 PuLID + 解耦融合 + Execution Planner V3

这个模块是现有 image_generator.py 的增强版本，
整合了新的架构组件以解决"人脸一致性 vs 环境丰富度"问题。

使用方式:
    from enhanced_image_generator import EnhancedImageGenerator
    
    gen = EnhancedImageGenerator("config.yaml")
    image = gen.generate_scene(scene_json)
"""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from PIL import Image
import logging
import warnings
import os
import gc

# ⚡ 关键修复：设置 PyTorch CUDA allocator 为可扩展段模式（解决显存碎片化问题）
# 这必须在导入任何 torch 模块之前设置
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ⚡ 抑制 CLIP tokenizer 的 77 token 警告（Flux 使用 T5 作为主编码器，支持 512 tokens，CLIP 只是辅助编码器）
warnings.filterwarnings("ignore", message=".*Token indices sequence length is longer than the specified maximum sequence length.*")
warnings.filterwarnings("ignore", message=".*The following part of your input was truncated because CLIP can only handle sequences up to 77 tokens.*")

# ⚡ 抑制 transformers 库直接打印的警告（通过环境变量）
# 这些警告是 transformers 库内部直接打印的，不是通过 warnings 模块
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # 设置为 error 级别，只显示错误

# 导入新的模块
from pulid_engine import PuLIDEngine, CharacterProfile
from decoupled_fusion_engine import DecoupledFusionEngine
from execution_planner_v3 import (
    ExecutionPlannerV3, 
    GenerationStrategy,
    GenerationMode,
    IdentityEngine,
    SceneEngine
)

logger = logging.getLogger(__name__)

# ⚡ 关键修复：在 logger 初始化后记录环境变量设置
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ and os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True":
    logger.info("  ✓ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 已设置（解决显存碎片化）")


class EnhancedImageGenerator:
    """
    增强型图像生成器
    
    整合了:
    - PuLID-FLUX (身份保持 + 环境融合)
    - 解耦融合引擎 (SAM2 + YOLO)
    - Execution Planner V3 (智能路由)
    - 角色档案系统 (多参考图)
    """
    
    def __init__(self, config_path: str = "config.yaml", enable_memory_manager: bool = True):
        """
        初始化增强型图像生成器
        
        Args:
            config_path: 配置文件路径
            enable_memory_manager: 是否启用显存管理器
        """
        self.config_path = config_path
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.image_config = self.config.get("image", {})
        
        # 提取各模块配置
        self.pulid_config = self.image_config.get("pulid", {})
        self.decoupled_config = self.image_config.get("decoupled_fusion", {})
        self.planner_config = self.image_config.get("execution_planner", {})
        self.profiles_config = self.image_config.get("character_profiles", {})
        
        # ⚡ 关键修复：传递完整的 config 给 ExecutionPlannerV3，确保能读取 prompt_engine 配置
        # ExecutionPlannerV3 需要读取 prompt_engine.scene_analyzer_mode 来初始化 LLM 客户端
        self.planner = ExecutionPlannerV3(self.config)  # 传递完整 config，而不是只有 execution_planner 部分
        self.pulid_engine = None  # 延迟加载
        self.fusion_engine = None  # 延迟加载
        self.flux_pipeline = None  # 延迟加载
        self.quality_analyzer = None  # 延迟加载
        self.image_generator = None  # 延迟加载 ImageGenerator 实例（用于 SDXL/InstantID）
        
        # 角色档案
        self.character_profiles = {}
        self._load_character_profiles()
        
        # 设备配置
        self.device = self.image_config.get("device", "cuda")
        
        # 显存管理器
        self.enable_memory_manager = enable_memory_manager
        self._memory_manager = None
        if enable_memory_manager:
            self._init_memory_manager()
        
        logger.info("EnhancedImageGenerator 初始化完成")
        logger.info(f"  PuLID 启用: {self.pulid_config.get('enabled', False)}")
        logger.info(f"  解耦融合启用: {self.decoupled_config.get('enabled', False)}")
        logger.info(f"  Planner 版本: V{self.planner_config.get('version', 3)}")
        logger.info(f"  显存管理器: {'启用' if enable_memory_manager else '禁用'}")
    
    def _init_memory_manager(self):
        """初始化显存管理器"""
        try:
            from utils.memory_manager import MemoryManager, MemoryPriority
            
            self._memory_manager = MemoryManager(
                warning_threshold=0.85,
                critical_threshold=0.95,
                auto_cleanup=True
            )
            
            # 注册模型加载器
            self._memory_manager.register_model(
                name="pulid_engine",
                loader=self._create_pulid_engine,
                unloader=self._unload_pulid_engine,
                priority=MemoryPriority.CRITICAL,
                estimated_size_gb=25.0  # PuLID + Flux 约占 25GB
            )
            
            self._memory_manager.register_model(
                name="fusion_engine",
                loader=self._create_fusion_engine,
                unloader=self._unload_fusion_engine,
                priority=MemoryPriority.HIGH,
                estimated_size_gb=3.0  # SAM2 + YOLO + InsightFace
            )
            
            self._memory_manager.register_model(
                name="quality_analyzer",
                loader=self._create_quality_analyzer,
                unloader=self._unload_quality_analyzer,
                priority=MemoryPriority.LOW,
                estimated_size_gb=1.0  # InsightFace only
            )
            
            logger.debug("显存管理器初始化完成")
            
        except ImportError:
            logger.warning("无法导入显存管理器，使用默认显存管理")
            self._memory_manager = None
    
    def _create_pulid_engine(self):
        """创建 PuLID 引擎（供显存管理器使用）"""
        engine_config = {
            "device": self.device,
            "quantization": self.pulid_config.get("quantization", "bfloat16"),
            "model_dir": os.path.dirname(os.path.dirname(
                self.pulid_config.get("model_path", "")
            )),
        }
        engine = PuLIDEngine(engine_config)
        engine.load_pipeline()
        return engine
    
    def _unload_pulid_engine(self, engine):
        """卸载 PuLID 引擎"""
        if engine:
            engine.unload()
    
    def _unload_pipeline_completely(self, pipeline):
        """
        完全卸载 pipeline（工程级实现）
        
        核心原则：同一时刻 GPU 上只允许一个 diffusion pipeline 存活
        
        Args:
            pipeline: 要卸载的 pipeline 对象
        """
        if pipeline is None:
            return
        
        logger.info("  开始完全卸载 pipeline...")
        
        try:
            # 1. 先移到 CPU（释放 GPU 显存）
            try:
                pipeline.to("cpu")
                logger.info("  ✓ Pipeline 已移到 CPU")
            except Exception as e:
                logger.warning(f"  移动 pipeline 到 CPU 时出错: {e}")
            
            # 2. 清理所有组件（逐个删除，确保彻底）
            components_to_clear = [
                'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2',
                'vae', 'unet', 'scheduler', 'image_encoder', 'feature_extractor',
                'controlnet', 'adapter', 'attention_processor'
            ]
            
            for comp_name in components_to_clear:
                if hasattr(pipeline, comp_name):
                    try:
                        comp = getattr(pipeline, comp_name)
                        if comp is not None:
                            # 先移到 CPU
                            try:
                                if hasattr(comp, 'to'):
                                    comp.to("cpu")
                            except:
                                pass
                            # 删除
                            delattr(pipeline, comp_name)
                            del comp
                    except Exception as e:
                        logger.debug(f"  清理组件 {comp_name} 时出错: {e}")
            
            # 3. 删除 pipeline 对象本身
            del pipeline
            
            logger.info("  ✓ Pipeline 组件已全部删除")
            
        except Exception as e:
            logger.warning(f"  卸载 pipeline 时出错: {e}")
        
        # 4. 硬 barrier：同步 + 清理缓存
        try:
            torch.cuda.synchronize()
        except:
            pass
        
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        
        logger.info("  ✓ Pipeline 卸载完成，GPU 缓存已清理")
    
    def _optimize_prompt_for_sdxl(self, prompt: str, image_generator) -> str:
        """
        为 SDXL 优化 prompt（SDXL 使用 CLIP tokenizer，77 tokens 限制）
        
        优先保留关键信息（场景、角色、姿态），确保不被截断
        """
        # 检查 prompt 长度（使用 CLIP tokenizer）
        if not hasattr(image_generator, '_clip_tokenizer') or image_generator._clip_tokenizer is None:
            # 如果没有 tokenizer，简单估算（平均每个词 1.3 tokens）
            estimated_tokens = len(prompt.split()) * 1.3
            if estimated_tokens <= 77:
                return prompt
            logger.warning(f"  ⚠ 无法精确计算 token 数，使用估算: {estimated_tokens:.1f} tokens")
        else:
            try:
                tokenizer = image_generator._clip_tokenizer
                tokens_obj = tokenizer(prompt, truncation=False, return_tensors="pt")
                actual_tokens = tokens_obj.input_ids.shape[1]
                if actual_tokens <= 77:
                    return prompt
                logger.warning(f"  ⚠ Prompt 长度 ({actual_tokens} tokens) 超过 77 tokens 限制，开始智能精简...")
            except Exception as e:
                logger.warning(f"  ⚠ Token 检查失败: {e}，使用简单估算")
                estimated_tokens = len(prompt.split()) * 1.3
                if estimated_tokens <= 77:
                    return prompt
        
        # 智能精简：优先保留关键信息
        import re
        
        # 1. 提取所有部分
        parts = [p.strip() for p in prompt.split(',')]
        
        # 2. 识别关键部分（场景、角色、姿态）
        key_keywords = [
            # 场景关键词
            'desert', '沙漠', 'gray-green', 'gray green', 'ground', '地面', 'floor',
            # 角色关键词
            'han li', 'hanli', '韩立', 'character', 'cultivator',
            # 姿态关键词
            'lying', '躺', 'motionless', '不动', 'prone', 'horizontal',
            # 镜头关键词
            'wide', '远景', 'full', '全身', 'visible', '可见'
        ]
        
        key_parts = []  # 必须保留的关键部分
        other_parts = []  # 其他部分
        
        for part in parts:
            part_lower = part.lower()
            is_key = any(kw in part_lower for kw in key_keywords)
            if is_key:
                key_parts.append(part)
            else:
                other_parts.append(part)
        
        # 3. 优先保留关键部分，然后添加其他部分直到达到限制
        selected_parts = []
        token_limit = 77
        
        # 先添加关键部分
        for part in key_parts:
            test_prompt = ', '.join(selected_parts + [part])
            try:
                if hasattr(image_generator, '_clip_tokenizer') and image_generator._clip_tokenizer is not None:
                    tokens_obj = image_generator._clip_tokenizer(test_prompt, truncation=False, return_tensors="pt")
                    test_tokens = tokens_obj.input_ids.shape[1]
                else:
                    test_tokens = len(test_prompt.split()) * 1.3
                
                if test_tokens <= token_limit:
                    selected_parts.append(part)
                else:
                    # 如果添加这个关键部分会超过，尝试精简它
                    # 移除权重语法和冗余描述
                    simplified = re.sub(r':\d+\.\d+', '', part)  # 移除权重
                    simplified = re.sub(r':\d+', '', simplified)  # 移除整数权重
                    simplified = simplified.strip()
                    test_prompt = ', '.join(selected_parts + [simplified])
                    if hasattr(image_generator, '_clip_tokenizer') and image_generator._clip_tokenizer is not None:
                        tokens_obj = image_generator._clip_tokenizer(test_prompt, truncation=False, return_tensors="pt")
                        test_tokens = tokens_obj.input_ids.shape[1]
                    else:
                        test_tokens = len(test_prompt.split()) * 1.3
                    
                    if test_tokens <= token_limit:
                        selected_parts.append(simplified)
            except Exception as e:
                logger.warning(f"  ⚠ 处理关键部分时出错: {e}，直接添加")
                selected_parts.append(part)
        
        # 4. 添加其他部分（如果还有空间）
        for part in other_parts:
            test_prompt = ', '.join(selected_parts + [part])
            try:
                if hasattr(image_generator, '_clip_tokenizer') and image_generator._clip_tokenizer is not None:
                    tokens_obj = image_generator._clip_tokenizer(test_prompt, truncation=False, return_tensors="pt")
                    test_tokens = tokens_obj.input_ids.shape[1]
                else:
                    test_tokens = len(test_prompt.split()) * 1.3
                
                if test_tokens <= token_limit:
                    selected_parts.append(part)
                else:
                    break  # 没有更多空间
            except Exception as e:
                logger.warning(f"  ⚠ 处理其他部分时出错: {e}，跳过")
                break
        
        optimized = ', '.join(selected_parts)
        
        # 最终验证
        try:
            if hasattr(image_generator, '_clip_tokenizer') and image_generator._clip_tokenizer is not None:
                tokens_obj = image_generator._clip_tokenizer(optimized, truncation=False, return_tensors="pt")
                final_tokens = tokens_obj.input_ids.shape[1]
            else:
                final_tokens = len(optimized.split()) * 1.3
            
            logger.info(f"  ✓ 智能精简完成: {len(selected_parts)} 个部分，{final_tokens:.1f} tokens")
            if final_tokens > token_limit:
                logger.warning(f"  ⚠ 精简后仍超过 {token_limit} tokens ({final_tokens:.1f} tokens)，可能会被截断")
        except Exception as e:
            logger.warning(f"  ⚠ 最终验证失败: {e}")
        
        return optimized
    
    def _unload_all_models(self):
        """
        卸载所有已加载的模型（Flux, PuLID, Fusion 等）
        
        这是切换模型前的"硬切换"操作
        """
        logger.info("  [硬切换] 开始卸载所有已加载的模型...")
        
        # 1. 卸载 Flux pipeline
        if self.flux_pipeline is not None:
            logger.info("  [硬切换] 卸载 Flux pipeline...")
            self._unload_pipeline_completely(self.flux_pipeline)
            self.flux_pipeline = None
        
        # 2. 卸载 PuLID 引擎
        if self.pulid_engine is not None:
            logger.info("  [硬切换] 卸载 PuLID 引擎...")
            try:
                if hasattr(self.pulid_engine, 'unload'):
                    self.pulid_engine.unload()
                # 清理所有可能的 GPU 对象
                if hasattr(self.pulid_engine, 'pipeline') and self.pulid_engine.pipeline is not None:
                    self._unload_pipeline_completely(self.pulid_engine.pipeline)
                if hasattr(self.pulid_engine, 'flux_model') and self.pulid_engine.flux_model is not None:
                    try:
                        self.pulid_engine.flux_model.to("cpu")
                        del self.pulid_engine.flux_model
                    except:
                        pass
                if hasattr(self.pulid_engine, 'face_analyzer') and self.pulid_engine.face_analyzer is not None:
                    try:
                        del self.pulid_engine.face_analyzer
                    except:
                        pass
                del self.pulid_engine
                self.pulid_engine = None
            except Exception as e:
                logger.warning(f"  [硬切换] 卸载 PuLID 引擎时出错: {e}")
        
        # 3. 卸载融合引擎
        if self.fusion_engine is not None:
            logger.info("  [硬切换] 卸载融合引擎...")
            try:
                if hasattr(self.fusion_engine, 'unload'):
                    self.fusion_engine.unload()
                # 清理 SAM2 和 YOLO
                if hasattr(self.fusion_engine, 'sam2_predictor') and self.fusion_engine.sam2_predictor is not None:
                    try:
                        del self.fusion_engine.sam2_predictor
                    except:
                        pass
                if hasattr(self.fusion_engine, 'yolo_model') and self.fusion_engine.yolo_model is not None:
                    try:
                        del self.fusion_engine.yolo_model
                    except:
                        pass
                del self.fusion_engine
                self.fusion_engine = None
            except Exception as e:
                logger.warning(f"  [硬切换] 卸载融合引擎时出错: {e}")
        
        # 4. 卸载质量分析器
        if self.quality_analyzer is not None:
            logger.info("  [硬切换] 卸载质量分析器...")
            try:
                if hasattr(self.quality_analyzer, 'unload'):
                    self.quality_analyzer.unload()
                if hasattr(self.quality_analyzer, 'face_analyzer') and self.quality_analyzer.face_analyzer is not None:
                    try:
                        del self.quality_analyzer.face_analyzer
                    except:
                        pass
                del self.quality_analyzer
                self.quality_analyzer = None
            except Exception as e:
                logger.warning(f"  [硬切换] 卸载质量分析器时出错: {e}")
        
        # 5. 卸载 ImageGenerator 的 pipeline
        if self.image_generator is not None:
            logger.info("  [硬切换] 卸载 ImageGenerator 的 pipeline...")
            try:
                if hasattr(self.image_generator, 'pipeline') and self.image_generator.pipeline is not None:
                    self._unload_pipeline_completely(self.image_generator.pipeline)
                    self.image_generator.pipeline = None
                if hasattr(self.image_generator, 'sdxl_pipeline') and self.image_generator.sdxl_pipeline is not None:
                    self._unload_pipeline_completely(self.image_generator.sdxl_pipeline)
                    self.image_generator.sdxl_pipeline = None
                if hasattr(self.image_generator, 'img2img_pipeline') and self.image_generator.img2img_pipeline is not None:
                    self._unload_pipeline_completely(self.image_generator.img2img_pipeline)
                    self.image_generator.img2img_pipeline = None
                if hasattr(self.image_generator, 'face_analyzer') and self.image_generator.face_analyzer is not None:
                    try:
                        del self.image_generator.face_analyzer
                        self.image_generator.face_analyzer = None
                    except:
                        pass
            except Exception as e:
                logger.warning(f"  [硬切换] 卸载 ImageGenerator pipeline 时出错: {e}")
        
        # 6. 最终硬 barrier
        try:
            torch.cuda.synchronize()
        except:
            pass
        
        for _ in range(5):  # 多次清理确保彻底
            gc.collect()
            torch.cuda.empty_cache()
        
        logger.info("  [硬切换] 所有模型已卸载，GPU 缓存已清理")
    
    def _create_fusion_engine(self):
        """创建融合引擎（供显存管理器使用）"""
        engine_config = {
            "device": self.device,
            "model_dir": os.path.dirname(
                self.decoupled_config.get("sam2_path", "")
            ),
        }
        return DecoupledFusionEngine(engine_config)
    
    def _unload_fusion_engine(self, engine):
        """卸载融合引擎"""
        if engine:
            engine.unload()
    
    def _create_quality_analyzer(self):
        """创建质量分析器（供显存管理器使用）"""
        from utils.image_quality_analyzer import ImageQualityAnalyzer
        return ImageQualityAnalyzer({
            "device": self.device,
            "insightface_root": os.path.dirname(
                self.decoupled_config.get("sam2_path", "models")
            )
        })
    
    def _unload_quality_analyzer(self, analyzer):
        """卸载质量分析器"""
        if analyzer:
            analyzer.unload()
    
    def _load_character_profiles(self):
        """加载角色档案"""
        if not self.profiles_config.get("enabled", False):
            return
        
        profiles_dir = self.profiles_config.get("profiles_dir", "")
        characters = self.profiles_config.get("characters", {})
        
        for char_id, char_config in characters.items():
            profile_path = os.path.join(profiles_dir, char_config.get("profile_dir", char_id))
            if os.path.exists(profile_path):
                self.character_profiles[char_id] = CharacterProfile(char_id, profile_path)
                logger.info(f"加载角色档案: {char_id}")
            else:
                logger.warning(f"角色档案目录不存在: {profile_path}")
    
    def _load_pulid_engine(self):
        """延迟加载 PuLID 引擎"""
        if self.pulid_engine is not None:
            return
        
        if not self.pulid_config.get("enabled", False):
            logger.warning("PuLID 未启用")
            return
        
        logger.info("加载 PuLID 引擎...")
        
        engine_config = {
            "device": self.device,
            "quantization": self.pulid_config.get("quantization", "bfloat16"),
            "model_dir": os.path.dirname(os.path.dirname(
                self.pulid_config.get("model_path", "")
            )),
        }
        
        self.pulid_engine = PuLIDEngine(engine_config)
        self.pulid_engine.load_pipeline()
        
        logger.info("PuLID 引擎加载完成")
    
    def _load_fusion_engine(self):
        """延迟加载解耦融合引擎"""
        if self.fusion_engine is not None:
            return
        
        if not self.decoupled_config.get("enabled", False):
            logger.warning("解耦融合未启用")
            return
        
        logger.info("加载解耦融合引擎...")
        
        engine_config = {
            "device": self.device,
            "model_dir": os.path.dirname(
                self.decoupled_config.get("sam2_path", "")
            ),
        }
        
        self.fusion_engine = DecoupledFusionEngine(engine_config)
        
        logger.info("解耦融合引擎加载完成")
    
    def _load_flux_pipeline(self):
        """延迟加载 Flux pipeline (用于场景生成)"""
        if self.flux_pipeline is not None:
            return
        
        # 检查 PuLID 是否已加载原生模式，如果是，复用其 Flux 模型
        if self.pulid_engine is not None and hasattr(self.pulid_engine, 'use_native') and self.pulid_engine.use_native:
            logger.info("PuLID 已使用原生模式，复用其 Flux 模型，跳过独立 Flux pipeline 加载")
            # 创建一个包装器，使用 PuLID 的 Flux 模型进行场景生成
            self.flux_pipeline = self._create_flux_wrapper_from_pulid()
            return
        
        logger.info("加载 Flux pipeline...")
        
        # 检查可用显存
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            logger.info(f"显存状态: 总计={total:.2f}GB, 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB, 可用={free:.2f}GB")
            
            # 如果可用显存少于 25GB，警告
            if free < 25:
                logger.warning(f"可用显存较少 ({free:.2f}GB)，可能会超出显存限制")
        
        try:
            from diffusers import FluxPipeline
            
            flux_path = self.pulid_config.get(
                "flux_path",
                self.image_config.get("model_selection", {}).get("scene", {}).get("flux1", {}).get("model_path", "")
            )
            
            self.flux_pipeline = FluxPipeline.from_pretrained(
                flux_path,
                torch_dtype=torch.bfloat16
            )
            self.flux_pipeline.enable_model_cpu_offload()
            
            logger.info("Flux pipeline 加载完成")
            
        except Exception as e:
            logger.error(f"Flux pipeline 加载失败: {e}")
            raise
    
    def _create_flux_wrapper_from_pulid(self):
        """从 PuLID 引擎创建 Flux pipeline 包装器
        
        创建一个包装器，使 PuLID 的 Flux 模型可以像 diffusers pipeline 一样使用
        这样可以避免重复加载 Flux 模型，节省显存
        """
        class FluxWrapper:
            def __init__(self, pulid_engine):
                self.pulid_engine = pulid_engine
                self.device = pulid_engine.device
                
            def __call__(self, prompt, width=768, height=1152, **kwargs):
                """使用 PuLID 的 Flux 模型生成场景（无身份注入）"""
                # 直接使用原生 Flux 模型生成（无身份注入）
                if hasattr(self.pulid_engine, 'flux_model') and self.pulid_engine.flux_model is not None:
                    image = self._generate_with_native_flux(prompt, width, height, **kwargs)
                    # 返回类似 diffusers pipeline 的对象（有 .images 属性）
                    class Result:
                        def __init__(self, img):
                            self.images = [img]
                    return Result(image)
                else:
                    raise RuntimeError("PuLID 原生 Flux 模型不可用")
            
            def _generate_with_native_flux(self, prompt, width, height, **kwargs):
                """使用原生 Flux 模型生成（无身份注入）"""
                from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
                import random
                
                engine = self.pulid_engine
                seed = kwargs.get('seed', random.randint(0, 2**32 - 1))
                
                # 准备噪声
                x = get_noise(
                    num_samples=1,
                    height=height,
                    width=width,
                    device=engine.device,
                    dtype=engine.dtype,
                    seed=seed
                )
                
                # 获取采样时间表
                num_steps = kwargs.get('num_inference_steps', 28)
                timesteps = get_schedule(
                    num_steps=num_steps,
                    image_seq_len=x.shape[1] * x.shape[2] // 4,
                    shift=True
                )
                
                # 准备输入 (临时移动编码器到 GPU)
                if hasattr(engine, 'use_cpu_offload') and engine.use_cpu_offload:
                    engine.t5.to(engine.device)
                    engine.clip.to(engine.device)
                
                inp = prepare(engine.t5, engine.clip, x, prompt)
                
                # 移回 CPU
                if hasattr(engine, 'use_cpu_offload') and engine.use_cpu_offload:
                    engine.t5.to("cpu")
                    engine.clip.to("cpu")
                    torch.cuda.empty_cache()
                
                # 执行去噪（无身份注入）
                # 使用带显存管理的去噪函数，避免 OOM
                guidance_scale = kwargs.get('guidance_scale', 3.5)
                
                # 检查显存情况，决定是否使用 aggressive_offload
                use_aggressive_offload = False
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    free = total - reserved
                    logger.info(f"  场景生成前显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB, 可用={free:.2f}GB")
                    
                    # 如果可用显存较少（<50GB），使用 aggressive_offload（更安全）
                    use_aggressive_offload = free < 50
                    if use_aggressive_offload:
                        logger.info(f"  可用显存较少 ({free:.2f}GB)，启用 aggressive_offload 模式")
                    else:
                        logger.info(f"  可用显存充足 ({free:.2f}GB)，使用标准模式（速度更快）")
                
                # 去噪前清理显存
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # 检查是否需要使用显存管理
                # 对于解耦融合模式的场景生成，总是使用显存管理
                if hasattr(engine, '_denoise_with_memory_management'):
                    x = engine._denoise_with_memory_management(
                        model=engine.flux_model,
                        img=inp["img"],
                        img_ids=inp["img_ids"],
                        txt=inp["txt"],
                        txt_ids=inp["txt_ids"],
                        vec=inp["vec"],
                        timesteps=timesteps,
                        guidance=guidance_scale,
                        id=None,  # 无身份注入
                        id_weight=0.0,
                        uncond_id=None,
                        aggressive_offload=use_aggressive_offload  # 根据显存情况动态决定
                    )
                else:
                    # 回退到原始 denoise（不推荐）
                    logger.warning("  PuLID 引擎不支持显存管理，使用原始 denoise（可能 OOM）")
                    x = denoise(
                        model=engine.flux_model,
                        img=inp["img"],
                        img_ids=inp["img_ids"],
                        txt=inp["txt"],
                        txt_ids=inp["txt_ids"],
                        vec=inp["vec"],
                        timesteps=timesteps,
                        guidance=guidance_scale,
                        id=None,  # 无身份注入
                        id_weight=0.0,
                        uncond_id=None,
                        aggressive_offload=use_aggressive_offload
                    )
                
                # 解包
                x = unpack(x.float(), height, width)
                
                # 使用 AutoEncoder 解码
                # 注意：AutoEncoder 对象没有 dtype 属性，需要从参数中获取
                ae_dtype = next(engine.ae.parameters()).dtype
                x = x.to(ae_dtype)
                with torch.no_grad():
                    x = engine.ae.decode(x)
                
                # 转换为图像
                x = (x + 1.0) / 2.0
                x = x.clamp(0, 1)
                x = x.cpu().permute(0, 2, 3, 1).numpy()[0]
                x = (x * 255).astype(np.uint8)
                
                from PIL import Image
                image = Image.fromarray(x)
                
                # 返回类似 diffusers 的格式
                class Result:
                    def __init__(self, image):
                        self.images = [image]
                
                return Result(image)
        
        return FluxWrapper(self.pulid_engine)
    
    def generate_scene(
        self,
        scene: Dict[str, Any],
        character_id: Optional[str] = None,
        face_reference: Optional[Union[str, Image.Image]] = None,
        original_prompt: Optional[str] = None,
        **kwargs
    ) -> Image.Image:
        """
        生成场景图像
        
        这是主入口方法，会根据场景自动选择最佳策略
        
        Args:
            scene: 场景 JSON (v2 格式)
            character_id: 角色 ID (可选，用于选择角色档案)
            face_reference: 人脸参考图 (可选，覆盖角色档案)
            original_prompt: 原始 prompt（如果提供，会优先使用它，而不是从 scene 构建）
            **kwargs: 额外参数
            
        Returns:
            生成的图像
        """
        # ⚡ 关键修复：将 scene 添加到 kwargs，确保传递给生成方法
        kwargs['scene'] = scene
        
        # ⚡ 关键修复：使用 print 确保日志输出到控制台（logger 可能被重定向）
        print("  [步骤0] 进入 generate_scene 方法...")
        logger.info("=" * 60)
        logger.info("开始生成场景图像")
        logger.info("=" * 60)
        
        import time
        start_time = time.time()
        
        # 1. 分析场景，获取策略
        print("  [步骤1] 分析场景，获取策略...")
        logger.info("  [步骤1] 分析场景，获取策略...")
        strategy_start = time.time()
        try:
            strategy = self.planner.analyze_scene(
                scene=scene,
                character_profiles=self.character_profiles
            )
            elapsed = time.time() - strategy_start
            print(f"  ✓ 场景分析完成 (耗时: {elapsed:.2f}秒)")
            logger.info(f"  ✓ 场景分析完成 (耗时: {elapsed:.2f}秒)")
        except Exception as e:
            elapsed = time.time() - strategy_start
            print(f"  ❌ 场景分析失败 (耗时: {elapsed:.2f}秒): {e}")
            logger.error(f"  ❌ 场景分析失败 (耗时: {elapsed:.2f}秒): {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 2. 准备参考图
        print("  [步骤2] 准备参考图...")
        logger.info("  [步骤2] 准备参考图...")
        ref_start = time.time()
        try:
            ref_image = self._prepare_reference(
                strategy=strategy,
                character_id=character_id,
                face_reference=face_reference
            )
            elapsed = time.time() - ref_start
            print(f"  ✓ 参考图准备完成 (耗时: {elapsed:.2f}秒)")
            logger.info(f"  ✓ 参考图准备完成 (耗时: {elapsed:.2f}秒)")
        except Exception as e:
            elapsed = time.time() - ref_start
            print(f"  ❌ 参考图准备失败 (耗时: {elapsed:.2f}秒): {e}")
            logger.error(f"  ❌ 参考图准备失败 (耗时: {elapsed:.2f}秒): {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 3. 构建 Prompt（如果提供了 original_prompt，优先使用它）
        print("  [步骤3] 构建 Prompt...")
        logger.info("  [步骤3] 构建 Prompt...")
        prompt_start = time.time()
        try:
            prompt = self.planner.build_weighted_prompt(scene, strategy, original_prompt=original_prompt)
            elapsed = time.time() - prompt_start
            print(f"  ✓ Prompt 构建完成 (耗时: {elapsed:.2f}秒)")
            logger.info(f"  ✓ Prompt 构建完成 (耗时: {elapsed:.2f}秒)")
            logger.info(f"完整 Prompt: {prompt}")
            logger.info(f"Prompt 预览: {prompt[:150]}...")
        except Exception as e:
            elapsed = time.time() - prompt_start
            print(f"  ❌ Prompt 构建失败 (耗时: {elapsed:.2f}秒): {e}")
            logger.error(f"  ❌ Prompt 构建失败 (耗时: {elapsed:.2f}秒): {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 4. 根据策略选择生成方式
        logger.info("  [步骤4] 根据策略选择生成方式...")
        logger.info(f"  策略决策: scene_engine={strategy.scene_engine.value}, identity_engine={strategy.identity_engine.value}, use_decoupled={strategy.use_decoupled_pipeline}")
        gen_start = time.time()
        
        # ⚡ 关键修复：优先检查 scene_engine，确保 Planner 的决策被正确执行
        # 1. 如果 Planner 决定使用 SDXL + InstantID（稳定性不足的场景）
        if strategy.scene_engine == SceneEngine.SDXL:
            if strategy.identity_engine == IdentityEngine.INSTANTID:
                logger.info("  ⚡ 使用 SDXL + InstantID（稳定方案，适用于不稳定场景）")
                image = self._generate_with_sdxl_instantid(
                    prompt=prompt,
                    face_reference=ref_image,
                    strategy=strategy,
                    **kwargs
                )
            else:
                logger.warning(f"  ⚠ Planner 决定使用 SDXL，但 identity_engine={strategy.identity_engine.value}，回退到标准 SDXL 生成")
                image = self._generate_with_sdxl(
                    prompt=prompt,
                    face_reference=ref_image,
                    strategy=strategy,
                    **kwargs
                )
        # 2. 如果没有参考图像，直接使用标准生成（不需要身份注入）
        elif ref_image is None:
            logger.info("  没有参考图像，使用标准生成模式（无身份注入）")
            image = self._generate_standard(
                prompt=prompt,
                face_reference=ref_image,
                strategy=strategy,
                **kwargs
            )
        # 3. 解耦模式（远景/中景，参考强度 < 70%）
        elif strategy.use_decoupled_pipeline and strategy.reference_strength < 70:
            logger.info("  使用解耦生成模式（远景/中景）")
            image = self._generate_decoupled(
                prompt=prompt,
                face_reference=ref_image,
                strategy=strategy,
                **kwargs
            )
        # 4. PuLID 模式（Flux + PuLID，稳定性良好的场景）
        elif strategy.identity_engine == IdentityEngine.PULID:
            logger.info("  ⚡ 使用 Flux + PuLID（上限方案，适用于稳定场景）")
            image = self._generate_with_pulid(
                prompt=prompt,
                face_reference=ref_image,
                strategy=strategy,
                **kwargs
            )
        # 5. 标准生成（InstantID 或无身份约束）
        else:
            logger.info("  使用标准生成模式 (InstantID 或无身份约束)")
            image = self._generate_standard(
                prompt=prompt,
                face_reference=ref_image,
                strategy=strategy,
                **kwargs
            )
        logger.info(f"  ✓ 图像生成完成 (耗时: {time.time()-gen_start:.2f}秒)")
        
        # 5. 质量验证
        # 注意：对于解耦模式，验证应该在最终图像上进行（已经在 _generate_decoupled 中完成）
        # 对于直接 PuLID 模式，在这里验证
        if strategy.verify_face_similarity and ref_image is not None and not strategy.use_decoupled_pipeline:
            self._verify_quality(image, ref_image, strategy)
        
        logger.info("场景图像生成完成")
        return image
    
    def _prepare_reference(
        self,
        strategy: GenerationStrategy,
        character_id: Optional[str],
        face_reference: Optional[Union[str, Image.Image]]
    ) -> Optional[Image.Image]:
        """准备参考图像"""
        # 优先使用传入的参考图
        if face_reference is not None:
            if isinstance(face_reference, str):
                return Image.open(face_reference).convert('RGB')
            return face_reference
        
        # 使用策略中的参考图
        if strategy.primary_reference:
            return Image.open(strategy.primary_reference).convert('RGB')
        
        # 使用角色档案
        if character_id and character_id in self.character_profiles:
            profile = self.character_profiles[character_id]
            ref_path = profile.references.get("front") or \
                       profile.references.get("three_quarter")
            if ref_path:
                return Image.open(ref_path).convert('RGB')
        
        # 使用默认参考图
        default_ref = self.pulid_config.get("default_face_reference")
        if default_ref and os.path.exists(default_ref):
            return Image.open(default_ref).convert('RGB')
        
        return None
    
    def _generate_with_pulid(
        self,
        prompt: str,
        face_reference: Optional[Image.Image],
        strategy: GenerationStrategy,
        **kwargs
    ) -> Image.Image:
        """使用 PuLID 生成"""
        import time
        # ⚡ 关键修复：如果没有参考图像，直接使用标准生成（不需要身份注入）
        if face_reference is None:
            logger.info("  没有参考图像，跳过 PuLID 身份注入，使用标准生成")
            return self._generate_standard(prompt, face_reference, strategy, **kwargs)
        
        logger.info("  开始加载 PuLID 引擎...")
        load_start = time.time()
        
        self._load_pulid_engine()
        
        logger.info(f"  ✓ PuLID 引擎加载完成 (耗时: {time.time()-load_start:.2f}秒)")
        
        if self.pulid_engine is None:
            logger.warning("PuLID 引擎不可用，回退到标准生成")
            return self._generate_standard(prompt, face_reference, strategy, **kwargs)
        
        # 判断是否需要增强服饰一致性
        # 1. 如果用户明确指定，使用用户设置
        # 2. 否则，根据 reference_mode 自动判断（full_body 模式通常需要服饰一致性）
        enhance_clothing = kwargs.get('enhance_clothing_consistency', None)
        if enhance_clothing is None:
            # 自动判断：full_body 模式通常需要服饰一致性
            # 或者参考强度在中等范围（50-75），说明是能看到服饰的场景
            if strategy.reference_mode == "full_body" or (50 <= strategy.reference_strength <= 75):
                enhance_clothing = True
                logger.info(f"  检测到需要服饰一致性的场景（reference_mode={strategy.reference_mode}, strength={strategy.reference_strength}），自动启用服饰一致性增强")
        
        return self.pulid_engine.generate_with_identity(
            prompt=prompt,
            face_reference=face_reference,
            reference_strength=strategy.reference_strength,
            width=self.pulid_config.get("width", 768),
            height=self.pulid_config.get("height", 1152),
            num_inference_steps=self.pulid_config.get("num_inference_steps", 28),
            guidance_scale=self.pulid_config.get("guidance_scale", 3.5),
            enhance_clothing_consistency=enhance_clothing,
            **kwargs
        )
    
    def _generate_decoupled(
        self,
        prompt: str,
        face_reference: Optional[Image.Image],
        strategy: GenerationStrategy,
        **kwargs
    ) -> Image.Image:
        """使用解耦生成"""
        logger.info("使用解耦生成...")
        
        # 先加载 PuLID 引擎（如果使用原生模式，会加载 Flux 模型）
        self._load_pulid_engine()
        
        # 然后加载 Flux pipeline（如果 PuLID 使用原生模式，会复用其模型）
        self._load_flux_pipeline()
        
        # 最后加载融合引擎（相对较小）
        self._load_fusion_engine()
        
        if self.fusion_engine is None:
            logger.warning("解耦融合引擎不可用，回退到 PuLID 生成")
            return self._generate_with_pulid(prompt, face_reference, strategy, **kwargs)
        
        image = self.fusion_engine.generate_decoupled(
            prompt=prompt,
            face_reference=face_reference,
            width=self.pulid_config.get("width", 768),
            height=self.pulid_config.get("height", 1152),
            scene_generator=self.flux_pipeline,
            identity_injector=self.pulid_engine,
            reference_strength=strategy.reference_strength,
            **kwargs
        )
        
        # 质量验证（在最终图像上进行）
        if strategy.verify_face_similarity and face_reference is not None:
            self._verify_quality(image, face_reference, strategy)
        
        return image
    
    def _generate_with_sdxl_instantid(
        self,
        prompt: str,
        face_reference: Optional[Image.Image],
        strategy: GenerationStrategy,
        **kwargs
    ) -> Image.Image:
        """使用 SDXL + InstantID 生成（稳定方案，适用于不稳定场景）"""
        logger.info("  使用 SDXL + InstantID 生成（稳定方案）...")
        
        # ⚡ 关键修复：对于"死刑组合"场景（lying + wide + desert），InstantID 几乎失效
        # 检测是否为"死刑组合"场景
        scene = kwargs.get("scene", {})
        is_death_combo = False
        if scene:
            character = scene.get("character", {})
            camera = scene.get("camera", {})
            environment = scene.get("environment", {})
            
            # 检测 lying + wide shot
            character_pose = str(character.get("pose", "")).lower()
            shot_type = str(camera.get("shot", "")).lower()
            is_lying = character_pose in ["lying_motionless", "lying", "lie", "prone"]
            is_wide = shot_type in ["wide", "extreme_wide", "full"]
            
            # 检测沙漠场景
            scene_text = str(scene.get("prompt", "")).lower() + " " + str(scene.get("description", "")).lower()
            is_desert = any(kw in scene_text for kw in ["desert", "sand", "沙地", "沙漠", "gray-green", "gray green"])
            
            # 检测俯拍角度
            camera_angle = str(camera.get("angle", "")).lower()
            is_top_down = camera_angle in ["top_down", "topdown", "bird_eye"]
            
            # 检测低可见性
            face_visible = character.get("face_visible", True)
            visibility = str(character.get("visibility", "") or "").lower()
            is_low_visibility = not face_visible or visibility == "low"
            
            # "死刑组合"：lying + wide + (desert OR top_down OR low_visibility)
            if is_lying and is_wide and (is_desert or is_top_down or is_low_visibility):
                is_death_combo = True
                logger.warning("  ⚠ 检测到'死刑组合'场景（lying + wide + desert/top_down/low_visibility），InstantID 几乎失效")
                logger.warning("  ⚠ 禁用 InstantID，使用纯 SDXL + prompt（确保场景和姿态正确）")
        
        # 如果是"死刑组合"，直接使用纯 SDXL（不使用 InstantID）
        if is_death_combo:
            logger.info("  ⚡ 使用纯 SDXL 生成（死刑组合场景，InstantID 失效）")
            return self._generate_with_sdxl(prompt, face_reference, strategy, **kwargs)
        
        # 准备参考图
        if face_reference is None:
            logger.warning("  没有参考图像，使用纯 SDXL 生成（无身份注入）")
            return self._generate_with_sdxl(prompt, face_reference, strategy, **kwargs)
        
        # ⚡ 关键修复：在加载 InstantID 之前，先清理所有已加载的模型（PuLID、Flux 等）
        from pathlib import Path
        import tempfile
        import torch
        import gc
        
        # ⚡ 关键修复：硬切换 - 在加载 InstantID 之前，完全卸载所有已加载的模型
        # 核心原则：同一时刻 GPU 上只允许一个 diffusion pipeline 存活
        self._unload_all_models()
        
        # ⚡ 关键修复：复用现有的 ImageGenerator 实例，避免重复加载模型导致内存不足
        # 如果 self.image_generator 已存在，先清理之前的 pipeline，然后加载 InstantID
        if self.image_generator is not None:
            logger.info("  复用现有的 ImageGenerator 实例...")
            # ⚡ 更彻底的内存清理：清理所有 pipeline 组件
            logger.info("  清理之前的 pipeline 以释放内存...")
            try:
                # 清理主 pipeline
                if hasattr(self.image_generator, 'pipeline') and self.image_generator.pipeline is not None:
                    # 尝试清理 pipeline 的所有组件
                    if hasattr(self.image_generator.pipeline, 'text_encoder'):
                        try:
                            del self.image_generator.pipeline.text_encoder
                        except:
                            pass
                    if hasattr(self.image_generator.pipeline, 'text_encoder_2'):
                        try:
                            del self.image_generator.pipeline.text_encoder_2
                        except:
                            pass
                    if hasattr(self.image_generator.pipeline, 'vae'):
                        try:
                            del self.image_generator.pipeline.vae
                        except:
                            pass
                    if hasattr(self.image_generator.pipeline, 'unet'):
                        try:
                            del self.image_generator.pipeline.unet
                        except:
                            pass
                    del self.image_generator.pipeline
                    self.image_generator.pipeline = None
                # 清理其他 pipeline
                if hasattr(self.image_generator, 'sdxl_pipeline') and self.image_generator.sdxl_pipeline is not None:
                    del self.image_generator.sdxl_pipeline
                    self.image_generator.sdxl_pipeline = None
                if hasattr(self.image_generator, 'img2img_pipeline') and self.image_generator.img2img_pipeline is not None:
                    del self.image_generator.img2img_pipeline
                    self.image_generator.img2img_pipeline = None
            except Exception as e:
                logger.warning(f"  清理 pipeline 时出错（继续执行）: {e}")
            
            # 清理 GPU 缓存（多次清理确保彻底）
            for _ in range(3):
                torch.cuda.empty_cache()
                gc.collect()
            
            temp_generator = self.image_generator
        else:
            # 如果不存在，创建新实例
            from image_generator import ImageGenerator
            logger.info("  创建新的 ImageGenerator 实例...")
            temp_generator = ImageGenerator(self.config_path)
            self.image_generator = temp_generator
        
        # 加载 InstantID pipeline
        if not hasattr(temp_generator, 'pipeline') or temp_generator.pipeline is None or \
           (hasattr(temp_generator, 'engine') and temp_generator.engine != "instantid"):
            logger.info("  加载 InstantID pipeline...")
            temp_generator.load_pipeline("instantid")
        
        # 保存 face_reference 到临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            face_reference.save(tmp_file.name)
            face_ref_path = Path(tmp_file.name)
        
        try:
            # ⚡ 关键修复：直接调用 InstantID pipeline，避免递归调用 generate_image
            # 原因：generate_image 会检测 scene 参数并再次调用 enhanced_generator.generate_scene，形成无限递归
            # 我们已经有了完整的 prompt（已经通过 Execution Planner 处理），直接使用 pipeline 即可
            
            # 确保 pipeline 已加载
            if not hasattr(temp_generator, 'pipeline') or temp_generator.pipeline is None:
                raise RuntimeError("InstantID pipeline 不可用")
            
            pipeline = temp_generator.pipeline
            
            # 提取 face embedding（如果还没有）
            # ⚡ 修复：ImageGenerator 使用 face_analysis 而不是 face_analyzer
            if not hasattr(temp_generator, 'face_analysis') or temp_generator.face_analysis is None:
                # ImageGenerator 在初始化时已经加载了 face_analysis
                # 如果没有，尝试重新初始化（使用与ImageGenerator相同的逻辑）
                try:
                    from insightface.app import FaceAnalysis
                    from pathlib import Path
                    
                    # 使用与ImageGenerator相同的路径查找逻辑
                    antelopev2_path = None
                    script_dir = Path(__file__).parent
                    antelopev2_path = script_dir / "models" / "antelopev2"
                    
                    if antelopev2_path.exists() and antelopev2_path.is_dir():
                        # antelopev2_path 是 models/antelopev2
                        # root 应该是当前目录（antelopev2_path.parent.parent）
                        antelopev2_root = str(antelopev2_path.parent.parent)
                        temp_generator.face_analysis = FaceAnalysis(
                            name='antelopev2',
                            root=antelopev2_root,
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                        )
                    else:
                        # 使用默认路径
                        temp_generator.face_analysis = FaceAnalysis(
                            name='antelopev2',
                            root='./',
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                        )
                    
                    temp_generator.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
                    logger.info("  ✓ FaceAnalysis 初始化成功")
                except Exception as e:
                    logger.warning(f"  ⚠ FaceAnalysis 初始化失败: {e}")
                    temp_generator.face_analysis = None
            
            face_emb = None
            face_kps = None
            if temp_generator.face_analysis and face_reference:
                # ⚡ 修复：face_analysis.get() 需要 numpy 数组，不是 PIL Image
                # 将 PIL Image 转换为 numpy 数组
                try:
                    import numpy as np
                    if hasattr(face_reference, 'shape'):
                        # 已经是 numpy 数组
                        face_img_array = face_reference
                        face_image = Image.fromarray(face_img_array) if len(face_img_array.shape) == 3 else face_reference
                    else:
                        # 是 PIL Image，转换为 numpy 数组
                        face_img_array = np.array(face_reference)
                        face_image = face_reference
                    
                    face_info_list = temp_generator.face_analysis.get(face_img_array)
                    if face_info_list and len(face_info_list) > 0:
                        # ⚡ 修复：face_analysis.get() 返回的是列表，每个元素可能是字典或对象
                        # 选择最大的人脸（与 image_generator.py 保持一致）
                        try:
                            # 尝试按字典方式访问（image_generator.py 的方式）
                            if isinstance(face_info_list[0], dict):
                                face_info = sorted(face_info_list, key=lambda x: (
                                    x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
                                face_emb = face_info['embedding']
                                face_kps_data = face_info.get('kps')
                            else:
                                # 按对象方式访问
                                face_data = face_info_list[0]
                                face_emb = face_data.normed_embedding if hasattr(face_data, 'normed_embedding') else face_data.embedding
                                face_kps_data = face_data.kps if hasattr(face_data, 'kps') else (face_data.landmark if hasattr(face_data, 'landmark') else None)
                            
                            # ⚡ 修复：提取关键点图像（InstantID pipeline 需要）
                            if face_kps_data is not None:
                                try:
                                    # 导入 draw_kps 函数
                                    from pathlib import Path
                                    instantid_repo_path = Path(__file__).parent.parent / "InstantID"
                                    if instantid_repo_path.exists():
                                        import sys
                                        if str(instantid_repo_path) not in sys.path:
                                            sys.path.insert(0, str(instantid_repo_path))
                                        from pipeline_stable_diffusion_xl_instantid import draw_kps
                                        
                                        # 生成关键点图像
                                        face_kps_raw = draw_kps(face_image, face_kps_data)
                                        face_kps = face_kps_raw
                                        logger.info("  ✓ 已提取人脸关键点图像")
                                    else:
                                        logger.warning("  ⚠ InstantID 仓库未找到，无法提取关键点")
                                except Exception as e:
                                    logger.warning(f"  ⚠ 提取关键点图像失败: {e}")
                                    import traceback
                                    traceback.print_exc()
                            else:
                                logger.warning("  ⚠ 无法获取关键点数据")
                        except Exception as e:
                            logger.warning(f"  ⚠ 处理人脸信息失败: {e}")
                            import traceback
                            traceback.print_exc()
                            face_emb = None
                        
                        if face_emb is not None:
                            logger.info("  ✓ 已从参考图提取人脸 embedding")
                    else:
                        logger.warning("  ⚠ 未能从参考图提取人脸信息")
                except Exception as e:
                    logger.warning(f"  ⚠ 提取人脸信息失败: {e}")
                    face_emb = None
            
            if face_emb is None:
                logger.warning("  没有 face embedding，使用纯 SDXL 生成（无身份注入）")
                return self._generate_with_sdxl(prompt, face_reference, strategy, **kwargs)
            
            if face_kps is None:
                logger.warning("  没有关键点图像，使用纯 SDXL 生成（无身份注入）")
                return self._generate_with_sdxl(prompt, face_reference, strategy, **kwargs)
            
            # 直接调用 InstantID pipeline
            logger.info("  直接调用 InstantID pipeline 生成图像（避免递归调用）...")
            
            # 准备参数
            negative_prompt = kwargs.get("negative_prompt", "low quality, blurry, distorted, deformed, bad anatomy, bad hands, text, watermark")
            guidance_scale = kwargs.get("guidance_scale", 5.0)
            num_inference_steps = kwargs.get("num_inference_steps", 30)
            seed = kwargs.get("seed", None)
            
            # 调整 IP-Adapter scale（根据 reference_strength）
            ip_adapter_scale = temp_generator.face_emb_scale if hasattr(temp_generator, 'face_emb_scale') else 0.8
            if hasattr(strategy, 'reference_strength'):
                ip_adapter_scale = ip_adapter_scale * (strategy.reference_strength / 100.0)
                ip_adapter_scale = max(0.3, min(1.0, ip_adapter_scale))
            
            # 生成图像
            output_path = Path(tempfile.mktemp(suffix='.png'))
            result = pipeline(
                prompt=prompt,
                image_embeds=face_emb,
                image=face_kps,  # ⚡ 修复：添加关键点图像参数
                controlnet_conditioning_scale=ip_adapter_scale,
                width=temp_generator.width if hasattr(temp_generator, 'width') else 1024,
                height=temp_generator.height if hasattr(temp_generator, 'height') else 1024,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator=torch.Generator(device=temp_generator.device).manual_seed(seed) if seed is not None else None,
            )
            
            # 提取图像
            if hasattr(result, 'images') and len(result.images) > 0:
                result_image = result.images[0]
            elif isinstance(result, Image.Image):
                result_image = result
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                result_image = result[0]
            else:
                raise RuntimeError(f"Pipeline 返回了未知格式: {type(result)}")
            
            # 保存图像
            result_image.save(output_path)
            result_path = output_path
            
            # 清理临时文件
            if face_ref_path.exists():
                face_ref_path.unlink()
            if output_path.exists() and output_path != result_path:
                output_path.unlink()
            
            return result_image
            
        except Exception as e:
            logger.error(f"  SDXL + InstantID 生成失败: {e}")
            import traceback
            traceback.print_exc()
            # 清理临时文件
            if face_ref_path.exists():
                face_ref_path.unlink()
            # 回退到纯 SDXL
            logger.warning("  回退到纯 SDXL 生成（无身份注入）")
            return self._generate_with_sdxl(prompt, face_reference, strategy, **kwargs)
    
    def _generate_with_sdxl(
        self,
        prompt: str,
        face_reference: Optional[Image.Image],
        strategy: GenerationStrategy,
        **kwargs
    ) -> Image.Image:
        """使用纯 SDXL 生成（无身份注入，适用于死刑组合场景）"""
        logger.info("  使用纯 SDXL 生成（无身份注入）...")
        
        from pathlib import Path
        import tempfile
        import torch
        import gc
        
        # ⚡ 关键修复：硬切换 - 在加载 SDXL 之前，完全卸载所有已加载的模型
        # 核心原则：同一时刻 GPU 上只允许一个 diffusion pipeline 存活
        self._unload_all_models()
        
        # ⚡ 关键修复：复用现有的 ImageGenerator 实例，避免重复加载模型导致内存不足
        # 如果 self.image_generator 已存在，先清理之前的 pipeline，然后加载 SDXL
        if self.image_generator is not None:
            logger.info("  复用现有的 ImageGenerator 实例...")
            temp_generator = self.image_generator
        else:
            # 如果不存在，创建新实例
            from image_generator import ImageGenerator
            logger.info("  创建新的 ImageGenerator 实例...")
            temp_generator = ImageGenerator(self.config_path)
            self.image_generator = temp_generator
        
        # 加载 SDXL pipeline
        if not hasattr(temp_generator, 'pipeline') or temp_generator.pipeline is None or \
           (hasattr(temp_generator, 'engine') and temp_generator.engine != "sdxl"):
            logger.info("  加载 SDXL pipeline...")
            temp_generator.load_pipeline("sdxl")
        
        if not hasattr(temp_generator, 'pipeline') or temp_generator.pipeline is None:
            raise RuntimeError("SDXL pipeline 不可用")
        
        # ⚡ 关键修复：直接调用 pipeline，避免递归调用 generate_image
        # 原因：generate_image 会检测 scene 参数并再次调用 enhanced_generator.generate_scene，形成无限递归
        # 我们已经有了完整的 prompt（已经通过 Execution Planner 处理），直接使用 pipeline 即可
        output_path = Path(tempfile.mktemp(suffix='.png'))
        
        try:
            # 确保 pipeline 已加载
            if not hasattr(temp_generator, 'pipeline') or temp_generator.pipeline is None:
                raise RuntimeError("SDXL pipeline 不可用")
            
            pipeline = temp_generator.pipeline
            
            # 直接调用 pipeline（不使用 generate_image，避免递归）
            logger.info("  直接调用 SDXL pipeline 生成图像（避免递归调用）...")
            
            # ⚡ 关键修复：SDXL 使用 CLIP tokenizer（77 tokens 限制），需要智能精简 prompt
            # 优先保留关键信息（场景、角色、姿态），确保不被截断
            # 注意：需要确保 temp_generator 已加载 CLIP tokenizer
            if not hasattr(temp_generator, '_clip_tokenizer') or temp_generator._clip_tokenizer is None:
                # 如果 tokenizer 未加载，尝试加载它
                try:
                    from transformers import CLIPTokenizer
                    # 尝试从 SDXL pipeline 获取 tokenizer
                    if hasattr(temp_generator, 'pipeline') and temp_generator.pipeline is not None:
                        if hasattr(temp_generator.pipeline, 'tokenizer'):
                            temp_generator._clip_tokenizer = temp_generator.pipeline.tokenizer
                            logger.info("  ✓ 从 SDXL pipeline 获取 CLIP tokenizer")
                        elif hasattr(temp_generator.pipeline, 'tokenizer_2'):
                            temp_generator._clip_tokenizer = temp_generator.pipeline.tokenizer_2
                            logger.info("  ✓ 从 SDXL pipeline 获取 CLIP tokenizer_2")
                    # 如果还是 None，尝试直接加载
                    if temp_generator._clip_tokenizer is None:
                        tokenizer_path = "/vepfs-dev/shawn/vid/fanren/gen_video/models/sdxl-base"
                        if os.path.exists(tokenizer_path):
                            temp_generator._clip_tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
                            logger.info(f"  ✓ 从本地路径加载 CLIP tokenizer: {tokenizer_path}")
                except Exception as e:
                    logger.warning(f"  ⚠ 加载 CLIP tokenizer 失败: {e}，将使用估算方法")
            
            optimized_prompt = self._optimize_prompt_for_sdxl(prompt, temp_generator)
            logger.info(f"  Prompt 优化: 原始长度={len(prompt.split())} 词，优化后长度={len(optimized_prompt.split())} 词")
            
            # 验证优化后的 prompt 长度
            if hasattr(temp_generator, '_clip_tokenizer') and temp_generator._clip_tokenizer is not None:
                try:
                    tokens_obj = temp_generator._clip_tokenizer(optimized_prompt, truncation=False, return_tensors="pt")
                    final_tokens = tokens_obj.input_ids.shape[1]
                    if final_tokens > 77:
                        logger.warning(f"  ⚠ 优化后的 prompt 仍然超过 77 tokens ({final_tokens} tokens)，可能会被截断")
                    else:
                        logger.info(f"  ✓ 优化后的 prompt 长度: {final_tokens} tokens（在限制内）")
                except Exception as e:
                    logger.warning(f"  ⚠ 验证优化后的 prompt 长度失败: {e}")
            
            # 准备参数
            negative_prompt = kwargs.get("negative_prompt", "low quality, blurry, distorted, deformed, bad anatomy, bad hands, text, watermark")
            guidance_scale = kwargs.get("guidance_scale", 7.5)
            num_inference_steps = kwargs.get("num_inference_steps", 30)
            seed = kwargs.get("seed", None)
            
            # 生成图像
            result = pipeline(
                prompt=optimized_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=temp_generator.device).manual_seed(seed) if seed is not None else None,
            )
            
            # 提取图像
            if hasattr(result, 'images') and len(result.images) > 0:
                result_image = result.images[0]
            elif isinstance(result, Image.Image):
                result_image = result
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                result_image = result[0]
            else:
                raise RuntimeError(f"Pipeline 返回了未知格式: {type(result)}")
            
            # 保存图像
            result_image.save(output_path)
            result_path = output_path
            
            # 读取生成的图像
            result_image = Image.open(result_path)
            
            # 清理临时文件
            if output_path.exists() and output_path != result_path:
                output_path.unlink()
            
            return result_image
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"  SDXL 生成失败: {e}")
            import traceback
            traceback.print_exc()
            # 清理临时文件
            if output_path.exists():
                output_path.unlink()
            
            # ⚡ 关键修复：如果是内存不足错误，清理 pipeline 并抛出明确的错误信息
            if "out of memory" in error_msg.lower() or "cuda error" in error_msg.lower():
                logger.error("  ❌ 检测到内存不足错误，清理 pipeline 并抛出错误（避免无限循环）")
                # 清理 pipeline
                try:
                    if hasattr(temp_generator, 'pipeline') and temp_generator.pipeline is not None:
                        del temp_generator.pipeline
                        temp_generator.pipeline = None
                except:
                    pass
                # 清理 GPU 缓存
                import torch
                import gc
                for _ in range(3):
                    torch.cuda.empty_cache()
                    gc.collect()
                raise RuntimeError(f"内存不足，无法加载 SDXL pipeline。请先清理 GPU 内存或减少并发数量。原始错误: {e}") from e
            
            raise
    
    def _generate_standard(
        self,
        prompt: str,
        face_reference: Optional[Image.Image],
        strategy: GenerationStrategy,
        **kwargs
    ) -> Image.Image:
        """标准生成 (使用 Flux，当 scene_engine 为 FLUX 时)"""
        logger.info("使用标准生成（Flux）...")
        
        self._load_flux_pipeline()
        
        if self.flux_pipeline is None:
            raise RuntimeError("Flux pipeline 不可用")
        
        # 使用 Flux 生成
        result = self.flux_pipeline(
            prompt=prompt,
            width=self.pulid_config.get("width", 768),
            height=self.pulid_config.get("height", 1152),
            num_inference_steps=self.pulid_config.get("num_inference_steps", 28),
            guidance_scale=self.pulid_config.get("guidance_scale", 3.5),
            **kwargs
        )
        
        # ⚡ 关键修复：处理 Result 对象（FluxWrapper 返回的）和 ImagePipelineOutput（diffusers 返回的）
        if hasattr(result, 'images') and isinstance(result.images, list) and len(result.images) > 0:
            return result.images[0]
        elif hasattr(result, 'save'):
            # 如果 result 本身就是 PIL Image（不应该发生，但为了安全）
            return result
        else:
            raise TypeError(f"Flux pipeline 返回了未知类型: {type(result)}")
    
    def _verify_quality(
        self,
        generated: Image.Image,
        reference: Image.Image,
        strategy: GenerationStrategy,
        expected_shot_type: Optional[str] = None,
        verbose: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        验证生成质量（增强版）
        
        使用 ImageQualityAnalyzer 进行全面的质量评估，包括：
        - 人脸相似度验证
        - 构图分析（远景/中景/近景）
        - 技术指标（清晰度/饱和度/亮度/对比度）
        - 综合评分
        
        Args:
            generated: 生成的图像
            reference: 参考图像
            strategy: 生成策略
            expected_shot_type: 期望的镜头类型
            verbose: 是否输出详细日志
            
        Returns:
            质量报告字典，如果分析失败则返回 None
        """
        try:
            from utils.image_quality_analyzer import ImageQualityAnalyzer, QualityLevel
            
            # 创建分析器
            analyzer_config = {
                "device": self.device,
                "insightface_root": os.path.dirname(
                    self.decoupled_config.get("sam2_path", "models")
                )
            }
            analyzer = ImageQualityAnalyzer(analyzer_config)
            
            # 确定期望的镜头类型
            if expected_shot_type is None and hasattr(strategy, 'shot_type'):
                expected_shot_type = strategy.shot_type
            
            # 执行分析
            report = analyzer.analyze(
                image=generated,
                reference_image=reference,
                similarity_threshold=strategy.similarity_threshold,
                expected_shot_type=expected_shot_type
            )
            
            # 输出日志
            logger.info("=" * 50)
            logger.info("📊 图像质量分析结果")
            logger.info("=" * 50)
            
            # 综合评分
            level_emoji = {
                QualityLevel.EXCELLENT: "🌟",
                QualityLevel.GOOD: "✅",
                QualityLevel.FAIR: "🟡",
                QualityLevel.POOR: "🟠",
                QualityLevel.BAD: "🔴"
            }
            emoji = level_emoji.get(report.overall_level, "❓")
            logger.info(f"🎯 综合评分: {report.overall_score:.1f}/100 {emoji} {report.overall_level.value.upper()}")
            
            # 人脸相似度
            if report.face_similarity:
                face = report.face_similarity
                if face.error:
                    logger.warning(f"👤 人脸相似度: ⚠️ {face.error}")
                else:
                    status = "✅ 通过" if face.passed else "❌ 未通过"
                    logger.info(f"👤 人脸相似度: {face.similarity:.3f} (阈值: {face.threshold}) {status}")
            
            # 构图分析
            if report.composition and verbose:
                comp = report.composition
                shot_emoji = {"extreme_close": "🔍", "close": "👁️", "medium": "📷", "wide": "🏞️", "unknown": "❓"}
                logger.info(f"🎬 镜头类型: {shot_emoji.get(comp.shot_type.value, '')} {comp.shot_type.value}")
                if comp.person_ratio > 0:
                    logger.info(f"   人物占比: {comp.person_ratio*100:.1f}%")
            
            # 技术指标（简要）
            if report.technical and verbose:
                tech = report.technical
                level_sym = {"excellent": "🟢", "good": "🟢", "fair": "🟡", "poor": "🟠", "bad": "🔴"}
                logger.info(f"📊 清晰度: {tech.sharpness:.1f} {level_sym.get(tech.sharpness_level.value, '')}")
                logger.info(f"   饱和度: {tech.saturation:.1f} {level_sym.get(tech.saturation_level.value, '')}")
            
            # 问题和建议
            if report.issues:
                logger.warning("⚠️ 发现问题:")
                for issue in report.issues:
                    logger.warning(f"   • {issue}")
            
            if report.suggestions and verbose:
                logger.info("💡 优化建议:")
                for suggestion in report.suggestions:
                    logger.info(f"   • {suggestion}")
            
            logger.info("=" * 50)
            
            # 清理
            analyzer.unload()
            
            return report.to_dict()
            
        except ImportError:
            # 回退到简单验证
            logger.debug("ImageQualityAnalyzer 不可用，使用简单验证")
            return self._verify_quality_simple(generated, reference, strategy)
        except Exception as e:
            logger.error(f"质量分析失败: {e}")
            return self._verify_quality_simple(generated, reference, strategy)
    
    def _verify_quality_simple(
        self,
        generated: Image.Image,
        reference: Image.Image,
        strategy: GenerationStrategy
    ) -> Optional[Dict[str, Any]]:
        """简单质量验证（回退方法）"""
        if self.fusion_engine is None:
            self._load_fusion_engine()
        
        if self.fusion_engine is None:
            logger.warning("无法验证人脸相似度")
            return None
        
        passed, similarity = self.fusion_engine.verify_face_similarity(
            generated_image=generated,
            reference_image=reference,
            threshold=strategy.similarity_threshold
        )
        
        if passed:
            logger.info(f"✅ 质量验证通过: 相似度 {similarity:.2f}")
        else:
            logger.warning(f"⚠️ 质量验证未通过: 相似度 {similarity:.2f} < 阈值 {strategy.similarity_threshold}")
        
        return {
            "face_similarity": {
                "similarity": similarity,
                "passed": passed,
                "threshold": strategy.similarity_threshold
            }
        }
    
    def unload_all(self):
        """卸载所有模型"""
        logger.info("开始卸载所有模型...")
        
        # 记录卸载前的显存
        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"卸载前显存: 已分配={allocated_before:.2f}GB, 已保留={reserved_before:.2f}GB")
        
        # 卸载 PuLID 引擎
        if self.pulid_engine is not None:
            try:
                self.pulid_engine.unload()
            except Exception as e:
                logger.warning(f"卸载 PuLID 引擎失败: {e}")
            self.pulid_engine = None
        
        # 卸载融合引擎
        if self.fusion_engine is not None:
            try:
                if hasattr(self.fusion_engine, 'unload'):
                    self.fusion_engine.unload()
            except Exception as e:
                logger.warning(f"卸载融合引擎失败: {e}")
            self.fusion_engine = None
        
        # 卸载 Flux pipeline（包括包装器）
        if self.flux_pipeline is not None:
            # 如果是 FluxWrapper，需要清理其内部引用
            if hasattr(self.flux_pipeline, 'pulid_engine'):
                # FluxWrapper 复用 pulid_engine 的模型，不需要单独卸载
                # 但需要清理包装器本身
                pass
            else:
                # 普通 pipeline，尝试卸载
                if hasattr(self.flux_pipeline, 'unload'):
                    try:
                        self.flux_pipeline.unload()
                    except:
                        pass
                del self.flux_pipeline
            self.flux_pipeline = None
        
        # ⚡ 关键修复：清理 quality_analyzer（可能持有 InsightFace 模型）
        if self.quality_analyzer is not None:
            try:
                if hasattr(self.quality_analyzer, 'face_analyzer'):
                    # InsightFace 模型可能占用显存
                    if self.quality_analyzer.face_analyzer is not None:
                        del self.quality_analyzer.face_analyzer
                self.quality_analyzer = None
                logger.info("已清理 quality_analyzer")
            except Exception as e:
                logger.warning(f"清理 quality_analyzer 失败: {e}")
        
        # ⚡ 关键修复：清理 planner 的 LLM 客户端引用（虽然不占显存，但有助于垃圾回收）
        if self.planner is not None:
            try:
                if hasattr(self.planner, 'llm_client'):
                    self.planner.llm_client = None
            except Exception as e:
                logger.warning(f"清理 planner LLM 客户端失败: {e}")
        
        # 强制清理所有 Python 对象
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            # 同步所有 CUDA 操作
            torch.cuda.synchronize()
            # 清空缓存
            torch.cuda.empty_cache()
            # 再次强制垃圾回收
            gc.collect()
            
            # ⚡ 关键修复：多次清理，确保彻底释放
            for i in range(5):
                torch.cuda.empty_cache()
                gc.collect()
                if i % 2 == 0:
                    torch.cuda.synchronize()
            
            # 记录卸载后的显存
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            reserved_after = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"卸载后显存: 已分配={allocated_after:.2f}GB, 已保留={reserved_after:.2f}GB")
            if reserved_before > 0:
                logger.info(f"释放显存: {reserved_before - reserved_after:.2f}GB")
        
        logger.info("所有模型已卸载")


# ==========================================
# 便捷函数
# ==========================================

def generate_scene_enhanced(
    scene: Dict[str, Any],
    config_path: str = "config.yaml",
    face_reference: Optional[str] = None,
    **kwargs
) -> Image.Image:
    """
    便捷函数: 生成增强场景图像
    
    Args:
        scene: 场景 JSON
        config_path: 配置文件路径
        face_reference: 人脸参考图路径
        **kwargs: 额外参数
        
    Returns:
        生成的图像
    """
    generator = EnhancedImageGenerator(config_path)
    
    try:
        image = generator.generate_scene(
            scene=scene,
            face_reference=face_reference,
            **kwargs
        )
        return image
    finally:
        generator.unload_all()


def batch_generate_scenes(
    scenes: List[Dict[str, Any]],
    config_path: str = "config.yaml",
    output_dir: str = "outputs/enhanced",
    face_reference: Optional[str] = None,
    **kwargs
) -> List[str]:
    """
    便捷函数: 批量生成场景图像
    
    Args:
        scenes: 场景 JSON 列表
        config_path: 配置文件路径
        output_dir: 输出目录
        face_reference: 人脸参考图路径
        **kwargs: 额外参数
        
    Returns:
        生成的图像路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generator = EnhancedImageGenerator(config_path)
    results = []
    
    try:
        for i, scene in enumerate(scenes):
            logger.info(f"\n处理场景 {i+1}/{len(scenes)}...")
            
            image = generator.generate_scene(
                scene=scene,
                face_reference=face_reference,
                **kwargs
            )
            
            # 保存图像
            output_path = os.path.join(output_dir, f"scene_{i+1:03d}.png")
            image.save(output_path)
            results.append(output_path)
            
            logger.info(f"保存: {output_path}")
        
        return results
        
    finally:
        generator.unload_all()


# ==========================================
# 测试代码
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 测试场景
    test_scene = {
        "camera": {
            "shot": "medium",
            "angle": "eye_level"
        },
        "character": {
            "present": True,
            "id": "hanli",
            "emotion": "neutral",
            "pose": "standing"
        },
        "environment": {
            "description": "ancient mountain temple with misty clouds, traditional Chinese architecture, dawn light filtering through bamboo forest",
            "lighting": "soft golden morning light",
            "atmosphere": "serene and mystical"
        },
        "visual": {
            "composition": "rule of thirds, character on left third"
        }
    }
    
    print("\n" + "=" * 60)
    print("增强型图像生成器测试")
    print("=" * 60)
    
    # 检查配置文件
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"\n⚠️ 配置文件不存在: {config_path}")
        print("请确保运行目录正确")
    else:
        print(f"\n✅ 配置文件存在: {config_path}")
        
        # 创建生成器
        try:
            generator = EnhancedImageGenerator(config_path)
            print("✅ EnhancedImageGenerator 创建成功")
            
            # 测试策略分析
            strategy = generator.planner.analyze_scene(test_scene)
            print(f"\n生成策略:")
            print(f"  参考强度: {strategy.reference_strength}%")
            print(f"  身份引擎: {strategy.identity_engine.value}")
            print(f"  解耦生成: {strategy.use_decoupled_pipeline}")
            
            # 测试 Prompt 构建
            prompt = generator.planner.build_weighted_prompt(test_scene, strategy)
            print(f"\n构建的 Prompt:")
            print(f"  {prompt[:150]}...")
            
            generator.unload_all()
            print("\n✅ 测试完成!")
            
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
