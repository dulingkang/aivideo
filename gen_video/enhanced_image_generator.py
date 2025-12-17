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


class EnhancedImageGenerator:
    """
    增强型图像生成器
    
    整合了:
    - PuLID-FLUX (身份保持 + 环境融合)
    - 解耦融合引擎 (SAM2 + YOLO)
    - Execution Planner V3 (智能路由)
    - 角色档案系统 (多参考图)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化增强型图像生成器
        
        Args:
            config_path: 配置文件路径
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
        
        # 初始化组件
        self.planner = ExecutionPlannerV3(self.planner_config)
        self.pulid_engine = None  # 延迟加载
        self.fusion_engine = None  # 延迟加载
        self.flux_pipeline = None  # 延迟加载
        
        # 角色档案
        self.character_profiles = {}
        self._load_character_profiles()
        
        # 设备配置
        self.device = self.image_config.get("device", "cuda")
        
        logger.info("EnhancedImageGenerator 初始化完成")
        logger.info(f"  PuLID 启用: {self.pulid_config.get('enabled', False)}")
        logger.info(f"  解耦融合启用: {self.decoupled_config.get('enabled', False)}")
        logger.info(f"  Planner 版本: V{self.planner_config.get('version', 3)}")
    
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
        **kwargs
    ) -> Image.Image:
        """
        生成场景图像
        
        这是主入口方法，会根据场景自动选择最佳策略
        
        Args:
            scene: 场景 JSON (v2 格式)
            character_id: 角色 ID (可选，用于选择角色档案)
            face_reference: 人脸参考图 (可选，覆盖角色档案)
            **kwargs: 额外参数
            
        Returns:
            生成的图像
        """
        logger.info("=" * 60)
        logger.info("开始生成场景图像")
        logger.info("=" * 60)
        
        # 1. 分析场景，获取策略
        strategy = self.planner.analyze_scene(
            scene=scene,
            character_profiles=self.character_profiles
        )
        
        # 2. 准备参考图
        ref_image = self._prepare_reference(
            strategy=strategy,
            character_id=character_id,
            face_reference=face_reference
        )
        
        # 3. 构建 Prompt
        prompt = self.planner.build_weighted_prompt(scene, strategy)
        logger.info(f"完整 Prompt: {prompt}")
        logger.info(f"Prompt 预览: {prompt[:150]}...")
        
        # 4. 根据策略选择生成方式
        # 注意：特写/近景（参考强度 > 70%）不使用解耦模式，直接使用 PuLID
        # 解耦模式更适合远景/中景（参考强度 < 60%）
        if strategy.use_decoupled_pipeline and strategy.reference_strength < 70:
            # 解耦生成（仅用于远景/中景）
            image = self._generate_decoupled(
                prompt=prompt,
                face_reference=ref_image,
                strategy=strategy,
                **kwargs
            )
        elif strategy.identity_engine == IdentityEngine.PULID:
            # PuLID 生成
            image = self._generate_with_pulid(
                prompt=prompt,
                face_reference=ref_image,
                strategy=strategy,
                **kwargs
            )
        else:
            # 标准生成 (InstantID 或无身份约束)
            image = self._generate_standard(
                prompt=prompt,
                face_reference=ref_image,
                strategy=strategy,
                **kwargs
            )
        
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
        logger.info("使用 PuLID 生成...")
        
        self._load_pulid_engine()
        
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
    
    def _generate_standard(
        self,
        prompt: str,
        face_reference: Optional[Image.Image],
        strategy: GenerationStrategy,
        **kwargs
    ) -> Image.Image:
        """标准生成 (使用现有的 InstantID 或 Flux)"""
        logger.info("使用标准生成...")
        
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
        
        return result.images[0]
    
    def _verify_quality(
        self,
        generated: Image.Image,
        reference: Image.Image,
        strategy: GenerationStrategy
    ):
        """验证生成质量"""
        if self.fusion_engine is None:
            self._load_fusion_engine()
        
        if self.fusion_engine is None:
            logger.warning("无法验证人脸相似度")
            return
        
        passed, similarity = self.fusion_engine.verify_face_similarity(
            generated_image=generated,
            reference_image=reference,
            threshold=strategy.similarity_threshold
        )
        
        if passed:
            logger.info(f"✅ 质量验证通过: 相似度 {similarity:.2f}")
        else:
            logger.warning(f"⚠️ 质量验证未通过: 相似度 {similarity:.2f} < 阈值 {strategy.similarity_threshold}")
    
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
            self.pulid_engine.unload()
            self.pulid_engine = None
        
        # 卸载融合引擎
        if self.fusion_engine is not None:
            self.fusion_engine.unload()
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
