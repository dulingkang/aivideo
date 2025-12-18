#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ID-Animator 引擎
实现基于 ID-Animator 的身份保持视频生成

核心功能:
1. 零样本身份保持 - 单张参考图即可
2. 与 AnimateDiff 集成 - 使用成熟的动画生成框架
3. 身份强度可调 - 控制身份保持程度
4. 自动策略调节 - 根据场景自动调整参数

技术架构:
  参考图 → Face Encoder → ID Embedding → AnimateDiff → 身份一致视频

Author: AI Video Team
Date: 2025-12-17
Project: M6 - 视频身份保持
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class IDAnimatorEngine:
    """
    ID-Animator 引擎
    
    基于 AnimateDiff + Face Adapter 实现身份保持视频生成
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 ID-Animator 引擎
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.device = self.config.get("device", "cuda")
        
        # 模型路径
        self.model_dir = self.config.get("model_dir", "models")
        self.animator_path = self.config.get(
            "animator_path", 
            os.path.join(self.model_dir, "ID-Animator", "id_animator.pth")
        )
        self.motion_module_path = self.config.get(
            "motion_module_path",
            os.path.join(self.model_dir, "AnimateDiff", "diffusion_pytorch_model.safetensors")
        )
        
        # ID 保持参数
        self.id_strength = self.config.get("id_strength", 0.7)
        self.num_frames = self.config.get("num_frames", 16)
        self.fps = self.config.get("fps", 8)
        self.guidance_scale = self.config.get("guidance_scale", 7.5)
        self.num_inference_steps = self.config.get("num_inference_steps", 25)
        
        # 模型引用
        self.pipeline = None
        self.face_encoder = None
        self.animator_adapter = None
        self.face_analyzer = None
        
        # 加载状态
        self.loaded = False
        
        logger.info("IDAnimatorEngine 初始化完成")
        logger.info(f"  ID 强度: {self.id_strength}")
        logger.info(f"  帧数: {self.num_frames}")
        logger.info(f"  FPS: {self.fps}")
    
    def load_models(self):
        """加载所有模型"""
        if self.loaded:
            logger.info("模型已加载，跳过")
            return
        
        logger.info("加载 ID-Animator 模型...")
        
        try:
            # 1. 加载 AnimateDiff Pipeline
            self._load_animatediff_pipeline()
            
            # 2. 加载 Face Adapter
            self._load_face_adapter()
            
            # 3. 加载人脸分析器（用于提取参考图特征）
            self._load_face_analyzer()
            
            self.loaded = True
            logger.info("✅ ID-Animator 模型加载完成")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def _load_animatediff_pipeline(self):
        """加载 AnimateDiff Pipeline"""
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
        
        logger.info("  加载 AnimateDiff Pipeline...")
        
        # 转换模型目录为绝对路径
        self.model_dir = os.path.abspath(self.model_dir)
        logger.info(f"  模型根目录: {self.model_dir}")
        
        # 本地模型路径
        local_sd15_path = os.path.join(self.model_dir, "stable-diffusion-v1-5")
        local_sdxl_path = os.path.join(self.model_dir, "sdxl-base")
        local_motion_path = os.path.join(self.model_dir, "AnimateDiff", "diffusion_pytorch_model.safetensors")
        local_motion_alt_path = os.path.join(self.model_dir, "AnimateDiff", "mm_sd_v15_v2.safetensors")
        local_motion_alt_path2 = os.path.join(self.model_dir, "AnimateDiff", "mm_sd_v15_v2.ckpt")
        local_motion_sdxl_path = self.motion_module_path
        
        # 检查使用 SDXL 还是 SD1.5（默认推荐 SD1.5）
        self.use_sdxl = "sdxl" in str(self.motion_module_path).lower()
        
        # 尝试导入 AnimateDiffSDXLPipeline
        try:
            from diffusers import AnimateDiffSDXLPipeline
        except ImportError:
            AnimateDiffSDXLPipeline = None
            logger.warning("  无法导入 AnimateDiffSDXLPipeline，尝试使用普通 AnimateDiffPipeline")
        
        if self.use_sdxl:
            # SDXL 版本
            logger.info("  使用 SDXL 版本")
            
            # 检查本地 SDXL 模型
            if os.path.exists(local_sdxl_path):
                logger.info(f"  使用本地模型: {local_sdxl_path}")
                base_model = local_sdxl_path
                is_local = True
            else:
                base_model = self.config.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
                is_local = False
            
            # 加载 Motion Adapter
            if os.path.exists(local_motion_sdxl_path):
                logger.info(f"  加载本地 Motion Adapter: {local_motion_sdxl_path}")
                adapter = MotionAdapter.from_single_file(
                    local_motion_sdxl_path,
                    torch_dtype=torch.float16
                )
                logger.info(f"  ✓ Motion Adapter 加载成功")
            else:
                logger.warning(f"  本地 Motion Adapter 不存在: {local_motion_sdxl_path}")
                try:
                    adapter = MotionAdapter.from_pretrained(
                        "guoyww/animatediff-motion-adapter-sdxl-beta",
                        torch_dtype=torch.float16
                    )
                except Exception as e:
                    logger.error(f"  Motion Adapter 加载失败: {e}")
                    raise
            
            # 加载 Pipeline
            pipeline_cls = AnimateDiffSDXLPipeline if AnimateDiffSDXLPipeline else AnimateDiffPipeline
            logger.info(f"  使用 Pipeline 类: {pipeline_cls.__name__}")
            
            self.pipeline = pipeline_cls.from_pretrained(
                base_model,
                motion_adapter=adapter,
                torch_dtype=torch.float16,
                local_files_only=is_local,
            )
        else:
            # SD 1.5 版本 - 优先使用本地模型
            logger.info("  使用 SD 1.5 版本")
            
            # 检查本地模型是否存在
            resolved_motion_path = None
            for p in [self.motion_module_path, local_motion_path, local_motion_alt_path, local_motion_alt_path2]:
                if p and os.path.exists(p):
                    resolved_motion_path = p
                    break

            if os.path.exists(local_sd15_path) and resolved_motion_path is not None:
                logger.info(f"  使用本地模型: {local_sd15_path}")
                base_model = local_sd15_path
                
                # 从本地加载 Motion Adapter
                logger.info(f"  加载本地 Motion Adapter: {resolved_motion_path}")
                adapter = MotionAdapter.from_single_file(
                    resolved_motion_path,
                    torch_dtype=torch.float16
                )
            else:
                # 回退到 HuggingFace
                logger.info("  本地模型不存在，使用 HuggingFace")
                base_model = self.config.get("base_model", "runwayml/stable-diffusion-v1-5")
                
                try:
                    adapter = MotionAdapter.from_pretrained(
                        "guoyww/animatediff-motion-adapter-v1-5-2",
                        torch_dtype=torch.float16
                    )
                except Exception as e:
                    logger.warning(f"  从 HuggingFace 加载失败: {e}")
                    if os.path.exists(local_motion_path):
                        adapter = MotionAdapter.from_single_file(
                            local_motion_path,
                            torch_dtype=torch.float16
                        )
                    elif os.path.exists(local_motion_alt_path):
                        adapter = MotionAdapter.from_single_file(
                            local_motion_alt_path,
                            torch_dtype=torch.float16
                        )
                    elif os.path.exists(local_motion_alt_path2):
                        adapter = MotionAdapter.from_single_file(
                            local_motion_alt_path2,
                            torch_dtype=torch.float16
                        )
                    else:
                        raise
            
            # 加载 Pipeline
            self.pipeline = AnimateDiffPipeline.from_pretrained(
                base_model,
                motion_adapter=adapter,
                torch_dtype=torch.float16,
                local_files_only=os.path.exists(local_sd15_path),  # 如果本地存在则不联网
            )
        
        # 设置 Scheduler
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config,
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        
        # 移动到设备
        if torch.cuda.is_available():
            self.pipeline.enable_model_cpu_offload()
            logger.info("  ✓ 启用 CPU Offload")
        
        logger.info("  ✓ AnimateDiff Pipeline 加载完成")
    
    def _load_face_adapter(self):
        """加载 ID-Animator Face Adapter"""
        logger.info("  加载 Face Adapter...")
        
        if not os.path.exists(self.animator_path):
            # 尝试在目录里自动探测
            candidate_dir = os.path.join(self.model_dir, "ID-Animator")
            if os.path.isdir(candidate_dir):
                for name in ["id_animator.pth", "id_animator.safetensors", "animator.ckpt", "animator.pth"]:
                    p = os.path.join(candidate_dir, name)
                    if os.path.exists(p):
                        self.animator_path = p
                        break
            if not os.path.exists(self.animator_path):
                logger.warning(f"  Face Adapter 文件不存在: {self.animator_path}")
                logger.warning("  将使用基础 AnimateDiff（当前实现不会注入身份权重）")
                return
        
        try:
            # 加载 animator.ckpt
            checkpoint = torch.load(self.animator_path, map_location="cpu")
            
            # 提取 adapter 权重
            if "state_dict" in checkpoint:
                self.animator_adapter = checkpoint["state_dict"]
            else:
                self.animator_adapter = checkpoint
            
            logger.info(f"  ✓ Face Adapter 加载成功 (keys: {len(self.animator_adapter)})")
            logger.warning("  ⚠ 当前版本仅完成权重加载，尚未将 Face Adapter 权重注入到 diffusers pipeline（身份保持仍以 prompt 稳定性为主）")
            
        except Exception as e:
            logger.error(f"  Face Adapter 加载失败: {e}")
            self.animator_adapter = None
    
    def _load_face_analyzer(self):
        """加载人脸分析器"""
        logger.info("  加载人脸分析器...")
        
        try:
            from insightface.app import FaceAnalysis
            
            # 正确设置 InsightFace 模型根目录
            # InsightFace 会在 root/models/{name} 下查找模型
            # 所以 root 应该是 gen_video 目录（包含 models 子目录）
            gen_video_dir = Path(__file__).parent
            model_root = str(gen_video_dir)
            
            logger.info(f"  InsightFace 根目录: {model_root}")
            
            self.face_analyzer = FaceAnalysis(
                name='antelopev2',
                root=model_root,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            
            logger.info("  ✓ 人脸分析器加载成功")
            
        except Exception as e:
            logger.warning(f"  人脸分析器加载失败: {e}")
            self.face_analyzer = None
    
    def extract_face_embedding(
        self, 
        reference_image: Union[str, Path, Image.Image]
    ) -> Optional[np.ndarray]:
        """
        提取参考图的人脸嵌入
        
        Args:
            reference_image: 参考图像
            
        Returns:
            人脸嵌入向量
        """
        if self.face_analyzer is None:
            logger.warning("人脸分析器未加载")
            return None
        
        # 加载图像
        if isinstance(reference_image, (str, Path)):
            img = Image.open(reference_image).convert('RGB')
        else:
            img = reference_image.convert('RGB')
        
        img_np = np.array(img)
        
        # 检测人脸
        faces = self.face_analyzer.get(img_np)
        
        if not faces:
            logger.warning("参考图中未检测到人脸")
            return None
        
        # 使用最大的人脸
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        return largest_face.embedding
    
    def generate_video(
        self,
        prompt: str,
        reference_image: Union[str, Path, Image.Image],
        negative_prompt: str = "",
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        id_strength: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
        width: int = 512,
        height: int = 512,
    ) -> List[Image.Image]:
        """
        生成身份保持视频
        
        Args:
            prompt: 文本提示词
            reference_image: 参考人脸图像
            negative_prompt: 负面提示词
            num_frames: 帧数
            fps: 帧率
            id_strength: 身份保持强度 (0-1)
            guidance_scale: CFG 引导强度
            num_inference_steps: 推理步数
            seed: 随机种子
            output_path: 输出视频路径
            width: 视频宽度
            height: 视频高度
            
        Returns:
            生成的帧列表
        """
        if not self.loaded:
            self.load_models()
        
        # 使用默认值
        num_frames = num_frames or self.num_frames
        fps = fps or self.fps
        id_strength = id_strength or self.id_strength
        guidance_scale = guidance_scale or self.guidance_scale
        num_inference_steps = num_inference_steps or self.num_inference_steps
        
        # 设置随机种子
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        else:
            generator = None
        
        logger.info(f"生成视频: {prompt[:50]}...")
        logger.info(f"  帧数: {num_frames}, FPS: {fps}, ID强度: {id_strength}")
        logger.info(f"  分辨率: {width}x{height}")
        
        # 加载参考图用于 IP-Adapter 风格的图像引导
        ref_img = self._load_reference_image(reference_image, width, height)
        
        # 提取人脸嵌入用于验证
        face_embedding = self.extract_face_embedding(reference_image)
        
        if face_embedding is not None:
            logger.info("  ✓ 人脸嵌入提取成功")
            self._reference_embedding = face_embedding  # 保存用于后续验证
        else:
            logger.warning("  ⚠ 无法提取人脸嵌入")
            self._reference_embedding = None
        
        # 增强 prompt 以保持身份
        enhanced_prompt = self._enhance_prompt_for_identity(prompt, id_strength)
        
        # 默认负面提示词
        if not negative_prompt:
            negative_prompt = (
                "bad quality, worst quality, blurry, deformed face, "
                "multiple faces, distorted, ugly, disfigured, "
                "bad anatomy, extra limbs, cloned face"
            )
        
        try:
            # 构建生成参数
            gen_kwargs = {
                "prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
                "width": width,
                "height": height,
            }
            
            # 生成视频
            # 检查是否使用 SDXL (默认为 False)
            use_sdxl = getattr(self, "use_sdxl", False)
            
            if use_sdxl:
                 # SDXL 可能不需要 ip_adapter 参数或者需要不同格式
                 # 暂时移除 ip_adapter 参数以修复基础生成
                 if "ip_adapter_image" in gen_kwargs:
                     logger.warning("  SDXL 模式下暂时移除 ip_adapter_image 参数")
                     gen_kwargs.pop("ip_adapter_image", None)
                     gen_kwargs.pop("ip_adapter_scale", None)

            output = self.pipeline(**gen_kwargs)
            
            frames = output.frames[0]  # 获取帧列表
            
            logger.info(f"  ✓ 生成完成，共 {len(frames)} 帧")
            
            # 保存视频
            if output_path:
                self._save_video(frames, output_path, fps)
                logger.info(f"  ✓ 视频保存到: {output_path}")
            
            return frames
            
        except Exception as e:
            logger.error(f"  ❌ 视频生成失败: {e}")
            raise
    
    def _load_reference_image(
        self, 
        reference_image: Union[str, Path, Image.Image],
        width: int = 512,
        height: int = 512
    ) -> Image.Image:
        """加载并预处理参考图"""
        if isinstance(reference_image, (str, Path)):
            img = Image.open(reference_image).convert('RGB')
        else:
            img = reference_image.convert('RGB')
        
        # 调整大小
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        return img
    
    def _enhance_prompt_for_identity(self, prompt: str, id_strength: float) -> str:
        """
        增强 prompt 以提高身份保持
        
        Args:
            prompt: 原始 prompt
            id_strength: 身份强度
            
        Returns:
            增强后的 prompt
        """
        # 根据强度添加身份相关的描述
        identity_keywords = []
        
        if id_strength >= 0.7:
            identity_keywords.extend([
                "same person throughout",
                "consistent face",
                "maintaining identity",
            ])
        elif id_strength >= 0.5:
            identity_keywords.extend([
                "consistent appearance",
                "same character",
            ])
        
        if identity_keywords:
            identity_prefix = ", ".join(identity_keywords) + ", "
            return identity_prefix + prompt
        
        return prompt
    
    def _save_video(self, frames: List[Image.Image], output_path: str, fps: int):
        """保存视频"""
        import imageio
        
        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为 numpy 数组
        frame_arrays = [np.array(frame) for frame in frames]
        
        # 保存视频
        imageio.mimsave(output_path, frame_arrays, fps=fps)
    
    def unload(self):
        """卸载模型"""
        import gc
        
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        if self.face_analyzer is not None:
            del self.face_analyzer
            self.face_analyzer = None
        
        if self.animator_adapter is not None:
            del self.animator_adapter
            self.animator_adapter = None
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.loaded = False
        logger.info("IDAnimatorEngine 已卸载")
    
    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            "loaded": self.loaded,
            "device": self.device,
            "id_strength": self.id_strength,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "has_face_adapter": self.animator_adapter is not None,
            "has_face_analyzer": self.face_analyzer is not None,
        }


def create_id_animator_engine(config_path: str = "config.yaml") -> IDAnimatorEngine:
    """
    创建 ID-Animator 引擎的工厂函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        IDAnimatorEngine 实例
    """
    import yaml
    
    # 加载配置
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
            config = full_config.get("video", {}).get("id_animator", {})
    
    return IDAnimatorEngine(config)


if __name__ == "__main__":
    """测试 ID-Animator 引擎"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("ID-Animator 引擎测试")
    print("=" * 60)
    
    # 创建引擎
    engine = IDAnimatorEngine({
        "model_dir": "models",
        "id_strength": 0.7,
        "num_frames": 16,
        "fps": 8,
    })
    
    # 显示状态
    status = engine.get_status()
    print(f"\n引擎状态:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # 测试人脸嵌入提取
    ref_path = Path("reference_image/hanli_mid.jpg")
    if ref_path.exists():
        print(f"\n测试人脸嵌入提取: {ref_path}")
        engine._load_face_analyzer()
        embedding = engine.extract_face_embedding(ref_path)
        if embedding is not None:
            print(f"  ✓ 嵌入维度: {embedding.shape}")
        else:
            print("  ✗ 嵌入提取失败")
    
    # 清理
    engine.unload()
    print("\n✅ 测试完成!")
