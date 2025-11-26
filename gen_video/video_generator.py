#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频生成器
使用 Stable Video Diffusion (SVD) 从图像生成视频
支持静态图像动画（Ken Burns效果）和SVD视频生成
"""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import imageio
from scene_motion_analyzer import SceneMotionAnalyzer


class VideoGenerator:
    """视频生成器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化视频生成器"""
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (Path.cwd() / self.config_path).resolve()
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.video_config = self.config.get('video', {})
        self.config_dir = self.config_path.parent
        
        # 初始化智能分析器
        self.motion_analyzer = SceneMotionAnalyzer()
        
        # 模型相关
        self.pipeline = None
        self.model_loaded = False
    
    def load_model(self):
        """加载SVD模型"""
        if self.model_loaded:
            print("  ℹ 模型已加载，跳过")
            return
        
        model_type = self.video_config.get('model_type', 'svd-xt')
        model_path = self.video_config.get('model_path')
        
        if not model_path:
            raise ValueError("video.model_path 未配置")
        
        print(f"  加载视频生成模型: {model_type}")
        print(f"  模型路径: {model_path}")
        
        try:
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import load_image
            import torch
            
            # 加载SVD pipeline
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            
            # 移动到GPU
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
                self.pipeline.enable_model_cpu_offload()
            
            self.model_loaded = True
            print("  ✓ 模型加载成功")
            
        except Exception as e:
            print(f"  ✗ 模型加载失败: {e}")
            raise
    
    def generate_video(
        self,
        image_path: str,
        output_path: str,
        num_frames: Optional[int] = None,
        fps: Optional[float] = None,
        motion_bucket_id: Optional[int] = None,
        noise_aug_strength: Optional[float] = None,
        scene: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        生成视频
        
        Args:
            image_path: 输入图像路径
            output_path: 输出视频路径
            num_frames: 帧数（可选，默认从配置读取）
            fps: 帧率（可选，默认从配置读取）
            motion_bucket_id: 运动参数（可选）
            noise_aug_strength: 噪声参数（可选）
            scene: 场景JSON数据（用于智能分析）
            
        Returns:
            输出视频路径
        """
        print(f"\n生成视频: {image_path} -> {output_path}")
        
        # ========== 1. 智能分析场景 ==========
        analysis = None
        if scene:
            analysis = self.motion_analyzer.analyze(scene)
            print(f"  ℹ 智能分析结果:")
            print(f"    - 物体运动: {analysis['has_object_motion']} ({analysis['object_motion_type']})")
            print(f"    - 镜头运动: {analysis['camera_motion_type']}")
            print(f"    - 运动强度: {analysis['motion_intensity']}")
            print(f"    - 使用SVD: {analysis['use_svd']}")
            
            # 根据分析结果覆盖参数
            if analysis['motion_bucket_id_override'] is not None:
                motion_bucket_id = analysis['motion_bucket_id_override']
                print(f"    - 覆盖 motion_bucket_id: {motion_bucket_id}")
            if analysis['noise_aug_strength_override'] is not None:
                noise_aug_strength = analysis['noise_aug_strength_override']
                print(f"    - 覆盖 noise_aug_strength: {noise_aug_strength}")
        
        # ========== 2. 确定使用哪种生成方式 ==========
        use_svd = True
        if analysis and not analysis['use_svd']:
            use_svd = False
            print(f"  ℹ 检测到完全静态场景，使用静态图像动画")
        elif scene:
            # 检查场景描述中是否有物体运动关键词
            description = (scene.get("description") or "").lower()
            action = (scene.get("action") or "").lower()
            visual = scene.get("visual") or {}
            composition = (visual.get("composition", "") or "").lower() if isinstance(visual, dict) else ""
            fx = (visual.get("fx", "") or "").lower() if isinstance(visual, dict) else ""
            
            all_text = f"{description} {action} {composition} {fx}".lower()
            object_motion_keywords = [
                "unfurling", "unfold", "unfolding", "展开", "舒展开", "展开来", "缓缓展开",
                "open", "opening", "打开", "开启", "张开",
                "rotate", "rotating", "spin", "spinning", "旋转", "转动", "翻转",
                "float", "floating", "drift", "drifting", "飘动", "漂浮", "流动"
            ]
            has_object_motion = any(keyword in all_text for keyword in object_motion_keywords)
            
            if has_object_motion:
                use_svd = True
                print(f"  ℹ 检测到物体运动，使用SVD生成视频")
            elif any(keyword in all_text for keyword in ["still", "motionless", "静止", "不动"]):
                use_svd = False
                print(f"  ℹ 检测到完全静态场景，使用静态图像动画")
        
        # ========== 3. 获取参数并根据场景类型优化 ==========
        if num_frames is None:
            num_frames = self.video_config.get('num_frames', 120)
        if fps is None:
            fps = self.video_config.get('fps', 24)
        if motion_bucket_id is None:
            motion_bucket_id = self.video_config.get('motion_bucket_id', 1.5)
        if noise_aug_strength is None:
            noise_aug_strength = self.video_config.get('noise_aug_strength', 0.00025)
        
        # 根据场景类型进一步优化参数（确保人物动作自然流畅，镜头移动明显，物体运动明显）
        if analysis and use_svd:
            motion_intensity = analysis.get('motion_intensity', 'gentle')
            camera_motion_type = analysis.get('camera_motion_type', 'static')
            has_object_motion = analysis.get('has_object_motion', False)
            object_motion_type = analysis.get('object_motion_type')
            
            # 如果有物体运动（如卷轴展开），确保运动参数适中，减少闪动
            if has_object_motion:
                # 物体运动需要适中的运动参数，确保运动明显但减少闪动
                if motion_bucket_id > 1.8:
                    motion_bucket_id = 1.8  # 限制到1.8，减少闪动
                    print(f"  ℹ 检测到物体运动（{object_motion_type}），限制 motion_bucket_id 至 {motion_bucket_id} 减少闪动")
                if noise_aug_strength > 0.0003:
                    noise_aug_strength = 0.0003  # 限制到0.0003，减少闪动
                    print(f"  ℹ 物体运动场景，限制 noise_aug_strength 至 {noise_aug_strength} 减少闪动")
            
            # 如果有镜头运动，确保运动参数适中，减少闪动
            elif camera_motion_type != 'static':
                # 镜头移动需要适中的运动参数，确保移动明显但减少闪动
                if motion_bucket_id > 1.8:
                    motion_bucket_id = 1.8  # 限制到1.8，减少闪动
                    print(f"  ℹ 检测到镜头运动（{camera_motion_type}），限制 motion_bucket_id 至 {motion_bucket_id} 减少闪动")
                if noise_aug_strength > 0.0003:
                    noise_aug_strength = 0.0003  # 限制到0.0003，减少闪动
                    print(f"  ℹ 镜头运动场景，限制 noise_aug_strength 至 {noise_aug_strength} 减少闪动")
            
            # 对于人物动作场景，确保有适中的运动参数，减少闪动
            elif motion_intensity in ['dynamic', 'moderate']:
                # 确保motion_bucket_id适中，使动作明显但减少闪动
                if motion_bucket_id > 1.8:
                    motion_bucket_id = 1.8  # 限制到1.8，减少闪动
                    print(f"  ℹ 人物动作场景，限制 motion_bucket_id 至 {motion_bucket_id} 减少闪动")
                
                # 确保noise_aug_strength适中，使动作自然流畅但减少闪动
                if noise_aug_strength > 0.0003:
                    noise_aug_strength = 0.0003  # 限制到0.0003，减少闪动
                    print(f"  ℹ 人物动作场景，限制 noise_aug_strength 至 {noise_aug_strength} 减少闪动")
            
            # 对于轻微动作场景，使用适中的参数
            elif motion_intensity == 'gentle':
                # 确保有轻微运动，但不过度
                if motion_bucket_id < 1.8:
                    motion_bucket_id = 1.8
                if noise_aug_strength < 0.00025:
                    noise_aug_strength = 0.0003
                print(f"  ℹ 轻微动作场景，使用适中参数（motion_bucket_id={motion_bucket_id}, noise_aug_strength={noise_aug_strength}）")
        
        # 限制最大帧数
        max_frames = self.video_config.get('max_frames', 384)
        if num_frames > max_frames:
            num_frames = max_frames
            print(f"  ⚠ 帧数超过最大值，限制为 {max_frames}")
        
        # 确保num_frames足够，使动作流畅（对于动作场景）
        if analysis and analysis.get('motion_intensity') in ['dynamic', 'moderate']:
            # 对于动作场景，确保有足够的帧数
            min_frames_for_action = int(fps * 4)  # 至少4秒的帧数
            if num_frames < min_frames_for_action:
                num_frames = min_frames_for_action
                print(f"  ℹ 动作场景，确保足够帧数（{num_frames}帧）使动作流畅")
        
        # ========== 4. 生成视频 ==========
        if use_svd:
            return self._generate_video_svd(
                image_path,
                output_path,
                num_frames=num_frames,
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
            )
        else:
            return self._generate_static_image_animation(
                image_path,
                output_path,
                num_frames=num_frames,
                fps=fps,
                scene=scene,
            )
    
    def _generate_video_svd(
        self,
        image_path: str,
        output_path: str,
        num_frames: int,
        fps: float,
        motion_bucket_id: float,
        noise_aug_strength: float,
    ) -> str:
        """使用SVD生成视频"""
        print(f"  使用SVD生成视频")
        print(f"    参数: num_frames={num_frames}, fps={fps}")
        print(f"          motion_bucket_id={motion_bucket_id}, noise_aug_strength={noise_aug_strength}")
        
        if not self.model_loaded:
            self.load_model()
        
        # 加载图像
        from diffusers.utils import load_image
        image = load_image(image_path)
        
        # 调整图像大小（SVD要求必须是64的倍数）
        width = self.video_config.get('width', 1280)
        height = self.video_config.get('height', 768)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        # 生成视频
        num_inference_steps = self.video_config.get('num_inference_steps', 40)
        decode_chunk_size = self.video_config.get('decode_chunk_size', 8)
        
        # 对于动作场景和镜头运动场景，增加推理步数以提高稳定性和减少闪动
        if motion_bucket_id >= 1.7:
            # 动作场景和镜头运动需要更多步数以确保稳定性和减少闪动
            num_inference_steps = max(num_inference_steps, 50)  # 提高到50步，减少闪动
            print(f"  ℹ 动作/镜头运动场景，增加推理步数至 {num_inference_steps} 提高稳定性和减少闪动")
        
        # 对于动作场景和镜头运动场景，适当调整decode_chunk_size以提高连贯性和减少闪动
        if motion_bucket_id >= 1.7:
            # 动作场景和镜头运动需要适中的chunk size以确保帧间连贯性，减少闪动
            decode_chunk_size = min(decode_chunk_size, 8)  # 保持8，SVD-XT最稳定
            # 不要太小，避免过度平滑导致不自然
            if decode_chunk_size < 7:
                decode_chunk_size = 7  # 最小7，保持流畅度
            print(f"  ℹ 动作/镜头运动场景，调整 decode_chunk_size 至 {decode_chunk_size} 提高连贯性和减少闪动")
        
        # 限制motion_bucket_id在SVD-XT的有效范围内（最大2）
        if motion_bucket_id > 2.0:
            motion_bucket_id = 2.0
            print(f"  ⚠ motion_bucket_id 超过SVD-XT最大值，限制为 2.0")
        
        frames = self.pipeline(
            image,
            decode_chunk_size=decode_chunk_size,
            num_frames=num_frames,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            num_inference_steps=num_inference_steps,
        ).frames[0]
        
        # 保存视频
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
        
        print(f"  ✓ 视频生成成功: {output_path}")
        return output_path
    
    def _generate_static_image_animation(
        self,
        image_path: str,
        output_path: str,
        num_frames: int,
        fps: float,
        scene: Optional[Dict[str, Any]] = None,
    ) -> str:
        """生成静态图像动画（Ken Burns效果）"""
        print(f"  使用静态图像动画（Ken Burns效果）")
        
        # 加载图像
        image = Image.open(image_path)
        width, height = image.size
        
        # 创建视频帧
        frames = []
        for i in range(num_frames):
            # 简单的缩放和平移效果
            scale = 1.0 + (i / num_frames) * 0.1  # 轻微放大
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 裁剪中心区域
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            right = left + width
            bottom = top + height
            
            frame = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            frame = frame.crop((left, top, right, bottom))
            frames.append(np.array(frame))
        
        # 保存视频
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
        
        print(f"  ✓ 静态图像动画生成成功: {output_path}")
        return output_path
