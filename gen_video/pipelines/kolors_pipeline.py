#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kolors Pipeline（真实感人脸生成）
标准 diffusers 格式
"""

import torch
from diffusers import DiffusionPipeline
from PIL import Image
from typing import Optional
from .base_pipeline import BasePipeline


class KolorsPipeline(BasePipeline):
    """Kolors Pipeline（真实感人脸生成）"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        super().__init__(model_path, device)
        self.loaded = False
    
    def load(self) -> None:
        """加载 Kolors 模型"""
        if self.loaded and self.pipe is not None:
            return
        
        print(f"加载 Kolors 模型: {self.model_path}")
        
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="balanced"
        )
        
        self.loaded = True
        print("✅ Kolors 模型加载完成")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 22,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """生成图像"""
        if not self.loaded:
            self.load()
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Kolors tokenizer 有严重 bug，即使很短的提示词也会溢出
        # 尝试使用最小化的提示词，并添加特殊处理
        # 限制到非常短的长度（20字符），并确保负面提示词为空或很短
        max_chars = 20  # 非常短的限制
        
        safe_prompt = prompt[:max_chars] if len(prompt) > max_chars else prompt
        # 负面提示词设置为空字符串，避免 tokenizer 问题
        safe_negative = ""
        
        if len(prompt) > max_chars:
            print(f"  ⚠ Kolors tokenizer 限制：提示词长度从 {len(prompt)} 字符截断到 {len(safe_prompt)} 字符（避免溢出）")
        if negative_prompt:
            print(f"  ⚠ Kolors tokenizer 限制：负面提示词已禁用（避免溢出）")
        
        # 尝试直接调用，如果失败则抛出更友好的错误
        try:
            result = self.pipe(
                prompt=safe_prompt,
                negative_prompt=safe_negative,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs
            )
            
            return result.images[0]
        except OverflowError as e:
            # Kolors tokenizer 有严重 bug，无法修复
            error_msg = (
                f"Kolors tokenizer 存在严重 bug，无法处理提示词（即使很短也会溢出）。\n"
                f"建议：\n"
                f"1. 使用其他模型（如 'sdxl' 或 'flux1'）\n"
                f"2. 或使用 ModelManager 的其他任务类型（如 'science_background' 使用 Flux.2）\n"
                f"3. 等待 Kolors 官方修复 tokenizer bug"
            )
            print(f"  ❌ {error_msg}")
            raise RuntimeError(error_msg) from e

