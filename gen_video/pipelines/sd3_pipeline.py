#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SD3.5 Large Turbo Pipeline（极速批量生成）
标准 diffusers 格式
"""

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from typing import Optional
from .base_pipeline import BasePipeline


class SD3TurboPipeline(BasePipeline):
    """SD3.5 Large Turbo Pipeline（极速批量生成）"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        super().__init__(model_path, device)
        self.loaded = False
    
    def load(self) -> None:
        """加载 SD3.5 Large Turbo 模型"""
        if self.loaded and self.pipe is not None:
            return
        
        print(f"加载 SD3.5 Large Turbo 模型: {self.model_path}")
        
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="balanced"
        )
        
        self.loaded = True
        print("✅ SD3.5 Large Turbo 模型加载完成")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 8,  # Turbo 模式，步数少
        guidance_scale: float = 1.0,  # Turbo 模式，低引导
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """生成图像"""
        if not self.loaded:
            self.load()
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs
        )
        
        return result.images[0]

