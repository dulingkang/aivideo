#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hunyuan-DiT Pipeline（中文场景生成）
使用官方 End2End 类
"""

import sys
from pathlib import Path
import torch
from PIL import Image
from typing import Optional
import argparse
from .base_pipeline import BasePipeline


class HunyuanPipeline(BasePipeline):
    """Hunyuan-DiT Pipeline（中文场景生成）"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        super().__init__(model_path, device)
        self.loaded = False
        self.gen = None
        
        # 添加 HunyuanDiT 到路径
        hunyuan_dir = Path(__file__).parent.parent.parent / "HunyuanDiT"
        if hunyuan_dir.exists():
            sys.path.insert(0, str(hunyuan_dir))
        else:
            raise RuntimeError(f"HunyuanDiT 仓库不存在: {hunyuan_dir}")
    
    def load(self) -> None:
        """加载 Hunyuan-DiT 模型"""
        if self.loaded and self.gen is not None:
            return
        
        try:
            from hydit.inference import End2End
        except ImportError as e:
            raise RuntimeError(
                f"无法导入 Hunyuan-DiT 模块。请确保已安装依赖: {e}\n"
                f"运行: cd HunyuanDiT && pip install -r requirements.txt"
            ) from e
        
        print(f"加载 Hunyuan-DiT 模型: {self.model_path}")
        
        # 设置参数
        models_root = Path(self.model_path).parent  # t2i 的父目录
        args = argparse.Namespace(
            model_root=str(models_root),
            enhance=False,  # 不使用增强模型
            infer_mode="torch",
            sampler="ddpm",
            load_4bit=False,
            model="dit-xl-2",
            image_size=[1024, 1024],
            text_len=77,
            text_states_dim=2048,
            noise_schedule="cosine",
            beta_start=0.0001,
            beta_end=0.02,
            predict_type="v_prediction",
            dit_weight=None,
            lora_ckpt=None,
            load_key="ema",
        )
        
        self.gen = End2End(args, models_root)
        self.loaded = True
        print("✅ Hunyuan-DiT 模型加载完成")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """生成图像"""
        if not self.loaded:
            self.load()
        
        # 设置随机种子
        if seed is not None:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        results = self.gen.predict(
            prompt=prompt,
            height=height,
            width=width,
            seed=seed if seed is not None else 42,
            enhanced_prompt=None,
            negative_prompt=negative_prompt,
            infer_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            batch_size=1,
            src_size_cond=None,
            use_style_cond=False,
        )
        
        images = results["images"]
        if not images:
            raise RuntimeError("Hunyuan-DiT 生成失败，未返回图像")
        
        return images[0]
    
    def unload(self) -> None:
        """卸载模型"""
        if self.gen is not None:
            del self.gen
            self.gen = None
            torch.cuda.empty_cache()
        self.loaded = False

