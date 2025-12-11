#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础 Pipeline 接口
所有模型 Pipeline 都继承这个基类
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from PIL import Image
import torch


class BasePipeline(ABC):
    """所有 Pipeline 的基类"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        初始化 Pipeline
        
        Args:
            model_path: 模型路径
            device: 设备 (cuda/cpu)，默认自动选择
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        
    @abstractmethod
    def load(self) -> None:
        """加载模型"""
        raise NotImplementedError
    
    @abstractmethod
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
        """
        生成图像
        
        Args:
            prompt: 提示词
            negative_prompt: 负面提示词
            width: 图像宽度
            height: 图像高度
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            seed: 随机种子
            **kwargs: 其他参数
            
        Returns:
            PIL Image
        """
        raise NotImplementedError
    
    def unload(self) -> None:
        """卸载模型，释放显存"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
    
    def __del__(self):
        """析构时自动卸载"""
        self.unload()

