#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flux Pipeline（Flux.1 和 Flux.2）
标准 diffusers 格式
"""

import torch
from diffusers import DiffusionPipeline
from PIL import Image
from typing import Optional
from .base_pipeline import BasePipeline


class FluxPipeline(BasePipeline):
    """Flux Pipeline（支持 Flux.1 和 Flux.2）"""
    
    def __init__(self, model_path: str, device: Optional[str] = None, model_type: str = "flux2"):
        """
        初始化 Flux Pipeline
        
        Args:
            model_path: 模型路径
            device: 设备
            model_type: 模型类型 ("flux1" 或 "flux2")
        """
        super().__init__(model_path, device)
        self.model_type = model_type
        self.loaded = False
    
    def load(self) -> None:
        """加载 Flux 模型"""
        if self.loaded and self.pipe is not None:
            return
        
        print(f"加载 Flux ({self.model_type}) 模型: {self.model_path}")
        
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="balanced"
        )
        
        self.loaded = True
        print(f"✅ Flux ({self.model_type}) 模型加载完成")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 18,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        lora_path: Optional[str] = None,
        lora_alpha: float = 1.0,
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
            lora_path: LoRA 权重路径（可选）
            lora_alpha: LoRA 权重（0.0-1.0，默认 1.0）
            **kwargs: 其他参数
        """
        if not self.loaded:
            self.load()
        
        # 加载 LoRA（如果提供）
        if lora_path:
            from pathlib import Path
            from safetensors import safe_open
            import torch
            
            lora_path_obj = Path(lora_path)
            if lora_path_obj.exists():
                try:
                    # 直接处理 PEFT 格式的 LoRA（跳过 diffusers 的自动转换，避免误判为 FAL/Kontext 格式）
                    print(f"  🔧 加载 PEFT 格式 LoRA: {lora_path_obj.name}")
                    
                    # 读取 LoRA 权重并转换键名（PEFT 格式 → diffusers 格式）
                    lora_state_dict = {}
                    with safe_open(str(lora_path_obj), framework="pt") as f:
                        for key in f.keys():
                            new_key = key
                            # 步骤 1：移除 base_model.model. 前缀（PEFT 格式）
                            if key.startswith("base_model.model."):
                                new_key = key.replace("base_model.model.", "")
                            
                            # 步骤 2：将 single_transformer_blocks 替换为 transformer_blocks
                            # LoRA: single_transformer_blocks.0.attn.to_k.lora_A.default.weight
                            # Flux: transformer_blocks.0.attn.to_k.weight
                            if "single_transformer_blocks" in new_key:
                                new_key = new_key.replace("single_transformer_blocks", "transformer_blocks")
                            
                            # 步骤 3：移除 .default 部分（PEFT 格式）
                            if ".default." in new_key:
                                new_key = new_key.replace(".default.", ".")
                            
                            # 步骤 4：添加 transformer. 前缀（如果还没有，且是 transformer_blocks 相关的键）
                            # diffusers 的 load_lora_weights 期望键名格式为 transformer.transformer_blocks...
                            # 这样可以正确匹配到 FluxTransformer2DModel
                            if "transformer_blocks" in new_key and not new_key.startswith("transformer."):
                                new_key = f"transformer.{new_key}"
                            
                            lora_state_dict[new_key] = f.get_tensor(key)
                    
                    # 保存转换后的权重到临时文件
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp_file:
                        from safetensors.torch import save_file
                        save_file(lora_state_dict, tmp_file.name)
                        tmp_path = tmp_file.name
                    
                    try:
                        # 方法 1：使用 load_lora_weights 加载转换后的权重，指定 prefix=None 以避免警告
                        try:
                            # 指定 prefix=None 让系统自动检测正确的键名格式
                            self.pipe.load_lora_weights(tmp_path, adapter_name="character_lora", weight_name=None)
                            self.pipe.set_adapters(["character_lora"], adapter_weights=[lora_alpha])
                            print(f"  ✅ 已加载 LoRA: {lora_path_obj.name} (alpha={lora_alpha})")
                        except Exception as e2:
                            # 方法 2：如果方法 1 失败，尝试不指定 adapter_name
                            print(f"  ⚠ 方法 1 失败，尝试方法 2: {e2}")
                            self.pipe.load_lora_weights(tmp_path)
                            # 获取加载的 adapter 名称
                            adapters = list(self.pipe.get_active_adapters()) if hasattr(self.pipe, 'get_active_adapters') else []
                            if adapters:
                                self.pipe.set_adapters(adapters, adapter_weights=[lora_alpha])
                            print(f"  ✅ 已加载 LoRA (方法2): {lora_path_obj.name} (alpha={lora_alpha})")
                    finally:
                        # 清理临时文件
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        
                except Exception as e:
                    print(f"  ⚠ LoRA 加载失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  ⚠ LoRA 文件不存在: {lora_path}")
        
        # 验证 LoRA 是否已激活（如果加载了 LoRA）
        if lora_path and hasattr(self.pipe, 'get_active_adapters'):
            active_adapters = self.pipe.get_active_adapters()
            if active_adapters:
                print(f"  ✅ LoRA 已激活: {active_adapters}, 权重: {lora_alpha}")
            else:
                print(f"  ⚠ LoRA 已加载但未激活，尝试重新激活...")
                # 尝试重新激活
                try:
                    self.pipe.set_adapters(["character_lora"], adapter_weights=[lora_alpha])
                    print(f"  ✅ LoRA 已重新激活，权重: {lora_alpha}")
                except Exception as e:
                    print(f"  ⚠ LoRA 激活失败: {e}")
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 打印最终使用的提示词（用于调试）
        if lora_path:
            print(f"  🔍 生成参数: prompt长度={len(prompt)}, LoRA权重={lora_alpha}, steps={num_inference_steps}, guidance={guidance_scale}")
        
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

