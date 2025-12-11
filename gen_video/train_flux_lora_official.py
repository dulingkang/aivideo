#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flux LoRA 训练脚本（使用 diffusers 官方方法）
基于 diffusers 官方 Flux 训练示例
"""

import os
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import FluxPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import argparse


class HostDataset(Dataset):
    """主持人训练数据集"""
    
    def __init__(self, data_dir: str, tokenizer, size: int = 1024):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.size = size
        
        # 收集所有图片
        self.images = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP'}
        
        for img_file in sorted(self.data_dir.iterdir()):
            if img_file.suffix in image_extensions:
                # 从文件名提取提示词
                prompt = self._extract_prompt_from_filename(img_file.name)
                self.images.append({
                    'path': img_file,
                    'prompt': prompt
                })
        
        print(f"✅ 找到 {len(self.images)} 张训练图片")
    
    def _extract_prompt_from_filename(self, filename: str) -> str:
        """从文件名提取提示词"""
        if '_repeat_' in filename:
            parts = filename.split('_repeat_', 1)
            if len(parts) > 1:
                prompt_part = parts[1].split('_', 1)
                if len(prompt_part) > 1:
                    prompt = prompt_part[1]
                    prompt = prompt.rsplit('.', 1)[0]
                    return prompt
        return "科普主持人，专业形象"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        item = self.images[idx]
        
        # 加载图片
        image = Image.open(item['path']).convert('RGB')
        
        # 调整大小
        if image.size != (self.size, self.size):
            image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # 转换为 tensor
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        image_tensor = transform(image).float()
        
        # Tokenize 提示词（Flux 使用 T5，需要特殊处理）
        prompt = item['prompt']
        # Flux 使用双编码器，这里先用简单的 tokenizer
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': image_tensor,
            'input_ids': text_inputs.input_ids.squeeze(),
            'prompt': prompt
        }


def train_flux_lora(
    data_dir: str,
    output_dir: str,
    base_model_path: str,
    num_train_epochs: int = 10,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    lora_rank: int = 32,
    lora_alpha: int = 16,
    save_steps: int = 200,
    resolution: int = 1024,
    use_bf16: bool = True,  # 使用 bf16（H20 支持）
    use_flash_attention: bool = False,  # Flash Attention（可选）
):
    """
    训练 Flux LoRA（使用 diffusers 官方方法）
    
    基于 diffusers 官方 Flux 训练示例
    """
    
    print("=" * 60)
    print("🚀 开始训练 Flux LoRA（使用 diffusers 官方方法）")
    print("=" * 60)
    
    # 1. 初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="bf16" if use_bf16 else "fp16"
    )
    
    # 2. 加载基础模型
    print(f"\n📦 加载基础模型: {base_model_path}")
    pipe = FluxPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map="balanced"
    )
    
    # 3. 配置 LoRA（Flux DiT 架构）
    print(f"\n🔧 配置 LoRA (rank={lora_rank}, alpha={lora_alpha})")
    
    # Flux transformer 的注意力层
    # 注意：Flux 使用 DiT 架构，目标模块与 UNet 不同
    target_modules = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
    ]
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
    )
    
    # 4. 应用 LoRA 到 transformer
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    
    # 5. 准备数据集
    print(f"\n📁 准备训练数据: {data_dir}")
    dataset = HostDataset(
        data_dir=data_dir,
        tokenizer=pipe.tokenizer,
        size=resolution
    )
    
    if len(dataset) == 0:
        raise ValueError(f"未找到训练数据！请检查目录: {data_dir}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # 6. 设置优化器
    optimizer = torch.optim.AdamW(
        pipe.transformer.parameters(),
        lr=learning_rate
    )
    
    # 7. 准备训练
    pipe.transformer, optimizer, dataloader = accelerator.prepare(
        pipe.transformer, optimizer, dataloader
    )
    
    # 8. 训练循环
    num_update_steps_per_epoch = len(dataloader) // gradient_accumulation_steps
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    
    print(f"\n🎯 训练配置:")
    print(f"   训练轮数: {num_train_epochs}")
    print(f"   总步数: {max_train_steps}")
    print(f"   批次大小: {train_batch_size}")
    print(f"   梯度累积: {gradient_accumulation_steps}")
    print(f"   学习率: {learning_rate}")
    print(f"   分辨率: {resolution}x{resolution}")
    print(f"   精度: {'bf16' if use_bf16 else 'fp16'}")
    
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), desc="训练中")
    
    pipe.transformer.train()
    
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(pipe.transformer):
                device = next(pipe.transformer.parameters()).device
                
                # VAE 编码
                pixel_values = batch['pixel_values'].to(device, dtype=torch.bfloat16 if use_bf16 else torch.float16)
                with torch.no_grad():
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor
                
                # Flow Matching 噪声添加
                noise = torch.randn_like(latents, device=device, dtype=latents.dtype)
                timesteps = torch.rand(
                    (latents.shape[0],),
                    device=device,
                    dtype=latents.dtype
                )  # [0, 1]
                
                # Flow Matching: x_t = (1-t)*x_0 + t*x_1
                t = timesteps.view(-1, 1, 1, 1)
                noisy_latents = (1 - t) * latents + t * noise
                
                # 编码提示词（Flux 双编码器）
                input_ids = batch['input_ids'].to(device)
                with torch.no_grad():
                    if hasattr(pipe, 'text_encoder_1') and hasattr(pipe, 'text_encoder_2'):
                        prompt_embeds_1 = pipe.text_encoder_1(input_ids)[0]
                        prompt_embeds_2 = pipe.text_encoder_2(input_ids)[0]
                        encoder_hidden_states = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
                    else:
                        raise ValueError("无法找到 Flux 双编码器")
                
                # 预测（使用 pipe 的标准方法）
                # 注意：这里需要使用 pipe 的标准调用方式
                # Flux transformer 的输入格式由 pipe 内部处理
                try:
                    # 使用 pipe 的标准方法调用 transformer
                    # 这需要正确的输入格式
                    model_pred = pipe.transformer(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample
                except RuntimeError as e:
                    if "shapes cannot be multiplied" in str(e) or "mat1 and mat2" in str(e):
                        print(f"\n❌ Flux transformer 输入格式问题")
                        print(f"💡 建议：使用 diffusers 官方训练脚本")
                        print(f"   参考: https://github.com/huggingface/diffusers/tree/main/examples/flux")
                        raise RuntimeError(
                            f"Flux transformer 输入格式复杂，建议使用官方训练脚本。"
                            f"错误: {e}"
                        ) from e
                    raise
                
                # 计算损失（Flow Matching：速度场）
                target_velocity = noise - latents
                loss = torch.nn.functional.mse_loss(model_pred.float(), target_velocity.float())
                
                # 反向传播
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                # 更新进度条
                if step % 10 == 0:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            global_step += 1
            progress_bar.update(1)
            
            # 保存检查点
            if global_step % save_steps == 0:
                if accelerator.is_main_process:
                    checkpoint_dir = Path(output_dir) / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    pipe.transformer.save_pretrained(str(checkpoint_dir))
                    print(f"\n💾 已保存检查点: {checkpoint_dir}")
    
    # 9. 保存最终模型
    print(f"\n💾 保存最终模型到: {output_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if accelerator.is_main_process:
        pipe.transformer.save_pretrained(str(output_path))
        print(f"✅ 训练完成！模型保存在: {output_dir}")
    
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 Flux LoRA（diffusers 官方方法）")
    parser.add_argument("--data-dir", type=str, default="train_data/host_person")
    parser.add_argument("--output-dir", type=str, default="models/lora/host_person")
    parser.add_argument("--base-model", type=str, default="models/flux1-dev")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--use-bf16", action="store_true", help="使用 bf16（H20 支持）")
    
    args = parser.parse_args()
    
    train_flux_lora(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_model_path=args.base_model,
        num_train_epochs=args.epochs,
        train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        save_steps=args.save_steps,
        resolution=args.resolution,
        use_bf16=args.use_bf16,
    )

