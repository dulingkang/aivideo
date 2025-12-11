#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练科普主持人 LoRA
使用 diffusers + PEFT 进行训练
"""

import os
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
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
        # 文件名格式：_repeat_10_提示词.jpg
        if '_repeat_' in filename:
            parts = filename.split('_repeat_', 1)
            if len(parts) > 1:
                prompt_part = parts[1].split('_', 1)
                if len(prompt_part) > 1:
                    prompt = prompt_part[1]
                    # 移除扩展名
                    prompt = prompt.rsplit('.', 1)[0]
                    return prompt
        
        # 如果没有找到，返回默认提示词
        return "科普主持人，专业形象"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        item = self.images[idx]
        
        # 加载图片
        image = Image.open(item['path']).convert('RGB')
        
        # 调整大小（如果已经是 1024x1024 可以跳过）
        if image.size != (self.size, self.size):
            image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # 转换为 tensor (归一化到 [-1, 1])
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 归一化到 [-1, 1]
        ])
        image_tensor = transform(image)
        
        # Tokenize 提示词
        prompt = item['prompt']
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


def train_lora(
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
    resolution: int = 1024
):
    """
    训练 LoRA
    
    Args:
        data_dir: 训练数据目录
        output_dir: 输出目录
        base_model_path: 基础模型路径（Flux.1）
        num_train_epochs: 训练轮数
        train_batch_size: 批次大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        lora_rank: LoRA 维度
        lora_alpha: LoRA alpha
        save_steps: 每多少步保存一次
        resolution: 图片分辨率
    """
    
    print("=" * 60)
    print("🚀 开始训练科普主持人 LoRA")
    print("=" * 60)
    
    # 1. 初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16"
    )
    
    # 2. 加载基础模型
    print(f"\n📦 加载基础模型: {base_model_path}")
    # 使用 "balanced" 而不是 "auto"（Flux 模型要求）
    pipe = DiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="balanced"
    )
    
    # 3. 配置 LoRA
    print(f"\n🔧 配置 LoRA (rank={lora_rank}, alpha={lora_alpha})")
    
    # Flux 模型使用 transformer 而不是 unet
    # 检查模型架构
    if hasattr(pipe, 'transformer'):
        # Flux 模型：使用 transformer
        model_component = pipe.transformer
        model_name = "transformer"
        
        # Flux transformer 的注意力层名称
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
        ]
    elif hasattr(pipe, 'unet'):
        # 标准 SDXL/SD 模型：使用 unet
        model_component = pipe.unet
        model_name = "unet"
        
        # UNet 的注意力层名称
        target_modules = [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ]
    else:
        raise ValueError("无法找到 transformer 或 unet 组件")
    
    print(f"  ℹ 检测到模型组件: {model_name}")
    print(f"  ℹ 目标模块: {target_modules}")
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
    )
    
    # 4. 应用 LoRA 到模型组件
    if model_name == "transformer":
        pipe.transformer = get_peft_model(pipe.transformer, lora_config)
        trainable_model = pipe.transformer
    else:
        pipe.unet = get_peft_model(pipe.unet, lora_config)
        trainable_model = pipe.unet
    
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
        trainable_model.parameters(),
        lr=learning_rate
    )
    
    # 7. 准备训练
    trainable_model, optimizer, dataloader = accelerator.prepare(
        trainable_model, optimizer, dataloader
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
    
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), desc="训练中")
    
    trainable_model.train()
    
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(trainable_model):
                # 获取设备（确保所有张量在同一设备）
                device = next(trainable_model.parameters()).device
                
                # 前向传播 - VAE 编码
                # 确保数据类型匹配（VAE 使用 float16）
                pixel_values = batch['pixel_values'].to(device, dtype=torch.float16)
                with torch.no_grad():
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor
                
                # 保存原始 latents 形状（用于后续处理）
                original_latent_shape = latents.shape
                
                # 添加噪声（Flux 使用 Flow Matching，需要不同的处理）
                noise = torch.randn_like(latents, device=device, dtype=latents.dtype)
                
                # 检查调度器类型
                scheduler_name = type(pipe.scheduler).__name__
                if "Flow" in scheduler_name or "FlowMatch" in scheduler_name:
                    # Flux Flow Matching：使用时间步采样
                    # Flow Matching 使用连续时间 t ∈ [0, 1]
                    timesteps = torch.rand(
                        (latents.shape[0],),
                        device=device,
                        dtype=latents.dtype
                    )  # 随机时间步 [0, 1]
                    
                    # Flow Matching 的噪声添加方式
                    # x_t = (1 - t) * x_0 + t * x_1，其中 x_1 是噪声
                    t = timesteps.view(-1, 1, 1, 1)  # 广播到空间维度
                    noisy_latents = (1 - t) * latents + t * noise
                else:
                    # 标准扩散模型（DDPM/DDIM）
                    timesteps = torch.randint(
                        0, pipe.scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=device
                    )
                    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                
                # 编码提示词（Flux 使用双 T5 编码器）
                input_ids = batch['input_ids'].to(device)
                with torch.no_grad():
                    # Flux 使用 text_encoder_1 和 text_encoder_2（T5）
                    if hasattr(pipe, 'text_encoder_1') and hasattr(pipe, 'text_encoder_2'):
                        # Flux 的双编码器
                        prompt_embeds_1 = pipe.text_encoder_1(input_ids)[0]
                        prompt_embeds_2 = pipe.text_encoder_2(input_ids)[0]
                        encoder_hidden_states = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
                    elif hasattr(pipe, 'text_encoder'):
                        # 标准编码器（备用）
                        encoder_hidden_states = pipe.text_encoder(input_ids)[0]
                    else:
                        raise ValueError("无法找到 text encoder")
                
                # 预测噪声（Flux 使用 transformer，SDXL 使用 unet）
                is_flow_matching = "Flow" in scheduler_name or "FlowMatch" in scheduler_name
                
                if model_name == "transformer":
                    # Flux transformer 的调用方式
                    # Flux transformer 需要特定的输入格式
                    # 需要将 latents 重塑为正确的形状
                    batch_size = noisy_latents.shape[0]
                    height, width = noisy_latents.shape[2], noisy_latents.shape[3]
                    
                    # Flux transformer 期望的输入格式
                    # 需要将 (B, C, H, W) 重塑为 (B*H*W, C) 或类似格式
                    # 但实际格式可能不同，需要根据模型要求调整
                    
                    if is_flow_matching:
                        # Flow Matching：时间步已经是 [0, 1] 范围
                        try:
                            # 尝试标准调用
                            model_pred = pipe.transformer(
                                hidden_states=noisy_latents,
                                timestep=timesteps,
                                encoder_hidden_states=encoder_hidden_states,
                            ).sample
                        except RuntimeError as e:
                            if "shapes cannot be multiplied" in str(e):
                                # 形状不匹配，可能需要不同的输入格式
                                # Flux 可能需要将 latents 展平或重塑
                                # 尝试使用 pipe 的 encode_prompt 和标准生成流程
                                # 或者使用 pipe 的 __call__ 方法
                                print(f"⚠️  Flux transformer 输入形状错误，尝试使用 pipe 的标准方法")
                                # 对于训练，我们需要直接调用 transformer
                                # 可能需要调整输入形状
                                # 尝试：将 latents 重塑为 transformer 期望的格式
                                # Flux transformer 可能需要 (B, H*W, C) 格式
                                latent_height, latent_width = noisy_latents.shape[2], noisy_latents.shape[3]
                                noisy_latents_reshaped = noisy_latents.permute(0, 2, 3, 1).reshape(
                                    batch_size, latent_height * latent_width, -1
                                )
                                model_pred = pipe.transformer(
                                    hidden_states=noisy_latents_reshaped,
                                    timestep=timesteps,
                                    encoder_hidden_states=encoder_hidden_states,
                                ).sample
                                # 重塑回原始形状
                                model_pred = model_pred.reshape(
                                    batch_size, latent_height, latent_width, -1
                                ).permute(0, 3, 1, 2)
                            else:
                                raise
                    else:
                        # 标准扩散
                        model_pred = pipe.transformer(
                            hidden_states=noisy_latents,
                            timestep=timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                        ).sample
                else:
                    # 标准 UNet
                    model_pred = pipe.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states
                    ).sample
                
                # 计算损失
                if is_flow_matching:
                    # Flow Matching：预测速度场 v_t = x_1 - x_0
                    # 目标速度场是 noise - latents
                    target_velocity = noise - latents
                    loss = torch.nn.functional.mse_loss(model_pred.float(), target_velocity.float())
                else:
                    # 标准扩散：预测噪声
                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())
                
                # 反向传播
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                # 更新进度条
                if step % 10 == 0:  # 每 10 步更新一次
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            global_step += 1
            progress_bar.update(1)
            
            # 保存检查点
            if global_step % save_steps == 0:
                if accelerator.is_main_process:
                    checkpoint_dir = Path(output_dir) / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 保存 LoRA 权重
                    trainable_model.save_pretrained(str(checkpoint_dir))
                    print(f"\n💾 已保存检查点: {checkpoint_dir}")
    
    # 9. 保存最终模型
    print(f"\n💾 保存最终模型到: {output_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if accelerator.is_main_process:
        trainable_model.save_pretrained(str(output_path))
        
        # 也保存为 safetensors 格式（如果可能）
        try:
            from safetensors.torch import save_file
            # 只保存 LoRA 权重（不是完整模型）
            state_dict = {}
            for name, param in trainable_model.named_parameters():
                if 'lora' in name.lower():
                    state_dict[name] = param.data.cpu()
            
            if state_dict:
                safetensors_path = output_path / "pytorch_lora_weights.safetensors"
                save_file(state_dict, str(safetensors_path))
                print(f"✅ 已保存 safetensors: {safetensors_path}")
            else:
                print("⚠️  未找到 LoRA 权重，使用 save_pretrained 保存")
        except ImportError:
            print("⚠️  未安装 safetensors，跳过 safetensors 格式保存")
        except Exception as e:
            print(f"⚠️  保存 safetensors 时出错: {e}，使用 save_pretrained 保存")
    
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)
    print(f"📁 模型保存在: {output_dir}")
    print(f"📝 使用方式:")
    print(f"   lora_path = '{output_dir}/pytorch_lora_weights.safetensors'")
    print(f"   或 '{output_dir}' (目录)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练科普主持人 LoRA")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="train_data/host_person",
        help="训练数据目录（默认: train_data/host_person）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/lora/host_person",
        help="输出目录（默认: models/lora/host_person）"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="models/flux1-dev",
        help="基础模型路径（默认: models/flux1-dev）"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数（默认: 10）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="批次大小（默认: 1，根据显存调整）"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="梯度累积步数（默认: 4）"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="学习率（默认: 1e-4）"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA 维度（默认: 32）"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha（默认: 16）"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="每多少步保存一次（默认: 200）"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="图片分辨率（默认: 1024）"
    )
    
    args = parser.parse_args()
    
    train_lora(
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
        resolution=args.resolution
    )

