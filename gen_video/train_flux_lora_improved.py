#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的 Flux LoRA 训练脚本
针对当前问题优化：人脸不对、形象不对
"""

import os
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import FluxPipeline
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm import tqdm
import argparse


class HostDataset(Dataset):
    """主持人训练数据集"""
    
    def __init__(self, data_dir: str, tokenizer, tokenizer_2, size: int = 512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer  # CLIP tokenizer
        self.tokenizer_2 = tokenizer_2  # T5 tokenizer
        self.size = size
        
        # 收集所有图片
        self.images = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP'}
        
        for img_file in sorted(self.data_dir.iterdir()):
            if img_file.suffix in image_extensions:
                prompt = self._extract_prompt_from_filename(img_file.name)
                self.images.append({
                    'path': img_file,
                    'prompt': prompt
                })
        
        print(f"✅ 找到 {len(self.images)} 张训练图片")
    
    def _extract_prompt_from_filename(self, filename: str) -> str:
        """从文件名提取提示词，并优化顺序（核心信息在前）"""
        if '_repeat_' in filename:
            parts = filename.split('_repeat_', 1)
            if len(parts) > 1:
                prompt_part = parts[1].split('_', 1)
                if len(prompt_part) > 1:
                    prompt = prompt_part[1]
                    prompt = prompt.rsplit('.', 1)[0]
                    
                    # 优化提示词顺序：核心信息在前，细节在后
                    # 这样即使被 CLIP tokenizer 截断（77 tokens），核心信息也能保留
                    prompt_parts = prompt.split("，")
                    core_keywords = []
                    detail_keywords = []
                    
                    # 识别核心关键词
                    core_patterns = ["科普主持人", "男性", "专业形象"]
                    for part in prompt_parts:
                        if any(pattern in part for pattern in core_patterns):
                            core_keywords.append(part)
                        else:
                            detail_keywords.append(part)
                    
                    # 重新排序：核心信息 + 细节
                    if core_keywords:
                        optimized_prompt = "，".join(core_keywords + detail_keywords)
                        return optimized_prompt
                    else:
                        return prompt
        return "科普主持人，男性，专业形象"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        item = self.images[idx]
        
        # 加载图片
        image = Image.open(item['path']).convert('RGB')
        if image.size != (self.size, self.size):
            image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # 转换为 tensor
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        image_tensor = transform(image).float()
        
        # Tokenize 提示词（Flux 使用 CLIP + T5 双编码器）
        prompt = item['prompt']
        
        # CLIP tokenizer 只能处理 77 tokens，需要截断
        # 先检查长度，如果太长则截断
        text_inputs_1 = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,  # CLIP tokenizer 最大长度
            truncation=True,  # 自动截断
            return_tensors="pt",
            return_length=False  # 不返回长度，避免警告
        )
        
        # T5 tokenizer 支持更长序列（512 tokens）
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=512,  # T5 支持更长序列
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': image_tensor,
            'input_ids_1': text_inputs_1.input_ids.squeeze(),
            'input_ids_2': text_inputs_2.input_ids.squeeze(),
            'prompt': prompt
        }


def train_flux_lora_improved(
    data_dir: str,
    output_dir: str,
    base_model_path: str,
    num_train_epochs: int = 20,  # 增加到 20 轮
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,  # 降低学习率，更保守
    lora_rank: int = 16,  # 降低 rank，更保守（数据少时）
    lora_alpha: int = 16,
    save_steps: int = 500,
    resolution: int = 512,  # 使用 512 分辨率，节省显存，加快训练
    use_bf16: bool = True,
):
    """
    改进的 Flux LoRA 训练（针对当前问题优化）
    
    优化点：
    - 增加训练轮数（20 轮）
    - 降低学习率（5e-5，更保守）
    - 降低 LoRA rank（16，数据少时更稳定）
    - 使用 512 分辨率（加快训练）
    """
    
    print("=" * 60)
    print("🚀 开始训练改进版 Flux LoRA")
    print("=" * 60)
    
    # 1. 初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="bf16" if use_bf16 else "fp16"
    )
    
    # 2. 加载基础模型（优化显存）
    print(f"\n📦 加载基础模型: {base_model_path}")
    pipe = FluxPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map="balanced"
    )
    
    # 启用梯度检查点以节省显存
    if hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()
        print("  ✅ 已启用梯度检查点（节省显存）")
    
    # 3. 配置 LoRA（更保守的参数）
    print(f"\n🔧 配置 LoRA (rank={lora_rank}, alpha={lora_alpha})")
    
    # Flux transformer 的注意力层（DiT 架构）
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
    
    # 启用梯度检查点（在应用 LoRA 后）
    if hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()
        print("  ✅ LoRA 模型已启用梯度检查点")
    
    # 5. 准备数据集
    print(f"\n📁 准备训练数据: {data_dir}")
    dataset = HostDataset(
        data_dir=data_dir,
        tokenizer=pipe.tokenizer,  # CLIP tokenizer
        tokenizer_2=pipe.tokenizer_2,  # T5 tokenizer
        size=resolution
    )
    
    if len(dataset) == 0:
        raise ValueError(f"未找到训练数据！请检查目录: {data_dir}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0  # 避免多进程问题
    )
    
    # 6. 设置优化器（使用 8bit AdamW 节省显存）
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            pipe.transformer.parameters(),
            lr=learning_rate
        )
        print("  ℹ 使用 8bit AdamW 优化器（节省显存）")
    except ImportError:
        optimizer = torch.optim.AdamW(
            pipe.transformer.parameters(),
            lr=learning_rate
        )
        print("  ℹ 使用标准 AdamW 优化器")
    
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
    print(f"   LoRA rank: {lora_rank}")
    print(f"   LoRA alpha: {lora_alpha}")
    print(f"   分辨率: {resolution}x{resolution}")
    print(f"   精度: {'bf16' if use_bf16 else 'fp16'}")
    print(f"   GPU: H20 (97GB 显存)")
    
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), desc="训练中")
    
    pipe.transformer.train()
    
    # 记录损失
    losses = []
    
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(pipe.transformer):
                device = next(pipe.transformer.parameters()).device
                dtype = torch.bfloat16 if use_bf16 else torch.float16
                
                # VAE 编码
                pixel_values = batch['pixel_values'].to(device, dtype=dtype)
                with torch.no_grad():
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor
                # 清理显存
                del pixel_values
                torch.cuda.empty_cache()
                
                # Flow Matching 噪声添加
                noise = torch.randn_like(latents, device=device, dtype=dtype)
                timesteps = torch.rand(
                    (latents.shape[0],),
                    device=device,
                    dtype=dtype
                )  # [0, 1]
                
                # Flow Matching: x_t = (1-t)*x_0 + t*x_1
                t = timesteps.view(-1, 1, 1, 1)
                noisy_latents = (1 - t) * latents + t * noise
                
                # 编码提示词（使用 pipe.encode_prompt 方法）
                prompts = batch['prompt']
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                        prompts,
                        num_images_per_prompt=1,
                        device=device
                    )
                    encoder_hidden_states = prompt_embeds
                # 清理显存
                torch.cuda.empty_cache()
                
                # 准备 latent image IDs（Flux 需要）
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    noisy_latents.shape[0],
                    noisy_latents.shape[2] // 2,
                    noisy_latents.shape[3] // 2,
                    device,
                    dtype
                )
                
                # 打包 latents（Flux 需要）
                packed_noisy_latents = FluxPipeline._pack_latents(
                    noisy_latents,
                    batch_size=noisy_latents.shape[0],
                    num_channels_latents=noisy_latents.shape[1],
                    height=noisy_latents.shape[2],
                    width=noisy_latents.shape[3],
                )
                
                # 处理 guidance（如果需要）
                if hasattr(pipe.transformer.config, 'guidance_embeds') and pipe.transformer.config.guidance_embeds:
                    guidance = torch.tensor([3.5], device=device).expand(noisy_latents.shape[0])
                else:
                    guidance = None
                
                # 预测（Flux transformer 调用）
                model_pred = pipe.transformer(
                    hidden_states=packed_noisy_latents,
                    timestep=timesteps / 1000.0,  # Flux 需要除以 1000
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                
                # 解包 latents
                vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=noisy_latents.shape[2] * vae_scale_factor,
                    width=noisy_latents.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                
                # 计算损失（Flow Matching：速度场）
                target_velocity = noise - latents
                loss = torch.nn.functional.mse_loss(model_pred.float(), target_velocity.float())
                
                # 记录损失值（在清理前）
                loss_value = loss.item()
                losses.append(loss_value)
                
                # 反向传播
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                # 清理显存
                del model_pred, target_velocity, loss
                torch.cuda.empty_cache()
                
                # 更新进度条
                if step % 10 == 0:
                    avg_loss = sum(losses[-10:]) / min(10, len(losses))
                    progress_bar.set_postfix({
                        "loss": f"{loss_value:.4f}",
                        "avg_loss": f"{avg_loss:.4f}",
                        "epoch": f"{epoch+1}/{num_train_epochs}"
                    })
            
            global_step += 1
            progress_bar.update(1)
            
            # 保存检查点
            if global_step % save_steps == 0:
                if accelerator.is_main_process:
                    checkpoint_dir = Path(output_dir) / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    pipe.transformer.save_pretrained(str(checkpoint_dir))
                    avg_loss = sum(losses[-100:]) / min(100, len(losses))
                    print(f"\n💾 已保存检查点: {checkpoint_dir} (平均损失: {avg_loss:.4f})")
    
    # 9. 保存最终模型
    print(f"\n💾 保存最终模型到: {output_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if accelerator.is_main_process:
        pipe.transformer.save_pretrained(str(output_path))
        
        # 保存为 safetensors
        try:
            from safetensors.torch import save_file
            state_dict = {}
            for name, param in pipe.transformer.named_parameters():
                if 'lora' in name.lower():
                    state_dict[name] = param.data.cpu()
            if state_dict:
                safetensors_path = output_path / "pytorch_lora_weights.safetensors"
                save_file(state_dict, str(safetensors_path))
                print(f"✅ 已保存 safetensors: {safetensors_path}")
        except Exception as e:
            print(f"⚠️  保存 safetensors 时出错: {e}")
        
        # 打印训练统计
        final_avg_loss = sum(losses[-100:]) / min(100, len(losses))
        initial_avg_loss = sum(losses[:100]) / min(100, len(losses))
        print(f"\n📊 训练统计:")
        print(f"   初始平均损失: {initial_avg_loss:.4f}")
        print(f"   最终平均损失: {final_avg_loss:.4f}")
        print(f"   损失下降: {initial_avg_loss - final_avg_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)
    print(f"📁 模型保存在: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="改进的 Flux LoRA 训练脚本")
    parser.add_argument("--data-dir", type=str, default="train_data/host_person")
    parser.add_argument("--output-dir", type=str, default="models/lora/host_person_v2")
    parser.add_argument("--base-model", type=str, default="models/flux1-dev")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数（增加到 20）")
    parser.add_argument("--batch-size", type=int, default=1, help="批次大小")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="学习率（降低到 5e-5）")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank（降低到 16，更保守）")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--save-steps", type=int, default=500, help="保存检查点的步数")
    parser.add_argument("--resolution", type=int, default=512, help="训练分辨率（512 更快）")
    parser.add_argument("--use-bf16", action="store_true", default=True, help="使用 bf16")
    
    args = parser.parse_args()
    
    train_flux_lora_improved(
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

