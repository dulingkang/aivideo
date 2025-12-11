#!/bin/bash
# 训练风格LoRA脚本（大batch版，适合H20 95GB显存）

# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 设置工作目录
cd /vepfs-dev/shawn/vid/fanren

# 创建输出目录
mkdir -p gen_video/models/lora/anime_style

echo "开始训练风格LoRA（大batch版，适合H20 95GB显存）..."
echo "使用本地模型: gen_video/models/sdxl-base"
echo "参考图像目录: gen_video/reference_materials/style_frames"
echo "输出目录: gen_video/models/lora/anime_style"
echo ""

# 大batch配置（充分利用95GB显存）：
# - batch_size=8, gradient_accumulation=1 (等效batch=8)
# - 1024分辨率（高质量）
# - 800步（足够学习风格）
# - 启用xformers和gradient_checkpointing

proxychains4 -q python scripts/train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path /vepfs-dev/shawn/vid/fanren/gen_video/models/sdxl-base \
    --instance_data_dir gen_video/reference_materials/style_frames \
    --output_dir gen_video/models/lora/anime_style \
    --instance_prompt "anime style, xianxia animation style, 凡人修仙传 style" \
    --resolution 1024 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --max_train_steps 800 \
    --checkpointing_steps 200 \
    --mixed_precision fp16 \
    --report_to tensorboard \
    --seed 42 \
    --validation_prompt "anime style, xianxia world, immortal cultivator, cinematic lighting" \
    --num_validation_images 2 \
    --validation_epochs 2 \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --train_text_encoder \
    --text_encoder_lr 5e-5 \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention

echo ""
echo "训练完成！模型保存在: gen_video/models/lora/anime_style"

