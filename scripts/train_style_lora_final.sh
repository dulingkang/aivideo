#!/bin/bash
# 训练风格LoRA脚本（最终版，已修复所有参数）

# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 设置工作目录
cd /vepfs-dev/shawn/vid/fanren

# 创建输出目录
mkdir -p gen_video/models/lora/anime_style

echo "开始训练风格LoRA..."
echo "使用本地模型: gen_video/models/sdxl-base"
echo "参考图像目录: gen_video/reference_materials/style_frames"
echo "输出目录: gen_video/models/lora/anime_style"
echo ""

# 使用proxychains4代理运行训练（使用本地模型路径）
proxychains4 -q python scripts/train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path /vepfs-dev/shawn/vid/fanren/gen_video/models/sdxl-base \
    --instance_data_dir gen_video/reference_materials/style_frames \
    --output_dir gen_video/models/lora/anime_style \
    --instance_prompt "anime style, xianxia animation style, 凡人修仙传 style" \
    --resolution 1024 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --max_train_steps 1000 \
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
    --text_encoder_lr 5e-5

echo ""
echo "训练完成！模型保存在: gen_video/models/lora/anime_style"

