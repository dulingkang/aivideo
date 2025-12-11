#!/bin/bash
# 训练风格LoRA脚本（使用proxychains4代理）

# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 设置工作目录
cd /vepfs-dev/shawn/vid/fanren

# 训练参数（使用本地模型路径，避免重复下载）
PRETRAINED_MODEL="/vepfs-dev/shawn/vid/fanren/gen_video/models/sdxl-base"
INSTANCE_DATA_DIR="gen_video/reference_materials/style_frames"
OUTPUT_DIR="gen_video/models/lora/anime_style"
INSTANCE_PROMPT="anime style, xianxia animation style, 凡人修仙传 style"
RESOLUTION=1024
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-4
MAX_TRAIN_STEPS=1000
CHECKPOINTING_STEPS=200

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 使用proxychains4代理运行训练（直接使用原始脚本）
proxychains4 -q python scripts/train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --instance_data_dir "$INSTANCE_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --instance_prompt "$INSTANCE_PROMPT" \
    --resolution $RESOLUTION \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --max_train_steps $MAX_TRAIN_STEPS \
    --checkpointing_steps $CHECKPOINTING_STEPS \
    --mixed_precision fp16 \
    --report_to tensorboard \
    --seed 42 \
    --validation_prompt "anime style, xianxia world, immortal cultivator, cinematic lighting" \
    --num_validation_images 4 \
    --validation_epochs 1 \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --train_text_encoder \
    --text_encoder_lr 5e-5

echo "训练完成！模型保存在: $OUTPUT_DIR"

