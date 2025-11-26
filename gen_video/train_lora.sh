#!/bin/bash
# LoRA 训练脚本

set -e

# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 设置环境变量
export HF_HOME=/vepfs-dev/shawn/hf-cache
export CUDA_VISIBLE_DEVICES=0

# 训练配置
PRETRAINED_MODEL_PATH="/vepfs-dev/shawn/vid/fanren/gen_video/models/sdxl-base"
INSTANCE_DATA_DIR="/vepfs-dev/shawn/vid/fanren/gen_video/datasets/hanli_lora/images"
OUTPUT_DIR="/vepfs-dev/shawn/vid/fanren/gen_video/models/lora/hanli"
INSTANCE_PROMPT="hanli, Han Li, chinese cultivator, long black hair, dark green traditional robe, rounded face, gentle features, determined gaze"
VALIDATION_PROMPT="hanli, Han Li standing in desert, long black hair, dark green robe, cinematic lighting, chinese fantasy style"
CLASS_PROMPT="chinese cultivator, traditional chinese character, fantasy character"

# 训练参数
RESOLUTION=1024
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-4
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0
MAX_TRAIN_STEPS=1000
CHECKPOINTING_STEPS=200
RANK=128
MIXED_PRECISION="fp16"
VALIDATION_EPOCHS=10
NUM_VALIDATION_IMAGES=4

# 使用 proxychains4 运行训练（如果需要代理）
# 如果需要代理，取消下面的注释并使用 proxychains4
# PROXY_CMD="proxychains4"

# 检查是否安装了 xformers
XFORMERS_FLAG=""
if python -c "import xformers" 2>/dev/null; then
    XFORMERS_FLAG="--enable_xformers_memory_efficient_attention"
    echo "使用 xformers 内存优化"
else
    echo "未安装 xformers，跳过内存优化选项"
fi

# 训练脚本路径
TRAIN_SCRIPT="/vepfs-dev/shawn/vid/fanren/scripts/train_dreambooth_lora_sdxl.py"

# 切换到脚本所在目录
cd "$(dirname "$0")"

${PROXY_CMD} accelerate launch \
    --mixed_precision=${MIXED_PRECISION} \
    --main_process_port=29512 \
    "${TRAIN_SCRIPT}" \
    --pretrained_model_name_or_path="${PRETRAINED_MODEL_PATH}" \
    --instance_data_dir="${INSTANCE_DATA_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --instance_prompt="${INSTANCE_PROMPT}" \
    --validation_prompt="${VALIDATION_PROMPT}" \
    --resolution=${RESOLUTION} \
    --train_batch_size=${TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate=${LEARNING_RATE} \
    --lr_scheduler=${LR_SCHEDULER} \
    --lr_warmup_steps=${LR_WARMUP_STEPS} \
    --max_train_steps=${MAX_TRAIN_STEPS} \
    --checkpointing_steps=${CHECKPOINTING_STEPS} \
    --rank=${RANK} \
    --mixed_precision=${MIXED_PRECISION} \
    --validation_epochs=${VALIDATION_EPOCHS} \
    --num_validation_images=${NUM_VALIDATION_IMAGES} \
    --seed=42 \
    --report_to="tensorboard" \
    --logging_dir="${OUTPUT_DIR}/logs" \
    ${XFORMERS_FLAG} \
    --gradient_checkpointing

echo "训练完成！LoRA 模型保存在: ${OUTPUT_DIR}"

