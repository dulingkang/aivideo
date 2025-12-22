#!/bin/bash
# Flux LoRA 训练脚本 - 韩立角色
# 使用 PEFT 直接训练，不依赖最新 diffusers

# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 设置代理
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# 工作目录
cd /vepfs-dev/shawn/vid/fanren/gen_video

echo "=============================================="
echo "从 GitHub 源码安装最新 diffusers..."
echo "=============================================="

# 从 GitHub 安装最新 diffusers（绕过 pypi SSL 问题）
pip install git+https://github.com/huggingface/diffusers.git --quiet 2>/dev/null || {
    echo "GitHub 安装失败，尝试清华镜像..."
    pip install diffusers -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --quiet
}

# 确认版本
echo "当前 diffusers 版本:"
pip show diffusers | grep Version

# 训练参数
MODEL_PATH="models/flux1-dev"
OUTPUT_DIR="models/lora/hanli_flux"
INSTANCE_PROMPT="hanli, young male cultivator with sharp composed eyes, wearing green daoist robe"

echo "=============================================="
echo "训练 Flux LoRA - 韩立角色"
echo "=============================================="
echo "模型: $MODEL_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "触发词: $INSTANCE_PROMPT"
echo "=============================================="

# 准备训练数据
TRAIN_DATA_DIR="datasets/hanli_flux_train"
rm -rf $TRAIN_DATA_DIR
mkdir -p $TRAIN_DATA_DIR

# 复制所有素材（使用绝对路径）
echo "复制训练素材..."
cp /vepfs-dev/shawn/vid/fanren/gen_video/character_profiles/hanli/front/*.jpg $TRAIN_DATA_DIR/ 2>/dev/null
cp /vepfs-dev/shawn/vid/fanren/gen_video/character_profiles/hanli/side/*.jpg $TRAIN_DATA_DIR/ 2>/dev/null
cp /vepfs-dev/shawn/vid/fanren/gen_video/character_profiles/hanli/three_quarter/*.jpg $TRAIN_DATA_DIR/ 2>/dev/null

# 列出复制的文件
echo "已复制的文件:"
ls -la $TRAIN_DATA_DIR/

TRAIN_COUNT=$(ls $TRAIN_DATA_DIR/*.jpg 2>/dev/null | wc -l)
echo "训练图片数量: $TRAIN_COUNT"

if [ "$TRAIN_COUNT" -lt 5 ]; then
    echo "警告: 训练图片数量较少"
fi

echo "=============================================="
echo "开始训练..."
echo "=============================================="

# 使用 accelerate 运行训练
accelerate launch --mixed_precision=bf16 \
    /vepfs-dev/shawn/vid/fanren/diffusers/examples/dreambooth/train_dreambooth_lora_flux.py \
    --pretrained_model_name_or_path="$MODEL_PATH" \
    --instance_data_dir="$TRAIN_DATA_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --instance_prompt="$INSTANCE_PROMPT" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --rank=16 \
    --seed=42 \
    --checkpointing_steps=500

echo "=============================================="
echo "训练完成！"
echo "LoRA 保存位置: $OUTPUT_DIR"
echo "=============================================="
