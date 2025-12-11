# 开始训练 Flux LoRA

## ✅ 准备完成

- ✅ 虚拟环境：`/vepfs-dev/shawn/venv/py312/bin/activate`
- ✅ GPU：H20 (97GB 显存)
- ✅ 训练数据：`train_data/host_person/`
- ✅ 基础模型：`models/flux1-dev`
- ✅ 官方训练脚本：`diffusers/examples/dreambooth/train_dreambooth_lora_flux.py`

---

## 🚀 方案 1：使用官方脚本（推荐）

### 步骤 1：升级 diffusers

```bash
source /vepfs-dev/shawn/venv/py312/bin/activate
cd /vepfs-dev/shawn/vid/fanren/gen_video

# 从源码安装最新版本（推荐）
proxychains4 git clone https://github.com/huggingface/diffusers.git --depth=1
cd diffusers
pip install -e .
cd examples/dreambooth
pip install -r requirements_flux.txt
```

### 步骤 2：配置 accelerate

```bash
accelerate config default
```

### 步骤 3：开始训练

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video/diffusers/examples/dreambooth

accelerate launch train_dreambooth_lora_flux.py \
    --pretrained_model_name_or_path=/vepfs-dev/shawn/vid/fanren/gen_video/models/flux1-dev \
    --instance_data_dir=/vepfs-dev/shawn/vid/fanren/gen_video/train_data/host_person \
    --output_dir=/vepfs-dev/shawn/vid/fanren/gen_video/models/lora/host_person \
    --mixed_precision="bf16" \
    --instance_prompt="科普主持人，专业形象" \
    --resolution=1024 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=2 \
    --optimizer="prodigy" \
    --learning_rate=1.0 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --validation_prompt="科普主持人，专业形象，微笑，正式着装" \
    --validation_epochs=25 \
    --seed=0 \
    --rank=32 \
    --lora_alpha=16
```

**注意：** 官方脚本使用统一的 `instance_prompt`，不支持从文件名提取提示词。如果需要使用文件名中的提示词，请使用方案 2。

---

## 🚀 方案 2：使用适配脚本（兼容当前版本）

### 步骤 1：直接开始训练

```bash
source /vepfs-dev/shawn/venv/py312/bin/activate
cd /vepfs-dev/shawn/vid/fanren/gen_video

python train_flux_lora_final.py \
    --data-dir train_data/host_person \
    --output-dir models/lora/host_person \
    --base-model models/flux1-dev \
    --epochs 10 \
    --batch-size 2 \
    --gradient-accumulation 2 \
    --learning-rate 1e-4 \
    --lora-rank 32 \
    --lora-alpha 16 \
    --use-bf16
```

**优点：**
- ✅ 兼容当前 diffusers 版本（0.35.2）
- ✅ 支持从文件名提取提示词（你的数据格式）
- ✅ 已优化 H20 GPU 配置

---

## ⚙️ H20 GPU 推荐配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `train_batch_size` | 2 | H20 显存充足，可以更大 |
| `gradient_accumulation_steps` | 2 | 梯度累积 |
| `learning_rate` | 1e-4 (AdamW) 或 1.0 (Prodigy) | 根据优化器选择 |
| `lora_rank` | 32 | LoRA 维度 |
| `lora_alpha` | 16 | LoRA alpha |
| `resolution` | 1024 | 训练分辨率 |
| `use_bf16` | true | H20 支持 bf16 |
| `max_train_steps` | 1000 | 训练步数 |

---

## 📝 训练数据格式

你的数据格式（已支持）：
```
train_data/host_person/
  _repeat_10_科普主持人，男性，专业形象，微笑，正式着装，正面，演播室背景.png
  _repeat_10_科普主持人，男性，专业形象，温和，商务正装，正面，纯色背景.png
  ...
```

脚本会自动从文件名提取提示词。

---

## ✅ 训练完成后

LoRA 模型保存在：
```
models/lora/host_person/pytorch_lora_weights.safetensors
```

在 `model_manager.py` 中配置：
```python
self.lora_configs = {
    "host_face": {
        "lora_path": "models/lora/host_person/pytorch_lora_weights.safetensors",
        "lora_alpha": 0.7
    }
}
```

---

## 🔗 参考文档

- `使用官方Flux训练脚本.md` - 官方脚本详细说明
- `使用diffusers官方训练Flux.md` - 通用训练指南

