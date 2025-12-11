# 使用 diffusers 官方 Flux LoRA 训练脚本

## ✅ 推荐方案：使用官方训练脚本

diffusers 官方提供了专门的 Flux LoRA 训练脚本，完全兼容 Flux DiT 架构。

---

## 📦 准备

### 1. 激活虚拟环境

```bash
source /vepfs-dev/shawn/venv/py312/bin/activate
```

### 2. 检查 diffusers 版本

```bash
python -c "import diffusers; print(diffusers.__version__)"
```

**需要 >= 0.36.0.dev0**（如果版本不够，需要升级）

### 3. 安装依赖（如果需要）

```bash
cd diffusers/examples/dreambooth
pip install -r requirements_flux.txt
```

---

## 🚀 使用官方脚本训练

### 方法 1：直接使用官方脚本（推荐）

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
source /vepfs-dev/shawn/venv/py312/bin/activate

cd diffusers/examples/dreambooth

accelerate launch train_dreambooth_lora_flux.py \
    --pretrained_model_name_or_path=/vepfs-dev/shawn/vid/fanren/gen_video/models/flux1-dev \
    --instance_data_dir=/vepfs-dev/shawn/vid/fanren/gen_video/train_data/host_person \
    --output_dir=/vepfs-dev/shawn/vid/fanren/gen_video/models/lora/host_person \
    --instance_prompt="科普主持人，专业形象" \
    --resolution=1024 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-4 \
    --max_train_steps=1000 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=100 \
    --use_bf16 \
    --save_steps=200
```

### 方法 2：使用我创建的适配脚本

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
source /vepfs-dev/shawn/venv/py312/bin/activate

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

---

## ⚙️ H20 GPU 优化配置

### 推荐配置（H20，97GB 显存）

```bash
train_batch_size=2          # 可以更大
gradient_accumulation_steps=2
learning_rate=1e-4
use_bf16=true               # H20 支持 bf16，性能更好
use_8bit_adam=true          # 节省显存（如果安装了 bitsandbytes）
resolution=1024
lora_rank=32
lora_alpha=16
max_train_steps=1000
```

### 如果显存不足

```bash
train_batch_size=1
gradient_accumulation_steps=4
use_8bit_adam=true
```

---

## 📝 训练数据格式

官方脚本支持两种数据格式：

### 格式 1：ImageFolder（推荐）

```
train_data/host_person/
  image1.png
  image2.png
  ...
```

配合 `--instance_prompt` 使用统一提示词。

### 格式 2：带提示词的文件名（你的格式）

```
train_data/host_person/
  _repeat_10_科普主持人，男性，专业形象，微笑，正式着装，正面，演播室背景.png
  ...
```

需要修改脚本以从文件名提取提示词（或使用我创建的适配脚本）。

---

## 🎯 参数说明

| 参数 | 说明 | H20 推荐值 |
|------|------|-----------|
| `train_batch_size` | 批次大小 | 2 |
| `gradient_accumulation_steps` | 梯度累积 | 2 |
| `learning_rate` | 学习率 | 1e-4 |
| `lora_rank` | LoRA 维度 | 32 |
| `lora_alpha` | LoRA alpha | 16 |
| `max_train_steps` | 最大步数 | 1000 |
| `use_bf16` | 使用 bf16 | true |
| `resolution` | 分辨率 | 1024 |

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

## 🔗 参考资源

- [官方 Flux LoRA 训练脚本](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth#flux)
- [Flux README](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md)

