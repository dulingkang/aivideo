# 开始训练科普主持人 LoRA

## ✅ 数据准备完成

- ✅ 训练图片：20 张
- ✅ 图片尺寸：1024x1024（已统一）
- ✅ 提示词：已从文件名提取

## 🚀 开始训练

### **方法 1：使用训练脚本（推荐）**

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video

# 激活虚拟环境（如果使用）
source /vepfs-dev/shawn/venv/py312/bin/activate

# 开始训练（使用默认参数）
python train_host_lora.py \
    --data-dir train_data/host_person \
    --output-dir models/lora/host_person \
    --base-model models/flux1-dev \
    --epochs 10 \
    --batch-size 1 \
    --gradient-accumulation 4 \
    --learning-rate 1e-4 \
    --lora-rank 32 \
    --lora-alpha 16 \
    --save-steps 200
```

### **参数说明：**

- `--data-dir`: 训练数据目录（默认: `train_data/host_person`）
- `--output-dir`: 输出目录（默认: `models/lora/host_person`）
- `--base-model`: 基础模型路径（默认: `models/flux1-dev`）
- `--epochs`: 训练轮数（默认: 10，约 1000 步）
- `--batch-size`: 批次大小（默认: 1，根据显存调整）
- `--gradient-accumulation`: 梯度累积步数（默认: 4）
- `--learning-rate`: 学习率（默认: 1e-4）
- `--lora-rank`: LoRA 维度（默认: 32）
- `--lora-alpha`: LoRA alpha（默认: 16）
- `--save-steps`: 每多少步保存一次（默认: 200）

### **根据显存调整：**

**24GB 显存（A100）：**
```bash
python train_host_lora.py \
    --batch-size 2 \
    --gradient-accumulation 2
```

**16GB 显存：**
```bash
python train_host_lora.py \
    --batch-size 1 \
    --gradient-accumulation 4
```

**12GB 显存：**
```bash
python train_host_lora.py \
    --batch-size 1 \
    --gradient-accumulation 8 \
    --lora-rank 16  # 降低 LoRA 维度
```

---

## ⏱️ 训练时间估算

- **20 张图片，10 轮，batch_size=1，gradient_accumulation=4**
- **总步数**: 约 1000 步（20 张 × 10 轮 / 4 梯度累积）
- **预计时间**: 
  - A100 (24GB): 约 1-2 小时
  - RTX 3090 (24GB): 约 2-3 小时
  - RTX 3080 (10GB): 约 3-4 小时

---

## 📊 训练过程监控

训练过程中会显示：
- 当前步数
- 损失值（loss）
- 每 200 步保存一次检查点

**检查点保存位置：**
```
models/lora/host_person/
  checkpoint-200/
  checkpoint-400/
  checkpoint-600/
  ...
  checkpoint-1000/  (最终)
```

---

## ✅ 训练完成后的步骤

1. **检查训练结果**
   ```bash
   ls -lh models/lora/host_person/
   ```

2. **测试 LoRA**
   - 在 `model_manager.py` 中配置 `lora_path`
   - 使用触发词"科普主持人"生成测试图像

3. **如果效果不理想**
   - 增加训练轮数（`--epochs 15`）
   - 调整 LoRA 权重（`lora_alpha`）
   - 检查训练数据质量

---

## ⚠️ 注意事项

1. **显存不足**
   - 降低 `batch_size` 到 1
   - 增加 `gradient_accumulation`
   - 降低 `lora_rank` 到 16

2. **训练中断**
   - 检查点会自动保存
   - 可以从检查点恢复训练（需要修改脚本）

3. **效果不理想**
   - 检查训练数据质量
   - 增加训练轮数
   - 调整学习率（尝试 5e-5 或 2e-4）

---

## 🔧 故障排除

### **错误：CUDA out of memory**
```bash
# 解决方案：降低 batch_size 或增加 gradient_accumulation
python train_host_lora.py --batch-size 1 --gradient-accumulation 8
```

### **错误：模型路径不存在**
```bash
# 检查模型路径
ls -la models/flux1-dev/
```

### **错误：找不到训练数据**
```bash
# 检查数据目录
ls -la train_data/host_person/
```

---

## 📝 快速启动命令

```bash
# 一键启动训练（使用默认参数）
cd /vepfs-dev/shawn/vid/fanren/gen_video
source /vepfs-dev/shawn/venv/py312/bin/activate
python train_host_lora.py
```

