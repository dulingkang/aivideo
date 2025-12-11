# Flux LoRA 训练问题说明

## ⚠️ 当前问题

Flux transformer 的输入形状要求与标准 UNet 不同，导致训练时出现形状不匹配错误。

## 🔧 已尝试的修复

1. ✅ 检测 Flow Matching 调度器
2. ✅ 使用 Flow Matching 的噪声添加方式
3. ✅ 修复损失计算（速度场 vs 噪声）
4. ✅ 添加输入形状重塑尝试

## 💡 推荐解决方案

### **方案 1：使用 Kohya_ss（强烈推荐）**

Kohya_ss 是专门为 Flux 等模型设计的训练工具，已经处理了所有形状和架构问题。

#### 安装步骤：

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

#### 训练配置：

创建 `train_config.yaml`：

```yaml
pretrained_model_name_or_path: /vepfs-dev/shawn/vid/fanren/gen_video/models/flux1-dev
output_dir: /vepfs-dev/shawn/vid/fanren/gen_video/models/lora/host_person
train_data_dir: /vepfs-dev/shawn/vid/fanren/gen_video/train_data/host_person

# 训练参数
resolution: 1024,1024
train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1e-4
lr_scheduler: cosine
lr_warmup_steps: 100
max_train_steps: 1000
save_every_n_steps: 200

# LoRA 参数
network_module: lycoris.kohya
network_dim: 32
network_alpha: 16
network_dropout: 0.1

# 优化器
optimizer_type: adamw8bit
mixed_precision: fp16
```

#### 启动训练：

```bash
cd kohya_ss
python train_network.py --config ../train_config.yaml
```

---

### **方案 2：使用 diffusers 官方训练脚本**

diffusers 提供了官方的 Flux 训练示例，可以参考：

```bash
# 查看 diffusers 官方示例
# https://github.com/huggingface/diffusers/tree/main/examples/text_to_image
```

---

### **方案 3：继续修复当前脚本**

如果坚持使用当前脚本，需要：

1. **检查 Flux transformer 的实际输入要求**
   - 查看 `pipe.transformer` 的文档
   - 检查输入/输出形状

2. **使用正确的输入格式**
   - Flux transformer 可能需要特定的输入格式
   - 可能需要使用 `pipe.prepare_latents()` 等方法

3. **参考官方训练代码**
   - 查看 diffusers 的 Flux 训练示例
   - 参考 Kohya_ss 的实现

---

## 📝 当前状态

- ✅ 数据准备完成（20 张图片）
- ✅ 脚本框架完成
- ⚠️ Flux transformer 输入形状问题（需要进一步调试）

## 🎯 建议

**强烈建议使用 Kohya_ss**，因为：
- ✅ 专门为 Flux 设计
- ✅ 已经处理了所有架构问题
- ✅ 更稳定、更成熟
- ✅ 社区支持更好

如果使用 Kohya_ss，训练应该可以顺利进行。

