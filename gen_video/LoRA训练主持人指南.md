# LoRA 训练科普主持人指南

## 📋 是否需要训练主持人 LoRA？

### ✅ **建议训练 LoRA 的情况：**

1. **需要固定主持人形象**
   - 每次生成都是同一个人
   - 保持人脸特征一致性
   - 适合批量生成科普视频

2. **需要控制更多细节**
   - 服装、发型、表情等
   - 比 InstantID 更灵活
   - 可以训练多个主持人（不同 LoRA）

3. **需要批量生成**
   - 生成速度快（LoRA 权重小）
   - 显存占用低
   - 适合流水线生产

### ❌ **可以不训练 LoRA 的情况：**

1. **使用 InstantID**
   - 只需要一张参考图
   - 适合快速测试
   - 但灵活性较低

2. **临时生成**
   - 不需要固定形象
   - 每次可以不同

---

## 🚀 训练方案

### **方案 1：使用 Kohya_ss（推荐）**

Kohya_ss 是最流行的 LoRA 训练工具，支持 Flux、SDXL、SD1.5 等。

#### 安装步骤：

```bash
# 1. 克隆仓库
cd /vepfs-dev/shawn/vid/fanren/gen_video
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. 安装 xformers（可选，加速训练）
pip install xformers
```

#### 准备训练数据：

```
train_data/
  host_person/
    _repeat_10_科普主持人，专业形象，微笑.jpg
    _repeat_10_科普主持人，正式着装，正面.jpg
    _repeat_10_科普主持人，温和表情，半身.jpg
    _repeat_10_科普主持人，商务正装，全身.jpg
    ...
```

**命名规则：**
- `_repeat_N_` 表示重复 N 次（建议 10-20）
- 文件名包含提示词，用于自动标注

#### 训练配置（Flux.1）：

```yaml
# train_config.yaml
pretrained_model_name_or_path: /vepfs-dev/shawn/vid/fanren/gen_video/models/flux1-dev
output_dir: /vepfs-dev/shawn/vid/fanren/gen_video/models/lora/host_person
train_data_dir: /vepfs-dev/shawn/vid/fanren/gen_video/train_data/host_person

# 训练参数
resolution: 1024,1024
train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1e-4
lr_scheduler: cosine
lr_warmup_steps: 100
max_train_steps: 1000
save_every_n_steps: 200

# LoRA 参数
network_module: lycoris.kohya
network_dim: 32  # LoRA 维度（16/32/64，越大越强但越容易过拟合）
network_alpha: 16  # 通常设为 network_dim 的一半
network_dropout: 0.1

# 优化器
optimizer_type: adamw8bit
mixed_precision: fp16
```

#### 启动训练：

```bash
cd kohya_ss
python train_network.py --config train_config.yaml
```

---

### **方案 2：使用 diffusers + PEFT（代码集成）**

适合直接在项目中集成训练功能。

#### 训练脚本示例：

```python
# train_host_lora.py
from diffusers import DiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

# 1. 加载基础模型
pipe = DiffusionPipeline.from_pretrained(
    "/vepfs-dev/shawn/vid/fanren/gen_video/models/flux1-dev",
    torch_dtype=torch.float16
)

# 2. 配置 LoRA
lora_config = LoraConfig(
    r=32,  # LoRA 维度
    lora_alpha=16,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Flux 的注意力层
    lora_dropout=0.1,
)

# 3. 应用 LoRA
pipe.unet = get_peft_model(pipe.unet, lora_config)

# 4. 准备数据集
class HostDataset(Dataset):
    def __init__(self, data_dir):
        self.images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))]
        self.prompts = ["科普主持人，专业形象"] * len(self.images)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        prompt = self.prompts[idx]
        return {"image": image, "prompt": prompt}

# 5. 训练循环（简化版）
dataset = HostDataset("/vepfs-dev/shawn/vid/fanren/gen_video/train_data/host_person")
# ... 训练代码 ...

# 6. 保存 LoRA
pipe.unet.save_pretrained("/vepfs-dev/shawn/vid/fanren/gen_video/models/lora/host_person")
```

---

## 📦 集成到 ModelManager

训练完成后，需要更新 `FluxPipeline` 以支持 LoRA：

### 更新 `pipelines/flux_pipeline.py`：

```python
def generate(
    self,
    prompt: str,
    negative_prompt: Optional[str] = None,
    lora_path: Optional[str] = None,  # 新增：LoRA 路径
    lora_alpha: float = 1.0,  # 新增：LoRA 权重
    **kwargs
) -> Image.Image:
    """生成图像"""
    if not self.loaded:
        self.load()
    
    # 加载 LoRA（如果提供）
    if lora_path and Path(lora_path).exists():
        self.pipe.load_lora_weights(lora_path, adapter_name="host_person")
        self.pipe.set_adapters(["host_person"], adapter_weights=[lora_alpha])
        print(f"  ℹ 已加载 LoRA: {lora_path} (alpha={lora_alpha})")
    
    # ... 生成代码 ...
```

### 更新 `model_manager.py`：

```python
# 在 ModelManager 中添加 LoRA 配置
self.lora_configs = {
    "host_face": {
        "lora_path": "/vepfs-dev/shawn/vid/fanren/gen_video/models/lora/host_person/pytorch_lora_weights.safetensors",
        "lora_alpha": 0.7
    }
}

# 在 generate 方法中使用
if task == "host_face" and "host_face" in self.lora_configs:
    lora_cfg = self.lora_configs["host_face"]
    image = pipeline.generate(
        prompt=optimized_prompt,
        lora_path=lora_cfg["lora_path"],
        lora_alpha=lora_cfg["lora_alpha"],
        **kwargs
    )
```

---

## 🎯 训练数据准备建议

### 1. **图片要求：**
- 分辨率：1024x1024 或更高
- 数量：20-50 张（越多越好）
- 质量：清晰、正面、光线均匀
- 多样性：不同角度、表情、服装

### 2. **标注要求：**
- 每张图片对应一个提示词
- 包含关键特征：性别、职业、风格
- 示例：`科普主持人，专业形象，微笑，正式着装`

### 3. **数据增强（可选）：**
- 水平翻转
- 轻微旋转
- 亮度调整

---

## ⚡ 快速开始（推荐流程）

1. **准备 20-50 张主持人图片**
   - 放在 `train_data/host_person/` 目录

2. **使用 Kohya_ss 训练**
   - 参考上面的配置
   - 训练 500-1000 步

3. **测试 LoRA**
   - 加载训练好的 LoRA
   - 生成测试图像

4. **集成到 ModelManager**
   - 更新 `FluxPipeline` 支持 LoRA
   - 配置 `model_manager.py`

---

## 📝 注意事项

1. **过拟合问题**
   - LoRA 维度不要太大（建议 16-32）
   - 训练步数不要太多（500-1000 步）

2. **显存占用**
   - Flux.1 训练需要 24GB+ 显存
   - 可以使用 `gradient_checkpointing` 降低显存

3. **训练时间**
   - 20 张图片，1000 步，约 1-2 小时（A100）

4. **效果对比**
   - LoRA：固定形象，灵活控制
   - InstantID：快速测试，一张图即可

---

## 🔗 参考资源

- [Kohya_ss 官方文档](https://github.com/bmaltais/kohya_ss)
- [Flux LoRA 训练指南](https://huggingface.co/docs/diffusers/training/lora)
- [PEFT 文档](https://huggingface.co/docs/peft)

