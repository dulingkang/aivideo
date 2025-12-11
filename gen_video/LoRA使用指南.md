# LoRA 使用指南

## ✅ 训练完成

LoRA 模型已训练完成，保存在：
```
models/lora/host_person/
  ├── pytorch_lora_weights.safetensors  (143 MB)
  ├── adapter_model.safetensors         (143 MB)
  ├── adapter_config.json
  └── README.md
```

---

## 🚀 自动应用 LoRA

### 方法 1：使用 ModelManager（推荐）

`ModelManager` 已配置为自动加载 LoRA，当任务类型为 `host_face` 或 `character_face` 时会自动应用：

```python
from model_manager import ModelManager

manager = ModelManager()

# 生成科普主持人（自动应用 LoRA）
image = manager.generate(
    task="host_face",
    prompt="科普主持人，专业形象，微笑，正式着装",
    width=1024,
    height=1024
)
```

### 方法 2：使用 API（前端调用）

通过 API 调用时，使用 `use_model_manager=true` 和 `task=host_face`：

```bash
curl -X POST "http://localhost:8000/api/generate-image" \
  -F "prompt=科普主持人，专业形象，微笑" \
  -F "use_model_manager=true" \
  -F "task=host_face" \
  -F "width=1024" \
  -F "height=1024"
```

### 方法 3：直接使用 FluxPipeline

```python
from pipelines.flux_pipeline import FluxPipeline

pipe = FluxPipeline(
    model_path="models/flux1-dev",
    model_type="flux1"
)

image = pipe.generate(
    prompt="科普主持人，专业形象，微笑",
    lora_path="models/lora/host_person/pytorch_lora_weights.safetensors",
    lora_alpha=0.7,  # LoRA 权重（0.0-1.0）
    width=1024,
    height=1024
)
```

---

## ⚙️ 调整 LoRA 权重

LoRA 权重（`lora_alpha`）控制 LoRA 的影响强度：

- **0.0-0.5**：轻微影响，保持更多原始模型特征
- **0.6-0.8**：平衡（推荐，当前设置为 0.7）
- **0.9-1.0**：强烈影响，更接近训练数据

### 修改权重

#### 方法 1：修改 model_manager.py

```python
self.lora_configs = {
    "host_face": {
        "lora_path": str(lora_root / "pytorch_lora_weights.safetensors"),
        "lora_alpha": 0.8  # 修改这里
    }
}
```

#### 方法 2：API 调用时指定（如果支持）

```python
# 在 API 调用中传递 lora_alpha 参数（需要 API 支持）
```

---

## 📝 测试 LoRA 效果

### 测试脚本

```python
from model_manager import ModelManager
from PIL import Image

manager = ModelManager()

# 测试 1：不使用 LoRA（对比）
image_without = manager.generate(
    task="science_background",  # 不使用 host_face，不加载 LoRA
    prompt="科普主持人，专业形象",
    width=1024,
    height=1024
)

# 测试 2：使用 LoRA
image_with = manager.generate(
    task="host_face",  # 使用 host_face，自动加载 LoRA
    prompt="科普主持人，专业形象",
    width=1024,
    height=1024
)

# 保存对比
image_without.save("test_without_lora.png")
image_with.save("test_with_lora.png")
```

---

## 🔍 验证 LoRA 是否加载

生成时会看到日志：
```
  ℹ 已加载 LoRA: pytorch_lora_weights.safetensors (alpha=0.7)
```

如果没有看到这个日志，说明 LoRA 未加载，检查：
1. LoRA 文件路径是否正确
2. 任务类型是否为 `host_face` 或 `character_face`
3. `model_manager.py` 中的 `lora_path` 是否已设置

---

## 🎯 使用场景

### 1. 科普主持人固定人设

```python
# 自动应用 LoRA
image = manager.generate(
    task="host_face",
    prompt="科普主持人，专业形象，微笑，正式着装，演播室背景"
)
```

### 2. 不同场景的主持人

```python
# 场景 1：演播室
image1 = manager.generate(
    task="host_face",
    prompt="科普主持人，专业形象，演播室背景"
)

# 场景 2：户外
image2 = manager.generate(
    task="host_face",
    prompt="科普主持人，专业形象，户外场景，自然光"
)
```

### 3. 不同表情/姿势

```python
# 微笑
image1 = manager.generate(
    task="host_face",
    prompt="科普主持人，专业形象，微笑，正面"
)

# 严肃
image2 = manager.generate(
    task="host_face",
    prompt="科普主持人，专业形象，严肃表情，正面"
)
```

---

## ⚠️ 注意事项

1. **LoRA 仅适用于 Flux.1**：当前 LoRA 是为 Flux.1 训练的，不能用于其他模型
2. **任务类型必须匹配**：只有 `host_face` 和 `character_face` 会自动加载 LoRA
3. **权重调整**：如果效果不理想，可以调整 `lora_alpha`（0.5-1.0 之间尝试）
4. **显存占用**：加载 LoRA 会增加少量显存占用（约 200-300 MB）

---

## 🔧 故障排除

### LoRA 未加载

1. 检查文件是否存在：
   ```bash
   ls -lh models/lora/host_person/pytorch_lora_weights.safetensors
   ```

2. 检查 model_manager.py 配置：
   ```python
   print(manager.lora_configs["host_face"]["lora_path"])
   ```

3. 检查任务类型：
   ```python
   # 确保使用 host_face 或 character_face
   image = manager.generate(task="host_face", ...)
   ```

### LoRA 效果不明显

1. 增加 `lora_alpha`（0.7 → 0.9）
2. 在提示词中明确提到"科普主持人"
3. 检查训练数据质量

### LoRA 效果过强

1. 降低 `lora_alpha`（0.7 → 0.5）
2. 在提示词中添加更多场景描述，平衡 LoRA 影响

---

## 📚 相关文件

- `model_manager.py` - ModelManager 配置
- `pipelines/flux_pipeline.py` - FluxPipeline LoRA 加载逻辑
- `models/lora/host_person/` - LoRA 模型文件

