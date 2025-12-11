# LoRA 问题诊断指南

## 🔍 当前问题

生成的图像：
- ❌ 风格不对
- ❌ 人脸不对
- ❌ 性别不对

## 📋 诊断步骤

### 1. 检查 LoRA 是否正确加载

从日志看，LoRA 已加载：
```
✅ 已加载 LoRA: pytorch_lora_weights.safetensors (alpha=1.0)
```

但有一个警告：
```
No LoRA keys associated to CLIPTextModel found with prefix='text_encoder'
```

这个警告可以忽略（CLIPTextModel 通常不需要 LoRA）。

### 2. 验证 LoRA 是否真正生效

**测试 1：不使用 LoRA 生成**

```bash
curl -X POST "http://localhost:8000/api/v1/images/generate" \
  -H "X-API-Key: test-key-123" \
  -F "prompt=科普主持人，专业形象，微笑" \
  -F "use_model_manager=false" \
  -F "model_engine=flux-instantid" \
  -F "width=1024" \
  -F "height=1024"
```

**测试 2：使用 LoRA 生成**

```bash
curl -X POST "http://localhost:8000/api/v1/images/generate" \
  -H "X-API-Key: test-key-123" \
  -F "prompt=科普主持人，专业形象，微笑" \
  -F "use_model_manager=true" \
  -F "task=host_face" \
  -F "width=1024" \
  -F "height=1024"
```

**对比结果：**
- 如果两个结果**完全不同** → LoRA 可能生效了，但效果不对
- 如果两个结果**几乎相同** → LoRA 可能没有正确应用

### 3. 检查训练数据质量

训练数据：20 张图片

**可能的问题：**
- 数据量可能不够（建议 30-50 张）
- 数据多样性可能不够
- 数据质量可能不够好

**检查训练数据：**
```bash
ls -lh train_data/host_person/*.png
```

### 4. 检查训练参数

**可能的问题：**
- 训练步数可能不够（当前：1000 步）
- 学习率可能不合适
- LoRA rank/alpha 可能不合适

**检查训练配置：**
```bash
cat models/lora/host_person/adapter_config.json
```

### 5. 检查提示词

**当前提示词优化：**
- 添加了 `(male:1.3), (man:1.2)` 权重标记
- 添加了 `男性, 男性主持人, 男性形象, 男性特征`
- 负面提示词：`(female:1.5), (woman:1.5)`

**注意：** Flux 可能不支持权重标记 `(word:1.3)`，这可能导致提示词解析错误。

### 6. 测试不同的 LoRA alpha

尝试不同的 LoRA 权重：

```python
# 在 model_manager.py 中修改
"lora_alpha": 1.5  # 超过 1.0，增强效果
# 或
"lora_alpha": 0.5  # 降低，减少影响
```

## 🔧 可能的解决方案

### 方案 1：重新训练 LoRA（如果数据质量不够）

1. **增加训练数据**（30-50 张）
2. **增加训练步数**（2000-3000 步）
3. **调整学习率**（尝试 5e-5 或 2e-4）

### 方案 2：修复 LoRA 应用问题

如果 LoRA 没有正确应用，可能需要：
1. 检查键名转换是否正确
2. 验证 LoRA 是否在生成时激活
3. 尝试不同的加载方式

### 方案 3：调整提示词（如果权重标记不支持）

移除权重标记，使用纯文本：

```python
optimized_prompt = f"male, man, 男性, {prompt}, 男性主持人, 男性形象"
```

### 方案 4：使用参考图像（InstantID）

如果 LoRA 效果不好，可以尝试使用 InstantID + 参考图像的方式。

## 📊 调试信息

已添加的调试输出：
- LoRA 激活状态
- 生成参数（prompt 长度、LoRA 权重、steps、guidance）
- 提示词优化结果

## 🎯 下一步

1. **先测试不使用 LoRA**，看看是否能生成正确的性别
2. **对比使用和不使用 LoRA 的结果**
3. **根据对比结果决定下一步**：
   - 如果不用 LoRA 也错 → 提示词问题
   - 如果不用 LoRA 对，用 LoRA 错 → LoRA 应用问题
   - 如果两个都错 → 需要重新训练或调整训练参数

