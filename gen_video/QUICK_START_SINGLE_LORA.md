# 快速开始：单LoRA配置（5分钟）

## 🎯 目标

**用你现有的单LoRA，加上3个运行时补丁，立即稳定运行。**

---

## ✅ 你的当前状态

- ✅ 有一个**人脸LoRA**（已训练）
- ✅ 包含**多表情、多角度**
- ❓ 是否需要训练服饰LoRA？**不需要**
- ❓ 是否需要训练风格LoRA？**不需要**

---

## 📝 配置步骤（5分钟）

### 步骤1: 更新JSON配置

在你的场景JSON中添加：

```json
{
  "version": "v2.1-exec",
  "character": {
    "id": "hanli",
    "gender": "male",
    "lora_path": "你的LoRA路径.safetensors",
    "lora_weight": 0.85,
    "temperament_anchor": "calm and restrained temperament, sharp but composed eyes",
    "costume_lock_words": [
      "wearing his iconic mid-late-stage green daoist robe"
    ]
  }
}
```

### 步骤2: 使用简化版管理器

```python
from utils.character_anchor_v2_1_simple import get_character_anchor_manager_simple

# 初始化
manager = get_character_anchor_manager_simple()

# 注册角色
anchor = manager.register_character_simple(
    character_id="hanli",
    gender="male",
    lora_path="你的LoRA路径.safetensors",
    lora_weight=0.85,
    temperament_anchor="calm and restrained temperament, sharp but composed eyes",
    costume_lock_words=[
        "wearing his iconic mid-late-stage green daoist robe"
    ]
)
```

### 步骤3: 生成图像

```python
# 基础Prompt
base_prompt = "Han Li, standing in a desert"

# 获取增强Prompt（自动添加气质锚点和显式锁词）
enhanced_prompt = manager.get_enhanced_prompt("hanli", base_prompt)

# 生成图像
image = image_generator.generate_image(
    prompt=enhanced_prompt,
    character_lora="hanli",
    lora_weight=0.85
)
```

---

## 🔧 3个运行时补丁说明

### 补丁1: 气质锚点（必须）

**作用**: 防止气质轻微漂

**配置**:
```json
{
  "temperament_anchor": "calm and restrained temperament, sharp but composed eyes"
}
```

**效果**: ✅ 同一个人感觉强，气质稳定

---

### 补丁2: FaceDetailer（可选，但推荐）

**作用**: 解决远景脸泛化问题

**配置**:
```json
{
  "face_refine": {
    "enable": true,
    "trigger": "shot_scale >= medium",
    "denoise": 0.35,
    "steps": 12
  }
}
```

**效果**: ✅ 远景人脸清晰，保持一致性

---

### 补丁3: 显式锁词（必须）

**作用**: 解决服饰细节跑偏问题

**配置**:
```json
{
  "costume_lock_words": [
    "wearing his iconic mid-late-stage green daoist robe"
  ]
}
```

**效果**: ✅ 服饰稳定，即使只有一套也能稳定

---

## 📊 推荐参数

### LoRA权重

- **单LoRA**: 0.85-0.95
- **推荐**: 0.85-0.9（安全范围）
- **如果LoRA训练很好**: 可以到0.95

### 气质锚点

- **必须**: 是
- **位置**: Prompt最前面
- **示例**: "calm and restrained temperament, sharp but composed eyes"

### 显式锁词

- **必须**: 是
- **位置**: Prompt最后面
- **示例**: "wearing his iconic mid-late-stage green daoist robe"

---

## ❓ 常见问题

### Q1: 服饰需要训练LoRA吗？

**A**: ❌ **不需要**。一套就够了，通过显式锁词稳定。

### Q2: 风格需要训练LoRA吗？

**A**: ❌ **不需要**。通过Prompt控制即可。

### Q3: LoRA权重多少合适？

**A**: **0.85-0.95**。单LoRA时可以高一点。

### Q4: 气质锚点必须加吗？

**A**: ✅ **必须**。不加的话气质会轻微漂。

### Q5: 显式锁词必须加吗？

**A**: ✅ **必须**。不加的话服饰细节会跑偏。

---

## 🎯 总结

### 可以直接使用

✅ **能，完全能！** 只需要：

1. **单LoRA**（你现有的，权重0.85-0.95）
2. **气质锚点**（必须，Prompt中）
3. **显式锁词**（必须，服饰描述）
4. **FaceDetailer**（可选，但推荐）

### 不需要的

- ❌ 不需要训练服饰LoRA
- ❌ 不需要训练风格LoRA
- ❌ 不需要重新训练LoRA

### 立即开始

1. 更新JSON配置（添加3个补丁）
2. 使用简化版管理器
3. 测试生成效果
4. 根据效果微调

---

## 🔗 相关文档

- `SINGLE_LORA_USAGE_GUIDE.md` - 详细使用指南
- `LORA_STACK_IMPLEMENTATION.md` - LoRA Stack指南（多LoRA场景）
- `utils/character_anchor_v2_1_simple.py` - 单LoRA管理器

