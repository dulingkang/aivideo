# 单LoRA使用指南（最小可行配置）

## 📋 总览

**问题**: 目前只有一个人脸的LoRA，能否直接使用？

**答案**: ✅ **能，完全能！** 只需要加上3个运行时补丁即可。

---

## ✅ 可以直接使用的情况

### 当前状态

- ✅ 有一个**人脸LoRA**（已训练）
- ✅ 包含**多表情、多角度**
- ✅ 服饰可以用**一套**（通过显式锁词）
- ⚠️ 风格LoRA**不是必须的**（可选）

### 最小可行配置

你只需要：

1. **单LoRA**（人脸LoRA，权重0.85-0.95）
2. **气质锚点**（运行时补丁1，Prompt中）
3. **显式锁词**（运行时补丁3，服饰描述）
4. **FaceDetailer**（运行时补丁2，可选，远景增强）

---

## 🔧 3个运行时补丁（必须）

### 补丁1: 气质锚点（必须）

**目的**: 防止气质轻微漂（年轻/冷酷）

**实现**:
```python
# 在Prompt最前面添加
temperament = "calm and restrained temperament, sharp but composed eyes"
prompt = f"{temperament}, {base_prompt}"
```

**配置**:
```json
{
  "character": {
    "id": "hanli",
    "temperament_anchor": "calm and restrained temperament, sharp but composed eyes"
  }
}
```

**效果**:
- ✅ 不影响表情
- ✅ 极大提升"同一个人"的感觉
- ✅ 抑制LoRA的"自由发挥"

---

### 补丁2: FaceDetailer（可选，但推荐）

**目的**: 解决远景/中景脸偶尔变"泛化脸"的问题

**触发条件**: 当Shot类型为medium或wide时

**配置**:
```json
{
  "character": {
    "face_refine": {
      "enable": true,
      "trigger": "shot_scale >= medium",
      "denoise": 0.35,
      "steps": 12
    }
  }
}
```

**关键参数**:
- `denoise`: **不要超过0.4**，否则会"换人"
- `steps`: 12步足够（快速）
- `lora`: 只使用脸部LoRA（权重0.9）

**效果**:
- ✅ 远景人脸清晰度提升
- ✅ 保持角色一致性
- ✅ 不影响构图和光影

---

### 补丁3: 显式锁词（必须）

**目的**: 解决服饰细节偶尔跑偏的问题

**实现**:
```python
# 在Prompt中显式添加服饰描述
costume_lock = "wearing his iconic mid-late-stage green daoist robe"
prompt = f"{base_prompt}, {costume_lock}"
```

**配置**:
```json
{
  "character": {
    "costume_lock_words": [
      "wearing his iconic mid-late-stage green daoist robe"
    ]
  }
}
```

**原理**:
> Flux的设计哲学是：**LoRA = 偏置，Prompt = 执行指令**

**效果**:
- ✅ 服饰细节稳定
- ✅ 颜色和花纹不跑偏
- ✅ 即使只有一套服饰也能稳定

---

## 📊 推荐配置（单LoRA）

### JSON配置示例

```json
{
  "version": "v2.1-exec",
  "character": {
    "id": "hanli",
    "gender": "male",
    "lora_path": "HanLi_Face_v3.safetensors",
    "lora_weight": 0.85,
    "temperament_anchor": "calm and restrained temperament, sharp but composed eyes",
    "costume_lock_words": [
      "wearing his iconic mid-late-stage green daoist robe"
    ],
    "face_refine": {
      "enable": true,
      "trigger": "shot_scale >= medium",
      "denoise": 0.35,
      "steps": 12
    }
  }
}
```

### LoRA权重建议

| LoRA类型 | 权重 | 说明 |
|---------|------|------|
| 人脸LoRA | **0.85-0.95** | 单LoRA时推荐0.85-0.95 |

**重要**: 
- 如果只有一个人脸LoRA，权重可以高一点（0.85-0.95）
- 如果LoRA训练得很好，可以到0.95
- 如果LoRA训练一般，建议0.85-0.9

---

## 🎯 关于服饰和风格

### 服饰：一套就可以

**答案**: ✅ **可以，一套就够了**

**原因**:
1. 通过**显式锁词**（补丁3）可以稳定服饰
2. 不需要训练服饰LoRA
3. 在Prompt中明确描述即可

**示例**:
```json
{
  "costume_lock_words": [
    "wearing his iconic mid-late-stage green daoist robe",
    "traditional Chinese daoist clothing",
    "green fabric with subtle patterns"
  ]
}
```

**效果**:
- ✅ 即使只有一套服饰，也能稳定生成
- ✅ 颜色、花纹、细节都稳定
- ✅ 不需要训练多个服饰LoRA

---

### 风格：不需要训练

**答案**: ✅ **不需要训练风格LoRA**

**原因**:
1. 风格可以通过**Prompt**控制
2. 风格LoRA是**可选的**（权重0.4）
3. 对于单LoRA场景，风格LoRA不是必须的

**替代方案**:
```json
{
  "prompt": {
    "style_hint": [
      "Chinese fantasy illustration",
      "xianxia_anime style",
      "cinematic lighting"
    ]
  }
}
```

**何时需要风格LoRA**:
- 需要**非常统一**的画风（如特定插画师风格）
- 需要**批量生成**时保持风格一致
- 需要**特殊效果**（如暗黑风格、明亮风格）

**结论**: 对于单LoRA场景，**风格LoRA不是必须的**

---

## 📝 使用步骤

### 步骤1: 配置角色锚

```python
from utils.character_anchor_v2_1_simple import get_character_anchor_manager_simple

manager = get_character_anchor_manager_simple()

# 注册角色（单LoRA）
anchor = manager.register_character_simple(
    character_id="hanli",
    gender="male",
    lora_path="HanLi_Face_v3.safetensors",
    lora_weight=0.85,
    temperament_anchor="calm and restrained temperament, sharp but composed eyes",
    costume_lock_words=[
        "wearing his iconic mid-late-stage green daoist robe"
    ]
)
```

### 步骤2: 构建增强Prompt

```python
# 基础Prompt
base_prompt = "Han Li, standing in a desert"

# 获取增强Prompt（自动添加气质锚点和显式锁词）
enhanced_prompt = manager.get_enhanced_prompt("hanli", base_prompt)

# 结果: "calm and restrained temperament, sharp but composed eyes, Han Li, standing in a desert, wearing his iconic mid-late-stage green daoist robe"
```

### 步骤3: 生成图像

```python
# 使用增强Prompt生成图像
image = image_generator.generate_image(
    prompt=enhanced_prompt,
    character_lora="hanli",
    lora_weight=0.85
)
```

---

## ⚠️ 注意事项

### 1. LoRA权重不要太高

- **单LoRA**: 0.85-0.95
- **如果太高**（>0.95）: 可能过拟合，表情僵硬
- **如果太低**（<0.8）: 角色特征不够明显

### 2. 气质锚点必须加

- **不加**: 气质会轻微漂（年轻/冷酷）
- **加了**: 气质稳定，同一个人感觉强

### 3. 显式锁词必须加

- **不加**: 服饰细节会跑偏（颜色、花纹）
- **加了**: 服饰稳定，即使只有一套也能稳定

### 4. FaceDetailer可选但推荐

- **不加**: 远景脸可能泛化
- **加了**: 远景脸清晰，但需要额外时间

---

## 📊 预期效果

### 稳定性提升

| 问题 | 不加补丁 | 加补丁 |
|------|---------|--------|
| 气质轻微漂 | ⚠️ 偶尔 | ✅ 基本消失 |
| 远景脸泛化 | ⚠️ 偶尔 | ✅ 大幅减少 |
| 服饰细节跑偏 | ⚠️ 偶尔 | ✅ 明显减少 |

### 灵活性

- ✅ 支持多表情（LoRA已训练）
- ✅ 支持多角度（LoRA已训练）
- ✅ 支持一套服饰（显式锁词）
- ✅ 不需要风格LoRA（Prompt控制）

---

## 🎯 总结

### 可以直接使用

✅ **能，完全能！** 只需要：

1. **单LoRA**（人脸LoRA，权重0.85-0.95）
2. **气质锚点**（必须，Prompt中）
3. **显式锁词**（必须，服饰描述）
4. **FaceDetailer**（可选，但推荐）

### 不需要的

- ❌ 不需要训练服饰LoRA（一套就够了）
- ❌ 不需要训练风格LoRA（Prompt控制）
- ❌ 不需要重新训练LoRA（现有LoRA即可）

### 下一步

1. 配置角色锚（使用`character_anchor_v2_1_simple.py`）
2. 应用3个运行时补丁
3. 测试生成效果
4. 根据效果微调权重和锁词

---

## 🔗 相关文档

- `LORA_STACK_IMPLEMENTATION.md` - LoRA Stack实施指南（多LoRA场景）
- `utils/character_anchor_v2_1_simple.py` - 单LoRA管理器（**新增**）
- `TECH_ARCHITECTURE_V2_1.md` - 技术架构文档

