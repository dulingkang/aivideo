# LoRA Stack（分层）实施指南

## 📋 总览

基于第三方分析报告（2025-12-20）的建议，v2.2架构升级支持**LoRA Stack（分层）**和**解耦训练策略**。

---

## 🎯 核心问题

### 问题1: Flux + LoRA 也会飘

**现象**:
- 同一个人，气质轻微漂（年轻/冷酷）
- 远景/中景，脸偶尔变"泛化脸"
- 服饰对，但细节偶尔跑偏

**原因**:
- Flux的T5很会"发挥"，没人锁就会自己演
- LoRA在远景token权重天然下降
- 单LoRA无法应对长篇连载（换地图、换衣服、年龄增长）

### 问题2: 单LoRA无法应对长篇连载

**场景**:
- 韩立从炼气期（青袍、少年）→ 筑基期（黄袍、青年）→ 结丹期（甲胄、严肃）

**问题**:
- 大杂烩LoRA会把"青袍"和"脸"炼死在一起
- 需要画"韩立洗澡（没衣服）"或"韩立穿战甲"时，LoRA里的"青袍"特征会干扰画面（Bleeding）

**结论**: **解耦（Disentanglement）是必须的**

---

## ✅ 解决方案：LoRA Stack（分层）

### 架构设计

```
LoRA Stack（分层）:
├─ Layer 0: 核心脸部LoRA（永远不变，权重0.85-0.95）
│  └─ 目的: 锁脸，不锁衣服
│
├─ Layer 1: 阶段性气质LoRA（随剧情变，权重0.6）
│  └─ 目的: 年龄/气质变化
│
├─ Layer 2: 服饰LoRA（动态，随装备变，权重0.8-0.9）
│  └─ 目的: 衣服结构，忽略穿衣服的人
│
└─ Layer 3: 画风LoRA（可选，权重0.4）
   └─ 目的: 风格统一
```

### JSON设计（v2.2）

```json
{
  "character": {
    "id": "hanli",
    "state": "foundation_phase",
    "costume": "battle_armor",
    "lora_stack": [
      {
        "layer": "face",
        "path": "HanLi_Face_v3.safetensors",
        "weight": 0.9,
        "trigger": "hanli_face"
      },
      {
        "layer": "age",
        "path": "HanLi_Youth_v1.safetensors",
        "weight": 0.6,
        "trigger": "young man"
      },
      {
        "layer": "costume",
        "path": "Costume_BattleArmor_v1.safetensors",
        "weight": 0.8,
        "trigger": "battle armor"
      }
    ]
  }
}
```

---

## 🔧 运行时补丁（3个MVP Fix）

### 补丁1: 气质锚点（Temperament Anchor）

**目的**: 给人脸一个"气质锚点"，抑制表情LoRA的"自由发挥"

**实现**:
```python
# 在Prompt最前面添加恒定气质锚
temperament = "calm and restrained temperament, sharp but composed eyes"
prompt = f"{temperament}, {base_prompt}"
```

**效果**:
- ✅ 不影响表情
- ✅ 极大提升"同一个人"的感觉
- ✅ T5会把这当作长期性格描述

**配置**:
```json
{
  "character": {
    "temperament_anchor": "calm and restrained temperament, sharp but composed eyes"
  }
}
```

---

### 补丁2: FaceDetailer（远景增强）

**目的**: 远景/中景时，自动增强人脸细节

**触发条件**:
```json
{
  "face_refine": {
    "enable": true,
    "trigger": "shot_scale >= medium",
    "model": "flux1-dev",
    "lora": [
      { "name": "HanLi_face_lora", "weight": 0.9 }
    ],
    "steps": 12,
    "denoise": 0.35
  }
}
```

**关键参数**:
- `denoise`: 不要超过0.4，否则会"换人"
- `steps`: 12步足够（快速）
- `lora`: 只使用脸部LoRA，不加载服饰LoRA

**效果**:
- ✅ 远景人脸清晰度提升
- ✅ 保持角色一致性
- ✅ 不影响构图和光影

---

### 补丁3: 显式锁词（Explicit Lock Words）

**目的**: 即使有服饰LoRA，也必须写清楚服饰描述

**实现**:
```python
# 在Prompt中显式添加服饰描述
costume_lock = "wearing his iconic mid-late-stage green daoist robe"
prompt = f"{base_prompt}, {costume_lock}"
```

**原理**:
> Flux的设计哲学是：**LoRA = 偏置，Prompt = 执行指令**

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

---

## 📊 LoRA权重建议（关键）

| LoRA类型 | 权重范围 | 说明 |
|---------|---------|------|
| 脸部LoRA | 0.85-0.95 | 最高优先级，永远不变 |
| 表情LoRA | 0.4-0.6 | **不要抢主导权** |
| 服饰LoRA | 0.8-0.9 | 动态，随装备变 |
| 画风LoRA | 0.4 | 可选，最低优先级 |

**重要**: 表情LoRA权重不要超过0.6，否则会干扰气质锚点

---

## 🎓 解耦训练策略

### LoRA A: 脸部核心

**素材要求**:
- 韩立的各种角度大头照、半身照
- **关键**: 素材里必须包含**多种衣服**（甚至赤膊）

**目的**: 让AI知道，"这张脸"是共性，衣服是变量

**训练提示**:
- 不要只训练"穿青袍的韩立"
- 要训练"韩立的脸"（不管穿什么）

---

### LoRA B: 服饰

**素材要求**:
- 该特定服装的设定图
- 或者用模特穿这件衣服的图，把脸打码或换成不同人的脸

**目的**: 让AI学会"这件衣服"的结构，而忽略穿衣服的人是谁

**训练提示**:
- 不要只训练"韩立穿青袍"
- 要训练"青袍"（不管谁穿）

---

## 🔄 迁移路径

### 从v2.1到v2.2

1. **保持v2.1硬规则**（不要丢掉）
   - Shot/Pose/Model硬规则继续使用
   - 这是Flux最缺的缰绳

2. **升级CharacterAnchorManager**
   - 从单LoRA升级到LoRA Stack
   - 添加气质锚点和显式锁词支持

3. **更新JSON格式**
   - 从`lora_path`升级到`lora_stack`
   - 添加`temperament_anchor`和`costume_lock_words`

4. **实施运行时补丁**
   - 在Prompt Builder中集成气质锚点
   - 在ImageGenerator中集成FaceDetailer
   - 在Prompt中显式添加服饰锁词

---

## 📝 实施检查清单

### 代码层面

- [ ] 升级`CharacterAnchorManager`支持LoRA Stack
- [ ] 更新`Prompt Builder`支持气质锚点
- [ ] 更新`Prompt Builder`支持显式锁词
- [ ] 集成FaceDetailer到`ImageGenerator`
- [ ] 更新JSON转换器支持v2.2格式

### 训练层面

- [ ] 准备脸部LoRA训练素材（多种衣服）
- [ ] 准备服饰LoRA训练素材（多种人脸）
- [ ] 训练脸部核心LoRA
- [ ] 训练阶段性气质LoRA
- [ ] 训练服饰LoRA

### 测试层面

- [ ] 测试LoRA Stack加载
- [ ] 测试气质锚点效果
- [ ] 测试FaceDetailer触发
- [ ] 测试显式锁词效果
- [ ] 批量测试稳定性

---

## 🎯 最终架构图（v2.2）

```
JSON Scene v2.2
    ↓
[规则引擎] → 锁定Shot/Pose/Model（v2.1硬规则）
    ↓
[LoRA Stack Builder] → 构建LoRA堆栈
    ├─ 脸部LoRA (0.9)
    ├─ 气质LoRA (0.6)
    ├─ 服饰LoRA (0.8)
    └─ 画风LoRA (0.4)
    ↓
[Prompt Builder] → 构建增强Prompt
    ├─ 气质锚点（最前面）
    ├─ 基础Prompt
    └─ 显式锁词（服饰描述）
    ↓
[ImageGenerator] → 生成图像
    ├─ 加载LoRA Stack
    ├─ 应用气质锚点
    └─ 应用显式锁词
    ↓
[FaceDetailer] → 远景增强（条件触发）
    └─ 只使用脸部LoRA
    ↓
最终图像
```

---

## 📊 预期效果

### 稳定性提升

| 问题 | v2.1 | v2.2（LoRA Stack） |
|------|------|-------------------|
| 气质轻微漂 | ⚠️ 偶尔 | ✅ 基本消失 |
| 远景脸泛化 | ⚠️ 偶尔 | ✅ 大幅减少 |
| 服饰细节跑偏 | ⚠️ 偶尔 | ✅ 明显减少 |
| 长篇连载稳定性 | ⚠️ 下降 | ✅ 保持稳定 |

### 灵活性提升

- ✅ 支持角色成长（年龄变化）
- ✅ 支持装备变化（服饰切换）
- ✅ 支持场景变化（画风调整）
- ✅ 保持角色一致性（脸部不变）

---

## 🔗 相关文档

- `TECH_ARCHITECTURE_V2_1.md` - 技术架构文档
- `V2_1_TO_V2_2_EVOLUTION.md` - v2.2演进建议
- `schemas/scene_v2_2_example.json` - v2.2 JSON示例
- `utils/character_anchor_v2_2.py` - LoRA Stack实现

---

## 总结

**这份第三方分析100%正确，且直击要害。**

**核心价值**:
1. ✅ 确认v2.1硬规则的价值（Flux最缺的缰绳）
2. ✅ 指出单LoRA的局限性（长篇连载会崩溃）
3. ✅ 提供LoRA Stack解决方案（解耦训练）
4. ✅ 提供3个运行时补丁（立即可用）

**下一步**:
1. 保持v2.1硬规则（不要丢掉）
2. 升级到LoRA Stack（支持分层）
3. 实施运行时补丁（气质锚点、FaceDetailer、显式锁词）
4. 开始解耦训练（脸部与服饰分离）

