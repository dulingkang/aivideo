# v2.2-final格式生成问题分析

> **分析日期**: 2025-12-21  
> **问题**: 生成的图片不符合预期

---

## 🔍 问题分析

根据终端输出，发现了以下关键问题：

### 1. ❌ 模型选择被覆盖

**问题**:
- JSON中指定: `flux + pulid`
- 实际使用: `SDXL + InstantID`

**原因** (第86行):
```
⚠ 场景稳定性不足 (0.40)，回退到 SDXL + InstantID（稳定方案）
```

**影响**:
- v2.2-final格式中明确指定的模型路由被忽略
- 系统自动回退到"稳定方案"，违背了v2.2-final的"硬规则"原则

---

### 2. ❌ Prompt被截断

**问题** (第201-204行):
```
Token indices sequence length is longer than the specified maximum sequence length for this model (186 > 77)
The following part of your input was truncated: 
['wearing his iconic mid - late - stage green daoist robe']
```

**原因**:
- CLIP tokenizer只能处理77个token
- 生成的prompt有186个token，超出限制
- 关键信息（服装描述）被截断

**影响**:
- 角色服装描述丢失
- 生成的图片可能缺少关键特征

---

### 3. ❌ 人脸相似度不足

**问题** (第219-220行):
```
⚠️ 发现问题: 人脸相似度不足 (-0.023 < 0.7)
```

**原因**:
- 相似度是负数，说明生成的人脸和参考图差异很大
- 可能是模型选择错误（SDXL vs Flux）导致

**影响**:
- 角色一致性差
- 不符合v2.2-final的"角色锚定"要求

---

### 4. ⚠️ Prompt构建过程复杂

**问题** (第91-131行):
- 使用了LLM进行场景分析（违背v2.2-final原则）
- 多次去重处理
- Prompt构建耗时3.79秒

**原因**:
- `ExecutionExecutorV21`调用了增强模式生成器
- 增强模式生成器内部使用了LLM和复杂的Prompt处理

**影响**:
- 违背了v2.2-final的"不使用LLM决策"原则
- 增加了不确定性和延迟

---

## 🔧 根本原因

### 问题1: ExecutionExecutorV21没有直接调用ImageGenerator

**当前流程**:
```
ExecutionExecutorV21.execute_scene()
  → _execute_image_generation()
    → 调用增强模式生成器 (generate_scene)
      → 使用LLM分析
      → 自动选择模型（忽略JSON中的model_route）
      → 复杂的Prompt处理
```

**应该的流程**:
```
ExecutionExecutorV21.execute_scene()
  → _execute_image_generation()
    → 直接调用ImageGenerator.generate()
      → 使用JSON中的model_route（硬规则）
      → 使用JSON中的prompt（模板填充）
      → 不使用LLM
```

---

### 问题2: Prompt过长

**当前Prompt** (186 tokens):
```
hanli, calm and restrained temperament, sharp but composed eyes, 
determined expression, traditional chinese cultivation attire, 
sitting in meditation pose, in 黄枫谷, serene and mysterious, 
ancient cultivation atmosphere, spiritual energy flowing, 
ancient chinese architecture, traditional pavilions, 
mountain peaks in distance, spiritual mist, cinematic lighting, 
high detail, epic atmosphere, chinese fantasy illustration style, 
wearing his iconic mid-late-stage green daoist robe
```

**CLIP限制**: 77 tokens

**解决方案**:
1. 精简Prompt，保留关键信息
2. 使用SDXL的CLIP（支持更长prompt）
3. 或者使用Flux的T5编码器（支持更长prompt）

---

## 💡 解决方案

### 方案1: 修复ExecutionExecutorV21直接调用ImageGenerator

**修改点**:
1. `_execute_image_generation()`应该直接调用`ImageGenerator.generate()`
2. 传递JSON中的`model_route`、`prompt`、`character`等信息
3. 不使用增强模式生成器

---

### 方案2: 精简Prompt模板

**修改点**:
1. 优化`_build_prompt()`方法
2. 确保生成的prompt不超过77 tokens（如果使用CLIP）
3. 或者使用支持更长prompt的编码器

---

### 方案3: 强制使用JSON中的model_route

**修改点**:
1. 在`ExecutionExecutorV21`中强制使用JSON中的`model_route`
2. 不允许自动回退
3. 如果模型不可用，应该报错而不是回退

---

## 📊 当前状态

| 项目 | 期望 | 实际 | 状态 |
|------|------|------|------|
| 模型 | Flux + PuLID | SDXL + InstantID | ❌ |
| Prompt长度 | < 77 tokens | 186 tokens | ❌ |
| 人脸相似度 | > 0.7 | -0.023 | ❌ |
| LLM使用 | 不使用 | 使用 | ❌ |
| 硬规则执行 | 是 | 否 | ❌ |

---

## 🎯 下一步行动

1. **立即修复**: 修改`ExecutionExecutorV21._execute_image_generation()`，直接调用ImageGenerator
2. **优化Prompt**: 精简prompt模板，确保不超过token限制
3. **强制模型路由**: 不允许自动回退，严格按照JSON执行
4. **测试验证**: 重新生成图片，验证修复效果

