# Prompt 优化技巧（Flux & SDXL 社区最佳实践）

## 一、Flux Prompt 优化技巧

### 1. 结构化 Prompt 格式（推荐）
Flux 对结构化 prompt 响应更好，建议使用以下格式：

```
[主体描述], [风格/质量], [镜头/构图], [环境/背景], [细节增强]
```

**示例：**
```
Golden scroll unfurling in Immortal realm sky, Chinese xianxia fantasy style, 
cinematic wide shot, clouds veiled background, highly detailed, 4k quality
```

### 2. 权重控制（使用括号）
Flux 支持权重调整，但语法略有不同：

- `(keyword:1.5)` - 提高权重到 1.5 倍
- `(keyword:0.8)` - 降低权重到 0.8 倍
- `[keyword]` - 负面提示（某些实现支持）

**最佳实践：**
- 主体元素：`(main subject:1.8)`
- 风格标签：`(style:1.3)`
- 细节增强：`(detailed:1.2)`

### 3. 分词与逗号分隔
Flux 的 CLIP 编码器对逗号分隔更敏感：

**✅ 推荐：**
```
xianxia fantasy, golden scroll, sky background, clouds, detailed, cinematic
```

**❌ 不推荐：**
```
xianxia fantasy golden scroll sky background clouds detailed cinematic
```

### 4. 固定 Seed（提升一致性）
Flux 对 seed 更敏感，相同 prompt + seed 可以提升批次一致性：

```python
# 为每个场景使用固定的 seed（基于场景ID）
seed = 42 + scene_id  # 确保相同场景每次生成一致
```

### 5. 步数优化（极速模式）
- **Flux.1-dev**: 20 步足够（推荐配置）
- **Flux.1-schnell**: 15-18 步即可
- **CFG Scale**: 3.5-5.0（Flux 不需要高 CFG）

### 6. 负面 Prompt 优化
Flux 的负面 prompt 效果不如 SDXL 明显，建议：

**简洁版（推荐）：**
```
low quality, blurry, distorted, deformed, bad anatomy
```

**避免过度复杂的负面 prompt**，Flux 可能会"混淆"。

---

## 二、SDXL Prompt 优化技巧

### 1. 双编码器策略
SDXL 使用两个 CLIP 编码器，可以分别优化：

- **CLIP-L**: 理解复杂语义（长文本）
- **CLIP-G**: 理解风格和视觉（短文本）

**推荐格式：**
```
[长描述，包含细节和语义] 
style: [风格关键词], quality: [质量关键词], composition: [构图关键词]
```

### 2. 风格标签位置
SDXL 对 prompt 的位置敏感：

**✅ 推荐（风格在前）：**
```
Chinese xianxia anime style, 3D rendered, detailed character, 
Han Li lying on gray-green sand, cinematic lighting
```

**❌ 不推荐（风格在后）：**
```
Han Li lying on gray-green sand, cinematic lighting, 
Chinese xianxia anime style, 3D rendered, detailed character
```

### 3. 权重语法（完整支持）
SDXL 完全支持 ComfyUI/A1111 风格的权重：

- `(keyword:1.5)` - 提高权重
- `((keyword))` - 等效于 `(keyword:1.1)`
- `[keyword]` - 降低权重（某些实现）

**组合使用：**
```
((Chinese xianxia anime style:1.3)), (detailed character:1.2), 
Han Li, (cinematic lighting:1.1)
```

### 4. 引导词（Leading Keywords）
SDXL 对 prompt 开头更敏感，建议：

**✅ 推荐：**
```
Chinese xianxia anime style, 3D rendered anime, detailed character, 
cinematic lighting, 4k, sharp focus, [主体描述]
```

### 5. 负面 Prompt 深度优化
SDXL 的负面 prompt 非常有效，可以使用详细的负面描述：

```
low quality, blurry, noise, overexposed, underexposed, deformed, 
distorted, mutated hands, deformed hands, extra fingers, missing fingers, 
fused fingers, bad anatomy, photorealistic, hyperrealistic, realistic, 
real photo, photograph, watermark, text, logo, compression artifacts
```

### 6. CFG Scale 优化
- **人物生成（InstantID）**: 7.0-7.5（平衡质量和一致性）
- **场景生成**: 5.0-6.0（更自然）
- **高质量模式**: 8.0-9.0（可能过饱和）

### 7. 步数优化
- **标准模式**: 25-30 步
- **快速模式**: 20 步（略有质量损失）
- **高质量模式**: 40-50 步（收益递减）

---

## 三、通用 Prompt 优化原则

### 1. 长度控制
- **Flux**: 建议 50-100 tokens（过长可能被截断）
- **SDXL**: 建议 75-77 tokens（CLIP-L 限制）

### 2. 关键词密度
避免关键词堆砌，优先使用：
- **主体**（1-2 个核心概念）
- **风格**（1-2 个风格标签）
- **质量**（1-2 个质量标签）
- **构图**（1 个镜头描述）

### 3. 语言选择
- **Flux**: 英文 prompt 效果更好
- **SDXL**: 中英文混合可接受（但英文更稳定）

### 4. 避免冲突
**❌ 避免：**
- `realistic` + `anime style` （冲突）
- `photorealistic` + `3D rendered` （冲突）
- `wide shot` + `close-up` （冲突）

**✅ 推荐：**
- `Chinese xianxia anime style, 3D rendered anime, detailed character`
- `cinematic wide shot, vast landscape, distant view`

---

## 四、仙侠场景专用优化

### 1. 风格标签组合
```
Chinese xianxia anime style, 3D rendered anime, 
traditional Chinese fantasy aesthetic, immortal cultivator style
```

### 2. 环境描述
```
immortal realm, vast golden desert, gray-green sand, 
clouds veiled sky, mystical atmosphere, ethereal lighting
```

### 3. 角色描述（SDXL + InstantID）
```
(male:1.5), Han Li, (tied long black hair:1.5), 
(deep cyan cultivator robe:1.3), xianxia cultivator style
```

### 4. 负面 Prompt（仙侠专用）
```
western style, european style, modern style, modern clothing, 
modern architecture, modern transportation, electronic products
```

---

## 五、实际应用建议

### 1. 场景生成（Flux.1）
```
[环境描述], xianxia fantasy style, cinematic [镜头类型], 
[背景元素], highly detailed, 4k quality
```

### 2. 人物生成（SDXL + InstantID）
```
[风格标签], [角色描述], [动作/姿态], [镜头类型], 
cinematic lighting, detailed character, sharp focus
```

### 3. 质量标签优先级
1. **必须包含**: `detailed`, `high quality`, `4k`
2. **推荐包含**: `cinematic lighting`, `sharp focus`
3. **可选**: `8k`, `masterpiece`, `best quality`

---

## 六、参考资源

- **Flux 官方文档**: https://github.com/black-forest-labs/FLUX.1-dev
- **SDXL 最佳实践**: https://github.com/Stability-AI/generative-models
- **ComfyUI Prompt 指南**: https://github.com/comfyanonymous/ComfyUI

