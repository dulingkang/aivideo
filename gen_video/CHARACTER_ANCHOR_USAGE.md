# 人设锚点图使用说明

## 概述

人设锚点图（Character Anchor）是整个视频的"DNA"，所有后续场景都必须引用这张图，确保形象一致性。

## 核心原则

> **人设只能"注入一次"，但必须在所有叙事之前**

## ⚡ 核心规则（工业界标准做法）

> **hanli_anchor.png = hanli_mid.jpg（直接复制，不生成）**

**原因**：
- ✅ 100% 相似度（直接复制，无任何失真）
- ✅ 简单可靠（不需要生成，避免生成质量问题）
- ✅ 工业界标准做法（所有专业流程都这样做）

---

## 一、创建人设锚点图（Scene 0）

### 1. 运行创建脚本

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
python generate_character_anchor.py --character hanli --config config.yaml
```

**参数说明：**
- `--character`: 角色ID（默认：hanli）
- `--config`: 配置文件路径（默认：config.yaml）
- `--output-dir`: 输出目录（默认：gen_video/character_anchors）

**注意**：不再需要 `--seed` 参数（因为不生成，直接复制）

### 2. 创建规则

- **源文件**：`hanli_mid.jpg`（或配置中的 `face_image_path`）
- **目标文件**：`gen_video/character_anchors/hanli_anchor.png`
- **操作**：直接复制（`shutil.copy2`）
- **相似度**：100%（完全一致）

### 3. 参考图优先级

1. **配置中的 `face_image_path`**（通常是高质量参考图）
2. **`gen_video/reference_image/hanli_mid.jpg`**（优先）
3. **`gen_video/reference_image/hanli_mid.png`**（备选）

---

## 二、三类镜头分流规则

### 🟢 A类：叙事/氛围镜头

**引擎**：FLUX  
**禁用**：InstantID / LoRA  
**reference**：人设锚点图  
**要求**：像"韩立"，但不追求脸

**适用场景：**
- 躺沙漠
- 远景
- 背影
- 剪影
- wide + top_down + lying

**特点：**
- 世界观一致性 > 人脸一致性
- 使用语义化 prompt（自然语言句子）
- 必须引用人设锚点图

---

### 🟡 B类：过渡人物镜头

**引擎**：SDXL  
**reference**：人设锚点图  
**不用**：InstantID  
**镜头**：中景为主

**适用场景：**
- 站立
- 走路
- 回头
- 过渡场景

**特点：**
- 使用风格 LoRA（fanren_style）
- 必须引用人设锚点图

---

### 🔴 C类：情绪/表情镜头

**引擎**：InstantID  
**reference**：人设锚点图  
**镜头**：中近景  
**姿态**：简单

**适用场景：**
- 回忆
- 痛苦
- 施法特写
- 对话镜头

**特点：**
- 锁脸（face lock）
- 使用风格 LoRA（fanren_style）
- 必须引用人设锚点图

---

## 三、使用流程

### 步骤 1：生成人设锚点图

```bash
python generate_character_anchor.py --character hanli
```

**输出**：`gen_video/character_anchors/hanli_anchor.png`

### 步骤 2：生成场景图像

系统会自动：
1. 检测 Execution Planner 返回的 `use_character_anchor: True` 标志
2. 查找对应的人设锚点图（`{character_id}_anchor.png`）
3. 使用人设锚点图作为 `reference_image`
4. 根据镜头类型选择对应的引擎（A/B/C类）

### 步骤 3：验证

检查日志中是否有：
```
✓ 使用人设锚点图: hanli_anchor.png
🎯 传递人设锚点图到 generate_image: hanli_anchor.png
```

---

## 四、Execution Planner 决策逻辑

### A类镜头检测

```python
is_narrative_shot = (
    camera_shot == "wide" or 
    visibility == "low" or
    is_wide_topdown_lying or
    character_pose in ["lying_motionless", "lying", "back_view"]
)
```

**决策**：
- `engine: "flux1"`
- `use_character_anchor: True`
- `use_semantic_prompt: True`
- `disable_character_lora: True`
- `disable_style_lora: True`

### B类镜头检测

```python
is_transition_shot = (
    camera_shot == "medium" and
    character_pose in ["standing", "walking", "turning"]
)
```

**决策**：
- `engine: "sdxl"`
- `use_character_anchor: True`
- `style_anchor.enabled: True`

### C类镜头检测

```python
is_emotion_shot = (
    camera_shot in ["close", "medium"] and
    character_pose in ["thinking", "pain", "casting", "expression"]
)
```

**决策**：
- `engine: "instantid"`
- `lock_face: True`
- `use_character_anchor: True`
- `style_anchor.enabled: True`

---

## 五、注意事项

### ⚠️ 绝对禁止

1. **禁止用"Scene N 的生成结果"作为下一个 scene 的 reference**
   - 必须始终使用人设锚点图
   - 前一个场景的图像只用于连贯性（img2img），不用于形象锚定

2. **禁止在叙事镜头里反复切换引擎**
   - A类镜头必须用 FLUX
   - B类镜头必须用 SDXL
   - C类镜头必须用 InstantID

3. **禁止在没有生成人设锚点图的情况下生成场景**
   - 系统会警告，但不会阻止
   - 建议先运行 `generate_character_anchor.py`

### ✅ 推荐做法

1. **在生成所有场景之前，先生成人设锚点图**
2. **确保人设锚点图保存在正确的位置**：`gen_video/character_anchors/{character_id}_anchor.png`
3. **检查日志，确认使用了人设锚点图**

---

## 六、故障排查

### 问题 1：找不到人设锚点图

**错误信息**：
```
⚠ 警告：Execution Planner 要求使用人设锚点图，但未找到: gen_video/character_anchors/hanli_anchor.png
```

**解决方案**：
```bash
python generate_character_anchor.py --character hanli
```

### 问题 2：形象不一致

**可能原因**：
1. 没有使用人设锚点图
2. 使用了错误的 reference_image（前一个场景的图像）
3. 引擎选择错误（A类用了 InstantID）

**解决方案**：
1. 检查日志，确认使用了人设锚点图
2. 检查 Execution Planner 的决策是否正确
3. 重新生成人设锚点图

---

## 七、示例

### 完整工作流

```bash
# 1. 创建人设锚点图（直接复制参考图）
python generate_character_anchor.py --character hanli

# 2. 生成场景图像（会自动使用人设锚点图）
python main.py --script lingjie/episode/1.v2.json --output outputs/images/lingjie_ep1_v2
```

### 日志示例

```
创建角色人设锚点图: hanli
============================================================
  ✓ 使用配置中的参考图: hanli_mid.jpg
  🎯 直接复制参考图作为人设锚点图（工业界标准做法）...
     源文件: /vepfs-dev/shawn/vid/fanren/gen_video/reference_image/hanli_mid.jpg
     目标文件: /vepfs-dev/shawn/vid/fanren/gen_video/gen_video/character_anchors/hanli_anchor.png
  ✅ 人设锚点图已创建: gen_video/character_anchors/hanli_anchor.png
  ℹ 所有后续场景将引用此图作为形象锚点（100% 相似度）

处理场景 2/22 (场景ID=1)
============================================================
  🟢 A类镜头（叙事/氛围）：使用 FLUX 引擎（世界观一致性 > 人脸一致性）
  ✓ 必须引用人设锚点图（确保形象一致性）
  ✓ 使用人设锚点图: hanli_anchor.png
  🎯 传递人设锚点图到 generate_image: hanli_anchor.png
```

---

## 八、总结

**核心原则**：
1. ✅ 人设锚点图是"整个视频的 DNA"
2. ✅ 所有场景必须引用人设锚点图
3. ✅ 三类镜头必须彻底分流（A/B/C）
4. ✅ 禁止用前一个场景的图像作为形象锚定

**这套方案的优势**：
- ✅ 形象一致性：所有场景都引用同一张人设锚点图
- ✅ 世界观一致性：A类镜头使用 FLUX，世界观理解更强
- ✅ 人脸一致性：C类镜头使用 InstantID，锁脸更准确
- ✅ 可规模化：适用于小说推文等需要形象一致性的场景

