# JSON场景描述优化指南

## 问题说明

当前JSON描述可能不够精确，导致视频生成时无法正确识别动作类型和镜头运动。本指南提供优化建议和智能分析功能。

## 优化方案

### 方案1：优化JSON描述（推荐）

在JSON中明确描述动作和镜头，系统会自动识别：

#### 1. 物体运动场景（需要SVD生成视频）

**示例：卷轴展开**
```json
{
  "description": "镜头淡入，云雾笼罩的仙域天空中金色卷轴舒展开来并闪烁灵光。",
  "action": "Title reveal",
  "camera": "Static shot",
  "visual": {
    "composition": "Golden scroll unfurling in immortal realm sky...",
    "fx": "Golden light,scroll-unfurling effect,spirit particles",
    "motion": {
      "type": "pan",
      "direction": "left_to_right",
      "speed": "slow"
    }
  },
  "prompt": "golden scroll unfurling prominently..."
}
```

**关键点**：
- ✅ `description` 中包含 `"展开"` 或 `"unfurling"`
- ✅ `composition` 中包含 `"unfurling"`
- ✅ `fx` 中包含 `"unfurling effect"`
- ✅ `prompt` 中包含 `"unfurling"`
- ✅ `motion.type` 不为 `"static"`（建议使用 `"pan"` 或 `"zoom"`）

#### 2. 人物静止场景（使用静态图像动画）

**示例：躺着不动**
```json
{
  "description": "韩立躺在青灰色沙地上一动不动，感受地面升腾的燥热。",
  "action": "Lying motionless",
  "camera": "Top-down wide shot",
  "visual": {
    "character_pose": "Lying flat on back,motionless,facing upward",
    "motion": {
      "type": "pan",
      "direction": "left_to_right",
      "speed": "slow"
    }
  }
}
```

**关键点**：
- ✅ `action` 中包含 `"motionless"` 或 `"still"`
- ✅ `description` 中包含 `"一动不动"` 或 `"不动"`
- ✅ `character_pose` 中包含 `"motionless"`

#### 3. 人物动态场景（使用SVD生成视频）

**示例：战斗场景**
```json
{
  "description": "韩立施展法术，青芒激射而出。",
  "action": "Casting spell",
  "camera": "Dynamic shot",
  "visual": {
    "character_pose": "Facing camera,front view,casting spell,determined expression",
    "fx": "Green streaks shooting out,spiritual energy",
    "motion": {
      "type": "static"
    }
  }
}
```

**关键点**：
- ✅ `action` 中包含动态动作（`"cast"`, `"attack"`, `"fight"` 等）
- ✅ `description` 中包含动态关键词（`"施展"`, `"攻击"`, `"战斗"` 等）

### 方案2：使用智能分析功能（自动推断）

系统已添加智能分析功能，会根据JSON描述自动推断：

1. **物体运动检测**：自动识别 `"展开"`, `"unfurling"`, `"打开"`, `"旋转"` 等关键词
2. **镜头运动推断**：从 `description`, `camera`, `motion` 字段自动推断
3. **运动强度判断**：根据动作类型自动设置 `motion_bucket_id` 和 `noise_aug_strength`

## JSON字段优化建议

### description 字段
- ✅ 明确描述动作：`"卷轴舒展开来"`, `"门缓缓打开"`, `"粒子飘动"`
- ✅ 明确描述镜头：`"镜头淡入"`, `"镜头推进"`, `"镜头横移"`
- ❌ 避免模糊描述：`"场景变化"`, `"画面切换"`

### action 字段
- ✅ 使用标准动作词：`"Title reveal"`, `"Lying motionless"`, `"Casting spell"`
- ✅ 包含运动状态：`"motionless"`, `"still"`, `"moving"`, `"attacking"`

### visual.composition 字段
- ✅ 包含物体运动：`"scroll unfurling"`, `"door opening"`, `"particles drifting"`
- ✅ 包含镜头信息：`"close-up"`, `"wide shot"`, `"top-down view"`

### visual.fx 字段
- ✅ 包含特效和运动：`"unfurling effect"`, `"spirit particles"`, `"golden light shimmering"`

### visual.motion 字段
- ✅ 明确指定镜头运动类型：`"pan"`, `"zoom"`, `"tilt"`, `"dolly"`
- ✅ 指定方向：`"left_to_right"`, `"right_to_left"`, `"in"`, `"out"`
- ✅ 指定速度：`"slow"`, `"medium"`, `"fast"`
- ❌ 避免使用 `"static"` 当有物体运动时

## 常见场景模板

### 模板1：物体展开/打开场景
```json
{
  "description": "金色卷轴缓缓展开，灵光闪烁",
  "action": "Object animation",
  "visual": {
    "composition": "Golden scroll unfurling slowly...",
    "fx": "Scroll-unfurling effect,spirit light",
    "motion": {
      "type": "pan",
      "direction": "left_to_right",
      "speed": "slow"
    }
  },
  "prompt": "...scroll unfurling..."
}
```

### 模板2：人物静止观察场景
```json
{
  "description": "韩立一动不动，静静观察天空",
  "action": "Observing",
  "visual": {
    "character_pose": "Motionless,observing",
    "motion": {
      "type": "pan",
      "direction": "left_to_right",
      "speed": "slow"
    }
  }
}
```

### 模板3：人物动态动作场景
```json
{
  "description": "韩立施展法术，青芒激射",
  "action": "Casting spell",
  "visual": {
    "character_pose": "Casting spell,determined",
    "fx": "Green streaks shooting out",
    "motion": {
      "type": "static"
    }
  }
}
```

## 智能分析功能使用

系统会在视频生成时自动调用智能分析，根据JSON描述推断：

1. **是否需要SVD生成视频**（有物体运动或人物动态动作）
2. **镜头运动类型和参数**（从description和motion字段推断）
3. **运动强度参数**（自动设置motion_bucket_id和noise_aug_strength）

## 检查清单

在优化JSON时，检查以下项目：

- [ ] `description` 中是否明确描述了动作（展开、打开、旋转等）？
- [ ] `action` 字段是否准确反映了人物状态（静止/动态）？
- [ ] `visual.composition` 中是否包含了物体运动描述？
- [ ] `visual.fx` 中是否包含了特效和运动效果？
- [ ] `visual.motion.type` 是否与场景需求匹配（有物体运动时不应为"static"）？
- [ ] `prompt` 中是否包含了关键动作关键词？

## 示例：场景0（卷轴展开）优化

**优化前**：
```json
{
  "motion": {
    "type": "static"
  }
}
```

**优化后**：
```json
{
  "description": "镜头淡入，云雾笼罩的仙域天空中金色卷轴舒展开来并闪烁灵光。",
  "visual": {
    "composition": "Golden scroll unfurling in immortal realm sky...",
    "fx": "Golden light,scroll-unfurling effect,spirit particles",
    "motion": {
      "type": "pan",
      "direction": "left_to_right",
      "speed": "slow"
    }
  },
  "prompt": "golden scroll unfurling prominently..."
}
```

**关键改进**：
- ✅ `motion.type` 从 `"static"` 改为 `"pan"`（允许镜头运动）
- ✅ 所有字段中都包含 `"unfurling"` 或 `"展开"` 关键词
- ✅ 系统会自动检测到物体运动，使用SVD生成视频

