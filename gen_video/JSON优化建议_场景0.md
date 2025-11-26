# JSON优化建议 - 场景0（卷轴展开）

## 当前问题

场景0的卷轴展开没有正确表现，因为：
1. `motion.type: "static"` 导致使用静态图像动画
2. 虽然有"展开"关键词，但系统可能没有正确识别

## 优化建议

### 方案1：修改motion字段（最简单）

将 `motion.type` 从 `"static"` 改为 `"pan"`：

```json
{
  "visual": {
    "motion": {
      "type": "pan",
      "direction": "left_to_right",
      "speed": "slow"
    }
  }
}
```

### 方案2：增强描述（更精确）

在所有相关字段中都明确包含"展开"关键词：

```json
{
  "description": "镜头淡入，云雾笼罩的仙域天空中金色卷轴缓缓舒展开来并闪烁灵光。",
  "action": "Title reveal",
  "camera": "Static shot",
  "visual": {
    "composition": "Golden scroll slowly unfurling in immortal realm sky,veiled in clouds,spirit light shimmering,no person,no character,scroll is the main element",
    "environment": "Immortal realm sky,mist-wreathed,golden radiance,spirit particles drifting",
    "character_pose": "",
    "fx": "Golden light,scroll-unfurling effect,spirit particles,scroll slowly opening",
    "motion": {
      "type": "pan",
      "direction": "left_to_right",
      "speed": "slow"
    }
  },
  "prompt": "Xianxia opening,golden scroll slowly unfurling prominently,mist-shrouded immortal sky,spirit particles drifting,wide cinematic shot,no person,no character,film-grade lighting,4K"
}
```

**关键改进**：
- ✅ `description` 中明确 `"缓缓舒展开来"`（强调展开动作）
- ✅ `composition` 中明确 `"slowly unfurling"`（强调展开过程）
- ✅ `fx` 中明确 `"scroll-unfurling effect"` 和 `"scroll slowly opening"`（双重强调）
- ✅ `prompt` 中明确 `"slowly unfurling"`（强调展开）
- ✅ `motion.type` 改为 `"pan"`（允许镜头运动，触发SVD生成）

## 其他需要优化的场景

检查所有有物体运动的场景，确保：
1. `motion.type` 不为 `"static"`
2. 描述中包含明确的动作关键词
3. `visual.composition` 和 `visual.fx` 中包含运动描述

