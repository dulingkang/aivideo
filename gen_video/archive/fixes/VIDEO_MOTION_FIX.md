# 视频运动描述修复说明

## 🔍 问题描述

用户反馈：生成的视频中只有镜头运动（相机运动），没有场景元素本身的运动。

例如，对于"山谷、瀑布、彩虹"的场景，视频中只有相机在移动，但瀑布没有流动，云彩没有飘动，彩虹没有闪烁。

## ✅ 修复方案

### 1. 问题根源

之前的 `_build_video_prompt` 方法只添加了：
- 相机运动描述（camera pan, zoom, dolly）
- 通用的运动强度描述（dynamic/moderate/gentle movement）

**缺少**：场景元素本身的运动描述（如瀑布流动、云彩飘动等）

### 2. 解决方案

添加了 `_extract_scene_motion` 函数，自动从prompt中识别场景元素，并生成相应的运动描述：

```python
def _extract_scene_motion(prompt_text: str) -> list:
    """
    从prompt中提取场景元素，并生成相应的运动描述
    """
    motion_keywords = {
        # 水相关
        '瀑布': ['waterfall flowing', 'water cascading', 'water rushing'],
        '河流': ['river flowing', 'water streaming'],
        '溪流': ['stream flowing', 'water trickling'],
        '水': ['water rippling', 'water flowing'],
        '湖': ['lake rippling', 'water gently moving'],
        '海': ['waves rolling', 'ocean waves'],
        
        # 天空相关
        '云': ['clouds drifting', 'clouds floating'],
        '云彩': ['clouds drifting', 'clouds floating'],
        '彩虹': ['rainbow shimmering', 'rainbow glowing'],
        '阳光': ['sunlight shifting', 'light rays moving'],
        '光线': ['light rays moving', 'light shifting'],
        
        # 植物相关
        '树': ['leaves swaying', 'trees gently moving'],
        '树叶': ['leaves swaying', 'leaves rustling'],
        '草': ['grass swaying', 'grass gently moving'],
        '花': ['flowers swaying', 'flowers gently moving'],
        
        # 风相关
        '风': ['wind blowing', 'breeze moving'],
        
        # 雾气相关
        '雾': ['mist rising', 'fog drifting'],
        '雾气': ['mist rising', 'fog drifting'],
        
        # 火相关
        '火': ['flames flickering', 'fire dancing'],
        '火焰': ['flames flickering', 'fire dancing'],
        
        # 雪相关
        '雪': ['snow falling', 'snowflakes drifting'],
        
        # 鸟相关
        '鸟': ['birds flying', 'birds soaring'],
    }
    
    # 检查prompt中的关键词，返回相应的运动描述
    ...
```

### 3. 运动描述优先级

修复后的运动描述添加顺序：

1. **场景元素运动**（最重要）
   - 从prompt中自动识别（如"瀑布" → "waterfall flowing"）
   - 这是物体本身的运动，是视频动感的主要来源

2. **运动强度描述**
   - 根据 `motion_intensity` 配置添加（dynamic/moderate/gentle）

3. **相机运动**（次要）
   - 根据 `camera_motion` 配置添加（pan/zoom/dolly）
   - 如果只有相机运动，视频会显得单调

4. **默认自然运动**（兜底）
   - 如果未检测到特定场景元素，添加默认的自然运动描述

### 4. 示例

**修复前**：
```
prompt: "一个美丽的山谷，有瀑布和彩虹"
video_prompt: "... gentle movement, subtle motion, smooth camera pan"
结果：只有相机在移动，瀑布和彩虹是静态的
```

**修复后**：
```
prompt: "一个美丽的山谷，有瀑布和彩虹"
video_prompt: "... waterfall flowing, rainbow shimmering, gentle movement, subtle motion, smooth camera pan"
结果：瀑布在流动，彩虹在闪烁，同时相机也在移动
```

## 📊 支持的运动类型

### 水相关
- 瀑布 → waterfall flowing
- 河流 → river flowing
- 溪流 → stream flowing
- 水 → water rippling
- 湖 → lake rippling
- 海 → waves rolling

### 天空相关
- 云/云彩 → clouds drifting
- 彩虹 → rainbow shimmering
- 阳光 → sunlight shifting
- 光线 → light rays moving

### 植物相关
- 树 → leaves swaying
- 树叶 → leaves swaying
- 草 → grass swaying
- 花 → flowers swaying

### 其他
- 风 → wind blowing
- 雾/雾气 → mist rising
- 火/火焰 → flames flickering
- 雪 → snow falling
- 鸟 → birds flying

## 🎯 关键改进

1. **自动识别场景元素**：从prompt中自动提取关键词，无需手动配置
2. **智能运动描述**：为每个场景元素添加最合适的运动描述
3. **优先级明确**：场景元素运动 > 运动强度 > 相机运动
4. **兜底机制**：如果未检测到特定元素，添加默认自然运动描述

## ⚠️ 注意事项

1. **关键词匹配**：使用简单的关键词匹配，可能无法识别所有场景元素
2. **运动描述语言**：使用英文运动描述（HunyuanVideo对英文理解更好）
3. **运动强度**：可以通过 `scene['motion_intensity']` 控制整体运动强度

## 📝 总结

- ✅ 自动识别场景元素（瀑布、云彩、彩虹等）
- ✅ 为每个元素添加相应的运动描述
- ✅ 确保视频中有物体运动，而不仅仅是相机运动
- ✅ 保持相机运动的灵活性（可通过配置控制）

现在生成的视频应该同时包含：
- **场景元素的运动**（瀑布流动、云彩飘动、彩虹闪烁等）
- **相机运动**（如果配置了camera_motion）

视频会更加生动自然！



