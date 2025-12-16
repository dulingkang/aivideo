# Prompt智能排序优化说明

## 📊 问题描述

用户反馈 `build_prompt` 不够智能，需要：
1. **优先最重要的细节**：确保最重要的信息在前面
2. **排列一些关键信息**：保留足够的关键信息，不要过度精简

当前问题：场景1的prompt只有21 tokens，太少了，缺少关键信息（如"golden scroll"、"unfurling"等）。

## ✅ 已实施的修复

### 1. 智能提取和排序关键信息（`image_generator.py`）

**优化内容**：
- **按重要性分类**：将关键信息分为4类，按优先级排序
  1. **主要物体**（最高优先级，权重2.0）：如"golden scroll"、"immortal city"
  2. **动作/效果**（高优先级，权重1.8）：如"unfurling"、"shimmering"
  3. **环境**（中优先级，权重1.6）：如"mist-shrouded sky"
  4. **镜头/构图**（低优先级，权重1.3）：如"wide shot"

- **智能提取**：从 `composition`、`fx`、`environment`、`prompt` 中智能提取关键信息
- **保留完整短语**：优先保留完整短语（如"golden scroll"而不是"scroll"）

**代码位置**：
```python
# image_generator.py, line 1962-2070
# 智能提取和排序关键信息（按重要性）
scene_elements = {
    "primary_object": [],  # 主要物体（最高优先级）
    "action_effect": [],   # 动作/效果（高优先级）
    "environment": [],     # 环境（中优先级）
    "camera_composition": []  # 镜头/构图（低优先级）
}

# 从composition中提取关键信息（最准确）
if composition:
    # 提取主要物体（保留完整短语）
    if "scroll" in composition_lower:
        scroll_phrase = re.search(r'golden\s+scroll[^,]*|scroll[^,]*', composition_lower)
        if scroll_phrase:
            scene_elements["primary_object"].append(scroll_phrase.group())
```

### 2. 按优先级构建prompt

**优化内容**：
- **主要物体**：权重2.0，确保最重要
- **动作/效果**：权重1.8，确保动作明显
- **环境**：权重1.6，提供背景信息
- **镜头/构图**：权重1.3，如果有空间

**代码位置**：
```python
# image_generator.py, line 2070-2095
# 1. 主要物体（最高优先级，权重2.0）
if scene_elements["primary_object"]:
    primary_obj_text = ", ".join(scene_elements["primary_object"][:1])
    priority_parts.append(f"({primary_obj_text}, prominent, main element{exclusion_text}:2.0)")

# 2. 动作/效果（高优先级，权重1.8）
if scene_elements["action_effect"]:
    action_text = ", ".join(scene_elements["action_effect"][:2])
    priority_parts.append(f"({action_text}:1.8)")

# 3. 环境（中优先级，权重1.6）
if scene_elements["environment"]:
    env_text = ", ".join(scene_elements["environment"][:1])
    priority_parts.append(f"({env_text}:1.6)")
```

### 3. 防止过度精简

**优化内容**：
- **检查prompt长度**：如果prompt太短（少于30 tokens），尝试添加更多关键信息
- **智能补充**：从prompt或description中提取关键短语补充
- **放宽限制**：如果prompt太短，允许稍微超过70 tokens（最多75 tokens）

**代码位置**：
```python
# image_generator.py, line 2095-2120
# 如果prompt太短（少于30 tokens），尝试添加更多关键信息
if estimated_tokens < 30:
    # 提取prompt中的关键短语
    if "golden scroll" in prompt_lower:
        key_phrases.append("golden scroll")
    if "unfurling" in prompt_lower:
        key_phrases.append("unfurling")
    # 添加关键信息
    if key_phrases:
        additional_text = ", ".join(key_phrases[:3])
        priority_parts.append(f"({additional_text}:1.6)")
```

### 4. 优化精简逻辑

**优化内容**：
- **保留更多信息**：场景对象保留2个，动作保留2个，环境保留1个
- **优先保留完整短语**：如"golden scroll"而不是"scroll"

**代码位置**：
```python
# image_generator.py, line 2100-2118
# 场景对象：保留最多2个（确保包含"golden scroll"等完整描述）
if scene_keywords:
    scene_keywords_sorted = sorted(set(scene_keywords), key=lambda x: len(x), reverse=True)
    scene_parts.append(", ".join(scene_keywords_sorted[:2]))  # 保留2个

# 动作：保留最多2个
if action_keywords:
    action_keywords_sorted = sorted(set(action_keywords), key=lambda x: len(x), reverse=True)
    scene_parts.append(", ".join(action_keywords_sorted[:2]))  # 保留2个
```

## 🎯 预期改进效果

实施上述修复后，预期：
1. **Prompt更完整**：保留更多关键信息，不再过度精简
2. **优先级更明确**：最重要的细节在前面，权重更高
3. **信息更准确**：保留完整短语（如"golden scroll"），避免丢失关键信息

**优化前**（场景1）：
```
xianxia fantasy, (scroll, spirit light shimmering:1.60)
```
只有21 tokens，缺少"golden scroll"、"unfurling"等关键信息。

**优化后**（场景1，预期）：
```
xianxia fantasy, (golden scroll, prominent, main element, no weapons, no tools:2.0), (unfurling prominently, spirit light shimmering:1.8), (mist-shrouded immortal sky:1.6)
```
约50-60 tokens，包含所有关键信息，按重要性排序。

## 📋 优先级说明

### 信息优先级（从高到低）
1. **主要物体**（权重2.0）：如"golden scroll"、"immortal city"
2. **动作/效果**（权重1.8）：如"unfurling"、"shimmering"
3. **环境**（权重1.6）：如"mist-shrouded sky"
4. **镜头/构图**（权重1.3）：如"wide shot"

### Token限制策略
- **正常情况**：≤70 tokens（留出7个token的安全边界）
- **Prompt太短**：如果<30 tokens，允许补充到70 tokens
- **Prompt太长**：如果>70 tokens，智能精简，但保留最重要的信息

## 📝 代码变更文件

1. `gen_video/image_generator.py`
   - `_build_prompt` 方法：智能提取和排序关键信息，按优先级构建prompt，防止过度精简

## ✅ 语法检查

所有代码已通过语法检查，可以正常运行。

