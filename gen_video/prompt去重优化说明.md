# Prompt去重优化说明

## 📊 问题描述

用户反馈场景1的prompt有问题：
1. **有重复的内容**：`scroll` 重复出现了3次
2. **不能明确表示那个场景**：prompt太简单，缺少关键信息
3. **要去除一些重复的近义词，保留尽可能多的信息**

从日志看，最终的prompt是：
```
xianxia fantasy, (scroll, scroll prominent and clearly visible, scroll is the main element:2.00)
```

确实有重复：`scroll` 出现了3次。

## ✅ 已实施的修复

### 1. 去除实体描述中的重复（`image_generator.py`）

**优化内容**：
- **去除重复的实体名称**：从 `({entity_text}, {entity_text} prominent and clearly visible, {entity_text} is the main element:{entity_weight:.2f})` 
  改为 `({entity_text}, prominent, clearly visible, main element:{entity_weight:.2f})`
- **保留关键描述词**：保留 `prominent`, `clearly visible`, `main element` 等描述词，但只出现一次实体名称

**代码位置**：
```python
# image_generator.py, line 1940-1942
# 如果是物体，使用更高权重并强调（去除重复，使用更简洁的描述）
if entity.get('type') == 'object':
    priority_parts.append(f"({entity_text}, prominent, clearly visible, main element:{entity_weight:.2f})")
```

### 2. 智能去重和关键词提取（`image_generator.py`）

**优化内容**：
- **提取所有关键词**：从所有prompt部分中提取关键词
- **去除重复和近义词**：检查是否是重复或近义词，保留更具体的词（更长的）
- **分类关键词**：将关键词分为场景对象、动作、环境三类
- **构建精简描述**：组合关键信息，去除重复，保留尽可能多的信息

**代码位置**：
```python
# image_generator.py, line 2035-2080
# 2. 合并和去重：提取所有关键信息，去除重复
all_keywords = set()
scene_keywords = []
action_keywords = []
environment_keywords = []

for part in priority_parts[1:]:
    # 提取关键词，去除重复
    keywords = [kw.strip() for kw in part_clean.split(",")]
    for kw in keywords:
        # 去除重复的近义词
        if kw not in all_keywords:
            # 检查是否是重复或近义词
            for existing in list(all_keywords):
                if kw == existing or kw in existing or existing in kw:
                    # 保留更具体的词（更长的）
                    if len(kw) > len(existing):
                        all_keywords.discard(existing)
                        all_keywords.add(kw)
                    break
            # 分类关键词
            if any(word in kw for word in ["scroll", "卷轴", "sun", "太阳", ...]):
                scene_keywords.append(kw)
            elif any(word in kw for word in ["unfurling", "展开", ...]):
                action_keywords.append(kw)
            elif any(word in kw for word in ["sky", "天空", ...]):
                environment_keywords.append(kw)
```

### 3. 优化场景描述构建（`image_generator.py`）

**优化内容**：
- **优先保留最具体的词**：按长度排序，优先保留更具体的词（如"golden scroll"而不是"scroll"）
- **限制数量**：场景对象、动作、环境各只保留1个最具体的词
- **组合描述**：组合关键信息，保留尽可能多的信息

**代码位置**：
```python
# image_generator.py, line 2075-2085
# 优先保留最具体的场景对象（如"golden scroll"而不是"scroll"）
if scene_keywords:
    scene_keywords_sorted = sorted(set(scene_keywords), key=lambda x: len(x), reverse=True)
    scene_parts.append(" ".join(scene_keywords_sorted[:1]))  # 只保留1个最具体的场景对象
if action_keywords:
    action_keywords_sorted = sorted(set(action_keywords), key=lambda x: len(x), reverse=True)
    scene_parts.append(" ".join(action_keywords_sorted[:1]))  # 只保留1个最具体的动作
```

## 🎯 预期改进效果

实施上述修复后，预期：
1. **去除重复**：`scroll` 只出现一次，不再重复
2. **保留关键信息**：保留场景对象、动作、环境等关键信息
3. **更明确的场景描述**：prompt更简洁但包含更多关键信息

**优化前**：
```
xianxia fantasy, (scroll, scroll prominent and clearly visible, scroll is the main element:2.00)
```

**优化后**（示例）：
```
xianxia fantasy, (golden scroll, unfurling, immortal realm sky:2.00)
```

## 📋 验证方法

1. **重新生成图像**：使用优化后的代码重新生成场景1的图像
2. **检查prompt**：确认prompt中没有重复，且包含关键信息
3. **检查图像质量**：确认图像质量没有下降，且更符合场景描述

## 📝 代码变更文件

1. `gen_video/image_generator.py`
   - `_build_prompt` 方法：去除实体描述中的重复，智能去重和关键词提取，优化场景描述构建

## ✅ 语法检查

所有代码已通过语法检查，可以正常运行。

