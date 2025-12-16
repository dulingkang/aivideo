# build_prompt 优化说明

## 优化目标

根据图像分析报告，优化 `build_prompt` 方法，解决以下问题：
1. **视角不匹配**：character_pose 指定了 "facing camera" 但生成了背面
2. **镜头距离问题**：期望特写/远景但实际不符
3. **角色检测问题**：很多场景角色未检测到

## 已实施的优化

### 1. 增强正面朝向处理（解决视角不匹配问题）

**位置**：`image_generator.py` 第 2401-2408 行

**优化内容**：
- 当 `character_pose` 包含 "facing camera" 时，提高权重从 `1.6` 到 `1.8`
- 添加更明确的正面朝向描述：`"character facing viewer"`
- 在负面提示构建中，强制添加防止背影的负面提示

**代码变更**：
```python
elif has_facing:
    # 优化：提高权重到1.8，确保正面朝向不被覆盖
    priority_parts.append("(facing camera, front view, face forward, character facing viewer:1.8)")
    
    # 在负面提示构建中，强制添加防止背影的负面提示
    enhanced_negative += ", back view, from behind, character back, rear view, turned away, facing away, back of head, back of character"
```

### 2. 优化镜头类型处理（解决镜头距离问题）

**位置**：`image_generator.py` 第 2488-2500 行

**优化内容**：
- 将镜头类型描述前置到第2位（在风格之后），确保高优先级
- 根据镜头类型（特写/远景）使用权重 `2.0`
- 在负面提示构建中，添加排除不想要的镜头类型的负面提示

**代码变更**：
```python
if is_eye_closeup:
    # 眼睛特写：最高权重，放在前面
    priority_parts.insert(1, f"({camera_prompt}:2.0)")  # 插入到第2位
elif is_wide_shot:
    # 远景：高权重，放在前面
    priority_parts.insert(1, f"({camera_prompt}:2.0)")
    # 在负面提示中添加：排除特写和中景
    enhanced_negative += ", close-up, extreme close-up, medium shot, mid shot"
elif is_close_shot:
    # 特写：高权重，放在前面
    priority_parts.insert(1, f"({camera_prompt}:2.0)")
    # 在负面提示中添加：排除远景
    enhanced_negative += ", wide shot, distant view, long shot, very long shot"
```

### 3. 增强负面提示构建（解决视角和镜头问题）

**位置**：`image_generator.py` 第 3766-3800 行

**优化内容**：
- 当 `character_pose` 包含 "facing camera" 时，强制添加防止背影的负面提示
- 当 `camera` 指定了特写/远景时，添加排除不想要的镜头类型的负面提示

**代码变更**：
```python
# 优化：根据分析报告，当character_pose包含"facing camera"时，强制添加防止背影的负面提示
visual = scene.get("visual", {})
if isinstance(visual, dict):
    character_pose = visual.get("character_pose", "")
    if character_pose:
        pose_lower = character_pose.lower()
        has_facing_camera = any(kw in pose_lower for kw in ["facing camera", "facing", "front view", "正面", "面向镜头"])
        if has_facing_camera:
            enhanced_negative += ", back view, from behind, character back, rear view, turned away, facing away, back of head, back of character"

# 优化：根据分析报告，当camera指定了特写/远景时，添加排除不想要的镜头类型的负面提示
camera_desc = scene.get("camera", "")
if camera_desc:
    camera_desc_lower = camera_desc.lower()
    is_wide_shot = any(kw in camera_desc_lower for kw in ['wide', 'distant', 'long shot', '远景', '远距离', '全景'])
    is_close_shot = any(kw in camera_desc_lower for kw in ['close-up', 'closeup', 'close up', '特写', '近景', 'extreme close'])
    
    if is_wide_shot:
        enhanced_negative += ", close-up, extreme close-up, medium shot, mid shot, 特写, 近景, 中景"
    elif is_close_shot:
        enhanced_negative += ", wide shot, distant view, long shot, very long shot, 远景, 远距离, 全景"
```

## 预期改进效果

### 1. 视角不匹配问题
- **优化前**：场景1和18，期望正面但生成了背面
- **优化后**：
  - 正面朝向权重从 `1.6` 提高到 `1.8`
  - 强制添加防止背影的负面提示
  - 预期：视角不匹配问题从2个场景减少到0个

### 2. 镜头距离问题
- **优化前**：场景7、8、10、16、18、20、21，期望特写/远景但实际不符
- **优化后**：
  - 镜头类型描述前置到第2位，权重提高到 `2.0`
  - 添加排除不想要的镜头类型的负面提示
  - 预期：镜头距离问题从7个场景减少到2-3个

### 3. 角色检测问题
- **优化前**：20个场景角色未检测到
- **优化后**：
  - 角色描述位置和权重保持不变（已在前面）
  - 通过优化视角和镜头，可能间接改善角色检测
  - 预期：需要进一步检查实际图像，确认是否真的包含角色

## 测试建议

1. **重新生成场景1和18**：验证视角不匹配问题是否解决
2. **重新生成场景7、8、10、16、18、20、21**：验证镜头距离问题是否解决
3. **重新运行图像分析**：验证改进效果
4. **检查场景2-10、12-17、19-22的实际图像**：确认是否真的包含角色

## 注意事项

1. **权重平衡**：提高权重可能会影响其他元素的生成，需要平衡
2. **Token限制**：前置镜头类型描述可能会占用更多token，需要确保不超过77个token
3. **负面提示长度**：添加负面提示可能会使负面提示过长，需要监控

## 后续优化方向

1. **角色检测优化**：如果确认角色确实存在但检测不到，需要进一步优化角色描述
2. **镜头类型检测**：如果镜头距离问题仍然存在，可能需要更精确的镜头类型检测
3. **权重动态调整**：根据场景类型动态调整权重，而不是固定值

