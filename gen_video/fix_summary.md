# 修复总结

## 已修复的问题

### 1. Scene_002 完全不对的问题 ✅（关键修复）

**问题**：scene_002.png 完全不对，角色未检测到，出现了飞船而不是韩立躺在沙地上

**根本原因**：
- 优化器在精简prompt时，移除了关键的composition描述 `Han Li lying on gray-green sand`
- 只保留了简单的标记 `(male:1.8)`，这个标记也被错误地识别为composition类型
- 导致最终prompt中完全没有"lying"（躺着）的动作描述

**修复**：
1. **改进了类型识别**：
   - 区分真正的composition描述（包含动作和场景，如"Han Li lying on gray-green sand"）
   - 和简单的标记（如"(male:1.8)"）
   - 真正的composition描述会被优先保留

2. **优化了保留优先级**：
   - 约束条件 > 真正的composition描述 > 其他 > 简单标记
   - 确保包含关键动作的composition描述不会被简单标记替代

3. **增强了关键动作检测**：
   - 检测"lying"、"sees"、"strains"等关键动作
   - 包含这些动作的composition描述必须完整保留，即使超过token限制

### 2. Scene_002 出现飞船问题 ✅

**问题**：scene_002.png 出现了飞船，没有人躺着

**原因**：
- 优化器在精简prompt时，移除了关键的composition描述 `Han Li lying on gray-green sand`
- 最终prompt中只有单人约束、背景、风格，但**缺少了关键的动作描述**（lying）

**修复**：
1. 修复了优化器，确保包含关键动作的composition描述不被精简
   - 在 `_select_parts` 方法中，检测composition类型且包含关键动作（lying, sees, strains等）的描述
   - 这些描述必须完整保留，即使超过token限制也要保留
2. 添加了现代交通工具和飞船的排除项到negative prompt

### 2. Scene_012 出现多个相同的人问题 ✅

**问题**：scene_012.png 还是出现了很多一样的人

**原因**：
- 虽然单人约束被保留了，但negative prompt中的多人排除项可能不够强烈
- 需要更强烈的多人排除项

**修复**：
1. 增强了negative prompt，对所有人物场景都添加强化的多人排除项
   - 添加了 `(duplicate person:2.0), (same person twice:2.0), (ten people:2.0)` 等高权重排除项
   - 权重提高到2.0，确保模型不会生成多个相同的人

### 3. Scene_004 和 Scene_005 完全不对的问题 ✅（关键修复）

**问题**：场景3（scene_004）和场景4（scene_005）完全不对

**根本原因**：
- **场景4**：期望"Close-up on face"（面部特写），但被错误转换为"medium shot"（中景）
- 代码逻辑中，除了眼睛特写，所有特写都会被转换为中景
- 但"Close-up on face"（面部特写）应该保持特写，不应该转换为中景

**修复**：
1. **添加了面部特写识别**：
   - 区分眼睛特写、面部特写和其他特写
   - 面部特写（"Close-up on face"、"portrait"、"headshot"等）现在会保持特写，不转换为中景
   - 添加了 `is_face_closeup` 标记

2. **修复了镜头类型转换逻辑**：
   - 眼睛特写：保持特写
   - 面部特写：保持特写（新增）
   - 其他特写：转换为中景

3. **增强了面部特写描述**：
   - 添加了 `(close-up on face:2.0)` 和 `(portrait shot, headshot, clear facial expression:1.8)`
   - 确保面部特写场景能正确表达

### 4. Scene_007 出现坦克问题 ✅

**问题**：scene_007.png 出现了坦克

**原因**：
- negative prompt中缺少现代交通工具的排除项

**修复**：
1. 添加了现代交通工具和飞船的排除项到negative prompt
   - 包括：vehicle, tank, spaceship, military vehicle, modern technology等
   - 权重1.8，确保不会生成不符合仙侠风格的现代元素

## 修改的文件

1. **gen_video/prompt/optimizer.py**（关键修复）
   - 改进了 `_infer_part_type` 方法，区分真正的composition描述和简单标记
   - 修复了 `_select_parts` 方法，确保包含关键动作的composition描述不被精简
   - 添加了关键动作检测逻辑（lying, sees, strains等）
   - 优化了保留优先级：真正的composition描述优先于简单标记

2. **gen_video/prompt/builder.py**（关键修复）
   - 添加了面部特写识别（`is_face_closeup` 标记）
   - 修复了镜头类型转换逻辑：面部特写现在会保持特写，不转换为中景
   - 增强了面部特写描述，确保"Close-up on face"场景能正确表达
   - 在 `_convert_camera_to_prompt` 方法中添加了面部特写检测

3. **gen_video/image_generator.py**
   - 增强了negative prompt构建逻辑
   - 对所有人物场景添加强化的多人排除项（权重2.0）
   - 添加了现代交通工具和飞船的排除项（权重1.8）

## 下一步

1. **重新生成scene_002和scene_012**，验证修复效果
2. **检查其他场景**，确保没有类似问题
3. **如果还有问题**，可以进一步调整权重或添加更多排除项

## 验证方法

生成后检查：
- **scene_002**: 应该看到韩立躺在沙地上，没有飞船
- **scene_004**: 应该是面部特写（close-up on face），显示韩立阴沉的表情，不是中景
- **scene_005**: 应该是中景，显示韩立痛苦的表情
- **scene_012**: 应该只有一个人，没有多个相同的人
- **scene_007**: 应该没有坦克或其他现代交通工具

