# Scene_007 和 Scene_012 问题分析报告

## 问题概述

1. **scene_007.png**: 出现了坦克（不正常）
2. **scene_012.png**: 出现了10个一样的人（不正常）

## 详细分析

### Scene_007 问题分析

#### 期望的场景描述
- **character_pose**: Head tilted,face partially visible,facing camera angle
- **composition**: Han Li strains to tilt his head,sees endless gray-green gravel
- **camera**: Wide pan
- **action**: Turning head

#### 实际使用的Prompt
```
(warm orange tones:1.2), (vast golden desert:1.3), (male:1.8), xianxia fantasy, (gravel:1.70), (strains, tilt, endless:1.8), (Straining to tilt head, facing camera, front view:1.8)
```

#### 问题根源

1. **智能优化移除了单人约束**
   - 日志显示：`✓ 人物场景：在prompt最前面添加单人约束（权重2.0，防止多个人物）`
   - 但智能优化后：`🧠 智能优化: 从 15 个部分精简至 6 个部分`
   - **单人约束在优化过程中被移除了！**

2. **缺少明确的排除项**
   - Prompt中没有 `no vehicles, no tanks, no military equipment` 等排除项
   - "gravel"（沙砾）可能被模型误解为军事场景

3. **Prompt优化器的问题**
   - 优化器在精简prompt时，可能将单人约束视为"低重要性"内容而移除
   - 需要确保单人约束在优化时具有最高优先级，不能被移除

### Scene_012 问题分析

#### 期望的场景描述
- **composition**: Monstrous birds dive to thirty-plus zhang above ground,targeting Han Li
- **camera**: Low-angle birds
- **action**: 应该是韩立面对怪鸟的场景

#### 实际使用的Prompt
```
(warm orange tones:1.2), (vast golden desert:1.3), (male:1.8), xianxia fantasy, (Monstrous birds dive to thirty-plus zhang above ground,targeting Han Li:1.80), (consistent background:1.2)
```

#### 问题根源

1. **智能优化移除了单人约束**
   - 日志显示：`✓ 人物场景：在prompt最前面添加单人约束（权重2.0，防止多个人物）`
   - 但智能优化后：`🧠 智能优化: 从 12 个部分精简至 5 个部分`
   - **单人约束在优化过程中被移除了！**

2. **Prompt中没有单人约束**
   - 最终prompt中完全没有 `single person, only one character` 等约束
   - 导致模型生成了多个人物

3. **场景描述可能被误解**
   - "Monstrous birds dive... targeting Han Li" 可能被理解为"多个人被鸟攻击"
   - 需要更明确地强调"只有韩立一个人"

## 核心问题

### Prompt优化器的缺陷

**问题**：`PromptOptimizer` 在精简prompt时，将单人约束视为可移除的内容。

**证据**：
- scene_007: 从15个部分精简到6个部分，单人约束被移除
- scene_012: 从12个部分精简到5个部分，单人约束被移除

**影响**：
- 所有人物场景都可能出现多人问题
- 关键约束在优化时被错误移除

## 修复方案

### 1. 修复Prompt优化器（优先级：高）

**位置**：`gen_video/prompt/optimizer.py`

**修改**：
- 在优化时，将单人约束标记为"不可移除"（最高优先级）
- 确保单人约束始终保留在最终prompt中
- 添加保护机制：如果检测到人物场景，强制保留单人约束

**代码修改建议**：
```python
# 在优化器中添加保护列表
PROTECTED_KEYWORDS = [
    "single person", "only one character", "lone figure",
    "单人", "只有一个角色", "独行"
]

def optimize(self, parts, max_tokens=70):
    # 先识别并保护单人约束
    protected_parts = []
    for part in parts:
        if any(kw in part.lower() for kw in PROTECTED_KEYWORDS):
            protected_parts.append(part)
    
    # 优化其他部分
    optimized = self._optimize_other_parts(parts, protected_parts, max_tokens)
    
    # 确保单人约束在最前面
    return protected_parts + optimized
```

### 2. 增强Negative Prompt（优先级：高）

**位置**：`gen_video/image_generator.py`

**修改**：
- 对于所有人物场景，在negative prompt中添加：
  - `multiple people, crowd, group, many characters, duplicate characters`
  - `vehicles, tanks, military equipment, modern technology`（针对scene_007）

### 3. 修复Scene_007的Prompt（优先级：中）

**问题**：`gravel` 可能被误解

**修复**：
- 将 `(gravel:1.70)` 改为更明确的描述：`(gray-green sand ground:1.70)`
- 添加排除项：`no vehicles, no tanks, no military equipment`

### 4. 修复Scene_012的Prompt（优先级：中）

**问题**：场景描述可能暗示多人

**修复**：
- 明确强调：`(Han Li alone, single person, only one character:2.5)`
- 在composition中强调：`Han Li (single person) sees monstrous birds diving...`

## 修复完成 ✅

### 已完成的修复

1. **修复Prompt优化器** ✅
   - 在 `_infer_part_type` 方法中添加了 "constraint" 类型检测
   - 单人约束现在会被识别为 "constraint" 类型
   - 在 `_analyze_importance` 中，constraint 类型的重要性设置为 20.0（最高优先级）
   - 在 `_select_parts` 中，constraint 类型被添加到 `must_keep_types` 列表
   - 确保约束条件始终在最前面，不会被优化掉

2. **修改位置**
   - 文件：`gen_video/prompt/optimizer.py`
   - 修改了三个方法：
     - `_infer_part_type()`: 添加约束类型检测
     - `_analyze_importance()`: 设置约束类型最高重要性
     - `_select_parts()`: 确保约束条件被保留

### 下一步操作

1. **重新生成scene_007和scene_012**
   ```bash
   cd /vepfs-dev/shawn/vid/fanren/gen_video
   source /vepfs-dev/shawn/venv/py312/bin/activate
   # 重新运行图像生成，只生成这两个场景
   ```

2. **验证修复效果**
   - 检查 scene_007 是否还有坦克
   - 检查 scene_012 是否只有一个人
   - 检查其他人物场景是否也正确（单人）

## 验证方法

生成后检查：
1. scene_007是否还有坦克
2. scene_012是否只有一个人
3. 其他人物场景是否也正确（单人）

## 总结

**根本原因**：Prompt优化器在精简prompt时，错误地移除了关键的单人约束，导致：
- scene_007: 可能因为缺少约束和排除项，生成了坦克
- scene_012: 因为缺少单人约束，生成了10个一样的人

**解决方案**：
1. 修复优化器，保护单人约束不被移除
2. 增强negative prompt
3. 改进场景描述，更明确地强调单人

