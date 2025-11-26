# Visual 字段优化方案

## 当前问题分析

1. **重复问题**：
   - `scene_profiles.yaml` 中的 `background_prompt`（包含 color_palette, terrain, sky, atmosphere）
   - JSON 中的 `visual.environment`
   - 两者都描述环境背景，存在重复

2. **空字段问题**：
   - `visual.environment` 有30%为空
   - `visual.fx` 有50%为空
   - `visual.character_pose` 有40%为空（无人物场景）

## 推荐方案：职责明确化

### 字段职责划分

#### 1. **scene_profiles.yaml**（场景模板级别）
**职责**：场景类型的整体环境风格
- `color_palette`：颜色调色板
- `terrain`：地形地貌
- `sky`：天空描述
- `atmosphere`：氛围描述
- `background_prompt`：组合后的场景背景模板

**使用场景**：适用于整个场景类型（如"青罗沙漠"的所有场景）

#### 2. **JSON visual.environment**（场景实例级别）
**职责**：**仅用于特殊环境细节**（与模板不同的部分）
- 如果场景环境与模板一致 → **留空**（由 YAML 统一管理）
- 如果场景有特殊环境变化 → **填写特殊部分**（如"突然出现的怪鸟"、"特殊的天空现象"）

**示例**：
- 场景1（沙漠场景）：留空（由 YAML 的"青罗沙漠"模板管理）
- 场景2（天空特写）：填写"三个夺目的太阳，四个朦胧的月亮虚影"（这是特殊天空现象）
- 场景10（出现怪鸟）：填写"鹰首蝠身的黑色怪鸟"（这是特殊环境元素）

#### 3. **JSON visual.composition**
**职责**：整体构图（主体+背景关系）
- 必须填写
- 描述画面布局和主体位置

#### 4. **JSON visual.character_pose**
**职责**：角色动作/姿势/表情
- 有角色的场景：必须填写
- 无角色的场景：留空

#### 5. **JSON visual.fx**
**职责**：视觉特效（光芒、声音效果、粒子等）
- 有特效的场景：填写
- 无特效的场景：留空（这是正常的）

#### 6. **JSON visual.motion**
**职责**：镜头运动方式
- 必须填写（用于视频生成和运镜）

## 优化后的字段结构

```json
{
  "id": 1,
  "description": "韩立躺在青灰色沙地上，一动不动，感受着地面的燥热。",
  "visual": {
    "composition": "韩立躺在青灰色沙地上",
    "environment": "",  // 留空，由 scene_profiles.yaml 的"青罗沙漠"模板管理
    "character_pose": "一动不动，感受着地面的燥热",
    "fx": "",  // 留空，无特效
    "motion": "缓慢俯视平移展现广阔沙漠"
  }
}
```

```json
{
  "id": 2,
  "description": "天空中出现三个夺目的太阳和四个朦胧的月亮虚影。",
  "visual": {
    "composition": "天空中出现三个夺目的太阳和四个朦胧的月亮虚影",
    "environment": "三个夺目的太阳，四个朦胧的月亮虚影",  // 特殊天空现象，需要特别强调
    "character_pose": "",  // 无角色
    "fx": "",  // 无特效
    "motion": "缓慢平移天空展现七个天体"
  }
}
```

```json
{
  "id": 14,
  "description": "韩立胸膛一鼓，喷出强风，沙砾化为青芒激射而出。",
  "visual": {
    "composition": "韩立胸膛一鼓，喷出强风",
    "environment": "",  // 留空，由模板管理
    "character_pose": "韩立胸膛一鼓",
    "fx": "喷出强风，沙砾化为青芒激射而出",  // 有特效
    "motion": "投射物发射动态镜头"
  }
}
```

## 实施建议

1. **清理 visual.environment**：
   - 如果环境与模板一致 → 清空
   - 如果环境有特殊细节 → 保留特殊部分

2. **保持 visual.fx 可以为空**：
   - 这是正常的，只在有特效时填写

3. **保持 visual.character_pose 可以为空**：
   - 无人物场景时留空，这是正常的

4. **代码层面**：
   - 图像生成时，优先使用 `scene_profiles.yaml` 的 `background_prompt`
   - 如果 `visual.environment` 有内容，作为补充添加
   - 这样避免重复，职责清晰

