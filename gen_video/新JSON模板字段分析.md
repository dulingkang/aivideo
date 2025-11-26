# 新 JSON 模板字段分析

## 📋 模板结构

```json
{
  "episode": 5,
  "title": "青罗沙漠·初至遗迹",
  "opening": { "duration": 6, "narration": "..." },
  "scenes": [
    {
      "id": 1,
      "duration": 5,
      "description": "...",
      "mood": "serious",
      "lighting": "day",
      "action": "walking_forward",
      "camera": "wide_shot_low_angle",
      "visual": { ... },
      "prompt": "...",
      "narration": "...",
      "face_style_auto": { ... }
    }
  ],
  "ending": { "duration": 5, "narration": "..." }
}
```

## 🔍 字段详细分析

### 1. `duration` 字段

**位置：** `opening.duration`, `scenes[].duration`, `ending.duration`

**用途：**
- 控制每个场景/片段的时长（秒）
- 用于计算视频生成帧数：`frames = duration * fps`
- 用于 FFmpeg 合成时控制每个片段的时长

**当前支持：** ⚠️ 部分支持
- `video_generator.py` 中有 `num_frames` 配置，但没有从 `duration` 计算
- `video_composer.py` 中可以使用 `duration` 控制片段时长

**建议实现：**
```python
# 在 video_generator.py 中
def generate_video(self, image_path, output_path, scene=None):
    if scene and scene.get("duration"):
        duration = scene["duration"]
        fps = self.video_config.get("fps", 12)
        num_frames = int(duration * fps)
    else:
        num_frames = self.video_config.get("num_frames", 20)
    # ... 使用 num_frames 生成视频
```

---

### 2. `visual` 字段

**位置：** `scenes[].visual`

**结构：**
```json
{
  "composition": "Han Li small silhouette vs vast golden desert",
  "environment": "rolling sand waves, intense sunlight, heat distortion",
  "character_pose": "steady forward walk, robe slightly fluttering",
  "fx": "subtle heat haze, drifting sand",
  "motion": "slow dolly-forward shot"
}
```

#### 2.1 `composition` - 构图描述

**含义：** 画面的整体构图，描述主体与背景的关系

**示例：**
- `"Han Li small silhouette vs vast golden desert"` - 韩立小剪影 vs 广阔沙漠
- `"close focus on Han Li's expression"` - 聚焦韩立表情

**用途：**
- 生成更准确的 prompt
- 控制镜头景别（远景/中景/近景）
- 指导构图平衡

**当前支持：** ❌ 未使用

**建议实现：**
```python
# 在 build_prompt 中
if scene.get("visual", {}).get("composition"):
    composition = scene["visual"]["composition"]
    prompt_parts.append(f"({composition}:1.2)")
```

#### 2.2 `environment` - 环境描述

**含义：** 场景环境的详细描述，包括天气、光线、氛围等

**示例：**
- `"rolling sand waves, intense sunlight, heat distortion"` - 翻滚沙浪，强烈阳光，热扭曲
- `"still desert, heat haze vibrating subtly"` - 静止沙漠，热浪轻微振动

**用途：**
- 补充环境 prompt
- 控制环境特效
- 生成更真实的环境

**当前支持：** ⚠️ 部分支持（通过 `description` 字段）

**建议实现：**
```python
# 在 build_prompt 中
if scene.get("visual", {}).get("environment"):
    env_visual = scene["visual"]["environment"]
    prompt_parts.append(env_visual)
```

#### 2.3 `character_pose` - 角色姿势

**含义：** 角色的具体姿势和动作状态

**示例：**
- `"steady forward walk, robe slightly fluttering"` - 稳步前行，长袍轻微飘动
- `"standing still, eyes slightly narrowed"` - 静止站立，眼睛微眯

**用途：**
- 生成更准确的角色动作
- 控制角色姿态
- 指导 OpenPose/动作模板（如果使用）

**当前支持：** ⚠️ 部分支持（通过 `action` 字段）

**建议实现：**
```python
# 在 build_prompt 中
if scene.get("visual", {}).get("character_pose"):
    pose = scene["visual"]["character_pose"]
    prompt_parts.append(f"({pose}:1.1)")
```

#### 2.4 `fx` - 特效

**含义：** 视觉特效，如粒子效果、光效、扭曲等

**示例：**
- `"subtle heat haze, drifting sand"` - 轻微热浪，飘沙
- `"sand particles drifting, faint energy ripple effect on ground"` - 沙粒飘动，地面微弱能量波纹

**用途：**
- 生成特效层
- 控制后期特效
- 增强画面氛围

**当前支持：** ❌ 未使用

**建议实现：**
```python
# 在 build_prompt 中
if scene.get("visual", {}).get("fx"):
    fx = scene["visual"]["fx"]
    prompt_parts.append(f"({fx}:0.9)")  # 特效权重稍低，避免过度
```

#### 2.5 `motion` - 镜头运动

**含义：** 摄像机的运动方式

**示例：**
- `"slow dolly-forward shot"` - 缓慢推镜
- `"gentle push-in to face"` - 轻柔推向面部

**用途：**
- 控制镜头运动（pan, dolly, push-in）
- 指导视频生成的运动幅度
- 用于 FFmpeg 后期处理

**当前支持：** ⚠️ 部分支持（通过 `camera` 字段）

**建议实现：**
```python
# 在 build_prompt 中
if scene.get("visual", {}).get("motion"):
    motion = scene["visual"]["motion"]
    # 可以转换为 camera prompt
    camera_prompt = self._convert_motion_to_camera(motion)
    prompt_parts.append(f"({camera_prompt}:1.2)")
```

---

### 3. `face_style_auto` 字段

**位置：** `scenes[].face_style_auto`

**结构：**
```json
{
  "expression": "focused",
  "lighting": "bright_normal",
  "detail": "natural"
}
```

#### 3.1 `expression` - 表情

**含义：** 角色的面部表情

**可能值：**
- `"focused"` - 专注
- `"serious"` - 严肃
- `"calm"` - 平静
- `"alert"` - 警觉
- `"determined"` - 坚定
- 等等

**用途：**
- 控制 InstantID 生成的表情
- 调整面部权重
- 生成更符合剧情的表情

**当前支持：** ❌ 未使用

**建议实现：**
```python
# 在 build_prompt 中
if scene.get("face_style_auto", {}).get("expression"):
    expression = scene["face_style_auto"]["expression"]
    prompt_parts.append(f"({expression} expression:1.1)")

# 在 InstantID 生成中调整面部权重
if expression == "focused" or expression == "alert":
    # 提高面部权重，确保表情清晰
    ip_adapter_scale *= 1.1
```

#### 3.2 `lighting` - 光照

**含义：** 面部的光照条件

**可能值：**
- `"bright_normal"` - 明亮正常
- `"soft"` - 柔和
- `"dramatic"` - 戏剧性
- `"rim_light"` - 边缘光
- 等等

**用途：**
- 控制面部光照效果
- 调整面部可见度
- 生成更符合场景的光照

**当前支持：** ❌ 未使用

**建议实现：**
```python
# 在 build_prompt 中
if scene.get("face_style_auto", {}).get("lighting"):
    lighting = scene["face_style_auto"]["lighting"]
    prompt_parts.append(f"({lighting} lighting on face:0.9)")
```

#### 3.3 `detail` - 细节级别

**含义：** 面部细节的详细程度

**可能值：**
- `"natural"` - 自然
- `"soft_concentrated"` - 柔和聚焦
- `"detailed"` - 详细
- `"subtle"` - 微妙
- 等等

**用途：**
- 控制面部细节程度
- 调整面部权重
- 控制面部在画面中的重要性

**当前支持：** ❌ 未使用

**建议实现：**
```python
# 在 InstantID 生成中
if scene.get("face_style_auto", {}).get("detail"):
    detail = scene["face_style_auto"]["detail"]
    if detail == "detailed":
        # 提高面部权重
        ip_adapter_scale *= 1.15
        face_kps_scale *= 1.1
    elif detail == "subtle":
        # 降低面部权重
        ip_adapter_scale *= 0.9
        face_kps_scale *= 0.9
```

---

## 🚀 实现建议

### 优先级 1：立即实现

1. **`duration` → `num_frames` 计算**
   - 在 `video_generator.py` 中根据 `duration` 计算帧数
   - 在 `video_composer.py` 中使用 `duration` 控制片段时长

2. **`visual.composition` 和 `visual.environment`**
   - 在 `build_prompt` 中优先使用这些字段
   - 比 `description` 更精确

3. **`visual.motion` → `camera` 转换**
   - 将 `motion` 转换为 camera prompt
   - 与现有的 `camera` 字段合并使用

### 优先级 2：后续实现

1. **`visual.character_pose`**
   - 用于生成更准确的动作
   - 可以指导 OpenPose（如果使用）

2. **`visual.fx`**
   - 用于特效层生成
   - 可以用于后期处理

3. **`face_style_auto` 所有字段**
   - 用于精确控制面部生成
   - 调整 InstantID 参数

## ❓ 需要确认的问题

1. **`face_style_auto.expression` 的完整值列表？**
   - 目前看到：`focused`, `serious`, `calm`, `alert`, `determined`
   - 还有其他值吗？

2. **`face_style_auto.lighting` 的完整值列表？**
   - 目前看到：`bright_normal`
   - 还有其他值吗？

3. **`face_style_auto.detail` 的完整值列表？**
   - 目前看到：`natural`, `soft_concentrated`
   - 还有其他值吗？

4. **`visual.motion` 的格式？**
   - 是自由文本还是固定格式？
   - 需要解析为具体的镜头运动参数吗？

5. **`action` 字段与 `visual.character_pose` 的关系？**
   - 两者有什么区别？
   - 应该优先使用哪个？

