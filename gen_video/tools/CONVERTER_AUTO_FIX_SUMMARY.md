# 转换工具自动修正功能总结

## 问题描述

转换后的 JSON 文件存在不合法的 shot+pose 组合（如 `wide + lying`），导致 JSON 校验失败。

## 修复方案

在转换工具中添加自动修正逻辑，使用规则引擎自动修正不合法组合。

## 修复内容

### 1. V1 到 v2.2-final 转换器 (`utils/v1_to_v22_converter.py`)

**修复前**:
- 只检测兼容性，不自动修正
- `shot_pose_compatible` 可能为 `false`
- `pose.auto_corrected` 始终为 `false`

**修复后**:
- 使用 `rules.validate_pose()` 自动修正不合法组合
- 当检测到不合法组合时，自动修正为合法组合
- 更新 `pose.type`、`pose.auto_corrected`、`pose.validated_by` 字段
- 输出修正信息到控制台

### 2. V2 到 v2.2-final 转换器 (`utils/v2_to_v22_converter.py`)

**修复前**:
- 只检测兼容性，不自动修正
- `shot_pose_compatible` 可能为 `false`
- `pose.auto_corrected` 始终为 `false`

**修复后**:
- 使用 `rules.validate_pose()` 自动修正不合法组合
- 当检测到不合法组合时，自动修正为合法组合
- 更新 `pose.type`、`pose.auto_corrected`、`pose.validated_by` 字段
- 输出修正信息到控制台

## 修正规则

根据 `execution_rules_v2_1.py` 中的规则：

### Shot → Pose 允许表

| Shot | 允许的 Pose | 禁止的 Pose |
|------|------------|------------|
| WIDE | STAND, WALK | LYING, KNEEL, SIT |
| MEDIUM | STAND, WALK, LYING, SIT, KNEEL | - |
| CLOSE_UP | FACE_ONLY | LYING |
| AERIAL | STAND, WALK | LYING, KNEEL, SIT |

### 修正示例

1. **wide + lying** → **wide + stand**
   - 原因: `WIDE` 禁止 `LYING`
   - 修正: 使用允许列表中的第一个 `STAND`

2. **close_up + lying** → **close_up + face_only**
   - 原因: `CLOSE_UP` 禁止 `LYING`
   - 修正: 使用允许列表中的 `FACE_ONLY`

## 转换后的 JSON 字段

### Pose 字段更新

```json
{
  "pose": {
    "type": "stand",  // 修正后的类型
    "locked": true,
    "validated_by": "shot_pose_rules",  // 如果被修正，使用 "shot_pose_rules"
    "auto_corrected": true,  // 如果被修正，设为 true
    "description": "standing pose, upright posture"
  }
}
```

### Validation 字段更新

```json
{
  "validation": {
    "shot_pose_compatible": true,  // 修正后应该为 true
    "model_route_valid": true,
    "character_anchor_complete": true,
    "prompt_template_valid": true
  }
}
```

## 使用示例

### 转换 V1 文件

```bash
cd gen_video
python3 utils/v1_to_v22_converter.py ../lingjie/episode/1.json --output-dir ../lingjie/v22
```

**输出示例**:
```
⚠ [Level 1] Pose lying 在 Shot wide 中不合法，自动修正为 stand
⚠ 自动修正: wide + lying → wide + stand (原因: shot_pose_conflict: wide禁止lying)
✓ 转换场景 1: scene_001_v22.json
```

### 转换 V2 文件

```bash
cd gen_video
python3 utils/v2_to_v22_converter.py ../lingjie/episode/1.v2.json --output-dir ../lingjie/v22
```

**输出示例**:
```
⚠ [Level 1] Pose lying 在 Shot wide 中不合法，自动修正为 stand
⚠ 自动修正: wide + lying → wide + stand (原因: shot_pose_conflict: wide禁止lying)
✓ 转换场景 1: scene_001_v22.json
```

### 批量转换

```bash
cd gen_video
python3 tools/batch_convert_episode_to_v22.py \
    --episode-dir ../lingjie/episode \
    --output-dir ../lingjie/v22
```

## 验证

转换后的 JSON 文件应该：
1. ✅ 通过 JSON 校验（`shot_pose_compatible: true`）
2. ✅ `pose.auto_corrected` 正确标记（如果被修正）
3. ✅ `pose.validated_by` 正确设置（如果被修正，使用 `"shot_pose_rules"`）
4. ✅ 可以直接用于生成，无需手动修正

## 注意事项

1. **修正策略**: 规则引擎优先修正 `pose` 而不是 `shot`，因为 `shot` 通常由场景意图决定
2. **修正级别**: 当前使用 Level 1 修正（硬规则冲突），Level 2 修正（语义修正）需要 `story_context`
3. **日志输出**: 修正信息会输出到控制台，方便追踪修正过程
4. **向后兼容**: 如果组合合法，不会进行修正，保持原始值

