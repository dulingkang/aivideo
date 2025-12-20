# v2.1 → v2.2 自然演进建议

## 📋 总览

基于反馈，v2.1架构方向完全正确，只需要在3个关键点进行增强，即可达到v2.2的工业级标准。

---

## ✅ v2.1已正确的地方（保持不变）

1. **Shot/Pose/Model硬规则** - 完全正确，继续坚持
2. **LoRA是Layer 0** - 完全正确，继续坚持
3. **LLM降级为润色器** - 完全正确，继续坚持

---

## ⚠️ v2.2需要增强的3个点

### 1. Pose修正策略增强（两级修正）

**v2.1现状**：
- 只有Level 1（硬规则修正）
- 缺少语义修正

**v2.2增强**：
- ✅ Level 1：无感修正（已实现）
- ✅ Level 2：语义修正（已实现）
- ✅ 记录修正来源和原因

**实现位置**：
- `execution_rules_v2_1.py` - `validate_pose`方法已增强

**JSON字段**：
```json
"pose": {
  "type": "sitting",
  "locked": true,
  "auto_corrected": true,
  "correction_level": "level2",
  "correction_reason": "story_flow_conflict: 非受伤剧情，lying改为sitting",
  "original_pose": "lying"
}
```

---

### 2. Model路由可解释性增强

**v2.1现状**：
- 有decision_reason字段
- 但不够详细

**v2.2增强**：
- ✅ 完整的决策trace
- ✅ 置信度评分
- ✅ 决策路径记录

**实现位置**：
- `execution_executor_v21.py` - `_build_decision_trace`方法

**JSON字段**：
```json
"model_route": {
  "base_model": "flux1-dev",
  "identity_engine": "pulid",
  "allow_fallback": false,
  "decision_reason": "character_present + medium_shot -> flux + pulid",
  "confidence": 0.95,
  "decision_trace": {
    "has_character": true,
    "shot_type": "medium",
    "face_visible": true,
    "rules_applied": ["model_routing_table"]
  }
}
```

---

### 3. 失败重试机制（同模型低风险重试）

**v2.1现状**：
- 提到不推荐模型fallback
- 但缺少具体实现

**v2.2增强**：
- ✅ 同模型重试策略
- ✅ 参数微调（CFG scale, InstantID strength）
- ✅ 重试条件明确

**实现位置**：
- `execution_executor_v21.py` - `_retry_execution`方法

**配置示例**：
```json
"retry_policy": {
  "enabled": true,
  "max_retry": 1,
  "retry_on": ["gender_mismatch", "composition_error"],
  "strategy": "same_model_low_risk",
  "parameter_adjustments": {
    "cfg_scale": -1.0,
    "instantid_strength": -0.1
  }
}
```

---

## 🔧 v2.2核心组件

### 1. JSON v2 → v2.1-exec 转换器 ✅

**文件**: `utils/json_v2_to_v21_converter.py`

**功能**：
- 自动转换v2格式到v2.1-exec
- 应用硬规则
- 添加决策trace

**使用**：
```python
from utils.json_v2_to_v21_converter import convert_json_file

convert_json_file("episode_v2.json", "episode_v21_exec.json")
```

---

### 2. Execution Validator ✅

**文件**: `utils/execution_validator.py`

**功能**：
- 校验JSON可执行性
- 检查所有必需字段
- 生成详细报告

**使用**：
```python
from utils.execution_validator import validate_json_file

is_valid, report = validate_json_file("scene_v21.json")
print(report)
```

---

### 3. Execution Executor V2.1 ✅

**文件**: `utils/execution_executor_v21.py`

**功能**：
- 瘦身版执行器（不计划，只执行）
- 完全确定性路径
- 失败重试机制

**使用**：
```python
from utils.execution_executor_v21 import ExecutionExecutorV21, ExecutionConfig, ExecutionMode

config = ExecutionConfig(mode=ExecutionMode.STRICT)
executor = ExecutionExecutorV21(config=config)
result = executor.execute_scene(scene, output_dir)
```

---

## 📊 v2.1 vs v2.2对比

| 特性 | v2.1 | v2.2 |
|------|------|------|
| Pose修正 | Level 1（硬规则） | Level 1 + Level 2（语义） |
| 决策trace | 基础 | 完整可解释 |
| 失败重试 | 无 | 同模型低风险 |
| JSON转换 | 手动 | 自动转换器 ✅ |
| JSON校验 | 无 | 完整校验器 ✅ |
| 执行器 | Planner（复杂） | Executor（瘦身） ✅ |

---

## 🎯 下一步行动

### 立即执行（已完成）

1. ✅ 创建JSON v2 → v2.1-exec转换器
2. ✅ 创建Execution Validator
3. ✅ 创建Execution Executor V2.1
4. ✅ 增强Pose修正策略（两级）

### 本周完成

5. ⏳ 集成Execution Executor到主流程
6. ⏳ 更新Execution Planner V3（调用Executor）
7. ⏳ 实现失败重试机制（完整版）

### 下周完成

8. ⏳ 批量测试v2.1-exec格式
9. ⏳ 性能优化
10. ⏳ 监控与日志

---

## 📝 使用示例

### 完整工作流

```python
# 1. 转换v2 → v2.1-exec
from utils.json_v2_to_v21_converter import convert_json_file
convert_json_file("episode_v2.json", "episode_v21_exec.json")

# 2. 校验JSON
from utils.execution_validator import validate_json_file
is_valid, report = validate_json_file("episode_v21_exec.json")
if not is_valid:
    print(report)
    exit(1)

# 3. 执行场景
from utils.execution_executor_v21 import ExecutionExecutorV21, ExecutionConfig, ExecutionMode
import json

with open("episode_v21_exec.json", 'r') as f:
    episode = json.load(f)

config = ExecutionConfig(mode=ExecutionMode.STRICT)
executor = ExecutionExecutorV21(config=config)

for scene in episode["scenes"]:
    result = executor.execute_scene(scene, "outputs/")
    if result.success:
        print(f"✓ 场景 {scene['scene_id']} 执行成功")
    else:
        print(f"✗ 场景 {scene['scene_id']} 执行失败: {result.error_message}")
```

---

## 🔗 相关文件

- ✅ `utils/json_v2_to_v21_converter.py` - JSON转换器
- ✅ `utils/execution_validator.py` - JSON校验器
- ✅ `utils/execution_executor_v21.py` - 瘦身版执行器
- ✅ `utils/execution_rules_v2_1.py` - 规则引擎（已增强）
- ✅ `utils/character_anchor_v2_1.py` - 角色锚系统

---

## 总结

v2.1架构方向完全正确，v2.2只是在3个关键点进行增强：

1. ✅ Pose修正策略（两级） - 已实现
2. ✅ 决策trace可解释性 - 已实现
3. ✅ 失败重试机制 - 框架已实现，待完整集成

**下一步**：集成Execution Executor到主流程，完成v2.2演进。

