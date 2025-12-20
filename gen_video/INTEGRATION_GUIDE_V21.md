# v2.1系统集成指南

## 📋 总览

本指南说明如何将v2.1执行型架构集成到现有系统中。

---

## ✅ 测试结果

**所有核心组件测试通过**：
- ✓ 规则引擎（Intent→Shot映射、Pose验证、Model路由）
- ✓ 角色锚系统（LoRA + InstantID + 性别负锁）
- ✓ JSON转换器（v2 → v2.1-exec）
- ✓ Execution Validator（JSON校验）

---

## 🔧 集成步骤

### 步骤1: 转换现有JSON

```python
from gen_video.utils.json_v2_to_v21_converter import convert_json_file

# 转换整集JSON
convert_json_file(
    "lingjie/episode/1.v2.json",
    "lingjie/episode/1.v21_exec.json"
)
```

### 步骤2: 校验JSON

```python
from gen_video.utils.execution_validator import validate_json_file

is_valid, report = validate_json_file("lingjie/episode/1.v21_exec.json")
if not is_valid:
    print(report)
    exit(1)
```

### 步骤3: 使用Execution Executor生成

#### 方式1: 直接使用Executor（推荐）

```python
from gen_video.utils.execution_executor_v21 import (
    ExecutionExecutorV21,
    ExecutionConfig,
    ExecutionMode
)
import json

# 加载v2.1-exec JSON
with open("lingjie/episode/1.v21_exec.json", 'r') as f:
    episode = json.load(f)

# 创建执行器（严格模式，不用LLM）
config = ExecutionConfig(mode=ExecutionMode.STRICT)
executor = ExecutionExecutorV21(
    config=config,
    image_generator=image_generator,  # 传入实际生成器
    video_generator=video_generator,
    tts_generator=tts_generator
)

# 执行所有场景
for scene in episode["scenes"]:
    result = executor.execute_scene(scene, "outputs/")
    if result.success:
        print(f"✓ 场景 {scene['scene_id']} 生成成功")
    else:
        print(f"✗ 场景 {scene['scene_id']} 生成失败: {result.error_message}")
```

#### 方式2: 使用适配器（向后兼容）

```python
from gen_video.utils.v21_executor_adapter import V21ExecutorAdapter

# 创建适配器
adapter = V21ExecutorAdapter(
    image_generator=image_generator,
    video_generator=video_generator,
    tts_generator=tts_generator
)

# 准备场景（转换为现有系统格式）
legacy_scene = adapter.prepare_scene_for_generation(scene_v21)

# 使用现有ImageGenerator生成
image = image_generator.generate_scene(legacy_scene, prompt, negative_prompt)
```

---

## 🔄 集成到现有流程

### 方案A: 完全替换（推荐）

**修改 `generate_novel_video.py`**：

```python
# 旧代码
def generate(self, prompt, scene=None, ...):
    # 使用Execution Planner V3分析
    strategy = self.planner.analyze_scene(scene)
    # 生成图像
    ...

# 新代码（v2.1-exec）
def generate(self, scene_v21=None, ...):
    if scene_v21:
        # 使用Execution Executor（v2.1-exec格式）
        executor = ExecutionExecutorV21(...)
        result = executor.execute_scene(scene_v21, output_dir)
        return result
    else:
        # 兼容旧格式（自动转换）
        converter = JSONV2ToV21Converter()
        scene_v21 = converter.convert_scene(scene)
        executor = ExecutionExecutorV21(...)
        result = executor.execute_scene(scene_v21, output_dir)
        return result
```

### 方案B: 渐进式集成（推荐用于过渡期）

**保持现有流程，添加v2.1-exec支持**：

```python
def generate(self, scene=None, use_v21=False, ...):
    if use_v21 and scene and scene.get("version", "").startswith("v2.1"):
        # 使用v2.1-exec流程
        executor = ExecutionExecutorV21(...)
        return executor.execute_scene(scene, output_dir)
    else:
        # 使用现有流程（兼容）
        strategy = self.planner.analyze_scene(scene)
        # 原有生成逻辑
        ...
```

---

## 📝 关键集成点

### 1. ImageGenerator集成

**需要修改的地方**：

```python
# 在ImageGenerator.generate_scene中
def generate_scene(self, scene, prompt, negative_prompt):
    # 检查是否是v2.1-exec格式
    if scene.get("_v21_metadata"):
        # 使用v2.1决策
        model_route = scene["_v21_metadata"]["model_route"]
        base_model = model_route["base_model"]
        identity_engine = model_route["identity_engine"]
        
        # 获取角色锚
        character_id = scene["character"]["id"]
        anchor = anchor_manager.get_anchor(character_id)
        
        # 使用硬规则生成的参数
        ...
    else:
        # 使用现有逻辑（兼容）
        ...
```

### 2. Execution Planner V3重构

**改为调用Execution Executor**：

```python
# 旧代码
def analyze_scene(self, scene):
    # LLM分析
    # 动态决策
    ...

# 新代码（v2.1-exec模式）
def analyze_scene_v21(self, scene):
    # 如果已经是v2.1-exec格式，直接返回决策trace
    if scene.get("version", "").startswith("v2.1"):
        return {
            "shot": scene["shot"],
            "pose": scene["pose"],
            "model_route": scene["model_route"],
            "decision_trace": scene.get("_v21_metadata", {}).get("decision_trace")
        }
    
    # 否则，使用规则引擎转换
    from utils.execution_rules_v2_1 import get_execution_rules
    rules = get_execution_rules()
    
    intent = scene.get("intent", {}).get("type", "character_intro")
    shot_decision = rules.get_shot_from_intent(intent)
    pose_decision = rules.validate_pose(shot_decision.shot_type, scene["character"]["pose"])
    model, identity = rules.get_model_route(
        has_character=scene["character"]["present"],
        shot_type=shot_decision.shot_type
    )
    
    return {
        "shot": shot_decision,
        "pose": pose_decision,
        "model_route": (model, identity)
    }
```

---

## 🧪 测试建议

### 1. 单元测试

```bash
# 测试核心组件
python3 gen_video/test_v21_simple.py
```

### 2. 集成测试

```bash
# 测试完整流程
python3 gen_video/test_v21_integration.py
```

### 3. 端到端测试

```python
# 使用真实JSON文件测试
from gen_video.utils.json_v2_to_v21_converter import convert_json_file
from gen_video.utils.execution_validator import validate_json_file

# 转换
convert_json_file("lingjie/episode/1.v2.json", "test_outputs/1.v21_exec.json")

# 校验
is_valid, report = validate_json_file("test_outputs/1.v21_exec.json")
print(report)

# 执行（需要实际生成器）
# executor.execute_scene(scene, "test_outputs/")
```

---

## ⚠️ 注意事项

1. **向后兼容**：
   - 保持现有API不变
   - 添加`use_v21`参数控制是否使用v2.1流程

2. **配置迁移**：
   - 角色LoRA路径需要配置
   - 角色档案需要注册

3. **性能考虑**：
   - v2.1-exec模式更快（无LLM调用）
   - 但需要预先转换JSON

---

## 📊 集成检查清单

- [ ] JSON转换器测试通过
- [ ] Execution Validator测试通过
- [ ] Execution Executor测试通过
- [ ] 适配器集成测试通过
- [ ] ImageGenerator支持v2.1-exec格式
- [ ] VideoGenerator支持v2.1-exec格式
- [ ] 主流程支持v2.1-exec格式
- [ ] 向后兼容性测试通过

---

## 🔗 相关文档

- `USAGE_V2_1.md` - 使用指南
- `V2_1_TO_V2_2_EVOLUTION.md` - v2.2演进建议
- `TECH_ARCHITECTURE_V2_1.md` - 技术架构文档

