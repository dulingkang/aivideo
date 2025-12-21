# v2.2-final 架构设计总结（简明版）

## 一、核心原则

### ✅ 规则驱动，非LLM决策
- 所有关键决策（Shot、Pose、Model）通过硬规则表确定
- v2.2-final JSON 中所有参数都已锁定
- LLM 仅用于"修辞"（文本增强），不用于"决策"

### ✅ Execution Planner V3 已退休
- **不再参与生成流程**：v2.2-final 格式的执行不需要Planner决策
- **仅用于转换**：只负责 v2 → v2.1/v2.2 格式转换（向后兼容）
- **遗留问题**：`EnhancedImageGenerator` 中仍在使用，需要移除

---

## 二、执行流程

```
v2.2-final JSON (所有参数已锁定)
    ↓
ExecutionExecutorV21
    ├─ 验证JSON结构
    ├─ 读取锁定参数（shot, pose, model_route）
    ├─ 构建Prompt（模板填充，无LLM）
    └─ 执行生成
        ↓
    ┌─────────────────┬──────────────────┐
    │   标准模式       │    增强模式       │
    │                 │                  │
    │ ImageGenerator  │ EnhancedImageGen │
    │ - Flux/SDXL     │ - PuLIDEngine    │
    │ - LoRA支持      │ - 不使用Planner   │
    └─────────────────┴──────────────────┘
```

---

## 三、当前问题

### ❌ 问题1：EnhancedImageGenerator 仍在使用 Planner

**位置**：`enhanced_image_generator.py`

**问题代码**：
- 第94行：`self.planner = ExecutionPlannerV3(self.config)` ❌
- 第815行：`strategy = self.planner.analyze_scene(scene)` ❌
- 第856行：`prompt = self.planner.build_weighted_prompt(...)` ❌

**影响**：
- Planner 可能会覆盖JSON中锁定的参数
- 违反了"规则驱动，非LLM决策"的原则
- 增加了不确定性和不稳定性

### ✅ 正确做法

**应该直接从JSON读取锁定参数**：
```python
# ✅ 正确：直接从JSON读取
base_model = scene.get("model_route", {}).get("base_model", "flux")
identity_engine = scene.get("model_route", {}).get("identity_engine", "pulid")
shot_type = scene.get("shot", {}).get("type", "medium")
pose_type = scene.get("pose", {}).get("type", "stand")
prompt = scene.get("prompt", {}).get("final", original_prompt)
```

---

## 四、改造计划

### 改造1：移除 EnhancedImageGenerator 中的 Planner

**步骤**：
1. 移除第94行的 `self.planner = ExecutionPlannerV3(self.config)`
2. 修改 `generate_scene` 方法：
   - 移除第815行的 `strategy = self.planner.analyze_scene(...)`
   - 移除第856行的 `prompt = self.planner.build_weighted_prompt(...)`
   - 直接从JSON读取锁定参数
3. 移除第1783行的清理代码

### 改造2：明确 Planner 的保留用途

**保留场景**：
- ✅ 格式转换工具：`v2_to_v21_converter.py` 可以使用Planner
- ✅ 向后兼容：帮助用户从旧格式迁移

**移除场景**：
- ❌ 生成流程中的任何决策
- ❌ 动态路由选择
- ❌ Shot/Pose自动选择

---

## 五、架构对比

### 当前流程（错误）❌
```
v2.2-final JSON
    ↓
ExecutionExecutorV21
    ↓
EnhancedImageGenerator
    ↓
ExecutionPlannerV3.plan_generation()  ❌ 不应该使用
    ↓
PuLIDEngine.generate()
```

### 正确流程（改造后）✅
```
v2.2-final JSON
    ↓
ExecutionExecutorV21
    ├─ 读取锁定参数（shot, pose, model_route）
    ├─ 构建Prompt（模板填充）
    └─ 执行生成
        ↓
EnhancedImageGenerator
    ├─ 直接从JSON读取锁定参数  ✅ 不使用Planner
    ├─ 使用PuLIDEngine生成
    └─ 使用DecoupledFusionEngine融合
```

---

## 六、关键组件

### 1. Execution Executor V2.1 ✅
- **位置**：`utils/execution_executor_v21.py`
- **状态**：已正确实现（不使用Planner）
- **职责**：纯执行，不决策

### 2. EnhancedImageGenerator ⚠️
- **位置**：`enhanced_image_generator.py`
- **状态**：仍在使用Planner（需要改造）
- **职责**：PuLID + 解耦融合生成

### 3. Execution Planner V3 ⚠️
- **位置**：`utils/execution_planner_v3.py`
- **状态**：已退休，但仍在被使用（需要移除）
- **职责**：仅用于格式转换（向后兼容）

---

## 七、下一步行动

1. **立即改造**：移除 `EnhancedImageGenerator` 中的 Planner 使用
2. **测试验证**：确保改造后功能正常
3. **文档更新**：更新相关文档，明确 Planner 的角色定位
4. **代码清理**：移除不必要的 Planner 依赖

---

## 八、详细文档

完整的设计文档请参考：`ARCHITECTURE_V22_FINAL.md`

