# Execution Planner V3 退休计划

## 📋 总览

**建议**: Execution Planner V3 可以直接"退休"

**理由**: ExecutionRulesV21 + ExecutionExecutorV21 已经100%覆盖Planner的职责

**新角色**: Planner只做v2 → v2.1/v2.2转换和向后兼容，不再参与任何生成路径

---

## 🎯 当前状态

### Planner的职责（当前）

1. ✅ 场景分析（LLM）
2. ✅ Shot/Pose/Model决策（LLM）
3. ✅ Prompt构建（LLM）
4. ✅ 生成路径规划（LLM）

### 问题

- ❌ 使用LLM做决策（不稳定）
- ❌ 动态切换模型（不可预测）
- ❌ 允许不合法组合（wide + lying）

---

## ✅ 新架构（v2.1/v2.2）

### ExecutionRulesV21

**职责**:
- ✅ SceneIntent → Shot硬映射（表驱动）
- ✅ Shot → Pose验证与修正（硬规则）
- ✅ Model路由（硬规则）

**覆盖**: 100%覆盖Planner的决策职责

### ExecutionExecutorV21

**职责**:
- ✅ 执行场景生成（确定性路径）
- ✅ 构建Prompt（模板填充）
- ✅ 调用ImageGenerator/VideoGenerator

**覆盖**: 100%覆盖Planner的执行职责

---

## 🔧 退休计划

### 阶段1: 保留转换功能（立即）

**Planner的新角色**:
```python
class ExecutionPlannerV3:
    """只做转换，不参与生成"""
    
    def convert_v2_to_v21(self, scene_v2: Dict) -> Dict:
        """v2 → v2.1-exec转换"""
        from utils.json_v2_to_v21_converter import JSONV2ToV21Converter
        converter = JSONV2ToV21Converter()
        return converter.convert_scene(scene_v2)
    
    def convert_v2_to_v22(self, scene_v2: Dict) -> Dict:
        """v2 → v2.2转换（支持LoRA Stack）"""
        # 先转换为v2.1
        scene_v21 = self.convert_v2_to_v21(scene_v2)
        # 然后升级到v2.2（添加LoRA Stack支持）
        # TODO: 实现v2.2转换逻辑
        return scene_v21
    
    def is_backward_compatible(self, scene: Dict) -> bool:
        """检查是否向后兼容"""
        version = scene.get("version", "")
        return version in ["v2", "v2.1-exec", "v2.2-lora-stack"]
```

### 阶段2: 移除生成路径逻辑（1周内）

**移除内容**:
- ❌ 移除LLM决策逻辑
- ❌ 移除动态切换逻辑
- ❌ 移除生成路径规划

**保留内容**:
- ✅ 转换功能（v2 → v2.1/v2.2）
- ✅ 向后兼容检查
- ✅ 错误处理和日志

### 阶段3: 更新调用点（1周内）

**更新所有调用Planner的地方**:
```python
# 旧代码
planner = ExecutionPlannerV3()
result = planner.plan_and_generate(scene)

# 新代码
# 1. 转换（如果需要）
if scene.get("version") == "v2":
    planner = ExecutionPlannerV3()
    scene = planner.convert_v2_to_v21(scene)

# 2. 执行（使用Executor）
executor = ExecutionExecutorV21(...)
result = executor.execute_scene(scene, output_dir)
```

---

## 📊 影响分析

### 受影响的文件

1. `execution_planner_v3.py` - 需要重构
2. `generate_novel_video.py` - 需要更新调用
3. `enhanced_image_generator.py` - 需要更新调用
4. `batch_novel_generator.py` - 需要更新调用

### 向后兼容

- ✅ 保留Planner作为转换器
- ✅ 自动检测v2格式并转换
- ✅ 保持API兼容性（如果可能）

---

## 🎯 实施步骤

### 步骤1: 重构Planner（立即）

1. 移除生成路径逻辑
2. 保留转换功能
3. 添加向后兼容检查

### 步骤2: 更新调用点（1周内）

1. 更新`generate_novel_video.py`
2. 更新`enhanced_image_generator.py`
3. 更新`batch_novel_generator.py`

### 步骤3: 测试验证（1周内）

1. 测试v2 → v2.1转换
2. 测试v2 → v2.2转换
3. 测试向后兼容性

---

## 💡 心理层面的"断舍离"

**这不是技术难题，而是心理层面的"断舍离"**

**关键点**:
- ✅ 承认Planner已经完成历史使命
- ✅ 接受ExecutionRulesV21 + ExecutionExecutorV21的替代
- ✅ 专注于转换和向后兼容

**好处**:
- ✅ 代码更简洁
- ✅ 维护成本更低
- ✅ 系统更稳定

---

## 🔗 相关文档

- `TECH_ARCHITECTURE_V2_1.md` - 技术架构文档
- `ARCHITECTURE_FINAL_V22.md` - 架构最终结论
- `utils/execution_rules_v2_1.py` - 规则引擎
- `utils/execution_executor_v21.py` - 执行器

---

## 总结

**Planner的新角色**:
- ✅ 只做转换（v2 → v2.1/v2.2）
- ✅ 向后兼容
- ❌ 不再参与任何生成路径

**这是一次心理层面的"断舍离"，不是技术难题。**

