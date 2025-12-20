# v2.1 使用指南

## 📋 快速开始

### 1. 转换v2 JSON到v2.1-exec

```bash
python gen_video/utils/json_v2_to_v21_converter.py \
    lingjie/episode/1.v2.json \
    lingjie/episode/1.v21_exec.json
```

### 2. 校验JSON可执行性

```bash
python gen_video/utils/execution_validator.py \
    lingjie/episode/1.v21_exec.json
```

### 3. 执行场景生成

```python
from gen_video.utils.execution_executor_v21 import (
    ExecutionExecutorV21,
    ExecutionConfig,
    ExecutionMode
)
import json

# 加载JSON
with open("lingjie/episode/1.v21_exec.json", 'r') as f:
    episode = json.load(f)

# 创建执行器（严格模式，不用LLM）
config = ExecutionConfig(mode=ExecutionMode.STRICT)
executor = ExecutionExecutorV21(config=config)

# 执行所有场景
for scene in episode["scenes"]:
    result = executor.execute_scene(scene, "outputs/")
    print(f"场景 {scene['scene_id']}: {'✓' if result.success else '✗'}")
```

---

## 🔧 核心组件说明

### 1. JSON转换器

**功能**：自动将v2格式转换为v2.1-exec执行型格式

**关键特性**：
- ✅ 自动应用硬规则（Shot/Pose/Model）
- ✅ 自动添加角色锚配置
- ✅ 自动添加性别负锁
- ✅ 记录决策来源

---

### 2. Execution Validator

**功能**：校验JSON是否可执行

**检查项**：
- ✅ Shot/Pose是否锁定
- ✅ Shot → Pose组合是否合法
- ✅ Model路由是否配置
- ✅ 角色锚是否注册
- ✅ 性别负锁是否配置

---

### 3. Execution Executor V2.1

**功能**：瘦身版执行器（不计划，只执行）

**核心特性**：
- ✅ 完全确定性路径
- ✅ 无LLM参与
- ✅ 失败重试机制（同模型低风险）
- ✅ 完整决策trace

---

## 📝 配置说明

### ExecutionConfig

```python
config = ExecutionConfig(
    mode=ExecutionMode.STRICT,  # 严格模式：完全不用LLM
    enable_retry=True,           # 启用重试
    max_retry=1,                 # 最大重试次数
    retry_on_artifact=[          # 重试条件
        "gender_mismatch",
        "composition_error"
    ]
)
```

---

## 🎯 最佳实践

1. **先转换，再校验，最后执行**
2. **使用STRICT模式**（生产环境）
3. **记录决策trace**（用于调试）
4. **启用重试机制**（提高成功率）

---

## 🔗 相关文档

- `TECH_ARCHITECTURE_V2_1.md` - 完整技术架构
- `ARCHITECTURE_FLOW_V2_1.md` - 系统流程图
- `V2_1_TO_V2_2_EVOLUTION.md` - v2.2演进建议
- `REFACTOR_PLAN_V2_1.md` - 重构计划

