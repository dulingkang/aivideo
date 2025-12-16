# Prompt 架构升级总结

## ✅ Phase 1 完成（2025-12-15）

### 核心成果

已成功将 Prompt 处理系统从 **"字符串 + 规则堆"** 升级为 **"语义 AST + 策略驱动的渲染器"**。

---

## 📦 新增模块

### 1. 语义层（Semantic Layer）

- **`semantic.py`** (120行)
  - `PromptNode`: 语义节点数据结构
  - `PromptAST`: Prompt 抽象语法树
  
- **`ast_builder.py`** (250行)
  - `ASTBuilder`: 将字符串解析为 PromptNode AST
  - 一次解析，后续不再做字符串猜测

- **`enhancer.py`** (120行)
  - `SemanticEnhancer`: 语义级增强器
  - 基于语义模式自动增强关键元素

### 2. 策略层（Policy Layer）

- **`policy.py`** (180行)
  - `PromptPolicy`: 策略基类
  - `InstantIDPolicy`: InstantID 专用策略
  - `FluxPolicy`: Flux 专用策略
  - `HunyuanVideoPolicy`: HunyuanVideo 专用策略
  - `SDXLPolicy`: SDXL 默认策略
  - `PolicyEngine`: 策略引擎

### 3. 渲染层（Render Layer）

- **`renderer.py`** (120行)
  - `PromptRenderer`: 将 PromptAST 渲染为字符串
  - 禁止任何语义判断出现在 renderer 中

---

## 🔄 重构的模块

### `optimizer.py`

**升级内容**：
- ✅ 新增 `_optimize_with_ast()` 方法（使用新架构）
- ✅ 保留 `_optimize_legacy()` 方法（向后兼容）
- ✅ `optimize()` 方法支持 `model_type` 和 `use_ast` 参数
- ✅ `enhance_prompt_part()` 改为使用 AST 架构

**接口变化**：
```python
# 旧接口（仍然支持）
optimizer.optimize(parts, max_tokens=70)

# 新接口（推荐使用）
optimizer.optimize(parts, max_tokens=70, model_type="instantid", use_ast=True)
```

---

## 🎯 解决的问题

### ❌ 问题 1：类型推断 + 重要性 + 改写全耦合

**之前**：
- 一个 part 在 5 个地方被反复改写字符串
- 无法保证语义不漂移
- Debug 非常困难

**现在**：
- ✅ 一次解析为 AST，后续只操作节点
- ✅ 不再反复读写字符串
- ✅ 语义清晰，易于调试

### ❌ 问题 2：Prompt 被当成「字符串」，而不是「结构」

**之前**：
- 权重、否定、语义关系全靠正则猜
- `text: "(Han Li lying on desert sand:1.6)"`

**现在**：
- ✅ 权重、优先级、标记都是节点属性
- ✅ `PromptNode(content="Han Li lying on desert sand", weight=1.6, tags={"horizontal_pose"})`

### ❌ 问题 3：模型无感知（Flux / SDXL / Hunyuan 一视同仁）

**之前**：
- 所有模型使用相同的优化逻辑

**现在**：
- ✅ 不同模型使用不同策略
- ✅ `InstantIDPolicy`: 角色优先，削弱远景+lying
- ✅ `FluxPolicy`: 风格优先，场景权重高
- ✅ `HunyuanVideoPolicy`: 镜头优先，动作清晰

---

## 📊 架构对比

### 旧架构（字符串操作）

```
字符串 → 类型推断 → 重要性分析 → 字符串改写 → 选择 → 字符串拼接
```

### 新架构（AST + 策略）

```
字符串 → AST（语义层）
         ↓
    语义增强（语义层）
         ↓
    策略应用（策略层，模型感知）
         ↓
    渲染为字符串（渲染层）
```

---

## 🚀 使用示例

### 基础用法

```python
from prompt import ASTBuilder, PolicyEngine, SemanticEnhancer, PromptRenderer

# 1. 解析为 AST
ast_builder = ASTBuilder(token_estimator)
ast = ast_builder.parse_parts(parts)

# 2. 语义增强
enhancer = SemanticEnhancer()
ast = enhancer.enhance_ast(ast)

# 3. 应用策略（模型感知）
policy_engine = PolicyEngine()
ast = policy_engine.apply_policy(ast, model_type="instantid")

# 4. 渲染为字符串
renderer = PromptRenderer(token_estimator)
final_prompt = renderer.render(ast, max_tokens=70)
```

### 在 PromptOptimizer 中使用

```python
optimizer = PromptOptimizer(token_estimator, parser)

# 使用新架构（默认，推荐）
optimized = optimizer.optimize(parts, max_tokens=70, model_type="instantid")

# 使用旧架构（向后兼容）
optimized = optimizer.optimize(parts, max_tokens=70, use_ast=False)
```

---

## 🔧 向后兼容性

### ✅ 完全兼容

- 所有现有代码无需修改即可使用
- `optimizer.optimize()` 默认使用新架构（`use_ast=True`）
- 可以通过 `use_ast=False` 切换回旧架构

### ⚠️ 建议升级

- 在调用 `optimizer.optimize()` 时传递 `model_type` 参数
- 这样可以利用模型感知的策略优化

---

## 📈 下一步（Phase 2）

### 计划中的改进

1. **模型类型传递**
   - 在 `PromptBuilder.build()` 中添加 `model_type` 参数
   - 自动从 `ImageGenerator` 传递模型类型

2. **策略规则细化**
   - 根据实际使用情况调整策略规则
   - 添加更多模型专用策略

3. **性能优化**
   - AST 缓存
   - 策略结果缓存

---

## 📝 关键设计决策

### 1. 为什么 content 不含权重标记？

**答案**：权重是节点属性，不是内容的一部分。这样可以：
- 避免字符串解析的复杂性
- 支持动态调整权重
- 保持语义清晰

### 2. 为什么要有 tags？

**答案**：tags 用于语义标记，供策略层使用。例如：
- `horizontal_pose`: 水平姿势
- `sky_object`: 天空物体
- `pose_sensitive`: 姿势敏感

策略层可以根据 tags 决定如何调整节点。

### 3. 为什么策略层要返回新 AST？

**答案**：保持不可变性，避免副作用。策略应用不会修改原始 AST。

---

## 🎉 总结

**架构升级成功**：
- ✅ 从字符串操作升级为 AST 操作
- ✅ 从单一规则升级为策略驱动
- ✅ 从模型无关升级为模型感知
- ✅ 保持完全向后兼容

**代码质量**：
- ✅ 无 linter 错误
- ✅ 类型注解完整
- ✅ 文档齐全

**下一步**：
- ⏳ 在实际使用中验证新架构
- ⏳ 根据反馈调整策略规则
- ⏳ 优化性能

---

**完成时间**: 2025-12-15  
**架构版本**: v2.0 (三层架构)  
**状态**: ✅ Phase 1 完成


