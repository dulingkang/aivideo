# Prompt 三层架构设计文档

## 📋 架构总览

```
[语义层 Semantic Layer]   ← 不碰字符串，只操作 PromptNode
        ↓
[策略层 Policy Layer]     ← 模型/场景感知，调整节点权重和优先级
        ↓
[渲染层 Render Layer]     ← 生成最终 prompt string
```

---

## 一、语义层（Semantic Layer）

### 核心组件

1. **PromptNode** (`semantic.py`)
   - 语义节点数据结构
   - `content`: 纯语义内容（不含权重标记）
   - `weight`: 权重（数值）
   - `priority`: 语义优先级
   - `hard`: 是否不可删除
   - `tags`: 语义标记集合

2. **PromptAST** (`semantic.py`)
   - Prompt 抽象语法树
   - 管理一组 PromptNode
   - 提供统一的语义操作接口

3. **ASTBuilder** (`ast_builder.py`)
   - 将字符串解析为 PromptNode AST
   - 一次解析，后续不再做字符串猜测

4. **SemanticEnhancer** (`enhancer.py`)
   - 语义级增强（不是字符串后处理）
   - 基于语义模式自动增强关键元素

---

## 二、策略层（Policy Layer）

### 核心组件

1. **PromptPolicy** (`policy.py`)
   - 策略基类
   - 所有策略必须实现 `apply(ast: PromptAST) -> PromptAST`

2. **模型专用策略**
   - `InstantIDPolicy`: InstantID 专用策略
   - `FluxPolicy`: Flux 专用策略
   - `HunyuanVideoPolicy`: HunyuanVideo 专用策略
   - `SDXLPolicy`: SDXL 默认策略

3. **PolicyEngine** (`policy.py`)
   - 策略引擎
   - 根据模型类型选择合适的策略并应用

### 策略规则示例

#### InstantIDPolicy
```python
- 角色描述：hard=True, priority+=5
- 远景+lying：priority-=2, 添加排除词
- 强场景描述：适当降低权重
```

#### FluxPolicy
```python
- 风格描述：priority+=2
- 环境描述：保持高权重
- 角色描述：如果场景为主，适当降低
```

---

## 三、渲染层（Render Layer）

### 核心组件

1. **PromptRenderer** (`renderer.py`)
   - 将 PromptAST 渲染为最终的 prompt 字符串
   - **禁止任何语义判断出现在 renderer 中**
   - 只做字符串拼接

### 渲染流程

```
AST → 选择节点（基于优先级和token限制） → 渲染为字符串 → 组合
```

---

## 四、使用示例

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

# 使用新架构（默认）
optimized = optimizer.optimize(parts, max_tokens=70, model_type="instantid", use_ast=True)

# 使用旧架构（向后兼容）
optimized = optimizer.optimize(parts, max_tokens=70, use_ast=False)
```

---

## 五、架构优势

### ✅ 解决的问题

1. **类型推断 + 重要性 + 改写不再耦合**
   - 一次解析为 AST，后续只操作节点
   - 不再反复读写字符串

2. **Prompt 是结构，不是字符串**
   - 权重、优先级、标记都是节点属性
   - 不再用正则猜

3. **模型感知**
   - 不同模型使用不同策略
   - 策略规则清晰、可维护

### ✅ 设计原则

1. **语义层不碰字符串**：只操作 PromptNode
2. **策略层模型感知**：根据模型类型调整策略
3. **渲染层只拼接**：禁止语义判断

---

## 六、迁移路径

### Phase 1（已完成）
- ✅ 创建 PromptNode 数据结构
- ✅ 创建 AST Builder
- ✅ 创建策略层接口和基础策略
- ✅ 创建渲染层
- ✅ 创建语义增强器

### Phase 2（进行中）
- ⏳ 重构 optimizer，支持 AST 架构
- ⏳ 更新 builder，使用新架构
- ⏳ 添加模型类型传递

### Phase 3（计划中）
- ⏳ Prompt → SceneGraph
- ⏳ 人设图 & 场景图分离

---

## 七、关键设计决策

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

## 八、扩展指南

### 添加新的增强规则

在 `SemanticEnhancer` 中添加新方法：

```python
def _enhance_your_rule(self, node: PromptNode) -> None:
    if node.type == "your_type":
        # 检测语义模式
        if some_pattern in node.content:
            node.tags.add("your_tag")
            # 调整内容或权重
```

### 添加新的策略

继承 `PromptPolicy`：

```python
class YourModelPolicy(PromptPolicy):
    def apply(self, ast: PromptAST) -> PromptAST:
        result_ast = ast.copy()
        for node in result_ast.nodes:
            # 应用策略规则
        return result_ast
```

然后在 `PolicyEngine` 中注册。

---

**最后更新**: 2025-12-15  
**架构版本**: v2.0 (三层架构)


