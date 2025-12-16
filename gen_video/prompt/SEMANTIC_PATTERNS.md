# 语义模式系统（Semantic Patterns）

## 📋 概述

语义模式系统将硬编码的词语列表替换为**可配置、可扩展的语义模式匹配**。

### ❌ 之前的问题

```python
# 硬编码的词语列表
if any(kw in content_lower for kw in [
    "han li", "long black hair", "dark green", "deep cyan", "cultivator robe",
    "xianxia cultivator", "immortal cultivator", "黑色长发", "深绿", "道袍", "修仙"
]):
    return "character"
```

**问题**：
- 词语列表分散在多个函数中
- 难以维护和扩展
- 无法针对不同项目/场景自定义
- 不符合"语义层"的设计理念

### ✅ 现在的解决方案

```python
# 使用语义模式注册表
pattern_registry = get_pattern_registry()
node_type = pattern_registry.infer_type(content)
```

**优势**：
- 集中管理所有语义模式
- 支持动态配置和扩展
- 支持正则表达式模式
- 支持模式权重和优先级
- 易于测试和维护

---

## 🏗️ 架构设计

### 核心组件

1. **SemanticPattern** (`semantic_patterns.py`)
   - 定义单个语义模式
   - 支持关键词列表和正则表达式
   - 支持最少匹配数和权重

2. **SemanticPatternRegistry** (`semantic_patterns.py`)
   - 集中管理所有语义模式
   - 按类型组织（constraint, character, action, ...）
   - 支持动态添加/更新/删除模式

3. **get_pattern_registry()** (`semantic_patterns.py`)
   - 全局单例模式
   - 确保所有组件使用相同的模式配置

---

## 📝 使用示例

### 基础使用

```python
from prompt import get_pattern_registry

# 获取模式注册表
registry = get_pattern_registry()

# 推断类型
node_type = registry.infer_type("Han Li with long black hair")
# 返回: "character"
```

### 自定义模式

```python
from prompt import SemanticPattern, get_pattern_registry
import re

registry = get_pattern_registry()

# 添加自定义角色模式（例如：新角色"张三"）
custom_pattern = SemanticPattern(
    keywords=["zhang san", "张三", "red robe", "红袍"],
    patterns=[re.compile(r'zhang\s+san', re.IGNORECASE)],
    min_matches=1,
    weight=8.0
)

registry.add_pattern("character", custom_pattern)

# 现在可以识别新角色
node_type = registry.infer_type("Zhang San in red robe")
# 返回: "character"
```

### 更新现有模式

```python
# 更新角色模式，添加新关键词
registry = get_pattern_registry()

# 获取第一个角色模式
character_patterns = registry.patterns["character"]
if character_patterns:
    # 更新第一个模式，添加新关键词
    first_pattern = character_patterns[0]
    first_pattern.keywords.append("new character name")
    first_pattern.keywords.append("新角色名")
```

### 重置模式注册表（用于测试）

```python
from prompt import reset_pattern_registry

# 重置为默认配置
reset_pattern_registry()
```

---

## 🎯 默认模式定义

### 约束条件（constraint）

```python
keywords = [
    "single person", "lone figure", "only one character", "one person only",
    "sole character", "single individual", "单人", "独行", "只有一个角色",
    "仅一人", "唯一角色", "单独个体"
]
weight = 10.0  # 最高优先级
```

### 角色描述（character）

**模式1：角色名称和特征**
```python
keywords = [
    "han li", "韩立",
    "long black hair", "tied long black hair", "forehead bangs",
    "黑色长发", "长发", "刘海"
]
weight = 8.0
```

**模式2：服饰和修仙特征**
```python
keywords = [
    "cultivator robe", "dark green", "deep cyan",
    "xianxia cultivator", "immortal cultivator",
    "道袍", "深绿", "修仙", "仙侠"
]
weight = 8.0
```

**模式3：多个角色特征组合**
```python
keywords = ["hair", "robe", "cultivator", "长发", "道袍", "修仙"]
min_matches = 2  # 至少匹配2个特征
weight = 9.0
```

**模式4：性别标记（正则表达式）**
```python
patterns = [re.compile(r'^\(?(male|female)', re.IGNORECASE)]
weight = 5.0
```

### 动作描述（action）

```python
keywords = [
    "lying", "lying on", "躺", "卧",
    "sitting", "sit", "坐",
    "standing", "stand", "站",
    "walking", "walk", "走",
    "动作", "姿势", "description"
]
weight = 6.0
```

### 构图描述（composition）

**模式1：动作动词**
```python
keywords = [
    "uses", "method", "flowing", "essence", "energy", "performing", "casting",
    "strains", "tilt", "sees", "revealing", "showing",
    "recalls", "tilts", "dive", "hovers", "expands", "changes", "recognizing",
    "躺", "看见", "转头", "回忆", "使用", "施展", "俯冲", "盘旋", "扩张", "变化"
]
weight = 7.0
```

**模式2：场景关系词**
```python
keywords = [
    "on", "above", "below", "in", "at", "with",
    "在", "上", "下", "中", "看到", "展现"
]
weight = 5.0
```

**模式3：特殊场景标记**
```python
keywords = ["composition", "nascent soul"]
weight = 8.0
```

### 特效（fx）

```python
keywords = [
    "essence", "energy flow", "spiritual light", "glow", "fx", "effect",
    "flooding", "visible", "flow", "light", "particles",
    "能量", "光效", "特效", "流动", "可见"
]
weight = 6.0
```

### 环境描述（environment）

```python
keywords = [
    "environment", "desert", "chamber", "sky", "background", "gravel", "plain",
    "环境", "沙漠", "天空", "遗迹", "背景", "沙地", "地面"
]
weight = 6.0
```

### 风格描述（style）

```python
keywords = ["xianxia", "chinese fantasy", "仙侠", "修仙", "古风"]
weight = 7.0
```

### 镜头描述（camera）

```python
keywords = [
    "camera", "shot", "镜头", "俯视", "远景", "中景",
    "facing camera", "front view", "top-down", "bird's eye"
]
weight = 5.0
```

### 背景一致性（background）

```python
keywords = ["consistent", "same", "背景一致"]
weight = 4.0
```

---

## 🔧 类型推断算法

### 优先级顺序

1. **constraint** (最高优先级)
2. **character**
3. **composition**
4. **fx**
5. **environment**
6. **style**
7. **action**
8. **camera**
9. **background**
10. **other** (默认)

### 匹配算法

1. 对每个类型，检查所有模式
2. 计算匹配分数（基于模式权重）
3. 返回得分最高的类型

```python
# 伪代码
type_scores = {}
for node_type in priority_order:
    for pattern in patterns[node_type]:
        if pattern.matches(text):
            type_scores[node_type] += pattern.weight

return max(type_scores, key=score)
```

---

## 🚀 扩展指南

### 添加新类型

```python
# 1. 在 SemanticPatternRegistry._init_default_patterns() 中添加
self.patterns["new_type"] = []

# 2. 添加模式
self.patterns["new_type"].append(SemanticPattern(
    keywords=["keyword1", "keyword2"],
    min_matches=1,
    weight=5.0
))

# 3. 在 infer_type() 的优先级列表中添加
type_priority = [
    "constraint",
    "character",
    "new_type",  # 添加新类型
    # ...
]
```

### 添加项目特定模式

```python
# 在项目初始化时
from prompt import SemanticPattern, get_pattern_registry

registry = get_pattern_registry()

# 添加项目特定的角色模式
project_pattern = SemanticPattern(
    keywords=["project specific character", "项目特定角色"],
    min_matches=1,
    weight=9.0
)
registry.add_pattern("character", project_pattern)
```

---

## 📊 优势总结

### ✅ 解决的问题

1. **硬编码词语列表** → **可配置语义模式**
2. **分散的规则** → **集中管理**
3. **难以扩展** → **动态添加/更新**
4. **难以测试** → **易于测试和验证**

### ✅ 设计优势

1. **语义清晰**：模式定义明确，易于理解
2. **可扩展性**：支持动态添加新模式
3. **可维护性**：集中管理，易于更新
4. **可测试性**：模式独立，易于单元测试
5. **灵活性**：支持关键词和正则表达式

---

**最后更新**: 2025-12-15  
**版本**: v1.0


