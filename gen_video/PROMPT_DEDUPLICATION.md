# Prompt 去重机制说明

## 📋 概述

为了避免 prompt 构建过程中出现重复描述，系统实现了统一的去重机制。

## 🔧 核心组件

### 1. PromptDeduplicator 类 (`utils/prompt_deduplicator.py`)

统一的去重工具类，提供以下功能：

- **关键词提取**：从文本中提取关键词（支持中英文）
- **重复检测**：检查新描述是否与已有文本重复（基于关键词重叠率）
- **同义词识别**：识别同义词（如"地面"和"ground"）
- **过滤重复**：过滤掉与已有文本重复的描述
- **合并去重**：合并多个 prompt 部分并去重

### 2. 使用场景

#### 场景 1: Execution Planner V3
- **位置**：`execution_planner_v3.py` 的 `build_weighted_prompt` 方法
- **功能**：检查场景分析器的增强描述是否与镜头类型描述或原始 prompt 重复
- **阈值**：60% 重叠认为是重复

#### 场景 2: 角色描述添加
- **位置**：`generate_novel_video.py` 的角色描述添加逻辑
- **功能**：检查角色描述（性别、服饰、发型）是否与优化后的 prompt 重复
- **阈值**：50% 重叠认为是重复（更严格，因为角色描述很重要）

#### 场景 3: 场景分析器
- **位置**：`utils/scene_analyzer.py` 的 `_generate_enhancements` 方法
- **功能**：检查地面增强描述是否与环境描述重复
- **阈值**：50% 重叠认为是重复

## 📊 去重策略

### 1. 关键词提取
- 使用中文逗号（"，"）和英文逗号（","）分割
- 移除权重标记 `(xxx:1.5)`
- 提取中英文关键词
- 过滤无意义词（"的"、"了"、"the"、"a" 等）

### 2. 重复检测算法
1. 提取新描述和已有文本的关键词
2. 计算关键词重叠率：`重叠关键词数 / 新描述关键词数`
3. 检查同义词（如"地面"和"ground"）
4. 检查包含关系（如"地面可见"包含"地面"）
5. 如果重叠率 >= 阈值，认为是重复

### 3. 同义词映射
系统维护了同义词映射表，包括：
- 地面相关：`地面` ↔ `ground` ↔ `floor` ↔ `地面可见`
- 全身相关：`全身` ↔ `full body` ↔ `全身可见`
- 沙漠相关：`沙漠` ↔ `desert` ↔ `沙漠景观`
- 躺相关：`躺` ↔ `lying` ↔ `躺在地上`
- 可见性相关：`可见` ↔ `visible` ↔ `清晰可见`

## 🎯 使用示例

### 示例 1: 过滤重复描述

```python
from utils.prompt_deduplicator import filter_duplicates

# 已有文本
existing_texts = [
    "全身镜头，人物中等大小，地面可见，脚可见",
    "广阔的沙漠景观，沙丘，沙漠地面清晰可见"
]

# 新描述
new_descriptions = [
    "地面清晰可见，地面可见，脚在地面上",  # 重复（与"地面可见"重复）
    "躺在地上，躺在沙地上，水平位置"  # 不重复
]

# 过滤
filtered = filter_duplicates(new_descriptions, existing_texts, threshold=0.6)
# 结果: ["躺在地上，躺在沙地上，水平位置"]
```

### 示例 2: 合并 prompt 部分

```python
from utils.prompt_deduplicator import merge_prompt_parts

parts = [
    "全身镜头，地面可见",
    "地面清晰可见，脚在地面上",  # 重复
    "躺在地上，水平位置"
]

merged = merge_prompt_parts(parts)
# 结果: "全身镜头，地面可见, 躺在地上，水平位置"
```

## ⚙️ 配置参数

### threshold（重复阈值）
- **默认值**：0.6（60%）
- **含义**：如果新描述的关键词中有 60% 以上与已有文本重叠，认为是重复
- **调整建议**：
  - 更严格（0.5）：用于角色描述等关键信息
  - 更宽松（0.7）：用于一般场景描述

## 🔍 调试

启用调试日志可以看到去重过程：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

日志会显示：
- 哪些描述被检测为重复
- 重叠率是多少
- 为什么被跳过

## 📝 注意事项

1. **去重工具是可选依赖**：如果导入失败，会回退到简单检查
2. **阈值可调整**：根据场景调整 threshold 参数
3. **同义词映射可扩展**：在 `PromptDeduplicator.synonyms` 中添加更多同义词

