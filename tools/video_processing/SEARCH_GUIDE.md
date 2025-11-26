# 场景检索使用指南

## 索引信息

- **总场景数**: 1,292 个场景
- **包含集数**: 5 集 (142, 151, 165, 170, 171)
- **索引文件**: `processed/global_index.faiss`
- **元数据文件**: `processed/index_metadata.json`

## 基本用法

```bash
source /vepfs-dev/shawn/venv/py312/bin/activate
cd /vepfs-dev/shawn/vid/fanren

python3 tools/video_processing/search_scenes.py \
  --query "你的查询内容" \
  --index processed/global_index.faiss \
  --metadata processed/index_metadata.json \
  --scenes processed/episode_*/scene_metadata.json \
  --method hybrid \
  --top-k 10
```

## 检索方法

### 1. 混合检索 (推荐) - `--method hybrid`
结合向量检索和关键词检索，平衡语义相似度和精确匹配。

**优点:**
- 同时考虑语义和关键词
- 结果最全面
- 适合大多数场景

**示例:**
```bash
--method hybrid --vector-weight 0.7 --keyword-weight 0.3
```

### 2. 向量检索 - `--method vector`
基于语义相似度，使用embedding向量搜索。

**优点:**
- 理解语义，能找到同义词
- 适合概念性查询

**示例:**
```bash
--method vector
```

### 3. 关键词检索 - `--method keyword`
基于TF-IDF权重，精确匹配关键词。

**优点:**
- 精确匹配关键词
- 适合查找特定名称、物品

**示例:**
```bash
--method keyword
```

## 检索示例

### 示例1: 查找特定物品
```bash
python3 tools/video_processing/search_scenes.py \
  --query "身份令牌" \
  --index processed/global_index.faiss \
  --metadata processed/index_metadata.json \
  --scenes processed/episode_*/scene_metadata.json \
  --method hybrid \
  --top-k 5
```

### 示例2: 查找地点
```bash
python3 tools/video_processing/search_scenes.py \
  --query "洞府" \
  --index processed/global_index.faiss \
  --metadata processed/index_metadata.json \
  --scenes processed/episode_*/scene_metadata.json \
  --method hybrid \
  --top-k 5
```

### 示例3: 查找动作或场景
```bash
python3 tools/video_processing/search_scenes.py \
  --query "战斗" \
  --index processed/global_index.faiss \
  --metadata processed/index_metadata.json \
  --scenes processed/episode_*/scene_metadata.json \
  --method vector \
  --top-k 10
```

## 结果解释

### 分数说明

- **分数范围**: 0.0 - 1.0
- **1.0**: 完美匹配（通常出现在混合检索中，表示向量和关键词都匹配）
- **0.7-0.99**: 高度相关
- **0.3-0.7**: 中等相关
- **< 0.3**: 低相关

### 输出字段

- `scene_id`: 场景ID（格式：`episode_id_scene_id`）
- `score`: 相关性分数
- `episode_id`: 所属集数
- `text`: 场景描述（字幕+视觉描述）
- `subtitle_text`: 字幕文本

## 性能优化

### 调整权重（混合检索）

如果结果偏向关键词或语义，可以调整权重：

```bash
# 更偏向量检索（语义）
--vector-weight 0.8 --keyword-weight 0.2

# 更偏关键词检索（精确）
--vector-weight 0.5 --keyword-weight 0.5
```

### 调整返回数量

```bash
--top-k 5   # 返回前5个
--top-k 20  # 返回前20个
```

## 添加新集到索引

当处理新集后，需要重新建立索引：

```bash
python3 tools/video_processing/build_index.py \
  --input processed/episode_142/scene_metadata.json \
         processed/episode_151/scene_metadata.json \
         processed/episode_165/scene_metadata.json \
         processed/episode_170/scene_metadata.json \
         processed/episode_171/scene_metadata.json \
         processed/episode_XXX/scene_metadata.json \
  --index processed/global_index.faiss \
  --metadata processed/index_metadata.json
```

## 保存检索结果

```bash
python3 tools/video_processing/search_scenes.py \
  --query "身份令牌" \
  --index processed/global_index.faiss \
  --metadata processed/index_metadata.json \
  --scenes processed/episode_*/scene_metadata.json \
  --method hybrid \
  --top-k 10 \
  --output search_results.json
```

## 常见问题

### Q: 为什么有些相关场景没有出现？
A: 可能是：
- 场景描述中没有包含相关关键词
- 语义相似度较低
- 可以尝试调整权重或增加top-k数量

### Q: 如何提高检索准确度？
A: 
- 使用混合检索（hybrid）
- 尝试不同的查询表达方式
- 调整vector-weight和keyword-weight

### Q: 检索速度慢怎么办？
A:
- 减少top-k数量
- 使用纯向量检索（vector）或纯关键词检索（keyword）
- 检查faiss是否使用GPU版本

