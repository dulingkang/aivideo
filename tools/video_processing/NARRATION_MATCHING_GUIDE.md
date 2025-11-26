# Narration文本匹配场景视频指南

## 问题描述

在制作171集时：
1. 已经写好了narration文本
2. 已经算出了每一段的时长
3. 根据narration检索到了相关场景视频
4. 需要匹配时长：
   - 如果检索到的视频太长：裁剪
   - 如果检索到的视频太短：拼接多个场景
   - 每个narration如何选择场景

## 解决方案

### 策略说明

#### 策略1: 单视频完美匹配
- 如果某个检索到的视频时长接近narration时长（容差范围内）
- 直接使用该视频

#### 策略2: 多视频拼接
- 如果单个视频时长 < narration时长
- 从检索结果中按顺序选择多个视频，拼接直到满足时长

#### 策略3: 单视频裁剪
- 如果某个视频时长 > narration时长
- 裁剪视频到narration时长

#### 策略4: 多视频拼接+裁剪
- 先拼接多个视频
- 如果总时长 > narration时长，裁剪到目标时长

## 完整工作流

### 步骤1: Narration文本 -> 检索场景

```bash
source /vepfs-dev/shawn/venv/py312/bin/activate
cd /vepfs-dev/shawn/vid/fanren

python3 tools/video_processing/narration_to_scenes_workflow.py \
  --narration episode_171_narration.json \
  --index processed/global_index.faiss \
  --metadata processed/index_metadata.json \
  --scenes processed/episode_*/scene_metadata.json \
  --output episode_171_search_results \
  --top-k 5
```

**输入格式 (episode_171_narration.json):**
```json
{
  "narration_parts": [
    {
      "text": "这是我宗的身份令牌",
      "duration": 5.0
    },
    {
      "text": "你拿着这个令牌去找他",
      "duration": 4.5
    }
  ]
}
```

**输出:**
- `episode_171_search_results/narration_1_search.json` - 每个narration段落的检索结果
- `episode_171_search_results/all_search_results.json` - 所有检索结果汇总

### 步骤2: 匹配时长 -> 生成视频列表

```bash
python3 tools/video_processing/match_scenes_to_narration.py \
  --narration episode_171_narration.json \
  --search-results-dir episode_171_search_results \
  --video-base-dir processed \
  --output episode_171_matched_videos \
  --top-k 5 \
  --tolerance 0.5
```

**输出:**
- `episode_171_matched_videos/scene_001.mp4` - 匹配后的视频片段
- `episode_171_matched_videos/matched_scenes.json` - 匹配结果元数据

## 匹配策略详解

### 示例1: 单视频完美匹配
- Narration时长: 5.0秒
- 检索结果: scene_A (5.1秒, 分数0.95)
- 结果: 直接使用scene_A

### 示例2: 单视频裁剪
- Narration时长: 5.0秒
- 检索结果: scene_B (8.5秒, 分数0.90)
- 结果: 裁剪scene_B到5.0秒

### 示例3: 多视频拼接
- Narration时长: 10.0秒
- 检索结果:
  - scene_C (3.5秒, 分数0.95)
  - scene_D (4.2秒, 分数0.92)
  - scene_E (3.8秒, 分数0.90)
- 结果: 拼接scene_C + scene_D + scene_E = 11.5秒，然后裁剪到10.0秒

### 示例4: 部分匹配
- Narration时长: 8.0秒
- 检索结果:
  - scene_F (6.0秒, 分数0.95)
- 结果: 使用scene_F (6.0秒)，稍短但可以接受

## 参数说明

### --top-k
每个narration段落使用的检索结果数量。默认5个，可以尝试更多以获得更好的匹配。

### --tolerance
时长容差（秒）。默认0.5秒，表示：
- 如果视频时长在 `[目标时长-0.5, 目标时长+0.5]` 范围内，认为是完美匹配
- 如果视频总时长 <= `目标时长 + tolerance`，可以拼接

## 优化建议

### 1. 调整检索数量
如果匹配效果不好，可以增加top-k：
```bash
--top-k 10  # 检索更多候选场景
```

### 2. 调整容差
如果匹配过于严格，可以放宽容差：
```bash
--tolerance 1.0  # 允许1秒的误差
```

### 3. 手动筛选
生成匹配结果后，可以手动检查 `matched_scenes.json`，调整不合适的匹配。

## 输出格式

### matched_scenes.json
```json
{
  "matched_scenes": [
    {
      "index": 1,
      "text": "这是我宗的身份令牌",
      "target_duration": 5.0,
      "actual_duration": 5.1,
      "video_path": "episode_171_matched_videos/scene_001.mp4",
      "strategy": "单视频完美匹配: 171_scene_002 (5.10s)"
    }
  ],
  "total_scenes": 1
}
```

## 后续步骤

匹配完成后，可以使用这些视频片段：
1. 直接拼接所有片段生成完整视频
2. 添加音频和字幕
3. 使用现有的视频合成流程

