# 根据 episode_171.json 制作视频的完整工作流

## 概述

完整的工作流程，集成TTS配音生成、智能视频匹配和AI生成功能。

## 核心流程

```
1. 读取JSON → 提取narration文本
   ↓
2. TTS生成配音（声音克隆）→ 获取精确音频时长
   ↓
3. 根据音频时长和场景描述：
   - 优先检索原视频（智能匹配）
   - 匹配不到时，AI生成图片+视频
   ↓
4. 添加开头和结尾视频
   ↓
5. 拼接所有视频片段（静音）
   ↓
6. （后续）添加BGM和配音
```

## 关键特性

### ✅ 先读配音，确定时长

**为什么重要：**
- TTS生成的音频时长是**100%精确**的
- 根据精确时长匹配视频，确保音视频同步
- 避免使用估算的duration导致不匹配

**实现方式：**
1. 从JSON提取所有`narration`文本
2. 使用TTS（CosyVoice/ChatTTS）生成配音
3. 获取每个音频文件的实际时长
4. 使用实际音频时长匹配视频

### ✅ 智能视频匹配

**策略：**
1. **优先检索原视频**
   - 使用混合检索（向量+关键词）
   - 智能决策（`decision_make`）
   - 根据音频时长裁剪/拼接

2. **匹配不到时AI生成**
   - 生成图片（ImageGenerator）
   - 生成视频（VideoGenerator）
   - 使用精确音频时长

### ✅ 声音克隆

**使用gen_video中的TTS功能：**
- CosyVoice2（推荐，支持zero-shot声音克隆）
- ChatTTS（备选）
- 使用参考音频进行声音克隆

## 使用方式

### 方式1：使用一键脚本（推荐）

```bash
./renjie/generate_episode_171_complete.sh
```

### 方式2：直接运行Python脚本

```bash
cd /vepfs-dev/shawn/vid/fanren
source /vepfs-dev/shawn/venv/py312/bin/activate

python3 generate_video_from_json_complete.py \
    --json renjie/episode_171.json \
    --index processed/global_index.faiss \
    --metadata processed/index_metadata.json \
    --scenes processed/episode_142/scene_metadata.json \
             processed/episode_151/scene_metadata.json \
             processed/episode_165/scene_metadata.json \
             processed/episode_170/scene_metadata.json \
             processed/episode_171/scene_metadata.json \
    --video-dir processed/ \
    --opening renjie/opening_video_raw.mp4 \
    --ending renjie/ending_video_raw.mp4 \
    --output renjie/episode_171_video.mp4 \
    --gen-video-config gen_video/config.yaml
```

## 参数说明

- `--json`: JSON配置文件（renjie/episode_*.json）
- `--index`: FAISS索引路径
- `--metadata`: 索引metadata路径
- `--scenes`: 场景metadata JSON文件（可多个）
- `--video-dir`: 视频文件基础目录（processed/）
- `--opening`: 开头视频路径
- `--ending`: 结尾视频路径
- `--output`: 输出视频路径
- `--gen-video-config`: gen_video配置文件（用于TTS和AI生成）
- `--skip-tts`: 跳过TTS配音生成（使用JSON中的duration）
- `--skip-opening`: 跳过开头视频
- `--skip-ending`: 跳过结尾视频

## 输出结果

### 主要输出
- **视频文件**: `renjie/episode_171_video.mp4`
  - 完全静音（无音频轨道）
  - 包含所有场景的视频片段
  - 已按顺序拼接

### 中间文件
- **音频文件**: `renjie/episode_171_work/audios/`
  - `audio_opening.wav` - 开头配音
  - `audio_scene_*.wav` - 各场景配音
  - `audio_ending.wav` - 结尾配音

- **AI生成文件**（如果使用）:
  - `renjie/episode_171_work/images/` - AI生成的图片
  - `renjie/episode_171_work/videos/` - AI生成的视频

- **临时文件**: `renjie/episode_171_work/temp/`
  - 裁剪的临时视频片段

## 工作流程详解

### 步骤1: TTS生成配音

```python
# 为每个场景生成配音
for scene in scenes:
    narration = scene.get("narration", "")
    if narration:
        # 使用TTS生成配音（声音克隆）
        audio_file = audio_dir / f"audio_scene_{scene_id:03d}.wav"
        tts_generator.generate(narration, str(audio_file))
        
        # 获取精确音频时长
        audio_duration = get_media_duration(audio_file)
```

**优势：**
- 音频时长100%精确
- 支持声音克隆（使用参考音频）
- 可缓存复用（跳过已存在的音频）

### 步骤2: 智能视频匹配

```python
# 1. 尝试检索原视频
search_results = hybrid_search(description, ...)
decision = decision_make(search_results, target_duration=audio_duration, ...)

if decision['decision'] == 'use_retrieved':
    # 使用检索到的原视频
    video_path = decision['video_path']
    
# 2. 如果匹配不到，AI生成
else:
    # 生成图片
    image_path = image_generator.generate_image(prompt, ...)
    
    # 生成视频（使用精确音频时长）
    video_path = video_generator.generate_video(image_path, duration=audio_duration, ...)
```

**策略：**
- 优先使用原视频（质量更好）
- 匹配不到时自动AI生成
- 使用精确音频时长确保同步

### 步骤3: 拼接视频片段

```python
# 按顺序拼接：开头 → 中间场景 → 结尾
concatenate_videos(video_segments, output_path)
```

**特点：**
- 使用 `-c copy` 快速拼接（所有视频已静音）
- 保持视频顺序
- 输出完全静音的视频

## 后续步骤

生成静音视频后，需要：

1. **添加配音**
   - 使用生成的音频文件
   - 与视频同步

2. **添加BGM**
   - 选择合适的背景音乐
   - 调整音量混合

3. **生成字幕**
   - 基于配音音频生成字幕
   - 确保时间轴对齐

## 注意事项

1. **TTS配置**
   - 确保 `gen_video/config.yaml` 中TTS配置正确
   - 确保参考音频文件存在（用于声音克隆）

2. **AI生成**
   - 需要加载ImageGenerator和VideoGenerator
   - 需要相应的模型文件

3. **视频匹配**
   - 优先使用原视频（质量更好）
   - AI生成作为备选方案

4. **时长匹配**
   - 使用TTS生成的精确音频时长
   - 确保视频与音频完全同步


