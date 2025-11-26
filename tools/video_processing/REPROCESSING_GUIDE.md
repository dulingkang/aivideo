# 重新处理已生成视频的指南

## 问题
如果之前已经处理过的视频，现在需要去除结尾，是否需要全部重新生成？

## 答案：不需要全部重新生成

### 情况分析

1. **已完成的步骤（无需重新生成）**：
   - ✅ 场景切分（scenes目录）
   - ✅ 字幕提取（subtitles.json）
   - ✅ 字幕对齐（aligned_subtitles.json）
   - ✅ 关键帧提取（keyframes目录）
   - ✅ 场景描述（scene_metadata.json）
   - ✅ 索引构建（global_index.faiss）

2. **需要重新生成的步骤**：
   - ⚠️ **清洗视频**（clean目录）- 需要去除结尾
   - ⚠️ **场景切分** - 依赖清洗后的视频

### 处理策略

#### 策略1：只重新清洗和切分（推荐）

如果只需要去除结尾，其他数据保持不变：

```bash
# 重新清洗视频（去除结尾）
python3 tools/video_processing/pipeline.py \
    --episode 171 \
    --input gen_video/raw_videos/171_1080P.mp4 \
    --output processed/ \
    --crop 1920:980:0:0 \
    --delogo 1650:40:200:150 \
    --auto-aspect \
    --trim-start 10 \
    --trim-end 30 \
    --skip-transcribe \      # 跳过字幕提取（已存在）
    --skip-align \           # 跳过字幕对齐（已存在）
    --skip-extract \         # 跳过关键帧提取（已存在）
    --skip-describe          # 跳过场景描述（已存在）
```

**注意**：由于场景切分依赖清洗后的视频，所以：
- 如果重新清洗，场景切分也需要重新运行
- 或者手动删除 `processed/episode_171/scenes/` 目录，让系统重新切分

#### 策略2：强制重新清洗（覆盖已存在文件）

如果需要完全重新生成清洗视频和切分：

```bash
python3 tools/video_processing/pipeline.py \
    --episode 171 \
    --input gen_video/raw_videos/171_1080P.mp4 \
    --output processed/ \
    --crop 1920:980:0:0 \
    --delogo 1650:40:200:150 \
    --auto-aspect \
    --trim-start 10 \
    --trim-end 30 \
    --skip-transcribe \
    --skip-align \
    --skip-extract \
    --skip-describe
# 不添加 --skip-existing，会重新生成清洗视频和场景切分
```

#### 策略3：手动只重新清洗

如果场景切分已经满足要求（不需要重新切分），可以只重新清洗：

```bash
# 直接使用 clean_video.py
python3 tools/video_processing/clean_video.py \
    --input gen_video/raw_videos/171_1080P.mp4 \
    --output processed/episode_171/clean/episode_171_clean.mp4 \
    --crop 1920:980:0:0 \
    --delogo 1650:40:200:150 \
    --auto-aspect \
    --trim-start 10 \
    --trim-end 30

# 然后手动决定是否需要重新切分场景
# 如果不需要重新切分，场景描述等其他数据都可以继续使用
```

### 关键点

1. **场景描述（scene_metadata.json）可以复用**
   - 只要场景切分没有变化，场景描述就可以继续使用
   - 如果重新切分，场景数量或顺序可能变化，需要重新生成描述

2. **字幕对齐可能需要更新**
   - 如果重新清洗/切分，场景时长可能变化
   - 如果场景切分变化，需要重新对齐字幕

3. **索引可以增量更新**
   - 如果只是重新清洗，场景描述不变，索引不需要更新
   - 如果场景切分变化，需要重新构建索引

### 最佳实践

**推荐流程**：

1. **首次处理**：完整流程
   ```bash
   python3 tools/video_processing/pipeline.py --episode 171 ...
   ```

2. **发现需要去除结尾**：只重新清洗和切分
   ```bash
   # 删除场景目录，强制重新切分
   rm -rf processed/episode_171/scenes/
   
   # 重新运行（跳过其他步骤）
   python3 tools/video_processing/pipeline.py \
       --episode 171 \
       --input ... \
       --trim-start 10 \
       --trim-end 30 \
       --skip-transcribe \
       --skip-align \
       --skip-extract \
       --skip-describe
   ```

3. **如果场景切分有变化**：重新生成描述和索引
   ```bash
   # 重新提取关键帧和生成描述
   python3 tools/video_processing/pipeline.py \
       --episode 171 \
       --input ... \
       --skip-clean \
       --skip-split \
       --skip-transcribe \
       --skip-align
   ```

### 检查是否需要重新切分

如果重新清洗后：
- 视频总时长变化
- 场景数量可能变化
- 场景时长可能变化

建议：**重新切分场景**，以确保场景边界准确。

### 快速判断

问自己：
- 场景描述是否需要更新？→ 如果需要，重新提取关键帧和描述
- 场景切分是否需要更新？→ 如果清洗后的视频时长变化，建议重新切分
- 字幕对齐是否需要更新？→ 如果场景切分变化，建议重新对齐

**最小化重新处理**：如果只是去除了结尾，且去除的部分不影响已有场景，可以只重新清洗，场景切分可以保持不变。
