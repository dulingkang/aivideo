# 原番视频处理流程

自动化处理原番视频，包括清洗、切分、标注和索引建立。

## 目录结构

```
tools/video_processing/
├── clean_video.py         # 步骤1: 清洗视频（去字幕/LOGO）
├── split_scenes.py        # 步骤2: 切分镜头（PySceneDetect）
├── transcribe_subtitles.py # 步骤3: 转写字幕（WhisperX）
├── align_subtitles.py     # 步骤4: 对齐字幕到镜头
├── extract_keyframes.py   # 步骤5: 提取关键帧
├── describe_scenes.py     # 步骤6: 生成描述和embedding（整合字幕）
├── build_index.py         # 步骤7: 建立跨集检索索引
└── pipeline.py            # 主流程脚本（整合所有步骤）
```

## 依赖安装

```bash
# 基础工具
pip install ffmpeg-python  # 或系统安装ffmpeg
pip install scenedetect[opencv]  # PySceneDetect

# AI模型
pip install transformers sentence-transformers
pip install torch torchvision pillow

# 字幕转写
pip install whisperx  # 或使用项目已有的WhisperX

# 检索索引
pip install faiss-cpu  # 或 faiss-gpu (如果有GPU)
```

## 使用方法

### 方法1: 使用主流程脚本（推荐）

处理单集视频的完整流程：

```bash
python tools/video_processing/pipeline.py \
  --episode 170 \
  --input gen_video/raw_videos/170\ 1080P.mp4 \
  --output gen_video/processed
```

### 方法2: 分步执行

#### 步骤1: 清洗视频
```bash
python tools/video_processing/clean_video.py \
  --input gen_video/raw_videos/170\ 1080P.mp4 \
  --output gen_video/processed/episode_170/clean/episode_170_clean.mp4 \
  --auto-detect
```

#### 步骤2: 切分镜头
```bash
python tools/video_processing/split_scenes.py \
  --input gen_video/processed/episode_170/clean/episode_170_clean.mp4 \
  --output gen_video/processed/episode_170/scenes
```
**输出**: `scene_list.json` (场景时间列表)

#### 步骤3: 转写字幕
```bash
python tools/video_processing/transcribe_subtitles.py \
  --input gen_video/processed/episode_170/clean/episode_170_clean.mp4 \
  --output gen_video/processed/episode_170/subtitles.json \
  --language zh \
  --model medium
```

#### 步骤4: 对齐字幕到镜头
```bash
python tools/video_processing/align_subtitles.py \
  --subtitles gen_video/processed/episode_170/subtitles.json \
  --scenes gen_video/processed/episode_170/scene_list.json \
  --output gen_video/processed/episode_170/aligned_subtitles.json
```

#### 步骤5: 提取关键帧
```bash
python tools/video_processing/extract_keyframes.py \
  --input gen_video/processed/episode_170/scenes \
  --output gen_video/processed/episode_170/keyframes
```

#### 步骤6: 生成描述和embedding（整合字幕）
```bash
python tools/video_processing/describe_scenes.py \
  --input gen_video/processed/episode_170/keyframes \
  --output gen_video/processed/episode_170/scene_metadata.json \
  --subtitles gen_video/processed/episode_170/aligned_subtitles.json
```
**说明**: 如果提供 `--subtitles` 参数，会将字幕文本整合到描述中，提高检索准确性

#### 步骤7: 建立跨集索引（处理多集后）
```bash
python tools/video_processing/build_index.py \
  --input gen_video/processed/episode_170/scene_metadata.json \
           gen_video/processed/episode_171/scene_metadata.json \
  --index gen_video/processed/global_index.faiss \
  --metadata gen_video/processed/index_metadata.json
```

## 输出结构

```
gen_video/processed/
├── episode_170/
│   ├── clean/
│   │   └── episode_170_clean.mp4
│   ├── scenes/
│   │   ├── scene_000.mp4
│   │   ├── scene_001.mp4
│   │   └── ...
│   ├── scene_list.json          # 场景时间列表
│   ├── subtitles.json           # 转写的字幕
│   ├── aligned_subtitles.json    # 对齐到镜头的字幕
│   ├── keyframes/
│   │   ├── scene_000_start.jpg
│   │   ├── scene_000_middle.jpg
│   │   └── ...
│   └── scene_metadata.json      # 场景描述（整合字幕）
├── episode_171/
│   └── ...
├── global_index.faiss
└── index_metadata.json
```

## scene_metadata.json 格式

```json
{
  "scene_000": {
    "text": "韩立：跟我走。孙火：是，师祖！ [Visual: A man talking with another in a shop]",
    "visual_caption": "A man talking with another in a shop",
    "subtitle_text": "韩立：跟我走。孙火：是，师祖！",
    "subtitles": [
      {
        "start": 123.4,
        "end": 125.1,
        "text": "韩立：跟我走。"
      },
      {
        "start": 125.2,
        "end": 126.0,
        "text": "孙火：是，师祖！"
      }
    ],
    "embedding": [0.12, -0.98, ...],
    "keyframes": [
      {
        "path": "keyframes/scene_000_start.jpg",
        "caption": "A man talking with another in a shop"
      }
    ]
  }
}
```

**说明**:
- `text`: 合并后的描述（字幕 + 视觉描述），用于生成embedding
- `visual_caption`: BLIP-2生成的图像描述（英文）
- `subtitle_text`: 该场景的所有字幕文本合并
- `subtitles`: 详细的字幕段列表（带时间戳）

## 注意事项

1. **清洗视频**: 自动检测字幕区域可能不准确，建议手动指定 `--crop w:h:x:y`
2. **切分镜头**: PySceneDetect的阈值可能需要根据视频调整
3. **GPU加速**: 描述生成步骤建议使用GPU (`--device cuda`)
4. **内存占用**: 处理大量视频时注意内存，可以分批处理

## 批量处理

```bash
# 处理多集
for ep in 170 171 172; do
  python tools/video_processing/pipeline.py \
    --episode $ep \
    --input "gen_video/raw_videos/${ep} 1080P.mp4" \
    --output gen_video/processed \
    --skip-existing
done
```

