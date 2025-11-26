# 视频处理工作流总结

## 完整流程

### 阶段1: 视频预处理 ✅
1. **清洗视频** (`clean_video.py`)
   - 裁剪字幕区域
   - 去除LOGO
   - 统一分辨率（1742x980）
   
2. **切分场景** (`split_scenes.py`)
   - 使用PySceneDetect自动切分
   - 生成场景列表

3. **提取字幕** (`transcribe_subtitles.py` 或 `extract_subtitles_ocr.py`)
   - WhisperX: 语音识别
   - OCR: 文字识别（硬编码字幕）

4. **对齐字幕** (`align_subtitles.py`)
   - 将字幕对齐到场景

### 阶段2: 场景标注 ✅
5. **提取关键帧** (`extract_keyframes.py`)
   - 每个场景提取start/middle/end帧

6. **生成描述和embedding** (`describe_scenes.py`)
   - BLIP-2: 图像描述
   - CLIP: 生成向量embedding
   - 整合字幕和视觉描述

### 阶段3: 索引和检索 ✅
7. **建立跨集索引** (`build_index.py`)
   - 使用FAISS建立向量索引
   - 支持快速检索

8. **场景检索** (`search_scenes.py`)
   - 混合检索（hybrid）: 向量+关键词
   - 支持多种检索方式

### 阶段4: 提取和组合 ✅ (新增)
9. **提取检索场景** (`extract_scenes_from_search.py`)
   - 根据检索结果提取视频片段
   - 支持复制或软链接

## 使用示例

### 1. 处理单集视频
```bash
python3 tools/video_processing/pipeline.py \
  --episode 142 \
  --input gen_video/raw_videos/142_4K.mp4 \
  --crop 3840:1960:0:0 \
  --delogo 3300:80:535:300 \
  --auto-aspect \
  --scale 1742:980 \
  --trim-start 5
```

### 2. 建立跨集索引
```bash
python3 tools/video_processing/build_index.py \
  --input processed/episode_*/scene_metadata.json \
  --index processed/global_index.faiss \
  --metadata processed/index_metadata.json
```

### 3. 检索场景
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

### 4. 提取检索到的场景视频
```bash
python3 tools/video_processing/extract_scenes_from_search.py \
  --search-results search_results.json \
  --base-dir processed \
  --output extracted_scenes \
  --link \
  --create-playlist
```

## 完整工作流脚本

```bash
#!/bin/bash
# 完整工作流：检索 -> 提取

source /vepfs-dev/shawn/venv/py312/bin/activate
cd /vepfs-dev/shawn/vid/fanren

QUERY="身份令牌"
OUTPUT_DIR="extracted_scenes"

# 1. 检索场景
echo "=== 检索场景 ==="
python3 tools/video_processing/search_scenes.py \
  --query "$QUERY" \
  --index processed/global_index.faiss \
  --metadata processed/index_metadata.json \
  --scenes processed/episode_*/scene_metadata.json \
  --method hybrid \
  --top-k 10 \
  --output /tmp/search_results.json

# 2. 提取场景视频
echo ""
echo "=== 提取场景视频 ==="
python3 tools/video_processing/extract_scenes_from_search.py \
  --search-results /tmp/search_results.json \
  --base-dir processed \
  --output "$OUTPUT_DIR" \
  --link \
  --create-playlist

echo ""
echo "✓ 完成！提取的场景在: $OUTPUT_DIR"
```

## 文件结构

```
processed/
├── episode_142/
│   ├── clean/
│   ├── scenes/          # 场景视频片段
│   ├── keyframes/       # 关键帧
│   ├── scene_metadata.json
│   └── ...
├── global_index.faiss   # FAISS向量索引
└── index_metadata.json  # 索引元数据
```

## 下一步建议

1. **批量处理**: 处理更多集的视频
2. **优化检索**: 调整权重参数，提高检索准确度
3. **场景组合**: 将检索到的场景组合成新视频
4. **自动化**: 创建自动化脚本，一键完成检索和提取

