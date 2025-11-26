# 4K视频裁剪参数指南

## 视频分辨率
- **4K视频**: 3840x2160 (16:9)
- **1080P视频**: 1920x1080 (16:9)

## 裁剪参数说明

### 参数格式
```
--crop W:H:X:Y
```
- `W`: 保留的宽度
- `H`: 保留的高度
- `X`: 水平起始位置（通常为0，不左右裁）
- `Y`: 垂直起始位置（通常为0，从顶部保留）

### 4K视频裁剪建议

#### 方案1：按比例换算（参考1080P）
如果1080P视频使用 `1920:980:0:0`（裁掉底部100像素），那么4K视频按比例：
- 宽度：3840（保持）
- 高度：2160 - 200 = 1960（裁掉底部200像素）
- **参数**: `3840:1960:0:0`

#### 方案2：保守裁剪（推荐先测试）
如果字幕区域较大，可能需要裁掉更多：
- **参数**: `3840:1860:0:0`（裁掉底部300像素）

#### 方案3：精确裁剪（需要实际测试）
根据实际字幕位置调整，通常：
- 字幕区域：底部150-300像素
- **参数范围**: `3840:1860:0:0` 到 `3840:2010:0:0`

## 测试步骤

### 1. 提取测试片段（带字幕）
```bash
# 从视频中间提取5秒测试片段
python3 tools/video_processing/extract_test_clip.py \
  --input gen_video/raw_videos/142_4K.mp4 \
  --output processed/test_clips/test_142_4k.mp4 \
  --duration 5
```

### 2. 测试不同裁剪参数
```bash
# 测试方案1：裁掉200像素
python3 tools/video_processing/clean_video.py \
  --input processed/test_clips/test_142_4k.mp4 \
  --output processed/test_clips/test_crop_1960.mp4 \
  --crop 3840:1960:0:0

# 测试方案2：裁掉300像素
python3 tools/video_processing/clean_video.py \
  --input processed/test_clips/test_142_4k.mp4 \
  --output processed/test_clips/test_crop_1860.mp4 \
  --crop 3840:1860:0:0

# 测试方案3：裁掉250像素（中间值）
python3 tools/video_processing/clean_video.py \
  --input processed/test_clips/test_142_4k.mp4 \
  --output processed/test_clips/test_crop_1910.mp4 \
  --crop 3840:1910:0:0
```

### 3. 查看测试结果
```bash
# 使用ffplay查看（需要X11转发或VNC）
ffplay processed/test_clips/test_crop_1960.mp4

# 或者提取关键帧查看
ffmpeg -i processed/test_clips/test_crop_1960.mp4 -vf "select=eq(n\,0)" -vframes 1 test_frame.jpg
```

### 4. 确定最佳参数后，应用到完整视频
```bash
python3 tools/video_processing/pipeline.py \
  --episode 142 \
  --input gen_video/raw_videos/142_4K.mp4 \
  --crop 3840:1960:0:0 \
  --delogo X:Y:W:H \
  --auto-aspect \
  --trim-start 5
```

## LOGO位置（delogo参数）

4K视频的LOGO位置需要按比例换算：
- 如果1080P视频LOGO在 `1650:40:200:150`
- 4K视频按2倍比例：`3300:80:400:300`

**但建议实际测试**，因为LOGO位置可能不完全按比例。

## 等比例裁剪（--auto-aspect）

使用 `--auto-aspect` 可以自动从裁剪后的区域中再左右裁一点，保持16:9比例：
- 裁剪后：3840x1960（约1.96:1）
- 自动等比例：约3484x1960（16:9）

## 完整示例

```bash
# 处理4K视频的完整流程
python3 tools/video_processing/pipeline.py \
  --episode 142 \
  --input gen_video/raw_videos/142_4K.mp4 \
  --crop 3840:1960:0:0 \
  --delogo 3300:80:400:300 \
  --auto-aspect \
  --trim-start 5 \
  --scale-medium  # 缩放到1600x900，适合后续超分
```

## 注意事项

1. **先测试再应用**：4K视频文件很大（2-3GB），处理时间长，建议先用测试片段确定参数
2. **字幕位置可能不同**：不同集的字幕位置可能略有差异，建议每集都测试
3. **LOGO位置**：需要实际查看视频确定，不能完全按比例换算
4. **分辨率选择**：
   - 如果后续要超分：使用 `--scale-medium`（1600x900）
   - 如果直接使用：可以保持4K或缩放到1080P

