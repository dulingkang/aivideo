# 小说推文视频生成工作流

## 概述

使用 **Flux** 生成图像，然后用 **HunyuanVideo** 生成视频，实现小说推文视频的自动化生成。

## 工作流程

```
文本提示词（小说场景描述）
    ↓
Flux 生成图像（文生图）
    ↓
图像
    ↓
HunyuanVideo 生成视频（图生视频）
    ↓
视频
```

## 配置

### 图像生成（Flux）

- **模型**: Flux 1.1 Dev
- **配置位置**: `config.yaml` → `image.model_selection.character.engine: flux-instantid`
- **状态**: ✅ 已集成

### 视频生成（HunyuanVideo）

- **模型**: HunyuanVideo-1.5（图生视频）
- **配置位置**: `config.yaml` → `video.model_type: hunyuanvideo`
- **状态**: ✅ 已集成

## 使用方法

### 方法1: 使用脚本

```bash
python3 generate_novel_video.py \
  --prompt "一个美丽的山谷，有瀑布和彩虹，阳光透过云层洒下" \
  --output-dir outputs/novel_videos \
  --width 1280 --height 768 \
  --num-frames 120 --fps 24
```

### 方法2: 使用 Python API

```python
from generate_novel_video import NovelVideoGenerator

generator = NovelVideoGenerator()

result = generator.generate(
    prompt="你的小说场景描述",
    output_dir="outputs/novel_videos",
    width=1280,
    height=768,
    num_frames=120,
    fps=24,
)

print(f"图像: {result['image']}")
print(f"视频: {result['video']}")
```

### 方法3: 测试脚本

```bash
python3 test_novel_video.py
```

## 参数说明

- `--prompt`: 文本提示词（小说场景描述）
- `--output-dir`: 输出目录（可选）
- `--width`: 图像宽度（默认: 1280）
- `--height`: 图像高度（默认: 768）
- `--num-frames`: 视频帧数（默认: 120）
- `--fps`: 视频帧率（默认: 24）

## 注意事项

1. **显存需求**:
   - Flux: 约 8-12GB
   - HunyuanVideo: 约 18-24GB
   - 总计: 建议 24GB+ 显存

2. **模型下载**:
   - 确保 Flux 模型已下载到 `models/flux1-dev`
   - 确保 HunyuanVideo 模型已下载到配置的路径

3. **性能优化**:
   - 如果显存不足，可以降低分辨率（如 640x480）
   - 可以减少帧数（如 60 帧）
   - 可以降低推理步数（如 20-30 步）

## 优势

- ✅ **稳定性**: HunyuanVideo 图生视频比 CogVideoX 更稳定
- ✅ **质量**: Flux + HunyuanVideo 组合质量高
- ✅ **灵活性**: 可以分别调整图像和视频参数
- ✅ **易用性**: 一键生成，自动化流程

## 故障排除

### 问题1: 显存不足

**解决方案**:
- 降低分辨率（如 640x480）
- 减少帧数（如 60 帧）
- 启用 CPU offload

### 问题2: 图像生成失败

**检查**:
- Flux 模型是否正确加载
- 显存是否充足
- 提示词是否有效

### 问题3: 视频生成失败

**检查**:
- HunyuanVideo 模型是否正确加载
- 图像路径是否正确
- 显存是否充足（需要 18-24GB）

## 下一步

1. 批量生成多个场景
2. 添加音频合成
3. 添加字幕生成
4. 优化提示词工程
