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

### 方法1.1：启用“角色一致 + 视频一致”（推荐用于凡人修仙人物出镜推文）

> 说明：脚本现在支持**自动识别是否包含韩立**：prompt/scene 中出现“韩立/Han Li/hanli”时，会自动按角色模式生成，并对韩立场景自动启用 M6（否则不启用）。
> 你仍然可以用参数强制开启/关闭。

```bash
python3 generate_novel_video.py \
  --shot-type medium_close --motion-intensity moderate \
  --prompt "韩立在竹林中快步穿行，衣袍随风摆动，神情警惕，远处有淡淡雾气与月光" \
  --output-dir outputs/novel_videos_hanli \
  --width 768 --height 1152 \
  --num-frames 120 --fps 24
```

如果你有更稳定的参考脸，可显式指定（否则会尝试自动找 `reference_image/<character_id>_mid.jpg`）：

```bash
python3 generate_novel_video.py \
  --reference-image-path reference_image/hanli_mid.jpg \
  --prompt "..." \
  --output-dir outputs/novel_videos_hanli
```

强制纯场景（即使 prompt 提到了韩立也不走角色/M6）：

```bash
python3 generate_novel_video.py --force-scene --prompt "..." --output-dir outputs/novel_scene_only
```

强制关闭 M6（仍可生成带韩立的画面，但不做身份验证/重试）：

```bash
python3 generate_novel_video.py --disable-m6-identity --prompt "韩立..." --output-dir outputs/novel_no_m6
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

### 角色一致 / 视频一致相关参数

- `--include-character`: 启用角色模式（人物出镜，复用角色一致系统）
- `--force-scene`: 强制纯场景模式（忽略自动推断/手动角色）
- `--auto-character/--no-auto-character`: 是否自动识别是否包含韩立（默认开启）
- `--character-id`: 角色ID（可选，覆盖自动推断）
- `--image-model-engine`: 覆盖图片引擎（默认：场景模式 `flux1`；角色模式 `auto`）
- `--enable-m6-identity`: 强制启用 M6
- `--disable-m6-identity`: 强制关闭 M6
- `--auto-m6-identity/--no-auto-m6-identity`: 是否对韩立场景自动启用 M6（默认开启）
- `--reference-image-path`: 身份验证参考图（可选）
- `--shot-type`: 镜头类型（wide/medium/medium_close/close/extreme_close）
- `--motion-intensity`: 运动强度（gentle/moderate/dynamic）
- `--m6-max-retries`: M6 最大重试次数（0=不重试；None=用 config.yaml）
- `--m6-quick`: M6 快速模式（更少步数/默认不重试，冒烟用）

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
