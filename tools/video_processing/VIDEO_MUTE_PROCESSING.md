# 视频静音处理说明

## 概述

所有视频拼接和裁剪操作默认都会**去掉音频轨道（静音）**，以便后续统一添加BGM和旁白。

## 已修改的函数

### 1. `trim_video()` - 裁剪视频

**位置**: `match_scenes_to_narration.py`

**修改前**:
- 使用 `-c copy` 直接复制流
- 保留原始音频

**修改后**:
- 默认 `mute=True`（静音）
- 使用 `-an` 去掉音频轨道
- 重新编码视频（`-c:v libx264`）确保兼容性

**函数签名**:
```python
def trim_video(input_path: Path, output_path: Path, duration: float, mute: bool = True):
```

### 2. `concatenate_videos()` - 拼接多个视频

**位置**: `match_scenes_to_narration.py`

**修改前**:
- 使用 `-c copy` 直接复制流
- 保留原始音频（可能导致音视频不同步）

**修改后**:
- 默认 `mute=True`（静音）
- 使用 `-an` 去掉音频轨道
- 重新编码视频（`-c:v libx264`）确保所有片段参数一致

**函数签名**:
```python
def concatenate_videos(video_paths: List[Path], output_path: Path, mute: bool = True):
```

### 3. `create_opening_video.py` - 开头视频合成

**已包含静音处理**:
- `trim_video()` - 支持静音参数
- `concatenate_videos()` - 支持静音参数
- 默认所有片段都静音

## 处理流程

```
原始场景视频（有音频）
    ↓
裁剪/拼接处理（去掉音频，静音）
    ↓
输出视频（无音频轨道）
    ↓
后续统一添加BGM + 旁白
```

## 优势

1. **统一音频处理**: 所有视频片段都是静音的，便于后续统一添加BGM和旁白
2. **避免音视频不同步**: 重新编码确保所有片段参数一致
3. **简化工作流**: 不需要担心原始音频的干扰
4. **更好的质量控制**: 统一编码参数，保证输出质量

## 使用示例

### 裁剪视频（默认静音）
```python
from pathlib import Path
from match_scenes_to_narration import trim_video

# 裁剪到5秒，默认静音
trim_video(
    input_path=Path("input.mp4"),
    output_path=Path("output.mp4"),
    duration=5.0
)

# 如果需要保留音频（不推荐）
trim_video(
    input_path=Path("input.mp4"),
    output_path=Path("output.mp4"),
    duration=5.0,
    mute=False
)
```

### 拼接视频（默认静音）
```python
from pathlib import Path
from match_scenes_to_narration import concatenate_videos

# 拼接多个视频，默认静音
concatenate_videos(
    video_paths=[
        Path("video1.mp4"),
        Path("video2.mp4"),
        Path("video3.mp4")
    ],
    output_path=Path("output.mp4")
)

# 如果需要保留音频（不推荐）
concatenate_videos(
    video_paths=[...],
    output_path=Path("output.mp4"),
    mute=False
)
```

## 技术细节

### 编码参数
- **视频编码**: `libx264`
- **视频预设**: `fast`（快速编码）
- **质量控制**: `CRF 23`（高质量）
- **音频**: 使用 `-an` 去掉音频轨道

### 为什么重新编码而不是 `-c copy`？

1. **兼容性问题**: 不同片段的编码参数可能不一致
2. **时间戳问题**: 直接复制可能导致时间戳错乱，画面定格
3. **静音处理**: `-c copy` 无法去掉音频轨道

### 性能考虑

- 重新编码会稍微慢一些，但能保证质量
- 使用 `fast` 预设平衡速度和质量
- CRF 23 提供高质量输出

## 注意事项

1. **默认行为**: 所有视频默认静音，如需保留音频需要显式设置 `mute=False`
2. **编码时间**: 重新编码需要一些时间，但比视频播放速度更快
3. **文件大小**: 重新编码后文件大小可能略有变化，但质量保持一致

