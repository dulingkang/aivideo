# 根据 episode_171.json 制作视频的完整流程

## 概述

根据 `renjie/episode_171.json` 文件，自动检索匹配的视频场景，拼接生成完整视频。

## 前置条件

1. ✅ 静音视频已处理完成（所有场景视频都在 `processed/episode_*/scenes/` 目录）
2. ✅ 开头视频已制作：`renjie/opening_video_raw.mp4`
3. ✅ 结尾视频已制作：`renjie/ending_video_raw.mp4`（如已制作）
4. ✅ 场景索引已构建：`processed/global_index.faiss`

## JSON 文件结构

`renjie/episode_171.json` 包含以下字段：

- `episode`: 集数（171）
- `title`: 标题
- `scenes`: 场景数组
  - `id`: 场景ID（0=开头，999=结尾，其他=普通场景）
  - `duration`: 时长（秒）
  - `description`: 场景描述（中文）
  - `narration`: 旁白文本
  - `visual`: 视觉描述（用于检索）

## 工作流程

### 步骤1: 根据JSON生成视频

使用 `generate_video_from_json.py` 脚本：

```bash
cd /vepfs-dev/shawn/vid/fanren
source /vepfs-dev/shawn/venv/py312/bin/activate

python3 tools/video_processing/generate_video_from_json.py \
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
    --output renjie/episode_171_video.mp4
```

**功能说明：**

1. **开头场景（id=0）**
   - 使用 `--opening` 指定的开头视频
   - 根据JSON中的 `duration` 裁剪到指定时长

2. **普通场景（id=1-998）**
   - 根据 `description` 或 `narration` 检索匹配的视频场景
   - 使用智能决策（`decision_make`）选择最佳视频
   - 根据 `duration` 匹配视频时长（裁剪或拼接）

3. **结尾场景（id=999）**
   - 使用 `--ending` 指定的结尾视频
   - 根据JSON中的 `duration` 裁剪到指定时长

4. **拼接所有片段**
   - 按顺序拼接：开头 → 中间场景 → 结尾
   - 输出完全静音的视频（无音频轨道）

### 步骤2: 添加BGM和旁白（后续）

生成的是静音视频，后续需要：

1. **生成旁白音频**（TTS）
   - 根据JSON中的 `narration` 字段生成音频

2. **添加背景音乐（BGM）**

3. **合成最终视频**
   - 将旁白和BGM混合
   - 添加到视频中

## 输出结果

- **视频文件**: `renjie/episode_171_video.mp4`
  - 完全静音（无音频轨道）
  - 包含所有场景的视频片段
  - 已按顺序拼接

- **临时文件**: `processed/temp/` 目录
  - 裁剪的临时视频片段
  - 可手动清理

## 注意事项

1. **视频路径查找**
   - 脚本会自动查找场景视频文件
   - 支持多种命名格式

2. **时长匹配**
   - 如果视频太长：裁剪到目标时长
   - 如果视频太短：使用智能决策选择是否拼接多个视频
   - 容差范围：默认0.5秒

3. **检索质量**
   - 使用混合检索（向量+关键词）
   - 优先使用检索到的原版视频
   - 尽量避免AI生成

4. **性能优化**
   - 所有视频已静音，可以使用 `-c copy` 快速拼接
   - 只在需要裁剪时才重新编码

## 故障排查

### 问题1: 找不到匹配的视频场景

**解决方法：**
- 检查索引是否正确构建
- 检查场景metadata文件是否存在
- 调整检索参数（top_k）

### 问题2: 视频时长不匹配

**解决方法：**
- 检查JSON中的duration是否正确
- 检查视频文件是否存在
- 查看日志中的匹配策略

### 问题3: 视频拼接失败

**解决方法：**
- 检查所有视频片段是否存在
- 检查视频编码格式是否一致
- 查看ffmpeg错误信息

## 下一步

生成静音视频后，需要：

1. **生成旁白音频**（TTS）
   - 提取所有 `narration` 文本
   - 使用TTS生成音频文件

2. **准备BGM**
   - 选择合适的背景音乐
   - 调整音量混合

3. **最终合成**
   - 将旁白和BGM混合
   - 添加到视频中
   - 输出最终视频

