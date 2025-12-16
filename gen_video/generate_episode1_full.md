# 生成第一集完整视频指南

## 使用的脚本
**`main.py`** - 这是完整的视频生成流水线脚本，可以处理所有场景并自动合并。

## 运行命令

### 方法1：从 gen_video 目录运行（推荐）
```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
python main.py --script lingjie/episode/1.json --output lingjie_ep1_full
```

### 方法2：使用绝对路径
```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
python main.py --script /vepfs-dev/shawn/vid/fanren/lingjie/episode/1.json --output lingjie_ep1_full
```

### 方法3：如果需要代理
```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
proxychains4 python main.py --script lingjie/episode/1.json --output lingjie_ep1_full
```

## 参数说明

- `--script`: 脚本文件路径（第一集的JSON文件）
- `--output`: 输出名称（最终视频会保存为 `lingjie_ep1_full/lingjie_ep1_full.mp4`）

## 可选的跳过参数

如果某些步骤已经完成，可以跳过：

```bash
# 跳过图像生成（如果图像已存在）
python main.py --script lingjie/episode/1.json --output lingjie_ep1_full --skip-video

# 跳过TTS生成（如果音频已存在）
python main.py --script lingjie/episode/1.json --output lingjie_ep1_full --skip-tts

# 跳过字幕生成
python main.py --script lingjie/episode/1.json --output lingjie_ep1_full --skip-subtitle

# 跳过视频合成（如果只是想重新生成某个步骤）
python main.py --script lingjie/episode/1.json --output lingjie_ep1_full --skip-compose
```

## 完整流程

`main.py` 会自动执行以下步骤：

1. ✅ **生成所有场景的图像**（如果不存在）
2. ✅ **生成所有场景的配音**（TTS，使用云卷仙音）
3. ✅ **生成所有场景的视频**（使用 HunyuanVideo）
4. ✅ **生成字幕**（基于完整音频）
5. ✅ **合并所有视频片段**（按顺序拼接）
6. ✅ **合成最终视频**（添加音频和字幕）

## 输出位置

最终视频会保存在：
```
gen_video/outputs/lingjie_ep1_full/lingjie_ep1_full.mp4
```

中间产物（图像、视频片段等）会保存在：
```
gen_video/outputs/lingjie_ep1_full/
```

## 注意事项

1. **生成时间**：第一集包含多个场景，完整生成可能需要较长时间
2. **显存占用**：确保有足够的GPU显存（建议20GB+）
3. **如果中途失败**：可以重新运行，已生成的文件会被跳过（除非使用 `--force`）

## 强制重新生成

如果需要强制重新生成所有内容：
```bash
python main.py --script lingjie/episode/1.json --output lingjie_ep1_full --force
```
