# 小说推文批量生成工具使用指南

## 概述

批量生成工具用于处理 JSON 格式的场景文件，自动生成多个场景的视频。

## 功能特性

- ✅ 批量处理 JSON 场景文件
- ✅ 自动提取场景信息（prompt、镜头、运动强度等）
- ✅ 支持 M6 身份验证（自动启用/禁用）
- ✅ 错误重试机制
- ✅ 进度保存（支持断点续传）
- ✅ 详细生成报告（JSON + Markdown）

## 使用方法

### 基本用法

```bash
# 批量生成所有场景
python3 gen_video/tools/batch_novel_generator.py \
  --json lingjie/episode/1.v2-1.json \
  --output-dir outputs/batch_novel_test

# 快速模式（减少帧数，用于测试）
python3 gen_video/tools/batch_novel_generator.py \
  --json lingjie/episode/1.v2-1.json \
  --quick

# 禁用 M6 身份验证（纯场景或测试）
python3 gen_video/tools/batch_novel_generator.py \
  --json lingjie/episode/1.v2-1.json \
  --disable-m6
```

### 高级用法

```bash
# 分批处理（处理场景 0-10）
python3 gen_video/tools/batch_novel_generator.py \
  --json lingjie/episode/1.v2-1.json \
  --start 0 \
  --end 10

# 断点续传（从场景 10 开始）
python3 gen_video/tools/batch_novel_generator.py \
  --json lingjie/episode/1.v2-1.json \
  --start 10

# 自定义重试次数
python3 gen_video/tools/batch_novel_generator.py \
  --json lingjie/episode/1.v2-1.json \
  --max-retries 3
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--json` | JSON 场景文件路径（必需） | - |
| `--output-dir` | 输出目录 | `outputs/batch_novel_<timestamp>` |
| `--enable-m6` | 启用 M6 身份验证 | 启用 |
| `--disable-m6` | 禁用 M6 身份验证 | - |
| `--quick` | 快速模式（24帧，用于测试） | 否 |
| `--start` | 开始索引（断点续传） | 0 |
| `--end` | 结束索引（分批处理） | 全部 |
| `--max-retries` | 最大重试次数 | 2 |

## 输出结构

```
outputs/batch_novel_<timestamp>/
├── scene_000/          # 场景 0
│   ├── novel_image.png
│   └── novel_video.mp4
├── scene_001/          # 场景 1
│   ├── novel_image.png
│   └── novel_video.mp4
├── progress.json       # 进度文件（实时更新）
├── batch_report.json   # JSON 格式报告
└── batch_report.md     # Markdown 格式报告
```

## 报告格式

### JSON 报告 (`batch_report.json`)

```json
{
  "timestamp": "2025-12-19T10:30:00",
  "summary": {
    "total": 10,
    "success": 9,
    "errors": 1,
    "success_rate": "90.0%"
  },
  "errors": [
    {
      "scene_id": 5,
      "prompt": "韩立...",
      "error": "CUDA out of memory"
    }
  ]
}
```

### Markdown 报告 (`batch_report.md`)

包含：
- 摘要统计
- 失败场景详情
- 成功场景列表

## 场景 JSON 格式要求

场景 JSON 文件应包含以下字段：

```json
{
  "scenes": [
    {
      "scene_id": 0,
      "visual_constraints": {
        "environment": "场景描述"
      },
      "narration": {
        "text": "旁白文本"
      },
      "character": {
        "present": true,
        "id": "hanli"
      },
      "camera": {
        "shot": "medium"
      },
      "quality_target": {
        "motion_intensity": "moderate"
      },
      "width": 768,
      "height": 1152,
      "num_frames": 120,
      "target_fps": 24
    }
  ]
}
```

## 最佳实践

1. **首次运行使用快速模式**：
   ```bash
   python3 gen_video/tools/batch_novel_generator.py \
     --json lingjie/episode/1.v2-1.json \
     --quick \
     --start 0 \
     --end 5
   ```

2. **检查报告后再全量生成**：
   ```bash
   # 查看快速模式报告
   cat outputs/batch_novel_*/batch_report.md
   
   # 确认无误后全量生成
   python3 gen_video/tools/batch_novel_generator.py \
     --json lingjie/episode/1.v2-1.json
   ```

3. **分批处理大量场景**：
   ```bash
   # 第一批：0-20
   python3 gen_video/tools/batch_novel_generator.py \
     --json lingjie/episode/1.v2-1.json \
     --start 0 --end 20
   
   # 第二批：20-40
   python3 gen_video/tools/batch_novel_generator.py \
     --json lingjie/episode/1.v2-1.json \
     --start 20 --end 40
   ```

4. **断点续传**：
   ```bash
   # 如果中途失败，从上次停止的地方继续
   python3 gen_video/tools/batch_novel_generator.py \
     --json lingjie/episode/1.v2-1.json \
     --start 15  # 从场景 15 开始
   ```

## 故障排查

### 问题：显存不足

**解决方案**：
- 使用 `--quick` 模式（减少帧数）
- 分批处理（`--start` / `--end`）
- 检查是否有其他进程占用显存

### 问题：M6 验证失败

**解决方案**：
- 检查参考图是否存在：`reference_image/hanli_mid.jpg`
- 尝试禁用 M6：`--disable-m6`
- 检查场景中角色信息是否正确

### 问题：生成中断

**解决方案**：
- 查看 `progress.json` 了解已完成场景
- 使用 `--start` 参数从断点继续
- 检查日志文件了解错误原因

## 性能优化建议

1. **批量生成时**：
   - 使用快速模式测试前几个场景
   - 确认无误后全量生成
   - 分批处理避免显存溢出

2. **显存管理**：
   - 每个场景生成后会自动清理显存
   - 如果仍有问题，可以手动重启脚本

3. **错误处理**：
   - 默认重试 2 次
   - 可以通过 `--max-retries` 调整
   - 失败场景会在报告中详细记录

## 示例

### 示例 1：快速测试

```bash
# 测试前 3 个场景
python3 gen_video/tools/batch_novel_generator.py \
  --json lingjie/episode/1.v2-1.json \
  --quick \
  --start 0 \
  --end 3 \
  --output-dir outputs/test_batch
```

### 示例 2：生产环境批量生成

```bash
# 全量生成，启用 M6，正常模式
python3 gen_video/tools/batch_novel_generator.py \
  --json lingjie/episode/1.v2-1.json \
  --enable-m6 \
  --output-dir outputs/production_batch
```

### 示例 3：断点续传

```bash
# 从场景 10 继续生成
python3 gen_video/tools/batch_novel_generator.py \
  --json lingjie/episode/1.v2-1.json \
  --start 10
```

## 相关文档

- [小说推文生成指南](../README_小说推文.md)
- [M6 视频身份保持研究](../docs/M6_视频身份保持研究.md)
- [角色一致性升级进度追踪](../../角色一致性升级-进度追踪.md)

