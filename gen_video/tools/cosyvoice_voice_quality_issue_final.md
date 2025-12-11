# CosyVoice 声音质量问题 - 最终排查

## 问题描述

即使使用官方示例的完整配置，生成的音频声音仍然不清楚（"wuluwulu"）。

## 当前状态

- ✅ 已使用官方示例的完整配置：
  - `prompt_speech`: `/vepfs-dev/shawn/vid/fanren/CosyVoice/asset/zero_shot_prompt.wav`
  - `prompt_text`: `"希望你以后能够做的比我还好呦。"`
  - `mode`: `zero_shot`
- ✅ 已修复 max_len 计算（使用 text_len 作为基准）
- ✅ 已添加 stop_token 检测逻辑
- ⚠️ 音频时长已正常（11.20 秒）
- ❌ 但声音仍然不清楚（"wuluwulu"）

## 关键发现

1. **模型没有在 stop_token 时停止**
   - stop_token 概率都在 -8 到 -10 之间
   - stop_token 不在 top_k 中
   - 模型生成到了 max_len (280)

2. **即使使用官方示例配置，问题仍然存在**
   - 说明问题可能不在 prompt_speech 和 prompt_text 的匹配
   - 可能是我们的修复导致的问题

## 可能的原因

1. **我们的修复导致的问题**
   - max_len 限制可能影响了生成质量
   - stop_token 检测逻辑可能有问题
   - 其他代码修改可能影响了生成

2. **生成的 token 序列无效**
   - 虽然时长正常，但 token 序列可能不正确
   - 可能需要检查生成的 token 是否在有效范围内（0-6560）

3. **采样方法问题**
   - ras_sampling 可能没有正确采样
   - 可能需要调整采样参数（top_k, top_p）

## 建议的解决方案

### 方案 1: 检查正常文件生成时的代码版本
- 查看 2025-11-17 时的代码版本
- 对比差异，找出可能的问题

### 方案 2: 尝试移除我们的修复
- 暂时移除 max_len 限制
- 暂时移除 stop_token 检测逻辑
- 看看是否能恢复声音质量

### 方案 3: 检查其他代码修改
- 检查 `tts_generator.py` 是否有其他修改
- 检查音频预处理步骤是否有问题

## 相关文件

- 修复文件：`CosyVoice/cosyvoice/llm/llm.py`
- 配置文件：`gen_video/config.yaml`
- 测试文件：`gen_video/outputs/test_dynamic_maxlen.wav`

