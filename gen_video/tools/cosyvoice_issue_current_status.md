# CosyVoice 生成时长异常问题 - 当前状态

## 问题描述

- **正常文件**: `test_cosyvoice.wav` (10.20 秒，2025-11-17 生成，41 字符)
- **当前生成**: 
  - 25 字符 → 20.0 秒（异常，预期约 7 秒）
  - 41 字符 → 25.84 秒（异常，预期约 12 秒）

## 关键观察

1. **生成时长与文本长度不成正比**
   - 25 字符生成 20.0 秒
   - 41 字符生成 25.84 秒
   - 说明问题可能不在 max_len 计算

2. **已尝试的修复**
   - ✅ 完全按照官方示例，不传递 `text_frontend` 和 `speed`
   - ✅ 使用 `stream=False`
   - ✅ 使用 `fp16=False`
   - ✅ 调整 prompt_text 长度（10-28 字符）
   - ❌ 问题仍然存在

## 当前配置

- **prompt_speech**: `zero_shot_prompt_clean.wav` (10秒，312.54 KB)
- **prompt_text**: "大家好，我是云卷仙音，今天我们要继续讲述凡人修仙转的故事" (28字符)
- **生成文本**: "大家好，我是科普主持人。今天我们来聊聊科学的奥秘。" (25字符)
- **比例**: 0.89x（正常范围 0.5-2.0x）

## 可能的原因

1. **prompt_speech 和 prompt_text 不匹配**
   - 当前使用的 `zero_shot_prompt_clean.wav` 可能不匹配 prompt_text
   - 需要检查音频实际内容

2. **模型停止条件问题**
   - 模型没有在 `speech_token_size` 时停止
   - 而是生成到了某个固定的长度

3. **token2wav 转换问题**
   - 生成的 token 数量可能正常，但转换为音频时出现问题

## 建议的下一步

1. **检查 prompt_speech 和 prompt_text 匹配**
   - 使用音频识别工具确认 `zero_shot_prompt_clean.wav` 的实际内容
   - 确保 prompt_text 与音频内容完全匹配

2. **尝试使用官方示例的完整配置**
   - 使用官方示例的 prompt_speech 和 prompt_text
   - 对比生成结果

3. **检查 token 生成过程**
   - 查看实际生成的 token 数量
   - 检查是否在 `speech_token_size` 时停止

## 相关文件

- 配置文件：`gen_video/config.yaml`
- TTS 生成器：`gen_video/tts_generator.py`
- CosyVoice LLM：`CosyVoice/cosyvoice/llm/llm.py`
- 正常文件：`outputs/test_cosyvoice.wav` (10.20 秒)

