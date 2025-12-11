# CosyVoice 生成时长异常问题 - 最终总结

## 问题描述

- **正常文件**: `test_cosyvoice.wav` (10.20 秒，2025-11-17 生成)
- **当前生成**: 25-32 秒（异常）

## 已完成的修复

1. ✅ **完全按照官方示例**：不传递 `text_frontend` 和 `speed` 参数
2. ✅ **使用 stream=False**：按照官方示例
3. ✅ **使用 fp16=False**：配置中 `quantization: fp32`
4. ✅ **代码调用方式**：完全匹配 GitHub 官方示例

## 测试结果

### 测试 1: 使用官方示例的 prompt_text（15字符）
- **prompt_text**: "希望你以后能够做的比我还好呦。"
- **生成时长**: 32.80 秒 ❌
- **结论**: 问题不在 prompt_text 长度

### 测试 2: 使用短 prompt_text（10字符）
- **prompt_text**: "大家好，我是云卷仙音"
- **生成时长**: 25.84 秒 ❌
- **结论**: 问题不在 prompt_text 长度

## 关键发现

### 1. max_len 计算问题

在 `CosyVoice/cosyvoice/llm/llm.py` 中：

```python
max_len = int((text_len - prompt_text_len) * max_token_text_ratio)
```

**关键问题**：
- `text_len` 是经过 encode 后的长度
- `prompt_text_len` 是原始长度（没有经过 encode）
- 如果 `text_len` 经过 encode 后变小，但 `prompt_text_len` 没有变小，`max_len` 会很大
- 如果模型没有正确停止（没有生成 `speech_token_size`），就会生成到 `max_len`，导致生成过多 token

### 2. prompt_speech 和 prompt_text 匹配问题

- **当前 prompt_speech**: `zero_shot_prompt_clean.wav` (10秒，46字符文本)
- **官方示例 prompt_speech**: 可能不同
- **问题**: 如果 prompt_speech 和 prompt_text 不匹配，可能导致生成异常

## 可能的原因

1. **prompt_speech 和 prompt_text 不匹配**
   - 当前使用的 `zero_shot_prompt_clean.wav` 可能不匹配官方示例的 prompt_text
   - 需要检查音频实际内容

2. **max_len 计算错误**
   - `text_len` 和 `prompt_text_len` 的 encode 差异导致 `max_len` 过大
   - 模型生成到 `max_len` 才停止，而不是在 `speech_token_size` 时停止

3. **模型停止条件问题**
   - 模型没有正确识别 `speech_token_size`，导致继续生成

## 建议的下一步

1. **检查调试输出**
   - 查看实际的 `max_len`、`text_len`、`prompt_text_len` 值
   - 检查是否在 `speech_token_size` 时停止

2. **检查 prompt_speech 和 prompt_text 匹配**
   - 使用音频识别工具确认 `zero_shot_prompt_clean.wav` 的实际内容
   - 确保 prompt_text 与音频内容完全匹配

3. **尝试使用官方示例的完整配置**
   - 使用官方示例的 prompt_speech 和 prompt_text
   - 对比生成结果

## 相关文件

- 配置文件：`gen_video/config.yaml`
- TTS 生成器：`gen_video/tts_generator.py`
- CosyVoice LLM：`CosyVoice/cosyvoice/llm/llm.py`
- 正常文件：`outputs/test_cosyvoice.wav` (10.20 秒)

