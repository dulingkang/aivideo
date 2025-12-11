# CosyVoice 声音克隆问题排查总结

## 问题描述

生成的语音时长异常（20秒而不是正常的3-4秒），即使：
1. `prompt_text` 与生成文本长度相近（26字符 vs 25字符）
2. 使用 `instruct2` 模式（不需要精确匹配）
3. 音频格式正确（16kHz WAV，已清理）

## 测试结果

### 测试1：zero_shot 模式（之前正常工作的配置）
- **prompt_speech**: `zero_shot_prompt_clean.wav` (10秒)
- **prompt_text**: "大家好, 我是云卷仙音，今天我们要继续讲述凡人修仙转的故事，在这个故事中,韩立经历了无数的挑" (46字符)
- **生成文本**: "欢迎来到本期知识探索，我们将继续介绍黑洞的基本概念。" (26字符)
- **结果**: ❌ 生成时长 20.80 秒（异常）

### 测试2：zero_shot 模式（haoran 音频，短 prompt_text）
- **prompt_speech**: `haoran_prompt_clean.wav` (7.72秒)
- **prompt_text**: "欢迎来到本期知识探索，我们将继续介绍黑洞的基本概念。" (26字符)
- **生成文本**: "大家好，我是科普主持人。今天我们来聊聊科学的奥秘。" (25字符)
- **结果**: ❌ 生成时长 20.00 秒（异常）

### 测试3：instruct2 模式
- **prompt_speech**: `haoran_prompt_clean.wav` (7.72秒)
- **instruction**: "用专业、清晰、亲切的科普解说风格，语速适中，吐字清晰"
- **生成文本**: "欢迎来到本期知识探索，我们将继续介绍黑洞的基本概念。" (26字符)
- **结果**: ❌ 生成时长 20.80 秒（异常）

## 问题分析

### 1. 生成 token 数量异常
- **正常**: 约 40-50 tokens（对应3-4秒音频）
- **异常**: 约 250 tokens（对应20秒音频）

### 2. 可能的原因
1. **模型内部 max_len 计算错误**
   - `text_len` 经过 `encode` 后长度可能改变，但 `prompt_text_len` 没有经过 encode
   - 导致 `max_len = (text_len - prompt_text_len) * max_token_text_ratio` 计算错误
   - 如果模型没有正确停止（没有生成 `speech_token_size`），就会生成到 `max_len`

2. **音频格式问题**
   - 虽然音频已清理并转换为 16kHz WAV，但可能仍有格式问题
   - 音频数值范围接近边界（最大值: 0.8490），但代码中已禁用归一化处理

3. **代码改动导致的问题**
   - 之前的修复可能引入了新问题
   - `_load_prompt_speech` 方法的改动可能影响了音频加载

## 建议的解决方案

### 方案1：检查 CosyVoice2 模型版本
- 确认使用的是正确的模型版本
- 检查是否有模型更新或修复

### 方案2：尝试不同的音频
- 使用之前正常工作的 `zero_shot_prompt_clean.wav` 进行对比
- 如果这个音频也异常，说明是代码问题
- 如果这个音频正常，说明是 haoran 音频的问题

### 方案3：检查模型参数
- 检查 `max_token_text_ratio` 参数
- 检查 `min_token_text_ratio` 参数
- 尝试调整这些参数

### 方案4：回退代码改动
- 检查之前的代码版本
- 回退可能导致问题的改动
- 特别是 `_load_prompt_speech` 方法的改动

### 方案5：使用其他 TTS 引擎
- 如果 CosyVoice2 问题无法解决，可以考虑使用其他 TTS 引擎
- 例如：ChatTTS、Coqui TTS 等

## 下一步行动

1. ✅ 已创建排查工具：`debug_cosyvoice_voice.py`
2. ✅ 已识别音频内容：使用 Whisper 识别了音频实际内容
3. ✅ 已测试多种配置：zero_shot、instruct2 模式
4. ⏳ 需要检查：CosyVoice2 模型版本和参数
5. ⏳ 需要检查：之前的代码版本，找出导致问题的改动

## 相关文件

- 排查工具：`gen_video/tools/debug_cosyvoice_voice.py`
- 测试脚本：`gen_video/tools/test_haoran_voice.py`
- 测试脚本：`gen_video/tools/test_instruct2_mode.py`
- 配置文件：`gen_video/config.yaml`
- TTS 生成器：`gen_video/tts_generator.py`

