# CosyVoice 声音克隆问题排查总结

## 问题描述

1. **生成时长异常**：生成 20-26 秒音频（正常应该 3-4 秒）
2. **声音完全听不清**：即使使用之前正常工作的配置，现在也出现问题

## 关键发现

### 1. test_cosyvoice.wav 是正常的
- **文件**: `outputs/test_cosyvoice.wav`
- **生成时间**: 2025-11-17 20:08
- **时长**: 10.20 秒（正常）
- **说明**: 这个文件在 2025-11-17 生成时是正常的，说明当时代码是正常工作的

### 2. 当前测试结果
- **test_cosyvoice_fixed.wav**: 26.20 秒（异常）
- **test_haoran_voice.wav**: 20.00 秒（异常）
- **使用相同配置**: `zero_shot_prompt_clean.wav` + 对应的 `prompt_text`

## 已完成的修复

1. ✅ **移除 speed 参数**：官方示例中没有传递该参数
2. ✅ **使用 stream=False**：按照官方示例
3. ✅ **不传递 text_frontend 参数**：使用 API 默认值 `True`
4. ✅ **代码简化**：移除了大量调试信息和检查

## 官方示例对比

### 官方示例代码
```python
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

for i, j in enumerate(cosyvoice.inference_zero_shot(
    '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
    '希望你以后能够做的比我还好呦。',
    prompt_speech_16k,
    stream=False
)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```

### 官方 README 重要提示
```python
# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
```

## 当前代码状态

代码已按照官方示例调整：
- ✅ 使用 `stream=False`
- ✅ 不传递 `speed` 参数
- ✅ 使用配置中的 `text_frontend` 值（当前设置为 `false`）

## 可能的原因

由于 `test_cosyvoice.wav` 在 2025-11-17 生成时是正常的，但现在使用相同配置也出现问题，可能的原因：

1. **依赖库版本变化**
   - PyTorch: 2.8.0+cu126（较新版本）
   - TorchAudio: 2.8.0+cu126
   - 可能某些版本更新导致行为变化

2. **CosyVoice 仓库更新**
   - 检查是否有代码更新影响生成逻辑

3. **环境配置变化**
   - CUDA 版本、驱动版本等

## 建议的下一步

1. **检查依赖库版本**：对比 2025-11-17 时的版本
2. **检查 CosyVoice 仓库**：确认是否有更新
3. **使用 text_frontend=False**：按照官方 README 建议测试
4. **检查生成音频质量**：虽然时长异常，但检查声音是否清晰

## 相关文件

- 正常工作的示例：`outputs/test_cosyvoice.wav` (10.20 秒)
- 异常示例：`outputs/test_cosyvoice_fixed.wav` (26.20 秒)
- 配置文件：`gen_video/config.yaml`
- TTS 生成器：`gen_video/tts_generator.py`

