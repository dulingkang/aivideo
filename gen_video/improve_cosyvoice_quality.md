# CosyVoice2 声音质量优化指南

## 发现的问题

1. **警告**: `synthesis text too short than prompt text` 
   - 生成的文本比 prompt_text 短很多，导致质量差
   - 建议：使用更短的 prompt_text（5-15秒对应）

2. **prompt_text 太长**: 112字符，但实际音频可能只有部分内容匹配
   - 建议：使用与 prompt_speech 实际内容精确匹配的短文本

3. **静音比例高**: 47.8%的静音可能影响质量
   - 已提取清晰片段：`assets/prompts/zero_shot_prompt_clean.wav` (10秒)

## 优化建议

### 方案1：使用清晰的 prompt_speech 片段（推荐）

1. 使用已提取的清晰片段：
   ```yaml
   prompt_speech: assets/prompts/zero_shot_prompt_clean.wav
   prompt_text: "（请根据音频实际内容填写，约5-15秒对应的文本）"
   ```

2. prompt_text 应该：
   - 与 prompt_speech 中的实际内容**完全匹配**
   - 长度适中（约20-50字符，对应5-15秒）
   - 包含清晰、有代表性的语音特征

### 方案2：优化现有配置

如果继续使用 `reference_audio.mp3`：
- 确保 prompt_text 与音频中的实际内容匹配
- 建议使用音频前10-15秒对应的文本

### 方案3：下载 CosyVoice-ttsfrd（可选）

CosyVoice-ttsfrd 可以改善文本标准化，可能提升生成质量：
```python
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
# 然后解压 resource.zip
```

## 测试建议

1. 使用短文本测试（20-30字符）
2. 确保 prompt_text 与 prompt_speech 内容匹配
3. 对比使用清晰片段和原始文件的效果
