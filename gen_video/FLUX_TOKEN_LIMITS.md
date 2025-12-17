# Flux vs SDXL Token 限制对比

## 📊 关键差异

### SDXL
- **编码器**：CLIP-L/14（单一编码器）
- **Token 限制**：**77 tokens**（硬限制）
- **原因**：CLIP tokenizer 的上下文长度固定为 77

### Flux
- **编码器**：**双编码器架构**
  1. **T5 编码器**（主要）：处理完整 prompt
     - 支持：**128, 256, 512 tokens**（可配置）
     - 当前代码设置：**128 tokens**
  2. **CLIP 编码器**（辅助）：提供额外语义
     - 限制：**77 tokens**（但主要用于辅助，不是主要限制）

## 🔍 代码中的实现

### 当前设置（`pulid_engine.py`）
```python
# T5 编码器：支持 128 tokens
self.t5 = load_t5(device="cpu", max_length=128)

# CLIP 编码器：77 tokens（辅助）
self.clip = load_clip(device="cpu")  # max_length=77
```

### 可调整范围
根据 `pulid_code/flux/util.py`：
```python
def load_t5(device: str = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work
```

**支持的 max_length 值**：
- 64 tokens（最小）
- **128 tokens**（当前默认，平衡性能和容量）
- 256 tokens（更长描述）
- 512 tokens（最长，但可能更慢）

## ✅ 你的 Prompt 是否超限？

你提供的 prompt：
```
Chinese ancient fairy style young man, wearing deep teal blue and light gray blue wide-sleeve robe, with gilded hollowed-out tangled branch patterns on the shoulders and neck, flowing cloud dark patterns on the clothes, inner black cross-collar束腰 garment, long black hair tied up with hairpins, standing in front of a fairy mountain with floating palaces and clouds, misty and ethereal atmosphere, hyper-detailed, realistic skin texture, cinematic lighting, 8k --ar 3:4 --style expressive
```

**估算 token 数**：
- 英文单词数：约 80-90 个单词
- 中文字符：2 个（"束腰"）
- **预计 token 数**：约 **100-110 tokens**

**结论**：
- ❌ **超过当前 128 tokens 限制**（如果包含所有细节）
- ✅ **但可以通过调整 max_length 到 256 来解决**

## 🛠️ 解决方案

### 方案 1：提高 T5 max_length（推荐）

修改 `pulid_engine.py`：
```python
# 从 128 提高到 256，支持更长的 prompt
self.t5 = load_t5(device="cpu", max_length=256)
```

**优点**：
- 支持更详细的 prompt
- 不需要精简描述
- 性能影响较小（256 仍然很快）

### 方案 2：精简 Prompt（如果不想改代码）

使用优化后的版本（已在 `optimized_prompt.txt` 中提供），去掉冗余描述。

### 方案 3：动态调整（最佳实践）

根据 prompt 长度动态选择 max_length：
```python
def _estimate_t5_tokens(self, prompt: str) -> int:
    """估算 T5 token 数"""
    # 简单估算：英文约 1.3 tokens/词，中文约 1.5 tokens/字
    words = len(prompt.split())
    chinese_chars = sum(1 for c in prompt if ord(c) > 127)
    estimated = int(words * 1.3 + chinese_chars * 1.5)
    return estimated

# 根据估算值选择 max_length
estimated = self._estimate_t5_tokens(prompt)
if estimated <= 128:
    max_length = 128
elif estimated <= 256:
    max_length = 256
else:
    max_length = 512

self.t5 = load_t5(device="cpu", max_length=max_length)
```

## ⚠️ 注意事项

1. **性能影响**：
   - 128 tokens：最快
   - 256 tokens：稍慢（约 +10-20%）
   - 512 tokens：明显更慢（约 +30-50%）

2. **显存影响**：
   - 更长的序列需要更多显存
   - 256 tokens 通常可以接受
   - 512 tokens 可能需要更多显存

3. **实际限制**：
   - T5 编码器理论上支持到 512 tokens
   - 但实际使用中，128-256 tokens 已经足够
   - 超过 256 tokens 的 prompt 通常可以精简

## 📝 建议

对于你的详细 prompt：
1. **短期**：使用优化后的精简版本（版本 2 或 3）
2. **长期**：将 T5 max_length 提高到 256，支持更详细的描述

这样既能保持 prompt 的完整性，又不会显著影响性能。

