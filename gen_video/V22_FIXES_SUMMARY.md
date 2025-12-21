# v2.2-final 问题修复总结

## 修复时间
2025-12-21

## 修复内容

### 1. ✅ 修复推理步数问题

**问题**：
- JSON 配置：`num_inference_steps: 50`
- 实际使用：28 步

**修复**：
- 在 `enhanced_image_generator.py` 的 `_generate_with_pulid` 方法中添加了调试日志
- 确保从 JSON 读取的 `num_steps` 正确传递到 PuLID 引擎
- 添加了 `print` 语句确认读取的值

**修改文件**：
- `gen_video/enhanced_image_generator.py` (lines 1122-1150)

**关键代码**：
```python
num_steps = gen_params.get("num_inference_steps", self.pulid_config.get("num_inference_steps", 28))
logger.info(f"使用JSON中的生成参数: {width}x{height}, {num_steps}步, guidance={guidance}")
print(f"  [调试] 从JSON读取的推理步数: {num_steps}")
```

---

### 2. ✅ 修复人脸相似度低的问题

**问题**：
- 场景3（scene_003）：0.412 < 0.7 阈值
- 场景5（scene_004）：0.033 < 0.7 阈值（严重）

**根本原因**：
- PuLID 引擎未加载 LoRA 权重

**修复**：
1. **在 `enhanced_image_generator.py` 中读取 LoRA 配置**：
   - 从 `scene.character.lora_config` 读取 LoRA 路径和权重
   - 通过 `kwargs` 传递给 PuLID 引擎

2. **在 `pulid_engine.py` 中添加 LoRA 加载逻辑**：
   - 在 `generate_with_identity` 方法中，在 `load_pipeline()` 之后加载 LoRA
   - 使用 `pipeline.load_lora_weights()` 和 `pipeline.set_adapters()` 加载和设置 LoRA

**修改文件**：
- `gen_video/enhanced_image_generator.py` (lines 1134-1150)
- `gen_video/pulid_engine.py` (lines 843-870)

**关键代码**：
```python
# enhanced_image_generator.py
lora_config = character_info.get("lora_config", {})
if lora_config:
    pulid_kwargs['lora_config'] = lora_config

# pulid_engine.py
lora_config = kwargs.get('lora_config', None)
if lora_config and self.pipeline is not None:
    lora_path = lora_config.get('lora_path', '')
    lora_weight = lora_config.get('weight', 0.9)
    if lora_path and Path(lora_path).exists():
        self.pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
        if hasattr(self.pipeline, 'set_adapters'):
            self.pipeline.set_adapters([adapter_name], adapter_weights=[lora_weight])
```

---

### 3. ✅ 修复 Prompt 截断问题

**问题**：
- 场景5（scene_004）：80 tokens > 77，部分内容被截断
- 警告：`Token indices sequence length is longer than the specified maximum sequence length for this model (80 > 77)`
- 被截断部分：`['fantasy illustration style']`

**根本原因**：
- 所有模型都使用了 CLIP 的 77 tokens 限制
- Flux 模型使用 T5 tokenizer，支持 512 tokens

**修复**：
- 在 `execution_executor_v21.py` 的 `_build_prompt` 方法中，根据模型类型选择 token 限制
- Flux 模型：512 tokens（T5 限制）
- SDXL 模型：77 tokens（CLIP 限制）

**修改文件**：
- `gen_video/utils/execution_executor_v21.py` (lines 391-420)

**关键代码**：
```python
# 根据模型类型选择token限制
model_route = scene.get("model_route", {})
base_model = model_route.get("base_model", "flux")
if base_model == "flux":
    max_tokens = 512  # Flux 使用 T5，支持更长的 prompt
else:
    max_tokens = 77   # SDXL 使用 CLIP，限制为 77
```

---

## 测试建议

### 1. 测试推理步数
- 创建一个测试场景，设置 `num_inference_steps: 50`
- 检查日志输出，确认显示 "使用JSON中的生成参数: ... 50步"
- 检查实际生成的步数是否为 50

### 2. 测试 LoRA 加载
- 创建一个测试场景，明确指定 LoRA 路径
- 检查日志输出，确认显示 "加载 LoRA: ..." 和 "✓ LoRA 加载成功"
- 检查生成图像的人脸相似度是否提高

### 3. 测试 Prompt 优化
- 创建一个长 Prompt 测试场景（>77 tokens）
- 检查日志输出，确认显示 "使用 Flux 模型，Prompt token 限制: 512"
- 检查生成图像是否包含所有关键信息

---

## 相关文件

### 修改的文件
1. `gen_video/enhanced_image_generator.py`
   - 添加 LoRA 配置读取和传递
   - 添加调试日志

2. `gen_video/pulid_engine.py`
   - 添加 LoRA 加载逻辑
   - 在 `generate_with_identity` 方法中加载 LoRA

3. `gen_video/utils/execution_executor_v21.py`
   - 修复 Prompt 截断问题
   - 根据模型类型选择 token 限制

### 测试文件
- `gen_video/V22_ISSUE_ANALYSIS.md` - 问题分析报告
- `gen_video/V22_FIXES_SUMMARY.md` - 修复总结（本文档）

---

## 预期效果

1. **推理步数**：
   - JSON 配置的步数应该被正确使用
   - 日志中应该显示正确的步数

2. **人脸相似度**：
   - 场景3 和场景5 的人脸相似度应该提高到 0.7 以上
   - LoRA 权重应该被正确加载和应用

3. **Prompt 截断**：
   - Flux 模型的 Prompt 不应该被截断（支持 512 tokens）
   - SDXL 模型的 Prompt 应该被正确优化（限制为 77 tokens）

---

## 下一步

1. 运行测试，验证修复效果
2. 如果仍有问题，检查日志输出，进一步调试
3. 根据测试结果，可能需要调整 LoRA 权重或其他参数

