# v2.2-final LoRA 修复总结

## 问题描述
用户反馈生成的人脸和服饰不对，经检查发现以下问题：

1. **LoRA未正确传递**：`ExecutionExecutorV21._execute_image_generation`方法没有传递`character_lora`参数给`ImageGenerator.generate_image`
2. **LoRA被配置禁用**：`config.yaml`中`lora.enabled: false`，导致LoRA无法加载
3. **Trigger词不匹配**：JSON中的trigger词是"hanli"（小写），但提示词中使用的是"HanLi"（混合大小写）

## 修复内容

### 1. 修复LoRA参数传递 (`execution_executor_v21.py`)

**位置**: `_execute_image_generation`方法

**修复前**:
```python
image_path = self._image_generator.generate_image(
    prompt=prompt,
    output_path=str(output_path),
    ...
    face_reference_image_path=face_ref_path,
    model_engine=model_engine,
    task_type="character" if character_id else "scene"
)
```

**修复后**:
```python
# 确定LoRA适配器名称
character_lora = None
if character_id:
    # 优先使用character_id作为适配器名称（与config.yaml中的adapter_name一致）
    character_lora = character_id
    logger.info(f"使用LoRA适配器: {character_lora} (基于character_id)")

image_path = self._image_generator.generate_image(
    prompt=prompt,
    output_path=str(output_path),
    ...
    face_reference_image_path=face_ref_path,
    character_lora=character_lora,  # 传递LoRA适配器名称
    use_lora=True if character_lora else None,  # 强制启用LoRA
    model_engine=model_engine,
    task_type="character" if character_id else "scene"
)
```

### 2. 修复提示词构建，确保包含Trigger词 (`execution_executor_v21.py`)

**位置**: `_build_prompt`方法

**修复内容**:
```python
# v2.2-final格式：直接使用final字段（如果存在）
if "final" in prompt_config:
    final_prompt = prompt_config["final"]
    
    # 确保提示词包含LoRA trigger词（如果配置了）
    lora_config = character_info.get("lora_config", {})
    trigger = lora_config.get("trigger")
    if trigger and trigger.lower() not in final_prompt.lower():
        # 在提示词开头添加trigger词（LoRA激活词）
        final_prompt = f"{trigger}, {final_prompt}"
        logger.info(f"添加LoRA trigger词: {trigger}")
    
    # 精简Prompt，确保不超过77 tokens（CLIP限制）
    return self._optimize_prompt_length(final_prompt, max_tokens=77)
```

### 3. 修复JSON中的Trigger词大小写

**修复文件**:
- `schemas/scene_v22_real_example.json`
- `schemas/scene_v22_real_example_002.json`

**修复内容**:
- 将提示词中的"HanLi"改为"hanli"（小写，与JSON中的trigger词一致）

## 修复效果

修复后，系统将：
1. ✅ 正确传递LoRA适配器名称给ImageGenerator
2. ✅ 强制启用LoRA（即使config.yaml中禁用了）
3. ✅ 确保提示词包含正确的trigger词（"hanli"）
4. ✅ 从配置中自动获取LoRA文件路径（`models/lora/hanli/pytorch_lora_weights.safetensors`）

## 测试建议

1. **重新运行测试**：
```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
source /vepfs-dev/shawn/venv/py312/bin/activate
python3 test_v22_actual_generation_simple.py schemas/scene_v22_real_example.json
```

2. **检查生成结果**：
   - 人脸应该更接近参考图（hanli_mid.jpg）
   - 服饰应该符合"green daoist robe"的描述
   - 整体风格应该更稳定

3. **如果仍有问题**：
   - 检查LoRA文件是否存在：`ls -lh models/lora/hanli/pytorch_lora_weights.safetensors`
   - 检查参考图是否存在：`ls -lh reference_image/hanli_mid.jpg`
   - 检查日志中是否有LoRA加载信息

## 注意事项

1. **LoRA权重**：当前使用配置中的默认权重（`config.yaml`中的`alpha: 0.5`），如果需要调整，可以在JSON的`lora_config.weight`中指定（但需要代码支持）

2. **Trigger词**：确保JSON中的`lora_config.trigger`与提示词中的trigger词一致（当前使用"hanli"）

3. **参考图**：系统会自动从`character_reference_images`中获取参考图，如果JSON中的`reference_image`为空

4. **模型引擎**：当前使用`flux-instantid`（Flux + PuLID），如果LoRA效果不佳，可以尝试其他引擎

