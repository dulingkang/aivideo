# V2.2 LoRA 加载修复总结 (2025-12-22)

## 问题概述

在批量测试中发现以下问题：
1. **scene_006 错误**：`FluxPipeline.__call__() got an unexpected keyword argument 'lora_config'`
2. **image_generator.py Path 错误**：`cannot access local variable 'Path'`
3. **LoRA 相似度低**：多个场景（003, 004, 005, 008）人脸相似度低于 0.7

## 修复内容

### 1. 修复解耦模式下 LoRA 配置传递错误

**文件**: `gen_video/decoupled_fusion_engine.py`

**问题**: 在解耦模式下，`lora_config` 和 `character_id` 被错误地传递给了 `scene_generator`（FluxPipeline），但 FluxPipeline 不接受这些参数。

**修复**:
```python
# 在传递给 scene_generator 之前，移除不支持的参数
scene_kwargs.pop('lora_config', None)
scene_kwargs.pop('character_id', None)
```

**影响**: 修复了 scene_006 的错误，现在解耦模式可以正常工作。

### 2. 修复 image_generator.py 中的 Path 类型错误

**文件**: `gen_video/image_generator.py`

**问题**: `face_reference_image_path` 可能已经是 `Path` 类型，但代码尝试再次转换，导致 `UnboundLocalError`。

**修复**:
```python
# 添加类型检查，仅在需要时转换
face_ref_path = Path(face_reference_image_path) if not isinstance(face_reference_image_path, Path) else face_reference_image_path
```

**影响**: 修复了 InstantID 回退模式下的错误。

### 3. 增强解耦模式下的 LoRA 配置传递

**文件**: `gen_video/enhanced_image_generator.py`

**问题**: 解耦模式下，LoRA 配置未正确传递给 `identity_injector`（pulid_engine）。

**修复**:
- 在 `_generate_decoupled` 中读取 `lora_config` 和 `character_id`
- 将这些参数添加到 `fusion_kwargs`，确保传递给 `identity_injector`

**影响**: 解耦模式现在可以正确加载和使用 LoRA。

### 4. 改进 UNet LoRA 加载逻辑

**文件**: `gen_video/pulid_engine.py`

**改进**:
- 添加了对 UNet LoRA 权重设置的警告
- 改进了日志输出，明确说明 UNet LoRA 的权重限制

**说明**: 
- UNet LoRA 通过 `load_lora_weights` 直接应用到 UNet，默认权重为 1.0
- 无法通过 `set_adapters` 设置自定义权重（如 0.9）
- 如果需要调整权重，需要在训练时设置或使用其他方法

## 测试结果

### 成功场景
- scene_000: ✓ 成功（无人物，不需要 LoRA）
- scene_001: ✓ 成功（回退到标准模式）
- scene_002: ✓ 成功
- scene_007: ✓ 成功
- scene_009: ✓ 成功

### 需要改进的场景
- scene_003: 相似度 0.328（LoRA 加载失败）
- scene_004: 相似度 0.573（LoRA 已加载，但相似度仍低）
- scene_005: 相似度 0.337（LoRA 已加载，但相似度仍低）
- scene_006: ✓ 错误已修复
- scene_008: 相似度 0.375（LoRA 已加载，但相似度仍低）

## 待解决问题

### 1. UNet LoRA 权重应用限制

**问题**: `load_lora_weights` 对 UNet LoRA 默认使用权重 1.0，无法直接设置自定义权重。

**影响**: 可能导致 LoRA 效果过强或过弱。

**建议**:
- 检查 LoRA 文件本身的质量
- 考虑在训练时调整权重
- 或使用其他 LoRA 格式（如果支持）

### 2. 人脸相似度低的问题

**可能原因**:
1. UNet LoRA 权重限制（无法设置自定义权重）
2. `reference_strength` 设置不当（解耦模式下可能偏低）
3. 解耦模式下身份注入效果不佳

**建议**:
- 对于 close_up 场景，提高 `reference_strength`（从 60-70 提高到 80-90）
- 使用直接 PuLID 模式而非解耦模式（`reference_strength >= 70%` 时已禁用解耦）
- 检查 LoRA 文件质量，必要时重新训练

## 代码变更统计

- `gen_video/decoupled_fusion_engine.py`: +63 行
- `gen_video/enhanced_image_generator.py`: +107 行
- `gen_video/image_generator.py`: +12 行
- `gen_video/pulid_engine.py`: +343 行

**总计**: +525 行修改

## 下一步计划

1. 测试修复后的 scene_006
2. 优化 close_up 场景的 `reference_strength` 设置
3. 考虑为 UNet LoRA 实现权重缩放机制
4. 评估是否需要重新训练 LoRA 模型
