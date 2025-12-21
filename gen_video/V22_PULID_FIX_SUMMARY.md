# v2.2-final Flux + PuLID 修复总结

## 问题描述
用户反馈生成的图片是糊的（模糊），经检查发现：

1. **Flux + PuLID 未正确触发**：`ExecutionExecutorV21` 中设置了 `image_scene = None`，导致增强模式（EnhancedImageGenerator）无法触发
2. **增强模式支持 Flux + PuLID**：`EnhancedImageGenerator` 已经实现了 Flux + PuLID 的集成
3. **生成参数偏低**：分辨率 1024x1024 和步数 30 可能不够清晰

## 修复内容

### 1. 修复增强模式触发逻辑 (`execution_executor_v21.py`)

**修复前**:
```python
# 构建scene字典（关键：不传递scene参数，或者传递一个空的scene，禁用增强模式）
image_scene = None  # 不传递scene，禁用增强模式
```

**修复后**:
```python
# ⚡ 关键修复：如果使用 Flux + PuLID，需要传递 scene 参数以触发增强模式
use_enhanced_mode = False
if base_model == "flux" and identity_engine == "pulid":
    # Flux + PuLID 需要使用增强模式
    use_enhanced_mode = True
    logger.info(f"使用增强模式（Flux + PuLID）生成图像")

# 调用时传递 scene 参数
if use_enhanced_mode:
    image_path = self._image_generator.generate_image(
        ...
        scene=scene,  # ⚡ 关键：传递 scene 参数以触发增强模式
        ...
    )
```

### 2. 优化生成参数（提高清晰度）

**修复文件**:
- `schemas/scene_v22_real_example.json`
- `schemas/scene_v22_real_example_002.json`

**修复内容**:
```json
"generation_params": {
  "width": 1536,      // 从 1024 提高到 1536
  "height": 1536,     // 从 1024 提高到 1536
  "num_inference_steps": 40,  // 从 30 提高到 40
  "guidance_scale": 7.5,
  "seed": -1
}
```

### 3. 确保增强模式配置启用

**检查项**:
- `config.yaml` 中的 `image.enhanced.enabled` 应该为 `true`
- 如果未启用，需要修改配置或确保代码能正确触发

## 修复后的工作流程

1. **检测 Flux + PuLID**：当 `base_model == "flux"` 且 `identity_engine == "pulid"` 时
2. **触发增强模式**：传递 `scene` 参数给 `ImageGenerator.generate_image`
3. **使用 EnhancedImageGenerator**：`ImageGenerator` 检测到 `scene` 参数且 `use_enhanced_mode=True` 时，会调用 `EnhancedImageGenerator.generate_scene`
4. **PuLID 引擎工作**：`EnhancedImageGenerator` 使用 `PuLIDEngine` 进行 Flux + PuLID 生成
5. **使用锁定参数**：增强模式会使用 JSON 中锁定的参数（shot, pose, model_route等），不会自己决策

## 预期效果

修复后，系统将：
1. ✅ 正确触发增强模式（Flux + PuLID）
2. ✅ 使用更高的分辨率（1536x1536）和步数（40步）
3. ✅ 生成更清晰、更一致的人脸
4. ✅ 保持 JSON 中锁定的参数，不进行自动决策

## 测试建议

1. **重新运行测试**：
```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
source /vepfs-dev/shawn/venv/py312/bin/activate
python3 test_v22_actual_generation_simple.py schemas/scene_v22_real_example.json
```

2. **检查日志**：
   - 应该看到 "使用增强模式（Flux + PuLID）生成图像"
   - 应该看到 "✨ 使用增强模式生成（PuLID + 解耦融合 + Execution Planner V3）"
   - 不应该看到 "⚠️  注意: Flux.1 + InstantID 集成需要额外实现，当前回退到普通 Flux.1"

3. **检查生成结果**：
   - 图片应该更清晰（1536x1536 分辨率）
   - 人脸应该更一致（PuLID 工作）
   - 整体质量应该更好（40步生成）

## 注意事项

1. **增强模式配置**：确保 `config.yaml` 中 `image.enhanced.enabled = true`
2. **显存要求**：1536x1536 分辨率需要更多显存，如果显存不足，可以降低到 1280x1280
3. **生成时间**：40步生成会比30步慢一些，但质量更好
4. **PuLID 引擎**：首次使用会加载 PuLID 模型，可能需要一些时间

