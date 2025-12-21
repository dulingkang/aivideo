# v2.2-final 问题分析报告

## 测试时间
2025-12-21 22:04:48

## 问题概述

### 1. 推理步数问题 ⚠️

**现象**：
- JSON 配置：`num_inference_steps: 50`
- 实际使用：28 步（日志显示 `28/28`）

**影响**：
- 生成质量可能不足
- 细节可能不够清晰

**可能原因**：
1. `_generate_with_pulid` 虽然读取了 `gen_params`，但 PuLID 引擎内部可能使用了默认值
2. 日志中未看到 "使用JSON中的生成参数" 的输出，说明可能未正确读取

**需要检查**：
- `enhanced_image_generator.py` 的 `_generate_with_pulid` 方法是否正确读取 `scene.generation_params`
- `pulid_engine.py` 的 `generate_with_identity` 方法是否正确使用传入的 `num_inference_steps`

---

### 2. 人脸相似度低 ⚠️⚠️

**现象**：
- 场景3（scene_003）：0.412 < 0.7 阈值
- 场景5（scene_004）：0.033 < 0.7 阈值（严重）

**影响**：
- 角色一致性差
- 用户体验差

**可能原因**：
1. **PuLID 未加载 LoRA**：`pulid_engine.py` 中未找到 `load_lora` 相关代码
2. 参考图路径不正确
3. PuLID 权重设置不当

**需要检查**：
- `pulid_engine.py` 是否支持加载 LoRA
- `enhanced_image_generator.py` 的 `_generate_with_pulid` 是否传递了 LoRA 配置
- 参考图路径是否正确

---

### 3. Prompt 截断问题 ⚠️

**现象**：
- 场景5（scene_004）：80 tokens > 77，部分内容被截断
- 警告：`Token indices sequence length is longer than the specified maximum sequence length for this model (80 > 77)`
- 被截断部分：`['fantasy illustration style']`

**影响**：
- 风格信息丢失
- 生成结果可能不符合预期

**可能原因**：
- `_optimize_prompt_length` 方法可能未正确工作
- Prompt 构建时未考虑 token 限制

**需要检查**：
- `execution_executor_v21.py` 的 `_optimize_prompt_length` 方法
- Prompt 构建逻辑

---

### 4. 图像模糊问题 ⚠️

**现象**：
- 场景3（scene_003）：清晰度 18.0（较低）

**可能原因**：
- 推理步数不足（28 步）
- 分辨率设置问题
- 生成参数不当

---

## 修复建议

### 优先级 1：修复推理步数问题

1. **检查 `_generate_with_pulid` 方法**：
   ```python
   # enhanced_image_generator.py:1122-1132
   scene = kwargs.get("scene", {})
   gen_params = scene.get("generation_params", {}) if scene else {}
   num_steps = gen_params.get("num_inference_steps", self.pulid_config.get("num_inference_steps", 28))
   ```
   - 确保 `scene` 正确传递
   - 添加调试日志确认读取的值

2. **检查 `pulid_engine.py` 的 `generate_with_identity` 方法**：
   - 确保 `num_inference_steps` 参数被正确使用
   - 检查是否有默认值覆盖

### 优先级 2：修复人脸相似度问题

1. **在 `pulid_engine.py` 中添加 LoRA 加载逻辑**：
   - 从 `scene.character.lora_config` 读取 LoRA 路径和权重
   - 在生成前加载 LoRA

2. **在 `_generate_with_pulid` 中传递 LoRA 配置**：
   - 从 `scene.character.lora_config` 读取配置
   - 传递给 PuLID 引擎

3. **验证参考图路径**：
   - 确保 `character.reference_image` 路径正确
   - 检查文件是否存在

### 优先级 3：修复 Prompt 截断问题

1. **优化 `_optimize_prompt_length` 方法**：
   - 确保正确计算 token 数
   - 优先保留关键信息（角色、动作、场景）

2. **调整 Prompt 构建逻辑**：
   - 减少冗余描述
   - 使用更简洁的风格标签

---

## 测试建议

1. **单独测试推理步数**：
   - 创建一个简单的测试场景，设置 `num_inference_steps: 50`
   - 检查日志输出和实际使用的步数

2. **单独测试 LoRA 加载**：
   - 创建一个测试场景，明确指定 LoRA 路径
   - 检查是否加载成功

3. **单独测试 Prompt 优化**：
   - 创建一个长 Prompt 测试场景
   - 检查是否被正确截断和优化

---

## 相关文件

- `/vepfs-dev/shawn/vid/fanren/gen_video/enhanced_image_generator.py`
- `/vepfs-dev/shawn/vid/fanren/gen_video/pulid_engine.py`
- `/vepfs-dev/shawn/vid/fanren/gen_video/utils/execution_executor_v21.py`
- `/vepfs-dev/shawn/vid/fanren/lingjie/v22/scene_003_v22.json`
- `/vepfs-dev/shawn/vid/fanren/lingjie/v22/scene_004_v22.json`

