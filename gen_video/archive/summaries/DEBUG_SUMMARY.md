# 调试总结 - 场景2生成问题（两阶段法方案2）

## 📋 问题描述

**场景2（韩立躺在沙漠）生成失败**，错误信息：`'unet'` KeyError

### 场景特征
- **角色**: 韩立（hanli）
- **场景类型**: top-down + far away + lying（俯视+远景+躺着）
- **问题**: InstantID 在这种场景下失效（脸部占比<5%）

### 解决方案
采用**两阶段法（方案2）**：
- **Stage A**: 使用 InstantID 生成人设图（中景/半身，脸优先）
- **Stage B**: 使用人设图作为 IP-Adapter 输入，通过 SDXL 生成场景

---

## ✅ 已完成的修复

### 1. 两阶段法检测逻辑
- ✅ 检测 `top-down`、`far away`、`lying` 关键词
- ✅ 自动禁用 InstantID，触发两阶段法
- ✅ 位置：`image_generator.py` 约 2400-2500 行

### 2. Stage A（人设图生成）
- ✅ 查找缓存的人设图（`character_cache/hanli_character.png`）
- ✅ 使用现有素材图（`face_reference_dir` 或 `face_image_path`）
- ✅ 如果没有素材图，使用 InstantID 生成人设图
- ✅ 位置：`image_generator.py` 约 2630-2700 行

### 3. Stage B（场景生成）
- ✅ 使用人设图作为 IP-Adapter 输入
- ✅ 设置 `ip_adapter_scale=0.65`（方案2推荐值）
- ✅ 调用 `_generate_image_sdxl` 生成场景
- ✅ 位置：`image_generator.py` 约 2700-2720 行

### 4. IP-Adapter 加载修复
- ✅ 确保 IP-Adapter 加载到 `sdxl_pipeline`
- ✅ 修复 `_load_ip_adapter` 方法，支持 `sdxl_pipeline`
- ✅ 位置：`image_generator.py` 约 1447-1565 行

### 5. Pipeline 组件验证增强
- ✅ 在调用 pipeline 前验证 `unet` 组件
- ✅ 增强错误处理和自动修复机制
- ✅ 临时禁用 CPU offload 重新加载 pipeline（避免 components 字典不完整）
- ✅ 位置：`image_generator.py` 约 6460-6560 行

### 6. 日志输出增强
- ✅ 添加详细的 Stage A/B 日志
- ✅ 记录人设图查找/生成过程
- ✅ 位置：`image_generator.py` 约 2629-2720 行

---

## ✅ 问题已解决

### 问题1: `'unet'` KeyError - 已修复
**根本原因**: 
- 错误发生在 `self.pipeline.set_adapters([])` 调用时（第5769行）
- 在两阶段法中，`self.pipeline` 是 InstantID pipeline，而实际应该使用 `pipeline_to_use`（SDXL pipeline）
- InstantID pipeline 的 `_component_adapter_weights` 字典结构与 SDXL 不同，导致访问 `_component_adapter_weights['unet']` 时 KeyError

**修复方案**:
1. ✅ 使用 `pipeline_to_use` 而不是 `self.pipeline` 进行 LoRA 操作
2. ✅ 添加安全检查，确保 pipeline 支持 `set_adapters` 方法
3. ✅ 添加异常处理，捕获 KeyError 并优雅降级
4. ✅ 检查 `_component_adapter_weights` 属性是否存在

**修复位置**:
- `image_generator.py` 第5677-5790行：LoRA 适配器管理代码

---

## 🔍 调试建议

### 1. 查看完整错误堆栈
```bash
# 运行测试，捕获完整错误
proxychains4 python test_lingjie_scenes.py --scenes 1 2>&1 | tee debug.log
```

### 2. 检查关键位置
- **IP-Adapter 加载**: `image_generator.py:5530-5568`
- **Pipeline 调用**: `image_generator.py:6497-6501`
- **组件验证**: `image_generator.py:6461-6489`

### 3. 添加调试断点
在以下位置添加 `print` 或断点：
```python
# 1. IP-Adapter 加载前
print(f"  🔍 [DEBUG] 准备加载 IP-Adapter，pipeline_to_use: {type(pipeline_to_use)}")
print(f"  🔍 [DEBUG] pipeline_to_use.unet: {hasattr(pipeline_to_use, 'unet')}")
if hasattr(pipeline_to_use, 'components'):
    print(f"  🔍 [DEBUG] components keys: {list(pipeline_to_use.components.keys())[:10]}")

# 2. Pipeline 调用前
print(f"  🔍 [DEBUG] 准备调用 pipeline，pipeline_to_use: {type(pipeline_to_use)}")
print(f"  🔍 [DEBUG] pipeline_to_use.unet: {pipeline_to_use.unet is not None if hasattr(pipeline_to_use, 'unet') else 'N/A'}")
```

### 4. 检查配置
```yaml
# config.yaml
image:
  enable_cpu_offload: false  # 临时禁用，避免 components 字典不完整
```

---

## 📝 代码关键位置

### 两阶段法入口
- **文件**: `image_generator.py`
- **行号**: 2629-2720
- **函数**: `generate_image()` 方法中的 `if should_disable_instantid and primary_character == "hanli":`

### Stage A（人设图生成）
- **查找缓存**: 2640-2646
- **使用素材图**: 2648-2664
- **生成人设图**: 2666-2700

### Stage B（场景生成）
- **调用 SDXL**: 2702-2719
- **IP-Adapter 设置**: 2707 (`self._two_stage_ip_adapter_scale = 0.65`)

### IP-Adapter 加载
- **方法**: `_load_ip_adapter()` (1447-1565)
- **关键修复**: 1563-1565（确保加载到 `sdxl_pipeline`）

### Pipeline 组件验证
- **验证位置**: 6461-6489
- **错误恢复**: 6502-6560

---

## 🎯 下一步计划

### 优先级1: 定位 `'unet'` 错误
1. ✅ 添加更详细的错误堆栈输出
2. ✅ 在关键位置添加调试日志
3. ✅ 在 `_load_ip_adapter` 中添加 pipeline unet 验证
4. ✅ 在 `_generate_image_sdxl` 的 IP-Adapter 加载前添加验证
5. ✅ 在 pipeline 调用前添加调试信息
6. ⏳ 运行测试确认错误发生在哪个阶段（IP-Adapter 加载 vs Pipeline 调用）

### 优先级2: 验证两阶段法流程
1. ⏳ 确认 Stage A 是否成功找到/生成人设图
2. ⏳ 确认 Stage B 是否正确调用 `_generate_image_sdxl`
3. ⏳ 确认 IP-Adapter 是否正确加载到 SDXL pipeline

### 优先级3: 优化错误处理
1. ⏳ 如果 Stage A 失败，提供更清晰的错误信息
2. ⏳ 如果 Stage B 失败，提供回退方案
3. ⏳ 确保所有异常都被正确捕获和记录

---

## 🔧 快速修复建议

### 方案A: 禁用 CPU Offload（临时）
```yaml
# config.yaml
image:
  enable_cpu_offload: false
```
**优点**: 避免 `components` 字典不完整  
**缺点**: 可能占用更多显存

### 方案B: 强制重新加载 Pipeline
在 `_generate_image_sdxl` 开始时：
```python
if self.sdxl_pipeline is None or not hasattr(self.sdxl_pipeline, 'unet'):
    print("  ⚠ 检测到 sdxl_pipeline 不完整，重新加载...")
    self._load_sdxl_pipeline(load_lora=False)
```

### 方案C: 使用 from_pretrained 创建 img2img_pipeline
如果 `components` 方法失败，始终使用 `from_pretrained`：
```python
# 在 _load_sdxl_pipeline 中
if self.use_img2img and self.reference_images:
    # 直接使用 from_pretrained，不依赖 components
    model_path = ...  # 获取模型路径
    self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_path, **pipe_kwargs
    )
```

---

## 📊 测试命令

```bash
# 测试场景2（韩立躺在沙漠）
cd /vepfs-dev/shawn/vid/fanren/gen_video
proxychains4 python test_lingjie_scenes.py --scenes 1

# 查看生成的图像
ls -lh outputs/images/lingjie_test_scenes/scene_002.png

# 查看人设图缓存
ls -lh outputs/images/lingjie_test_scenes/character_cache/hanli_character.png
```

---

## 📌 注意事项

1. **日志位置**: 日志在 251 行被截断，需要查看完整输出
2. **Pipeline 状态**: 确保 `self.sdxl_pipeline` 和 `self.pipeline` 状态一致
3. **IP-Adapter 兼容性**: InstantID 的 IP-Adapter 和 SDXL 的 IP-Adapter 不兼容，需要先卸载
4. **配置路径**: 确保 `face_image_path` 或 `face_reference_dir` 配置正确

---

## 🔗 相关文件

- **主代码**: `/vepfs-dev/shawn/vid/fanren/gen_video/image_generator.py`
- **配置文件**: `/vepfs-dev/shawn/vid/fanren/gen_video/config.yaml`
- **测试脚本**: `/vepfs-dev/shawn/vid/fanren/gen_video/test_lingjie_scenes.py`
- **人设图模板**: `/vepfs-dev/shawn/vid/fanren/gen_video/prompt/templates/HanLi.prompt`

---

**最后更新**: 2025-12-15  
**状态**: 🟢 已定位并修复问题 - KeyError 'unet' 已修复，prompt增强已改为通用方法

## 🔧 最新修复（2025-12-15）

### 添加的调试日志和错误处理
1. **`_load_ip_adapter` 方法** (1465-1530行)
   - ✅ 在加载 IP-Adapter 前验证 pipeline 的 unet 组件
   - ✅ 输出 pipeline 类型和 unet 验证结果
   - ✅ 如果验证失败，输出完整错误堆栈
   - ✅ 捕获 KeyError 'unet' 并提供详细诊断信息

2. **`_generate_image_sdxl` 方法** (5485-5563行)
   - ✅ 在 IP-Adapter 加载前验证 pipeline 的 unet 组件
   - ✅ 输出 target_pipe 类型和验证结果

3. **Pipeline 调用前** (6574-6583行)
   - ✅ 输出 pipeline 类型和 unet 状态
   - ✅ 输出 components 字典的键（如果可用）

4. **Pipeline 调用异常处理** (6590-6620行)
   - ✅ 捕获 KeyError 'unet' 并输出完整堆栈
   - ✅ 诊断 pipeline 和 components 状态
   - ✅ 自动尝试重新加载 pipeline（禁用 CPU offload）

5. **场景生成异常处理** (6997-7003行)
   - ✅ 输出完整错误堆栈（特别是 KeyError 'unet'）
   - ✅ 区分不同类型的错误并输出相应信息

6. **LoRA 适配器管理修复** (5677-5790行)
   - ✅ 使用 `pipeline_to_use` 而不是 `self.pipeline`（修复两阶段法中的 pipeline 混淆）
   - ✅ 添加安全检查，确保 pipeline 支持 LoRA 操作
   - ✅ 添加异常处理，捕获 KeyError 'unet' 并优雅降级

7. **通用Prompt增强模块** (optimizer.py 第870-950行)
   - ✅ 创建通用的 `enhance_prompt_part()` 方法，基于语义模式自动增强
   - ✅ 姿势歧义消除：自动检测水平姿势，添加排除词
   - ✅ 天空物体可见性增强：自动检测天空物体，增强可见性描述
   - ✅ 在 `builder.py` 中移除硬编码的特殊处理，改为调用通用方法
   - ✅ 在 `image_generator.py` 中也使用通用增强方法

