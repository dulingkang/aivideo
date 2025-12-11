# ComfyUI SDXL 配置说明

## ✅ 当前状态

### SDXL 模型位置

1. **原始位置**：`/vepfs-dev/shawn/vid/fanren/gen_video/models/sdxl-base/`
   - `sd_xl_base_1.0.safetensors` (6.5GB)
   - `sd_xl_base_1.0_0.9vae.safetensors` (6.5GB)

2. **ComfyUI checkpoints 目录**：`/vepfs-dev/shawn/vid/fanren/ComfyUI/models/checkpoints/`
   - ✅ 已复制：`sd_xl_base_1.0.safetensors` (6.5GB)

### AnimateDiff Motion Model

- **位置**：`/vepfs-dev/shawn/vid/fanren/ComfyUI/models/animatediff_models/`
- **文件**：`diffusion_pytorch_model.fp16.safetensors` (1.7GB)
- **注意**：这是 SD1.5 的 motion model，**不支持 SDXL**

## ⚠️ 重要发现

### AnimateDiff 与 SDXL 的兼容性

根据 ComfyUI-AnimateDiff-Evolved 的 README：

1. **AnimateDiff-SDXL 支持**：
   - ✅ ComfyUI 支持 AnimateDiff-SDXL
   - ⚠️ 但需要专门的 SDXL motion module
   - ⚠️ 当前下载的是 SD1.5 的 motion model

2. **当前 motion model**：
   - 下载的是 `guoyww/animatediff-motion-adapter-v1-5-2`
   - 这是 **SD1.5 的 motion adapter**，不支持 SDXL

3. **SDXL motion module**：
   - 需要下载专门的 SDXL motion module
   - 例如：`mm_sdxl_v10_beta.ckpt` 或类似的 SDXL 版本

## 🔧 解决方案

### 方案1：使用 SDXL + SDXL Motion Module（推荐）

**步骤**：

1. **SDXL checkpoint**：✅ 已就绪
   - `ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors`

2. **下载 SDXL motion module**：
   ```bash
   cd /vepfs-dev/shawn/vid/fanren/ComfyUI/models/animatediff_models
   
   # 下载 SDXL motion module
   proxychains4 -q huggingface-cli download guoyww/animatediff \
       --local-dir . \
       --include "mm_sdxl_v10_beta.ckpt"
   ```

3. **更新工作流**：
   - 使用 SDXL checkpoint
   - 使用 SDXL motion module
   - 使用 `autoselect` 或 `linear (AnimateDiff-SDXL)` beta_schedule

### 方案2：使用 SD1.5 + SD1.5 Motion Module（简单但降级）

**步骤**：

1. **下载 SD1.5 checkpoint**：
   ```bash
   cd /vepfs-dev/shawn/vid/fanren/ComfyUI/models/checkpoints
   
   proxychains4 -q huggingface-cli download runwayml/stable-diffusion-v1-5 \
       --local-dir . \
       --include "v1-5-pruned-emaonly.safetensors"
   ```

2. **Motion module**：✅ 已就绪
   - `ComfyUI/models/animatediff_models/diffusion_pytorch_model.fp16.safetensors`

3. **更新工作流**：
   - 使用 SD1.5 checkpoint
   - 使用现有的 SD1.5 motion module

**缺点**：
- 分辨率限制：SD1.5 最大 768x768（不如 SDXL 的 1024x1024）
- 质量略低：SD1.5 不如 SDXL

## 🎯 推荐方案

### **使用 SDXL + 下载 SDXL Motion Module**

**理由**：
1. ✅ 保持高质量（1024x1024）
2. ✅ 与现有系统一致（InstantID、LoRA 都是 SDXL）
3. ✅ 只需要下载 motion module，不需要重新训练

**实施步骤**：

```bash
# 1. 下载 SDXL motion module
cd /vepfs-dev/shawn/vid/fanren/ComfyUI/models/animatediff_models

# 方法1：从 HuggingFace 下载
proxychains4 -q huggingface-cli download guoyww/animatediff \
    --local-dir . \
    --include "mm_sdxl_v10_beta.ckpt"

# 方法2：如果已有，检查是否在 animatediff-sdxl-1080p 目录
ls -lh /vepfs-dev/shawn/vid/fanren/gen_video/models/animatediff-sdxl-1080p/
```

## 📋 检查清单

- [x] SDXL checkpoint 已复制到 ComfyUI
- [ ] SDXL motion module 需要下载
- [ ] 更新工作流以使用 SDXL motion module
- [ ] 测试 ComfyUI AnimateDiff 生成

## 💡 提示

1. **SDXL motion module 位置**：
   - 检查 `gen_video/models/animatediff-sdxl-1080p/` 是否已有
   - 如果有，可以复制到 ComfyUI 的 `models/animatediff_models/`

2. **工作流配置**：
   - 使用 `ADE_AnimateDiffLoaderWithContext` 节点
   - 设置 `beta_schedule` 为 `autoselect` 或 `linear (AnimateDiff-SDXL)`
   - 使用 SDXL checkpoint

3. **测试**：
   - 先用简单的工作流测试
   - 确认 SDXL + SDXL motion module 能正常工作

