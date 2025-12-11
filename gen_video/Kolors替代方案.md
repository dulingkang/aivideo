# Kolors 模型替代方案

## ⚠️ 问题说明

Kolors 模型在 HuggingFace 上不存在（404 错误），无法直接下载。

## 💡 替代方案

### 方案一：使用 Realistic Vision（推荐）

**Realistic Vision** 是一个专注于真实感的 Stable Diffusion 模型，手部和光影表现优秀。

#### 下载方式

```bash
# 使用 huggingface-cli
huggingface-cli download SG161222/Realistic_Vision_V5.1_noVAE \
    --local-dir /vepfs-dev/shawn/vid/fanren/gen_video/models/realistic-vision \
    --local-dir-use-symlinks False
```

#### 模型信息
- **模型ID**: `SG161222/Realistic_Vision_V5.1_noVAE`
- **大小**: ~6GB
- **特点**: 真实感强，手部细节好，光影自然

### 方案二：使用 SDXL + 真实感 LoRA

使用现有的 SDXL 模型，配合真实感 LoRA 实现类似效果。

#### 推荐 LoRA
- **Realistic Vision LoRA**: 增强真实感
- **Hands Detail LoRA**: 改善手部细节
- **Lighting LoRA**: 优化光影效果

### 方案三：使用 Flux + 真实感 LoRA

使用 Flux 1-dev 模型（已下载），配合真实感 LoRA。

#### 优势
- Flux 模型质量更高
- 配合 LoRA 可以实现优秀的真实感效果

## 🔧 更新配置

### 更新 config.yaml

将 Kolors 配置替换为 Realistic Vision：

```yaml
image:
  model_selection:
    scene:
      # 替换 kolors 为 realistic_vision
      realistic_vision:
        model_path: /vepfs-dev/shawn/vid/fanren/gen_video/models/realistic-vision
        base_model: SG161222/Realistic_Vision_V5.1_noVAE
        width: 1536
        height: 864
        num_inference_steps: 40
        guidance_scale: 7.0
        realism_boost: true
        quantization: fp16
```

### 更新模型选择器

在 `model_selector.py` 中，将 `kolors` 替换为 `realistic-vision`：

```python
elif engine == "realistic-vision":
    scene_config = model_selection.get("scene", {})
    return scene_config.get("realistic_vision", {})
```

## 📝 实施步骤

1. **下载 Realistic Vision 模型**
   ```bash
   source /vepfs-dev/shawn/venv/py312/bin/activate
   huggingface-cli download SG161222/Realistic_Vision_V5.1_noVAE \
       --local-dir /vepfs-dev/shawn/vid/fanren/gen_video/models/realistic-vision \
       --local-dir-use-symlinks False
   ```

2. **更新配置文件**
   - 将 `kolors` 替换为 `realistic-vision`
   - 更新模型路径

3. **更新代码**
   - 更新 `model_selector.py`
   - 更新 `image_generator.py` 中的 pipeline 加载方法

4. **测试**
   - 测试真实感场景生成效果
   - 验证手部和光影表现

## 🎯 推荐方案

**推荐使用方案一（Realistic Vision）**，因为：
- ✅ 模型可直接下载
- ✅ 真实感强，手部细节好
- ✅ 与 SDXL 兼容，易于集成
- ✅ 社区支持好，文档完善

## 📚 参考资源

- Realistic Vision: https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE
- SDXL 真实感 LoRA: https://civitai.com/models?query=realistic+sdxl
- Flux 真实感 LoRA: https://civitai.com/models?query=realistic+flux

---

**最后更新**: 2024年12月

