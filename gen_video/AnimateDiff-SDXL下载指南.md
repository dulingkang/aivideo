# AnimateDiff-SDXL 模型下载指南

## 📋 当前状态

根据流水线分析（`分析chatgpt.md` 243-271行），当前缺少 **AnimateDiff-SDXL** 模型。

**当前使用：** SVD (Stable Video Diffusion)  
**需要切换为：** AnimateDiff-SDXL-1080P

## 🎯 需要下载的模型

### 1. AnimateDiff SDXL Motion Module（必需）

**文件名：** `mm_sdxl_v10_beta.ckpt`  
**目标路径：** `/vepfs-dev/shawn/vid/fanren/gen_video/models/animatediff-sdxl-1080p/mm_sdxl_v10_beta.ckpt`

### 2. 模型说明

AnimateDiff-SDXL 需要以下组件：
- ✅ **SDXL Base Model** - 已存在 (`models/sdxl-base/`)
- ⚠️ **Motion Module** - 需要下载 (`mm_sdxl_v10_beta.ckpt`)
- ⚠️ **AnimateDiff Pipeline** - 可能需要（取决于实现方式）

## 📥 下载方式

### 方式 1：从 HuggingFace 下载（推荐）

1. 访问：https://huggingface.co/guoyww/animatediff/tree/main
2. 查找并下载 `mm_sdxl_v10_beta.ckpt` 文件
3. 放置到：`models/animatediff-sdxl-1080p/mm_sdxl_v10_beta.ckpt`

### 方式 2：从 GitHub Releases 下载

1. 访问：https://github.com/guoyww/AnimateDiff/releases
2. 查找 SDXL 相关的 release
3. 下载 `mm_sdxl_v10_beta.ckpt` 文件
4. 放置到：`models/animatediff-sdxl-1080p/mm_sdxl_v10_beta.ckpt`

### 方式 3：使用 proxychains4 + 下载脚本（推荐）

如果系统已安装 proxychains4，可以使用以下命令：

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
source /vepfs-dev/shawn/venv/py312/bin/activate
proxychains4 -q python -c "
from download_stage1_models import download_animatediff_models, load_config
config = load_config()
download_animatediff_models(config)
"
```

或者下载所有阶段2模型：

```bash
proxychains4 -q python download_stage1_models.py
```

**注意：** 
- `-q` 参数用于静默模式，减少 proxychains 的输出
- 确保 proxychains4 已正确配置代理
- 如果网络无法访问 HuggingFace，也可以配置环境变量代理（`HTTP_PROXY` 和 `HTTPS_PROXY`）

## 🔍 验证模型

下载完成后，验证模型文件：

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
ls -lh models/animatediff-sdxl-1080p/
```

应该看到：
- `mm_sdxl_v10_beta.ckpt` 文件（大小约 907MB）

**✅ 已成功下载：** 模型文件已通过 proxychains4 下载完成！

## 📝 后续步骤

下载完成后，还需要：

1. **更新 config.yaml**
   - 将 `video.model_type` 从 `svd-xt` 改为 `animatediff-sdxl`
   - 更新 `video.model_path` 指向 AnimateDiff 模型路径
   - 调整参数（num_frames: 64, fps: 16, width: 1920, height: 1080）

2. **更新 video_generator.py**
   - 实现 AnimateDiff pipeline 加载逻辑
   - 实现 AnimateDiff 视频生成逻辑
   - 实现 FreeInit 去闪烁功能（可选）

3. **测试生成**
   - 使用 AnimateDiff 生成测试视频
   - 对比 SVD 和 AnimateDiff 的效果

## 🔗 参考资源

- AnimateDiff 官方仓库: https://github.com/guoyww/AnimateDiff
- HuggingFace 模型: https://huggingface.co/guoyww/animatediff
- AnimateDiff-SDXL 文档: 查看 `AnimateDiff切换计划.md`

## ⚠️ 注意事项

1. **网络问题：** 如果无法访问 HuggingFace，需要配置代理或使用手动下载
2. **模型大小：** Motion Module 文件较大（约 700MB-1GB），确保有足够空间
3. **显存要求：** AnimateDiff-SDXL 需要较高显存（建议 13GB+ VRAM）
4. **依赖关系：** AnimateDiff 需要配合 SDXL base model 使用（已存在）

## ✅ 检查清单

- [x] 创建模型目录：`models/animatediff-sdxl-1080p/` ✅
- [x] 下载 Motion Module：`mm_sdxl_v10_beta.ckpt` ✅ (907MB)
- [x] 验证模型文件完整性 ✅
- [ ] 更新 config.yaml 配置
- [ ] 实现 AnimateDiff 支持代码
- [ ] 测试视频生成功能

