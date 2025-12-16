# CogVideoX 测试指南

## 📋 概述

CogVideoX-5B 已集成到视频生成系统中，作为**量产产线**的核心模型，用于快速批量生成视频（短剧、推文、爆款视频等）。

## ✅ 集成状态

- ✅ CogVideoX模型加载（`_load_cogvideox_model`）
- ✅ CogVideoX视频生成（`_generate_video_cogvideox`）
- ✅ Prompt Engine集成（自动优化prompt）
- ✅ 模型路由器集成（自动选择模型）
- ✅ 显存优化（CPU offload, VAE tiling）

## 🚀 快速测试

### 1. 准备测试图像

```bash
# 创建测试图像目录
mkdir -p outputs/test_images

# 将测试图像放入该目录（支持 .png 或 .jpg）
# 例如：outputs/test_images/test_scene.png
```

### 2. 运行基础测试

```bash
cd gen_video
python3 test_cogvideox.py
```

### 3. 测试内容

测试脚本包含以下测试：

1. **CogVideoX基础生成功能**
   - 测试模型加载
   - 测试视频生成
   - 检查输出质量

2. **CogVideoX + Prompt Engine**
   - 测试Prompt Engine优化效果
   - 对比优化前后的prompt质量

3. **模型路由自动选择**
   - 测试不同场景类型的模型选择
   - 测试用户等级对模型选择的影响
   - 测试显存限制对模型选择的影响

4. **不同场景类型测试**（可选）
   - 测试novel、drama、daily等不同场景类型

## 📊 预期结果

### 生成参数

- **帧数**: 81帧（CogVideoX推荐）
- **帧率**: 16fps（CogVideoX推荐）
- **分辨率**: 1360x768（CogVideoX推荐）
- **推理步数**: 50步
- **引导尺度**: 6.0

### 性能指标

- **生成时间**: 约2-5分钟（取决于GPU）
- **显存占用**: 约12-15GB（启用CPU offload后）
- **视频时长**: 约5秒（81帧 @ 16fps）

## 🔧 配置说明

### config.yaml 配置

```yaml
video:
  model_type: cogvideox  # 或 auto（自动选择）
  
  cogvideox:
    model_path: /vepfs-dev/shawn/vid/fanren/gen_video/models/CogVideoX-5b
    num_frames: 81
    fps: 16
    width: 1360
    height: 768
    num_inference_steps: 50
    guidance_scale: 6.0
    use_dynamic_cfg: true
    enable_model_cpu_offload: true
    enable_tiling: true
```

### 强制使用CogVideoX

```python
from video_generator import VideoGenerator

generator = VideoGenerator()
generator.video_config['model_type'] = 'cogvideox'

result = generator.generate_video(
    image_path="path/to/image.png",
    output_path="output.mp4",
    num_frames=81,
    fps=16,
    scene={
        "type": "novel",
        "description": "a character in a fantasy world"
    }
)
```

## 🎯 使用场景

### 适合使用CogVideoX的场景

- ✅ 短剧生成（novel, drama）
- ✅ 推文视频（social, daily）
- ✅ 批量生成（需要快速产出）
- ✅ 基础用户（free, basic, professional）

### 不适合使用CogVideoX的场景

- ❌ 政府宣传片（应使用HunyuanVideo）
- ❌ 企业广告（应使用HunyuanVideo）
- ❌ 科普教育（应使用HunyuanVideo）
- ❌ 高端场景（应使用HunyuanVideo）

## 📈 质量对比

### CogVideoX vs HunyuanVideo

| 特性 | CogVideoX | HunyuanVideo |
|------|-----------|--------------|
| 生成速度 | ⚡ 快（2-5分钟） | 🐌 慢（15-30分钟） |
| 视频质量 | ⭐⭐⭐ 良好 | ⭐⭐⭐⭐⭐ 优秀 |
| 显存需求 | 💾 12-15GB | 💾 20-24GB |
| 适用场景 | 批量生成 | 高端场景 |
| 成本 | 💰 低 | 💰 高 |

## 🐛 常见问题

### 1. 模型加载失败

**问题**: `ImportError: cannot import name 'CogVideoXImageToVideoPipeline'`

**解决**: 确保已安装最新版本的diffusers
```bash
pip install --upgrade diffusers transformers
```

### 2. 显存不足

**问题**: `CUDA out of memory`

**解决**: 
- 启用CPU offload: `enable_model_cpu_offload: true`
- 启用VAE tiling: `enable_tiling: true`
- 降低分辨率或帧数

### 3. 生成视频质量不佳

**问题**: 视频质量不如预期

**解决**:
- 使用Prompt Engine优化prompt
- 提供详细的场景配置
- 调整`guidance_scale`（推荐6.0-7.0）
- 启用`use_dynamic_cfg`

## 📝 测试检查清单

- [ ] 模型加载成功
- [ ] 视频生成成功
- [ ] 输出视频可播放
- [ ] Prompt Engine正常工作
- [ ] 模型路由正确选择
- [ ] 显存占用在预期范围内
- [ ] 生成时间在预期范围内

## 🔗 相关文件

- `gen_video/video_generator.py`: VideoGenerator主类
- `gen_video/utils/prompt_engine.py`: Prompt Engine
- `gen_video/utils/model_router.py`: 模型路由器
- `gen_video/test_cogvideox.py`: 测试脚本
- `gen_video/config.yaml`: 配置文件

## 📚 参考文档

- [双模型产线开发计划.md](./双模型产线开发计划.md)
- [Prompt Engine使用指南.md](./Prompt_Engine使用指南.md)
- [模型选择分析.md](./模型选择分析.md)

