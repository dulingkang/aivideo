# RIFE 和 ComfyUI AnimateDiff 安装完成

## ✅ 已完成的安装

### 1. RIFE 官方实现
- ✅ 仓库已克隆：`/vepfs-dev/shawn/vid/fanren/RIFE`
- ✅ 依赖已安装（修复了 numpy 版本限制）
- ✅ 模型权重已下载并解压：`RIFE/train_log/`
- ✅ 代码已集成到 `video_generator.py`

### 2. ComfyUI
- ✅ 仓库已克隆：`/vepfs-dev/shawn/vid/fanren/ComfyUI`
- ✅ 依赖已安装

### 3. ComfyUI-AnimateDiff-Evolved
- ✅ 插件已克隆：`ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved`
- ✅ AnimateDiff Motion Adapter 已下载：`ComfyUI/models/animatediff_models/`

## 🎯 使用方法

### RIFE 插帧（自动使用）

系统会自动检测并使用 RIFE 官方实现，无需额外配置。

**验证**：运行视频生成时，如果看到以下输出，说明正在使用 RIFE：
```
✓ RIFE 模型加载成功（使用官方实现 v3 HD）
```

### ComfyUI AnimateDiff（需要启动服务器）

#### 1. 启动 ComfyUI 服务器

```bash
cd /vepfs-dev/shawn/vid/fanren/ComfyUI
source /vepfs-dev/shawn/venv/py312/bin/activate
python main.py --port 8188
```

**后台运行**：
```bash
cd /vepfs-dev/shawn/vid/fanren/ComfyUI
source /vepfs-dev/shawn/venv/py312/bin/activate
nohup python main.py --port 8188 > comfyui.log 2>&1 &
```

#### 2. 验证连接

```bash
curl http://127.0.0.1:8188/system_stats
```

或使用 Python：
```python
from gen_video.comfyui_integration import test_comfyui_connection
if test_comfyui_connection():
    print("ComfyUI 连接成功")
```

## 📋 文件结构

```
/vepfs-dev/shawn/vid/fanren/
├── RIFE/                          # RIFE 官方实现
│   ├── model/                     # 模型定义
│   ├── train_log/                 # 模型权重（已下载）
│   └── inference_video.py          # 推理脚本
├── ComfyUI/                       # ComfyUI
│   ├── custom_nodes/
│   │   └── ComfyUI-AnimateDiff-Evolved/  # AnimateDiff 插件
│   └── models/
│       └── animatediff_models/     # AnimateDiff 模型（已下载）
└── gen_video/
    ├── video_generator.py         # 已集成 RIFE
    └── comfyui_integration.py     # ComfyUI API 集成
```

## 🔧 配置

### RIFE 插帧

在 `config.yaml` 中：
```yaml
video:
  rife:
    enabled: true  # 启用插帧
    interpolation_scale: 2.0  # 插帧倍数
```

系统会自动使用 RIFE 官方实现（如果已安装）。

### ComfyUI AnimateDiff

需要先启动 ComfyUI 服务器，然后通过 API 调用。

## 📊 效果对比

| 方法 | 效果 | 速度 | 状态 |
|------|------|------|------|
| **RIFE 官方** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ 已安装 |
| OpenCV 光流 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ 降级方案 |
| 线性插值 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ 降级方案 |

## 🚀 下一步

### 测试 RIFE 插帧

```bash
python gen_video/test_full_pipeline_optimized.py --script lingjie/1.json --max-scenes 1
```

应该看到：
```
✓ RIFE 模型加载成功（使用官方实现 v3 HD）
✓ 插帧完成: 60 帧 → 120 帧
```

### 启动 ComfyUI（可选）

如果需要使用 ComfyUI AnimateDiff：
```bash
cd /vepfs-dev/shawn/vid/fanren/ComfyUI
source /vepfs-dev/shawn/venv/py312/bin/activate
python main.py --port 8188
```

## ✅ 总结

**RIFE 插帧**：
- ✅ 已安装并集成
- ✅ 自动检测和使用
- ✅ 无需额外配置

**ComfyUI AnimateDiff**：
- ✅ 已安装
- ✅ 模型已下载
- ⚠️ 需要启动服务器才能使用

**现在可以直接使用 RIFE 插帧了！**

