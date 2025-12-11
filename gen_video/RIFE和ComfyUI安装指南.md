# RIFE 和 ComfyUI AnimateDiff 安装指南

## ✅ 已完成的步骤

### 1. RIFE 仓库克隆
- ✅ 已克隆到 `/vepfs-dev/shawn/vid/fanren/RIFE`

### 2. ComfyUI 仓库克隆
- ✅ 已克隆到 `/vepfs-dev/shawn/vid/fanren/ComfyUI`

### 3. ComfyUI-AnimateDiff-Evolved 插件
- ✅ 已克隆到 `/vepfs-dev/shawn/vid/fanren/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved`

## 📋 待完成的步骤

### 步骤1：安装 RIFE 依赖（修复版本限制）

RIFE 的 `requirements.txt` 中 numpy 版本限制太旧（<=1.23.5），已修复为支持 Python 3.12。

**安装命令**：
```bash
cd /vepfs-dev/shawn/vid/fanren/RIFE
source /vepfs-dev/shawn/venv/py312/bin/activate
proxychains4 -q pip install -r requirements.txt -i https://pypi.org/simple
```

### 步骤2：下载 RIFE 模型权重

RIFE 需要预训练模型权重，需要手动下载：

**方法1：从 Google Drive 下载**
```bash
cd /vepfs-dev/shawn/vid/fanren/RIFE
mkdir -p train_log

# 使用 gdown 或手动下载
# Google Drive 链接：https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view?usp=sharing
# 百度网盘：https://pan.baidu.com/share/init?surl=u6Q7-i4Hu4Vx9_5BJibPPA 密码:hfk3

# 下载后解压到 train_log 目录
unzip train_log.zip -d train_log/
```

**方法2：使用 gdown（如果已安装）**
```bash
pip install gdown
cd /vepfs-dev/shawn/vid/fanren/RIFE
gdown "https://drive.google.com/uc?id=1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_" -O train_log.zip
unzip train_log.zip -d train_log/
```

### 步骤3：安装 ComfyUI 依赖

**安装命令**：
```bash
cd /vepfs-dev/shawn/vid/fanren/ComfyUI
source /vepfs-dev/shawn/venv/py312/bin/activate
proxychains4 -q pip install -r requirements.txt -i https://pypi.org/simple
```

### 步骤4：安装 ComfyUI-AnimateDiff-Evolved 依赖

**检查是否有 requirements.txt**：
```bash
cd /vepfs-dev/shawn/vid/fanren/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved
ls requirements.txt
```

如果有，安装：
```bash
proxychains4 -q pip install -r requirements.txt -i https://pypi.org/simple
```

### 步骤5：下载 AnimateDiff 模型

**下载 Motion Adapter**：
```bash
cd /vepfs-dev/shawn/vid/fanren/ComfyUI/models
mkdir -p animatediff

# 下载 SD1.5 motion adapter
proxychains4 -q huggingface-cli download guoyww/animatediff-motion-adapter-v1-5-2 \
    --local-dir models/animatediff/motion_adapter_v1_5_2

# 下载 SDXL motion adapter（如果使用 SDXL）
proxychains4 -q huggingface-cli download guoyww/animatediff-motion-adapter-sdxl \
    --local-dir models/animateddiff/motion_adapter_sdxl
```

### 步骤6：启动 ComfyUI 服务器

**启动命令**：
```bash
cd /vepfs-dev/shawn/vid/fanren/ComfyUI
source /vepfs-dev/shawn/venv/py312/bin/activate
python main.py --port 8188
```

**后台运行**：
```bash
nohup python main.py --port 8188 > comfyui.log 2>&1 &
```

## 🔧 验证安装

### 验证 RIFE

```bash
cd /vepfs-dev/shawn/vid/fanren/RIFE
source /vepfs-dev/shawn/venv/py312/bin/activate
python -c "from model.RIFE_HDv3 import Model; print('RIFE 导入成功')"
```

### 验证 ComfyUI

```bash
cd /vepfs-dev/shawn/vid/fanren/ComfyUI
source /vepfs-dev/shawn/venv/py312/bin/activate
python -c "import comfy; print('ComfyUI 导入成功')"
```

### 验证 ComfyUI 连接

```bash
python gen_video/comfyui_integration.py
# 或
curl http://127.0.0.1:8188/system_stats
```

## 📝 配置更新

安装完成后，系统会自动检测并使用：
- **RIFE**：自动检测 `RIFE/train_log` 目录
- **ComfyUI**：通过 API 调用（需要启动服务器）

## 🚀 使用方式

### RIFE 插帧

系统会自动使用 RIFE（如果已安装），无需额外配置。

### ComfyUI AnimateDiff

需要先启动 ComfyUI 服务器，然后通过 API 调用。

## ⚠️ 注意事项

1. **RIFE 模型权重**：必须下载并解压到 `train_log` 目录
2. **ComfyUI 服务器**：需要先启动服务器才能使用
3. **端口冲突**：确保 8188 端口未被占用
4. **依赖版本**：某些依赖可能需要调整版本以支持 Python 3.12

## 📚 参考文档

- RIFE 官方：https://github.com/hzwer/arXiv2020-RIFE
- ComfyUI 官方：https://github.com/comfyanonymous/ComfyUI
- AnimateDiff-Evolved：https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved

