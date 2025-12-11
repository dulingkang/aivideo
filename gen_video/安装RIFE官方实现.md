# 安装 RIFE 官方实现指南

## 📋 当前状态

系统当前使用的是 **OpenCV 光流插帧**（降级方案），因为 RIFE 官方实现未安装。

## 🎯 安装 RIFE 官方实现（最佳效果）

### 步骤1：克隆 RIFE 仓库

```bash
cd /vepfs-dev/shawn/vid/fanren
git clone https://github.com/hzwer/arXiv2020-RIFE.git RIFE
cd RIFE
```

### 步骤2：安装依赖

```bash
# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 步骤3：下载模型权重

RIFE 需要模型权重文件，可以从以下位置下载：

1. **从 GitHub Releases 下载**：
   ```bash
   # 创建模型目录
   mkdir -p train_log
   
   # 下载模型权重（需要手动下载）
   # 访问：https://github.com/hzwer/arXiv2020-RIFE/releases
   # 下载最新的模型权重文件
   ```

2. **或使用 HuggingFace**：
   ```bash
   # 如果有 huggingface-cli
   huggingface-cli download hzwer/RIFE train_log
   ```

### 步骤4：验证安装

安装完成后，系统会自动检测并使用 RIFE 官方实现。

## 🔍 验证是否使用 RIFE

运行视频生成时，如果看到以下输出，说明正在使用 RIFE：

```
✓ RIFE 模型加载成功（使用官方实现）
```

如果看到以下输出，说明使用的是 OpenCV：

```
✓ 使用 OpenCV 光流插帧（简单但有效）
```

## 📊 效果对比

| 方法 | 效果 | 速度 | 安装难度 |
|------|------|------|----------|
| **RIFE 官方** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| OpenCV 光流 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐（自动可用） |
| 线性插值 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐（自动可用） |

## 💡 建议

### 如果追求最佳效果

**安装 RIFE 官方实现**：
- ✅ 插帧质量最高
- ✅ 运动最自然
- ⚠️ 需要手动安装

### 如果追求简单

**使用 OpenCV 光流插帧**（当前方案）：
- ✅ 自动可用
- ✅ 效果较好
- ✅ 无需安装

## 🚀 快速安装命令

```bash
# 一键安装（需要手动下载模型权重）
cd /vepfs-dev/shawn/vid/fanren
git clone https://github.com/hzwer/arXiv2020-RIFE.git RIFE
cd RIFE
source /vepfs-dev/shawn/venv/py312/bin/activate
pip install -r requirements.txt

# 然后下载模型权重到 train_log 目录
```

## ✅ 安装后

安装完成后，系统会自动检测并使用 RIFE，无需修改配置。

下次运行视频生成时，会自动使用 RIFE 官方实现。

