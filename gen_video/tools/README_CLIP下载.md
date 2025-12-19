# CLIP 模型下载指南

## 🚀 快速开始（推荐）

### 方法 1: 使用快速下载脚本（自动尝试多种方式）
```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
./tools/download_clip_fast.sh
```

这个脚本会按顺序尝试：
1. ModelScope（如果已安装，速度最快）
2. HuggingFace 镜像站
3. proxychains4（如果前两种失败）

### 方法 2: 使用 HuggingFace 镜像站（推荐，速度快）
```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
export HF_ENDPOINT=https://hf-mirror.com
python3 tools/download_clip_model.py
```

### 方法 3: 使用 ModelScope（国内最快）
```bash
# 先安装 ModelScope
pip install modelscope

# 然后运行
cd /vepfs-dev/shawn/vid/fanren/gen_video
python3 tools/download_clip_with_mirror.py
```

### 方法 4: 使用 proxychains4（如果镜像站不可用）
```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
proxychains4 python3 tools/download_clip_model.py
```

## 📊 速度对比

| 方法 | 速度 | 稳定性 | 推荐度 |
|------|------|--------|--------|
| ModelScope | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| HuggingFace 镜像站 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| proxychains4 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 直接下载 | ⭐ | ⭐⭐ | ⭐ |

## 💡 推荐方案

**首选：ModelScope**
- 国内速度最快
- 无需代理
- 稳定性好

**备选：HuggingFace 镜像站**
- 设置简单
- 速度较快
- 无需安装额外工具

## 🔧 故障排除

### 如果下载速度很慢（8k/s）
1. 尝试使用镜像站：`export HF_ENDPOINT=https://hf-mirror.com`
2. 或安装 ModelScope：`pip install modelscope`
3. 或使用 proxychains4

### 如果文件损坏
运行清理脚本后重新下载：
```bash
./tools/clean_all_clip_cache.sh
./tools/download_clip_fast.sh
```

### 如果所有方法都失败
可以手动从百度网盘下载后放到缓存目录：
- 缓存路径：`/vepfs-dev/shawn/.cache/huggingface/hub/models--openai--clip-vit-large-patch14`

