# SD3.5 Large Turbo 下载状态

更新时间: 2025-12-07 12:10

## 当前状态

❌ **下载未完成**

- 目录大小: 39K（只有元数据文件）
- 缺少: `model_index.json` 和权重文件
- 状态: 正在重新下载

## 下载信息

- **模型 ID**: `stabilityai/stable-diffusion-3.5-large-turbo`
- **目标目录**: `/vepfs-dev/shawn/vid/fanren/gen_video/models/sd3-turbo`
- **下载方式**: 使用 proxychains4 代理，支持断点续传

## 检查下载进度

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video/models/sd3-turbo

# 检查目录大小
du -sh .

# 检查是否有 model_index.json
test -f model_index.json && echo "✅ 下载完成" || echo "⏳ 还在下载"

# 检查文件数量
find . -type f | wc -l
```

## 预期文件结构

下载完成后应该包含：

```
sd3-turbo/
├── model_index.json          # ✅ 必需
├── transformer/              # ✅ 必需
├── vae/                      # ✅ 必需
├── text_encoder/             # ✅ 必需
├── text_encoder_2/           # ✅ 必需
├── tokenizer/                # ✅ 必需
├── tokenizer_2/              # ✅ 必需
└── scheduler/                # ✅ 必需
```

## 下载命令

如果下载中断，可以重新运行：

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
source /vepfs-dev/shawn/venv/py312/bin/activate

proxychains4 python << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="stabilityai/stable-diffusion-3.5-large-turbo",
    local_dir="/vepfs-dev/shawn/vid/fanren/gen_video/models/sd3-turbo",
    local_dir_use_symlinks=False,
    resume_download=True
)
EOF
```

## 验证下载

下载完成后，运行验证：

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
source /vepfs-dev/shawn/venv/py312/bin/activate
python test_model_loading.py
```

## 注意事项

1. **下载时间**: 模型较大，可能需要较长时间
2. **断点续传**: 支持断点续传，如果中断可以重新运行
3. **网络**: 使用 proxychains4 代理，确保网络连接稳定

