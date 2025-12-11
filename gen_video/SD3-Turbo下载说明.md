# SD3.5 Large Turbo 下载说明

## ⚠️ 重要提示

`stabilityai/stable-diffusion-3.5-large-turbo` 是**受限模型（gated repo）**，需要：

1. **申请访问权限**: https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo
2. **等待批准**: 通常需要几分钟到几小时
3. **登录 HuggingFace**: 使用 `huggingface-cli login` 或 Python API

## 🔐 登录 HuggingFace

### 方式 1: 使用命令行

```bash
huggingface-cli login
# 输入你的 HuggingFace token
```

### 方式 2: 使用 Python

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
source /vepfs-dev/shawn/venv/py312/bin/activate

python << 'EOF'
from huggingface_hub import login
login()  # 会提示输入 token
EOF
```

### 获取 Token

1. 访问 https://huggingface.co/settings/tokens
2. 创建新的 token（需要有 read 权限）
3. 复制 token 并在登录时使用

## 📥 下载命令

登录后，运行：

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

## ✅ 验证下载

下载完成后，检查：

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video/models/sd3-turbo

# 检查 model_index.json
test -f model_index.json && echo "✅ 下载完成" || echo "⏳ 还在下载"

# 检查目录大小（应该有几个 GB）
du -sh .
```

## 🔍 检查下载进度

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video/models/sd3-turbo

# 实时查看目录大小变化
watch -n 5 'du -sh . && find . -type f | wc -l'
```

## ⚠️ 如果下载失败

如果遇到 403 错误：

1. **确认权限已批准**: 访问模型页面，确认显示 "You have been granted access"
2. **确认已登录**: 运行 `huggingface-cli whoami` 检查
3. **重新登录**: 如果 token 过期，重新登录
4. **等待**: 有时权限批准后需要等待几分钟才能生效

## 📝 当前状态

- **模型 ID**: `stabilityai/stable-diffusion-3.5-large-turbo`
- **目标目录**: `/vepfs-dev/shawn/vid/fanren/gen_video/models/sd3-turbo`
- **状态**: ⏳ 等待权限批准后下载

