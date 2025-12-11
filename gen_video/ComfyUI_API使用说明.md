# ComfyUI API 使用说明（服务器环境）

## ✅ 当前状态

- ✅ ComfyUI 服务器已启动：`http://127.0.0.1:8188`
- ✅ AnimateDiff 插件已安装
- ✅ API 模块已创建：`comfyui_animatediff_api.py`
- ⚠️ **需要下载 checkpoint 模型**：ComfyUI 的 checkpoints 目录为空

## 🔧 问题解决

### 问题1：Checkpoint 模型缺失

**错误信息**：
```
ckpt_name: 'v1-5-pruned-emaonly.safetensors' not in []
```

**解决方案**：

需要下载 Stable Diffusion 1.5 模型到 ComfyUI 的 checkpoints 目录：

```bash
cd /vepfs-dev/shawn/vid/fanren/ComfyUI/models/checkpoints

# 使用 proxychains4 下载
proxychains4 -q huggingface-cli download runwayml/stable-diffusion-v1-5 \
    --local-dir . \
    --include "v1-5-pruned-emaonly.safetensors"
```

或者使用已有的模型：

```bash
# 如果已有 SDXL 模型，可以创建软链接
ln -s /vepfs-dev/shawn/vid/fanren/gen_video/models/sdxl-base/*.safetensors \
    /vepfs-dev/shawn/vid/fanren/ComfyUI/models/checkpoints/
```

### 问题2：工作流节点连接

当前工作流结构需要根据实际的 ComfyUI 节点结构调整。建议：

1. **先通过 Web UI 构建工作流**（如果有 SSH 隧道）
2. **导出工作流 JSON**（File -> Export (API)）
3. **使用导出的 JSON 作为模板**

## 🚀 使用方法

### 方法1：使用 API 模块（推荐）

```python
from gen_video.comfyui_animatediff_api import ComfyUIAnimateDiffAPI

# 创建 API 客户端
api = ComfyUIAnimateDiffAPI(server_url="http://127.0.0.1:8188")

# 生成视频帧
frames = api.generate_video_from_image(
    image_path="path/to/image.png",
    prompt="anime style, xianxia fantasy",
    negative_prompt="blurry, low quality",
    num_frames=16,
    width=512,
    height=512,
    output_dir="outputs/comfyui_test",
)
```

### 方法2：直接调用 API

```python
import requests
import json

# 1. 上传图像
with open("image.png", "rb") as f:
    files = {"image": ("image.png", f, "image/png")}
    response = requests.post("http://127.0.0.1:8188/upload/image", files=files)
    image_filename = response.json()["name"]

# 2. 构建工作流（需要根据实际节点结构）
workflow = {
    # ... 工作流 JSON
}

# 3. 提交任务
response = requests.post(
    "http://127.0.0.1:8188/prompt",
    json={"prompt": workflow}
)
prompt_id = response.json()["prompt_id"]

# 4. 等待完成
# ... 轮询检查任务状态

# 5. 下载结果
# ... 从 /view 端点下载生成的图像
```

## 📋 下一步

1. **下载 checkpoint 模型**（必须）
2. **修复工作流结构**（根据实际节点）
3. **测试 API 调用**

## 💡 提示

- **获取可用节点**：`curl http://127.0.0.1:8188/object_info`
- **查看节点结构**：在 Web UI 中查看节点输入/输出
- **导出工作流**：在 Web UI 中 File -> Export (API)

## ⚠️ 注意事项

1. **Checkpoint 必须存在**：ComfyUI 需要 checkpoint 模型才能工作
2. **节点连接必须正确**：确保节点之间的输入/输出类型匹配
3. **AnimateDiff 需要 motion model**：已下载在 `models/animatediff_models/`

