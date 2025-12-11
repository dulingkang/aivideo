# ComfyUI 使用指南

## ✅ 安装状态

- ✅ ComfyUI 已安装：`/vepfs-dev/shawn/vid/fanren/ComfyUI`
- ✅ AnimateDiff 插件已安装：`ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved`
- ✅ AnimateDiff 模型已下载：`ComfyUI/models/animatediff_models/` (1.7GB)
- ✅ 服务器已启动：运行在 `http://127.0.0.1:8188`

## 🚀 使用方法

### 方法1：Web UI（推荐，最简单）

1. **访问 Web UI**：
   ```
   http://127.0.0.1:8188
   ```

2. **在 Web UI 中构建工作流**：
   - 添加 "LoadImage" 节点加载图像
   - 添加 "AnimateDiff" 相关节点
   - 配置参数（帧数、提示词等）
   - 运行生成

3. **保存工作流**：
   - 可以保存工作流 JSON 供后续使用

### 方法2：API 调用（程序化）

使用 `comfyui_integration.py` 中的 `ComfyUIAPI` 类：

```python
from gen_video.comfyui_integration import ComfyUIAPI

# 创建 API 客户端
api = ComfyUIAPI(server_url="http://127.0.0.1:8188")

# 构建工作流（需要根据实际节点结构）
workflow = {
    # ... 工作流 JSON
}

# 提交任务
prompt_id = api.queue_prompt(workflow)

# 等待完成
api.wait_for_completion(prompt_id)

# 获取结果
history = api.get_history(prompt_id)
```

## 📋 服务器管理

### 启动服务器

```bash
# 方法1：使用脚本
bash gen_video/启动ComfyUI服务器.sh

# 方法2：手动启动
cd /vepfs-dev/shawn/vid/fanren/ComfyUI
source /vepfs-dev/shawn/venv/py312/bin/activate
python main.py --port 8188

# 方法3：后台运行
nohup python main.py --port 8188 > comfyui.log 2>&1 &
```

### 停止服务器

```bash
# 如果知道 PID
kill $(cat /tmp/comfyui.pid)

# 或查找进程
pkill -f "python main.py"
```

### 查看日志

```bash
tail -f /vepfs-dev/shawn/vid/fanren/ComfyUI/comfyui.log
```

## 🔍 验证安装

### 测试连接

```bash
python gen_video/test_comfyui.py
```

### 测试 AnimateDiff

```bash
python gen_video/test_comfyui_animatediff.py
```

## 📚 参考资源

- **ComfyUI 官方文档**：https://github.com/comfyanonymous/ComfyUI
- **AnimateDiff-Evolved 文档**：https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
- **Web UI**：http://127.0.0.1:8188

## 💡 提示

1. **首次使用建议使用 Web UI**：
   - 更直观
   - 可以查看节点结构
   - 可以保存工作流

2. **工作流 JSON 结构**：
   - 每个节点有唯一的 ID
   - 节点之间通过输入/输出连接
   - 需要根据实际节点类型构建

3. **AnimateDiff 节点**：
   - 需要 Motion Adapter 模型（已下载）
   - 支持多种参数（帧数、运动强度等）
   - 可以与 ControlNet、IP-Adapter 等结合使用

## ✅ 当前状态

- ✅ 服务器运行中：http://127.0.0.1:8188
- ✅ AnimateDiff 节点已加载
- ✅ 模型文件已就绪

**可以开始使用 ComfyUI 了！**

