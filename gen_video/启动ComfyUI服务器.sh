#!/bin/bash
# 启动 ComfyUI 服务器脚本

# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 切换到 ComfyUI 目录
cd /vepfs-dev/shawn/vid/fanren/ComfyUI

# 启动 ComfyUI 服务器
# --port 8188: 指定端口
# --listen 0.0.0.0: 允许外部访问（可选）
python main.py --port 8188

# 后台运行版本（取消注释使用）：
# nohup python main.py --port 8188 > comfyui.log 2>&1 &
# echo "ComfyUI 服务器已在后台启动，日志文件：comfyui.log"
# echo "查看日志：tail -f comfyui.log"
# echo "停止服务器：pkill -f 'python main.py'"

