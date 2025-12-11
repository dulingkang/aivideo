#!/bin/bash
# 使用 proxychains4 运行科普视频生成工具

# 激活主环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 切换到 gen_video 目录
cd /vepfs-dev/shawn/vid/fanren/gen_video

# 使用 proxychains4 运行命令
proxychains4 "$@"

