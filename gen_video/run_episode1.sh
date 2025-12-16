#!/bin/bash
# 生成第一集完整视频的脚本

cd /vepfs-dev/shawn/vid/fanren/gen_video

# 运行主程序
python main.py \
  --script lingjie/episode/1.json \
  --output lingjie_ep1_full \
  --force

