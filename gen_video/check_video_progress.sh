#!/bin/bash
echo "=== 视频生成进度监控 ==="
echo ""
echo "进程状态:"
ps aux | grep "test_5_json.py" | grep -v grep || echo "  进程未运行"
echo ""
echo "最新日志 (最后20行):"
tail -20 /tmp/video_gen.log 2>/dev/null | tail -5
echo ""
echo "已生成的视频文件:"
ls -lh /vepfs-dev/shawn/vid/fanren/gen_video/outputs/lingjie_ep5_full_test/videos/ 2>/dev/null || echo "  暂无视频文件"
echo ""
echo "输出目录结构:"
find /vepfs-dev/shawn/vid/fanren/gen_video/outputs/lingjie_ep5_full_test -type f 2>/dev/null | head -10
