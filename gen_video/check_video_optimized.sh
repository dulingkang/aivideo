#!/bin/bash
echo "=== 优化后的视频生成进度 ==="
echo ""
echo "进程状态:"
ps aux | grep "test_5_json.py" | grep -v grep || echo "  进程未运行"
echo ""
echo "最新日志 (视频生成相关):"
tail -100 /tmp/video_gen_optimized.log 2>/dev/null | grep -E "(生成视频|motion_bucket_id|noise_aug_strength|检测到静态|减少晃动|生成参数|完成)" | tail -10
echo ""
echo "已生成的视频文件:"
ls -lh /vepfs-dev/shawn/vid/fanren/gen_video/outputs/lingjie_ep5_full_test/videos/*.mp4 2>/dev/null | tail -5 || echo "  暂无视频文件"
