#!/bin/bash
echo "=== 修复后的流程进度 ==="
echo ""
echo "进程状态:"
ps aux | grep "test_5_json.py" | grep -v grep || echo "  进程未运行"
echo ""
echo "TTS状态:"
grep -E "(TTS|CosyVoice|配音)" /tmp/video_gen_fixed.log 2>/dev/null | tail -3
echo ""
echo "场景2图像生成参数:"
grep -E "(场景图像 2|中景|面部权重.*0\.6|ControlNet.*0\.4)" /tmp/video_gen_fixed.log 2>/dev/null | tail -3
echo ""
echo "最新进度:"
tail -5 /tmp/video_gen_fixed.log 2>/dev/null
