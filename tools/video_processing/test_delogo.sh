#!/bin/bash
# 测试4K视频的delogo参数

INPUT="processed/test_clips/test_142_4k.mp4"
BASE_OUTPUT="processed/test_clips/test_delogo"

# 参考1080P的LOGO位置: 1650:40:200:150
# 4K按2倍比例: 3300:80:400:300

echo "=== 测试4K视频delogo参数 ==="
echo "输入文件: $INPUT"
echo ""

# 方案1: 按2倍比例 (3300:80:400:300)
echo "测试方案1: 3300:80:400:300 (按2倍比例)"
python3 tools/video_processing/clean_video.py \
  --input "$INPUT" \
  --output "${BASE_OUTPUT}_3300_80_400_300.mp4" \
  --crop 3840:1960:0:0 \
  --delogo 3300:80:400:300 \
  --auto-aspect

# 方案2: 稍微调整X位置 (3200:80:400:300)
echo ""
echo "测试方案2: 3200:80:400:300 (X向左移100)"
python3 tools/video_processing/clean_video.py \
  --input "$INPUT" \
  --output "${BASE_OUTPUT}_3200_80_400_300.mp4" \
  --crop 3840:1960:0:0 \
  --delogo 3200:80:400:300 \
  --auto-aspect

# 方案3: 稍微调整X位置 (3400:80:400:300)
echo ""
echo "测试方案3: 3400:80:400:300 (X向右移100)"
python3 tools/video_processing/clean_video.py \
  --input "$INPUT" \
  --output "${BASE_OUTPUT}_3400_80_400_300.mp4" \
  --crop 3840:1960:0:0 \
  --delogo 3400:80:400:300 \
  --auto-aspect

# 方案4: 调整Y位置 (3300:60:400:300)
echo ""
echo "测试方案4: 3300:60:400:300 (Y向上移20)"
python3 tools/video_processing/clean_video.py \
  --input "$INPUT" \
  --output "${BASE_OUTPUT}_3300_60_400_300.mp4" \
  --crop 3840:1960:0:0 \
  --delogo 3300:60:400:300 \
  --auto-aspect

# 方案5: 调整大小 (3300:80:450:350)
echo ""
echo "测试方案5: 3300:80:450:350 (增大LOGO区域)"
python3 tools/video_processing/clean_video.py \
  --input "$INPUT" \
  --output "${BASE_OUTPUT}_3300_80_450_350.mp4" \
  --crop 3840:1960:0:0 \
  --delogo 3300:80:450:350 \
  --auto-aspect

echo ""
echo "=== 测试完成 ==="
echo "请查看以下文件，选择效果最好的:"
ls -lh "${BASE_OUTPUT}"_*.mp4 2>/dev/null | awk '{print $9, "(" $5 ")"}'

