#!/bin/bash
# 重新生成音频、字幕和视频的脚本

set -e

OUTPUT_NAME="${1:-lingjie_ep5_auto_v2}"
MARKDOWN="${2:-../lingjie/5.md}"
IMAGE_DIR="${3:-../lingjie/img2}"

echo "============================================================"
echo "重新生成音频、字幕和视频"
echo "============================================================"
echo "输出名称: $OUTPUT_NAME"
echo "Markdown: $MARKDOWN"
echo "图像目录: $IMAGE_DIR"
echo ""

# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

cd /vepfs-dev/shawn/vid/fanren/gen_video

# 步骤 1: 删除旧文件
echo "[1/4] 清理旧文件..."
rm -f outputs/$OUTPUT_NAME/audio.wav
rm -f outputs/$OUTPUT_NAME/subtitle.srt
echo "✓ 已删除旧音频和字幕文件"

# 步骤 2: 重新生成配音
echo ""
echo "[2/4] 重新生成配音..."
python run_pipeline.py \
  --markdown "$MARKDOWN" \
  --image-dir "$IMAGE_DIR" \
  --output "$OUTPUT_NAME" \
  --skip-image \
  --skip-video \
  --skip-subtitle \
  --skip-compose

if [ ! -f "outputs/$OUTPUT_NAME/audio.wav" ]; then
    echo "✗ 音频生成失败"
    exit 1
fi

echo "✓ 配音已生成"

# 步骤 3: 重新生成字幕
echo ""
echo "[3/4] 重新生成字幕..."
python run_pipeline.py \
  --markdown "$MARKDOWN" \
  --image-dir "$IMAGE_DIR" \
  --output "$OUTPUT_NAME" \
  --skip-image \
  --skip-video \
  --skip-tts \
  --skip-compose

if [ ! -f "outputs/$OUTPUT_NAME/subtitle.srt" ]; then
    echo "✗ 字幕生成失败"
    exit 1
fi

echo "✓ 字幕已生成"

# 步骤 4: 重新合成视频
echo ""
echo "[4/4] 重新合成视频..."
python run_pipeline.py \
  --markdown "$MARKDOWN" \
  --image-dir "$IMAGE_DIR" \
  --output "$OUTPUT_NAME" \
  --skip-image \
  --skip-video \
  --skip-tts \
  --skip-subtitle

if [ ! -f "outputs/$OUTPUT_NAME/${OUTPUT_NAME}.mp4" ]; then
    echo "✗ 视频合成失败"
    exit 1
fi

echo ""
echo "============================================================"
echo "✓ 完成！"
echo "============================================================"
echo "最终视频: outputs/$OUTPUT_NAME/${OUTPUT_NAME}.mp4"
echo ""
echo "请播放视频检查："
echo "  1. 是否为女声"
echo "  2. 语速是否正常"
echo "  3. 内容是否正确"
echo "  4. 是否有背景音乐"
echo "  5. 字幕是否同步"

