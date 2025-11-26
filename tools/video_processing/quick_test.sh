#!/bin/bash
# 快速测试脚本：检查依赖和运行测试

set -e

echo "=== 检查依赖 ==="

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "✗ python3 未安装"
    exit 1
fi
echo "✓ python3: $(python3 --version)"

# 检查ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "✗ ffmpeg 未安装"
    exit 1
fi
echo "✓ ffmpeg: $(ffmpeg -version | head -1)"

# 检查scenedetect
if ! command -v scenedetect &> /dev/null; then
    echo "⚠ scenedetect 未安装，请运行: pip install scenedetect[opencv]"
else
    echo "✓ scenedetect: $(scenedetect --version 2>&1 | head -1)"
fi

# 检查Python包
echo ""
echo "=== 检查Python包 ==="
python3 -c "import torch; print(f'✓ torch: {torch.__version__}')" 2>/dev/null || echo "✗ torch 未安装"
python3 -c "from transformers import BlipProcessor; print('✓ transformers')" 2>/dev/null || echo "✗ transformers 未安装"
python3 -c "from sentence_transformers import SentenceTransformer; print('✓ sentence-transformers')" 2>/dev/null || echo "✗ sentence-transformers 未安装"
python3 -c "import faiss; print('✓ faiss')" 2>/dev/null || echo "✗ faiss 未安装"

echo ""
echo "=== 测试完成 ==="

