#!/bin/bash
# 快速验证模型状态

echo "=========================================="
echo "模型快速验证"
echo "=========================================="
echo ""

models=(
    "SD3.5 Large Turbo:/vepfs-dev/shawn/vid/fanren/gen_video/models/sd3-turbo"
    "Flux.1:/vepfs-dev/shawn/vid/fanren/gen_video/models/flux1-dev"
    "Flux.2:/vepfs-dev/shawn/vid/fanren/gen_video/models/flux2-dev"
    "Hunyuan-DiT:/vepfs-dev/shawn/vid/fanren/gen_video/models/hunyuan-dit"
    "Kolors:/vepfs-dev/shawn/vid/fanren/gen_video/models/kolors"
)

for model_info in "${models[@]}"; do
    IFS=':' read -r name path <<< "$model_info"
    
    if [ -d "$path" ]; then
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        incomplete=$(find "$path" -name "*.incomplete" 2>/dev/null | wc -l)
        
        if [ "$incomplete" -gt 0 ]; then
            status="❌ 不完整 ($incomplete 个不完整文件)"
        elif [ "$size" = "0" ] || [ "$size" = "0B" ]; then
            status="❌ 未下载"
        else
            status="✅ 已下载"
        fi
        
        echo "$name: $status ($size)"
    else
        echo "$name: ❌ 目录不存在"
    fi
done

echo ""
echo "=========================================="
echo "验证完成"
echo "=========================================="
