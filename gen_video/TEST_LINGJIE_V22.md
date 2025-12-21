# lingjie v2.2-final 测试指南

## 转换完成

### 已转换的场景
- **输入**: `lingjie/1.v2.json` (22个场景)
- **输出**: `lingjie/v22/` 目录
- **已转换**: 前5个场景（可扩展）

### 转换后的文件
```
lingjie/v22/
├── scene_000_v22.json  # 标题揭示（无角色）
├── scene_001_v22.json  # 韩立躺在沙地
├── scene_002_v22.json  # 场景2
├── scene_003_v22.json  # 场景3
├── scene_004_v22.json  # 场景4
└── all_scenes_v22.json # 合并文件
```

## 测试方法

### 1. 单个场景测试
```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
source /vepfs-dev/shawn/venv/py312/bin/activate
python3 test_v22_actual_generation_simple.py ../lingjie/v22/scene_001_v22.json
```

### 2. 批量场景测试
```bash
python3 test_lingjie_v22_batch.py --scenes-dir lingjie/v22 --max-scenes 5
```

### 3. 转换更多场景
```bash
# 转换前10个场景
python3 utils/v2_to_v22_converter.py ../lingjie/1.v2.json --output-dir ../lingjie/v22 --max-scenes 10

# 转换所有22个场景
python3 utils/v2_to_v22_converter.py ../lingjie/1.v2.json --output-dir ../lingjie/v22
```

## 已修复的问题

### 1. CLIPTokenizer 本地加载 ✅
- 优先使用 `models/sdxl-base/tokenizer`
- 避免网络下载失败

### 2. Pose 描述修复 ✅
- 场景1的pose从 "standing pose" 修复为 "lying on the ground, motionless"
- Prompt 中的 pose 描述也已修复

### 3. SceneEngine 映射修复 ✅
- `"flux"` → `SceneEngine.FLUX1`（不是 FLUX）

## 场景说明

### 场景0：标题揭示
- **类型**: 无角色场景
- **Shot**: medium
- **Model**: flux + none
- **环境**: 仙域天空，金色卷轴

### 场景1：韩立躺在沙地
- **类型**: 韩立场景
- **Shot**: wide
- **Pose**: lying
- **Model**: flux + pulid
- **环境**: 青灰色沙漠地面

## 预期结果

1. ✅ 使用本地 CLIPTokenizer，无网络下载
2. ✅ 正确触发增强模式（Flux + PuLID）
3. ✅ 生成清晰图像（1536x1536，40步）
4. ✅ 正确的人脸和服饰（LoRA + PuLID）

## 下一步

1. 运行批量测试，验证多个场景
2. 检查生成质量
3. 根据需要调整转换规则

