# v2 到 v2.2-final 转换总结

## 转换完成

### 转换的文件
- **输入**: `lingjie/1.v2.json` (22个场景)
- **输出**: `lingjie/v22/` 目录
  - `scene_000_v22.json` ~ `scene_004_v22.json` (前5个场景)
  - `all_scenes_v22.json` (合并文件)

### 转换器
- **脚本**: `utils/v2_to_v22_converter.py`
- **功能**: 自动将 v2 格式转换为 v2.2-final 格式

## 转换规则

### 1. Shot 转换
- 从 `camera.shot` 直接映射
- 添加 `locked: true` 和 `source: "v2_conversion"`

### 2. Pose 转换
- `lying_motionless` → `lying`
- `sitting` → `sit`
- `standing` → `stand`
- `walking` → `walk`
- `kneeling` → `kneel`

### 3. Model Route 转换
- `generation_policy.image_model` → `model_route.base_model`
- 如果角色是 `hanli` 且 `present: true` → `identity_engine: "pulid"`
- 否则 → `identity_engine: "none"`

### 4. Character 配置
- 韩立角色：自动添加完整的 `lora_config` 和 `anchor_patches`
- 其他角色：简化配置

### 5. Prompt 构建
- 从 `character.pose`、`visual_constraints.environment` 等构建
- 自动添加 `temperament_anchor` 和 `explicit_lock_words`（韩立）

## 修复的问题

### 1. CLIPTokenizer 本地加载
**问题**: 代码尝试从 HuggingFace 下载 CLIP tokenizer，但本地已有模型

**修复**:
- `execution_executor_v21.py`: 优先使用 `models/sdxl-base/tokenizer`
- `image_generator.py`: 优先使用本地路径

**修复后的逻辑**:
```python
# 1. 尝试从本地SDXL模型加载
local_sdxl_path = Path(__file__).parent / "models" / "sdxl-base"
if local_sdxl_path.exists() and (local_sdxl_path / "tokenizer").exists():
    tokenizer = CLIPTokenizer.from_pretrained(
        str(local_sdxl_path),
        subfolder="tokenizer",
        local_files_only=True
    )

# 2. 如果失败，尝试使用缓存
if tokenizer is None:
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14",
        local_files_only=True
    )
```

## 测试建议

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
python3 utils/v2_to_v22_converter.py ../lingjie/1.v2.json --output-dir ../lingjie/v22 --max-scenes 10
```

## 转换后的场景示例

### 场景0：标题揭示（无角色）
- Shot: medium
- Pose: stand (默认，无角色)
- Model: flux + none (无角色)
- 环境: 仙域天空，金色卷轴

### 场景1：韩立躺在沙地
- Shot: wide
- Pose: lying
- Model: flux + pulid (韩立)
- 环境: 青灰色沙漠地面

## 注意事项

1. **Pose 描述**: 已修复，现在正确显示 "lying on the ground, motionless" 而不是 "standing pose"

2. **参考图路径**: 转换后的 JSON 中 `reference_image` 为空字符串，系统会自动从 `character_reference_images` 获取

3. **LoRA 路径**: 转换后的 JSON 中 `lora_path` 为空字符串，系统会使用配置中的默认 LoRA

4. **生成参数**: 已优化为 1536x1536 分辨率，40步，确保清晰度

