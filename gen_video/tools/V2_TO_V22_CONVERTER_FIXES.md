# v2 到 v2.2-final 转换工具修复总结

## 修复的问题

### 1. ✅ lora_path 为空问题
**问题**: 转换后的 JSON 中 `lora_path` 是空字符串 `""`，导致 LoRA 无法加载

**修复**: 
- 将默认 `lora_path` 设置为 `"models/lora/hanli/pytorch_lora_weights.safetensors"`
- 确保韩立角色的 LoRA 配置正确

### 2. ✅ reference_image 为空问题
**问题**: 转换后的 JSON 中 `reference_image` 是空字符串，导致参考图无法加载

**修复**:
- 优先从 `assets.reference_images` 中提取参考图路径
- 如果没有，使用默认路径 `"character_references/hanli_reference.jpg"`
- 修改 `_build_character_config` 函数，接收 `v2_scene` 参数以访问 `assets`

### 3. ✅ generation_params 缺失问题
**问题**: 转换后的 JSON 中 `generation_params` 被注释掉，导致无法使用场景特定的生成参数

**修复**:
- 添加 `_build_generation_params` 函数
- 根据 `shot_type` 设置合理的默认值：
  - `close_up`: `num_inference_steps = 50`（特写需要更多步数）
  - `wide`: `num_inference_steps = 40`（远景可以少一些）
  - 默认: `num_inference_steps = 50`
- 默认值：`width=1536, height=1536, guidance_scale=7.5, seed=-1`

### 4. ✅ pose 类型转换不完整
**问题**: v2 格式中的特殊 pose 类型（如 "recalling"、"enduring_pain"）无法正确转换

**修复**:
- 扩展 `_convert_pose_type` 函数的映射表：
  - `"recalling"` → `"stand"`（回想时通常是站立）
  - `"enduring_pain"` → `"stand"`（忍受痛苦时通常是站立）
  - `"motionless"` → `"lying"`（不动通常是躺着）

### 5. ✅ environment.location 为空问题
**问题**: 当 `visual_constraints.environment` 为空时，转换后的 `location` 也是空字符串

**修复**:
- 添加 `_build_environment` 函数
- 如果 `location` 为空，从 `notes` 中智能提取：
  - 包含 "沙地" 或 "沙漠" → `"desert wasteland, gray-green sand"`
  - 包含 "天空" 或 "仙域" → `"Sky of the immortal realm, mist-wreathed"`
  - 如果还是空，使用 `"unknown location"`
- 同时改进 `atmosphere` 构建，从 `intent.emotion` 中提取情绪信息

## 修复后的转换逻辑

### Character 配置
```python
# 修复前
"lora_path": "",
"reference_image": "",

# 修复后
"lora_path": "models/lora/hanli/pytorch_lora_weights.safetensors",
"reference_image": "character_references/hanli_reference.jpg"  # 或从 assets.reference_images 提取
```

### Generation Params
```python
# 修复前
# "generation_params": { ... }  # 被注释掉

# 修复后
"generation_params": {
    "width": 1536,
    "height": 1536,
    "num_inference_steps": 50,  # 根据 shot_type 调整
    "guidance_scale": 7.5,
    "seed": -1
}
```

### Environment
```python
# 修复前
"location": "",  # 可能为空

# 修复后
"location": "desert wasteland, gray-green sand",  # 从 notes 中智能提取
"atmosphere": "xianxia_anime style, soft_cinematic lighting, tense mood"  # 包含情绪信息
```

## 使用建议

1. **重新转换现有文件**: 如果之前转换的文件有问题，建议重新转换
   ```bash
   cd gen_video
   python3 utils/v2_to_v22_converter.py lingjie/episode/1.v2.json --output-dir lingjie/v22
   ```

2. **验证转换结果**: 检查转换后的 JSON 文件，确保：
   - `lora_path` 不为空
   - `reference_image` 不为空（如果有角色）
   - `generation_params` 存在
   - `environment.location` 不为空

3. **批量转换**: 使用批量转换脚本
   ```bash
   python3 tools/batch_convert_episode_to_v22.py \
       --episode-dir ../lingjie/episode \
       --output-dir ../lingjie/v22
   ```

## 注意事项

1. **默认路径**: 修复后的转换工具使用相对路径，确保在 `gen_video` 目录下运行
2. **参考图**: 如果 `assets.reference_images` 中有参考图，会优先使用；否则使用默认路径
3. **Location 提取**: 从 `notes` 中提取 location 的逻辑比较简单，可能需要根据实际情况调整

