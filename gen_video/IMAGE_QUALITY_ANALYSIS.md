# 人像生成质量分析报告

## 一、带人像场景概览

根据脚本 `lingjie/episode/1.json`，前9个场景中带人像的场景：

1. **scene_002** (id:1)
   - 描述：韩立躺在青灰色沙地上一动不动
   - 镜头：Top-down wide shot（俯视远景）
   - 姿态：Motionless, feeling the heat

2. **scene_004** (id:3)
   - 描述：韩立回想空间节点的遭遇，面色渐渐阴沉
   - 镜头：Close-up on face（面部特写）
   - 姿态：Unpleasant expression

3. **scene_005** (id:4)
   - 描述：他忆起与冰凤分离以及禁制爆发时的剧痛
   - 镜头：Medium shot（中景）
   - 姿态：Wracked by the restriction's pain

4. **scene_006** (id:5)
   - 描述：韩立施展元婴虚化之法，将庞大的元婴精元灌注全身
   - 镜头：Energy close-up（能量特写）
   - 姿态：Performing the Nascent Soul Void method

5. **scene_007** (id:6)
   - 描述：他费力偏过头，只见一望无际的青灰砂砾
   - 镜头：Wide pan（宽景平移）
   - 姿态：Straining to tilt head

6. **scene_009** (id:8)
   - 描述：韩立睁大双眼盯着高空，瞳孔里隐隐闪烁蓝芒
   - 镜头：Extreme eye close-up（眼部极特写）
   - 姿态：Eyes wide, staring

## 二、潜在问题分析

### 2.1 竖屏模式下的镜头距离优化问题

**问题**：刚才的优化强制将未指定的镜头类型转换为中景，可能导致：
- **scene_002**（俯视远景）：可能被误判为中景，失去远景的震撼感
- **scene_004**（面部特写）：可能被转换为中景，失去特写的表现力
- **scene_009**（眼部极特写）：虽然有特殊处理，但可能仍然受影响

**原因**：
```python
# prompt/builder.py 第578-597行
# 如果没有明确指定镜头类型，默认使用中景
if not has_explicit_shot_type and include_character:
    shot_type_for_prompt["is_medium"] = True
```

### 2.2 InstantID 配置问题

**当前配置**：
- `face_emb_scale: 0.95`（人脸相似度权重）
- `face_kps_scale: 0.70`（面部关键点缩放）
- `num_inference_steps: 40`（推理步数，使用DPM++采样器）

**潜在问题**：
1. **推理步数可能不足**：40步对于复杂场景（如scene_006的能量特效）可能不够
2. **面部关键点缩放**：0.70可能在某些场景下导致面部控制不足

### 2.3 LoRA权重冲突

**当前配置**：
- `lora.alpha: 0.75`（角色LoRA权重）
- `style_lora.alpha: 0.65`（风格LoRA权重）
- `face_emb_scale: 0.95`（InstantID人脸权重）

**潜在问题**：
- LoRA权重0.75可能在某些场景下与InstantID的0.95权重冲突
- 风格LoRA可能影响角色特征（特别是服饰和发型）

### 2.4 Prompt构建问题

**可能的问题**：
1. **中英文混用**：脚本使用英文，但prompt构建可能使用中文
2. **镜头描述转换**：camera字段（如"Close-up on face"）可能没有正确转换为prompt
3. **姿态描述**：character_pose可能没有充分利用

### 2.5 竖屏分辨率问题

**当前配置**：768x1152（竖屏）

**潜在问题**：
- 竖屏模式下人像的宽高比可能导致：
  - 面部被拉长
  - 身体比例不协调
  - 远景场景中人物太小

## 三、优化建议

### 3.1 镜头距离优化调整

**建议**：
1. **保留明确的镜头类型**：不要强制转换已明确的镜头类型（如"Top-down wide shot"、"Close-up on face"）
2. **只对未指定的场景使用默认中景**：如果camera字段为空或模糊，才使用默认中景

### 3.2 InstantID配置优化

**建议**：
1. **提高推理步数**：从40步提高到50-60步，特别是在复杂场景
2. **动态调整face_kps_scale**：
   - 特写场景：0.75-0.80（增强控制）
   - 中景场景：0.70（保持当前）
   - 远景场景：0.65（降低控制，更自然）

### 3.3 LoRA权重优化

**建议**：
1. **场景自适应权重**：
   - 特写场景：降低LoRA权重到0.65，提高InstantID权重到0.98
   - 中景场景：保持当前0.75
   - 远景场景：提高LoRA权重到0.80（角色特征更重要）

### 3.4 Prompt优化

**建议**：
1. **统一语言**：根据脚本使用英文prompt（ascii_only_prompt: true）
2. **充分利用camera字段**：确保所有camera描述都正确转换为prompt
3. **增强姿态描述**：character_pose应该与prompt紧密结合

### 3.5 分辨率优化

**建议**：
1. **测试不同的宽高比**：考虑使用更接近16:9的竖屏比例（如640x1024）
2. **使用面部检测**：如果生成的人脸变形，自动调整resolution或face_kps_scale

## 四、具体优化方案

### 方案1：修复镜头距离逻辑（立即实施）

```python
# prompt/builder.py
# 只对真正未指定的场景使用默认中景
if not has_explicit_shot_type and include_character:
    # 检查camera字段是否包含镜头关键词
    camera_desc = scene.get("camera") or ""
    if camera_desc and any(kw in camera_desc.lower() for kw in 
        ["wide", "close", "medium", "远景", "特写", "中景"]):
        # 已有镜头描述，不强制转换
        pass
    else:
        # 真正未指定，使用默认中景
        shot_type_for_prompt["is_medium"] = True
```

### 方案2：提高推理步数（针对复杂场景）

```yaml
# config.yaml
instantid:
  num_inference_steps: 50  # 从40提高到50，改善质量
  # 或者根据场景复杂度动态调整
```

### 方案3：场景自适应配置

根据场景类型（特写/中景/远景）动态调整：
- face_emb_scale
- face_kps_scale
- lora.alpha

## 五、验证测试

建议重新生成以下场景进行对比：
1. scene_002（远景）：验证是否保持远景效果
2. scene_004（特写）：验证是否保持特写效果
3. scene_006（复杂场景）：验证提高步数后的质量

