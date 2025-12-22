# V22 架构分析与优化建议（基于 V1 格式）

## 更新时间
2025-12-22

## 数据格式说明

`lingjie/episode/` 目录下有 **55+ 个 V1 格式 JSON 文件**（如 `1.json`, `2.json`...），只有 1-2 个 V2 格式文件（`1.v2.json`）。

**分析重点应放在 V1 → V22 的转换优化上。**

---

## V1 格式详解

V1 格式信息非常丰富，示例 (`1.json` scene_3):

```json
{
  "id": 3,
  "duration": 4,
  "description": "韩立回想空间节点的遭遇，面色渐渐阴沉。",  // ✅ 中文场景描述
  "mood": "Tense",                                    // ✅ 情绪
  "lighting": "Daytime",                              // ✅ 光照
  "action": "Recalling",                              // ✅ 动作
  "camera": "Close-up on face",                       // ✅ 镜头
  "visual": {
    "composition": "Han Li recalls the spatial node ordeal,his expression darkens",  // ✅ 构图描述
    "environment": "",                                // ⚠️ 有时为空
    "character_pose": "Unpleasant expression",        // ✅ 姿态
    "fx": "",
    "motion": {"type": "push_in", "speed": "slow"}
  },
  "prompt": "Han Li recalls the spatial-node ordeal,his expression turning grim.",  // ✅ 已有英文 prompt！
  "narration": "每当想起空间节点发生的一切，韩立的脸色就会暗沉下来，心底泛起寒意。"  // ✅ 旁白
}
```

### V1 数据的优势

| 字段 | 内容 | 利用价值 |
|------|------|----------|
| `prompt` | 已有英文场景 prompt | ⭐⭐⭐ 直接可用！ |
| `visual.composition` | 英文构图描述 | ⭐⭐⭐ 场景/人物完整描述 |
| `visual.environment` | 英文环境描述 | ⭐⭐⭐ 背景信息 |
| `description` | 中文场景描述 | ⭐⭐ 补充理解 |
| `mood` | 情绪 (Tense/Calm/...) | ⭐⭐ 表情引导 |
| `action` | 动作 (Recalling/Casting...) | ⭐⭐ 姿态引导 |
| `visual.character_pose` | 角色姿态描述 | ⭐⭐ 姿态细节 |
| `visual.fx` | 特效描述 | ⭐⭐ 视觉效果 |

---

## 问题概述

使用 `proxychains4 python3 test_lingjie_v22_batch.py --scenes-dir lingjie/v22/episode_1 --max-scenes 10` 测试时发现效果不佳：

1. **人脸相似度低**：多个场景相似度 < 0.7
2. **场景单调**：生成的图片"只展示一个人"，缺乏场景感
3. **人物一致性不够**：LoRA 效果未充分发挥

---

## 根因分析

### 1. V1 已有的 `prompt` 字段被忽略（最核心问题 ⚠️）

**问题**：V1 格式已经有现成的英文 `prompt` 字段，但 `v1_to_v22_converter.py` 完全忽略了它！

**V1 原始数据** (`1.json` scene_1):
```json
{
  "prompt": "Han Li lying on gray-green sand,motionless,feeling the ground's dry heat."  // ✅ 已有完整 prompt
}
```

**转换后的 V22** (`scene_001_v22.json`):
```json
{
  "prompt": {
    "final": "hanli, calm and restrained temperament, sharp but composed eyes, determined expression, wearing his iconic mid-late-stage green daoist robe, traditional Chinese cultivation attire, lying on the ground, motionless, cinematic lighting, high detail, epic atmosphere, Chinese fantasy illustration style"
    // ❌ 使用了固定模板，丢失了原始 prompt 的场景信息！
  }
}
```

**对比**：
- 原始 V1 prompt: `"Han Li lying on gray-green sand,motionless,feeling the ground's dry heat."`
- 转换后 V22: 只有角色描述，没有 "gray-green sand" 等环境信息

### 2. `visual.composition` 未被利用

V1 的 `visual.composition` 字段包含非常详细的场景描述，例如：
```json
"composition": "Han Li strains to tilt his head,sees endless gray-green gravel"
```
但转换器没有使用这个字段。

### 3. 转换器使用固定模板

`v1_to_v22_converter.py` 的 `_build_prompt_config()` 函数使用固定模板：
```python
# 第 209-251 行
prompt_parts = []
prompt_parts.append("hanli")
prompt_parts.append("calm and restrained temperament...")  # 固定描述
prompt_parts.append("wearing his iconic mid-late-stage green daoist robe...")  # 固定服饰
# ... 完全忽略了 v1_scene["prompt"] 和 v1_scene["visual"]["composition"]
```

### 2. LoRA 权重应用受限

**问题**：`load_lora_weights` 默认使用权重 1.0，无法设置自定义权重

```python
# pulid_engine.py 第 1038-1046 行
if lora_weight != 1.0:
    logger.warning(f"  ⚠ UNet LoRA 权重设置为 {lora_weight}，但 load_lora_weights 默认使用 1.0")
    # UNet LoRA 的权重实际上无法应用
```

**影响**：配置的 `weight: 0.9` 没有生效，可能导致 LoRA 效果过强或不协调

### 3. 解耦模式使用策略不合理

**当前策略**：
```python
# enhanced_image_generator.py 第 826-836 行
# 参考强度 >= 70% 时禁用解耦模式
if reference_strength < 70:
    use_decoupled = True  # wide/medium 用解耦
else:
    use_decoupled = False  # close_up 直接用 PuLID
```

**问题**：
- close_up 镜头不使用解耦模式，但 Prompt 中没有环境描述
- 导致 PuLID 只能生成"纯人像"，缺乏场景感
- 即使是特写，也应该有**模糊背景暗示**

### 4. V2 转换器环境提取不足

**问题**：`v2_to_v22_converter.py` 的环境提取逻辑过于简单：

```python
# 当 environment 为空时的处理
if not environment:
    # 只从 composition 尝试提取
    if composition:
        environment = composition
    else:
        environment = description or ""  # 直接用 description
```

**缺失**：没有利用 `notes`、`narration.text`、`intent` 等字段来推断/补充环境

---

## 架构合理性评估

### 优点 ✅

1. **解耦融合思想正确**：先场景后身份，避免身份约束压制环境表达
2. **参考强度控制合理**：根据镜头类型调整（wide=50%, medium=70%, close_up=85%）
3. **LoRA 加载逻辑完善**：支持 PEFT 格式转换、UNet LoRA 自动检测
4. **验证机制完整**：人脸相似度验证、质量分析器

### 问题 ❌

1. **Prompt 模板过于死板**：没有充分利用原始场景描述
2. **环境信息丢失严重**：转换过程中丢失了大量有价值的信息
3. **close_up 场景缺乏上下文**：特写也需要环境暗示
4. **LoRA 权重不可控**：diffusers API 限制

---

## 优化建议（针对 V1 格式）

### 1. 直接使用 V1 的 `prompt` 字段（优先级：最高 ⭐⭐⭐）

**问题**：V1 已有完整的英文 prompt，但转换器完全忽略了它。

**修改文件**：`gen_video/utils/v1_to_v22_converter.py`

**当前代码问题** (第 209-251 行):
```python
def _build_prompt_config(v1_scene, character, environment, pose_type):
    prompt_parts = []
    prompt_parts.append("hanli")  # ❌ 固定模板
    prompt_parts.append("calm and restrained temperament...")  # ❌ 固定描述
    # ... 完全忽略 v1_scene["prompt"]
```

**优化方案**：
```python
def _build_prompt_config_v1_enhanced(v1_scene: Dict[str, Any], character: Dict[str, Any], 
                                      environment: Dict[str, Any], pose_type: str) -> Dict[str, Any]:
    """
    V1 增强版 Prompt 构建
    
    策略：优先使用 V1 原始 prompt，然后补充角色锚定词
    """
    # 1. 获取 V1 原始 prompt（这是最有价值的信息！）
    v1_prompt = v1_scene.get("prompt", "").strip()
    v1_composition = v1_scene.get("visual", {}).get("composition", "").strip()
    
    prompt_parts = []
    
    # 2. 角色锚定词（如果角色存在）- 放在最前面确保身份
    if character.get("present", False):
        # 触发词 + 简短身份描述
        prompt_parts.append("hanli")
        prompt_parts.append("young male cultivator with sharp composed eyes")
        prompt_parts.append("wearing green daoist robe")
    
    # 3. ⭐ 核心：使用 V1 原始 prompt 或 composition
    if v1_prompt:
        # 清理 prompt，避免重复角色名
        cleaned_prompt = v1_prompt.replace("Han Li", "he").replace("Han Li's", "his")
        prompt_parts.append(cleaned_prompt)
    elif v1_composition:
        # 使用 composition 作为备选
        cleaned_composition = v1_composition.replace("Han Li", "he").replace("Han Li's", "his")
        prompt_parts.append(cleaned_composition)
    
    # 4. 添加情绪描述（从 mood 字段）
    mood = v1_scene.get("mood", "").strip()
    mood_map = {
        "tense": "tense expression",
        "solemn": "solemn contemplative expression",
        "alert": "alert vigilant expression, eyes slightly narrowed",
        "calm": "calm serene expression",
        "agony": "expression showing pain",
        "fierce": "fierce determined expression",
        "mysterious": "mysterious contemplative expression",
        "resolute": "resolute determined expression"
    }
    if mood.lower() in mood_map:
        prompt_parts.append(mood_map[mood.lower()])
    
    # 5. 添加环境（如果 V1 prompt 中没有）
    v1_environment = v1_scene.get("visual", {}).get("environment", "").strip()
    if v1_environment and "in " not in prompt_parts[-1].lower():
        prompt_parts.append(f"in {v1_environment}")
    
    # 6. 添加特效（如果有）
    fx = v1_scene.get("visual", {}).get("fx", "").strip()
    if fx:
        prompt_parts.append(fx)
    
    # 7. 风格锚定词（放在最后）
    prompt_parts.append("xianxia anime style, Chinese fantasy, cinematic lighting, high detail, 4K")
    
    final_prompt = ", ".join([p for p in prompt_parts if p])
    
    return {
        "base_template": "...",
        "final": final_prompt,
        "v1_original": v1_prompt,  # 保留原始 prompt 供参考
        "llm_enhancement": {"enable": False}
    }
```

### 2. 优化转换函数 `convert_v1_to_v22`（优先级：高 ⭐⭐⭐）

**目标**：在 `convert_v1_to_v22()` 函数中调用增强版 prompt 构建

**修改位置**：`v1_to_v22_converter.py` 第 403 行附近

```python
# 当前代码（第 403 行）
"prompt": _build_prompt_config(v1_scene, character, environment, pose_type),

# 改为
"prompt": _build_prompt_config_v1_enhanced(v1_scene, character, environment, pose_type),
```

### 3. 添加背景暗示（针对 close_up 镜头）（优先级：高 ⭐⭐⭐）

**问题**：close_up 镜头即使有 V1 prompt，也可能缺少背景信息

**在 `_build_prompt_config_v1_enhanced` 中添加**：

```python
# 检查镜头类型，为 close_up 添加背景暗示
camera = v1_scene.get("camera", "").lower()
if "close" in camera and "background" not in final_prompt.lower():
    # 从 environment 或 description 推断背景
    desc = v1_scene.get("description", "")
    env = v1_scene.get("visual", {}).get("environment", "")
    
    if "沙" in desc or "desert" in env.lower():
        prompt_parts.append("with soft blurred desert background")
    elif "天" in desc or "sky" in env.lower():
        prompt_parts.append("with blurred celestial sky background")
    elif "回想" in desc or "recall" in v1_prompt.lower():
        prompt_parts.append("with soft bokeh background suggesting memories")
    else:
        prompt_parts.append("with soft bokeh background")
```

### 4. 示例：转换前后对比

**V1 原始数据** (`1.json` scene_1):
```json
{
  "id": 1,
  "description": "韩立躺在青灰色沙地上一动不动，感受地面升腾的燥热。",
  "mood": "Solemn",
  "camera": "Top-down wide shot",
  "visual": {
    "composition": "Han Li lying on gray-green sand",
    "environment": "Gray-green desert floor",
    "character_pose": "Motionless,feeling the heat"
  },
  "prompt": "Han Li lying on gray-green sand,motionless,feeling the ground's dry heat."
}
```

**当前转换结果** (问题版本):
```
"final": "hanli, calm and restrained temperament, sharp but composed eyes, determined expression, wearing his iconic mid-late-stage green daoist robe, traditional Chinese cultivation attire, lying on the ground, motionless, cinematic lighting, high detail, epic atmosphere, Chinese fantasy illustration style"
```
❌ 丢失了 "gray-green sand"、"dry heat" 等场景信息

**优化后的转换结果**:
```
"final": "hanli, young male cultivator with sharp composed eyes, wearing green daoist robe, lying on gray-green sand, motionless, feeling the ground's dry heat, solemn contemplative expression, in Gray-green desert floor, xianxia anime style, Chinese fantasy, cinematic lighting, high detail, 4K"
```
✅ 保留了原始场景描述，同时添加了角色锚定词

### 2. 为 close_up 添加环境暗示（优先级：高 ⭐⭐⭐）

**目标**：特写镜头也应有模糊背景，避免"只展示一个人"

**修改文件**：`gen_video/enhanced_image_generator.py`

```python
# 在 _generate_with_pulid 中添加背景暗示
def _add_background_hint(self, prompt: str, scene: Dict[str, Any]) -> str:
    """为特写场景添加背景暗示"""
    shot_type = scene.get("shot", {}).get("type", "medium")
    
    if shot_type == "close_up":
        # 检查是否已有背景描述
        if "background" not in prompt.lower() and "in " not in prompt.lower():
            # 从 notes 或 narration 提取环境线索
            notes = scene.get("notes", "")
            intent_type = scene.get("intent", {}).get("type", "")
            
            if "闪回" in notes or intent_type == "flashback":
                prompt += ", with soft blurred background suggesting memories"
            elif "沙地" in notes or "沙漠" in notes:
                prompt += ", with blurred gray-green desert in background"
            else:
                prompt += ", with soft bokeh background"
    
    return prompt
```

### 3. 实现 LoRA 权重缩放（优先级：中 ⭐⭐）

**问题**：`load_lora_weights` 不支持自定义权重

**解决方案**：在加载前手动缩放 LoRA 权重

```python
def _scale_lora_weights(self, lora_path: str, scale: float) -> str:
    """缩放 LoRA 权重并返回临时文件路径"""
    from safetensors import safe_open
    from safetensors.torch import save_file
    import tempfile
    
    scaled_dict = {}
    with safe_open(lora_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # 只缩放权重，不缩放偏置
            if "weight" in key or "lora" in key:
                scaled_dict[key] = tensor * scale
            else:
                scaled_dict[key] = tensor
    
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        save_file(scaled_dict, tmp.name)
        return tmp.name
```

### 4. 增强 V2 转换器的环境提取（优先级：高 ⭐⭐⭐）

**修改文件**：`gen_video/utils/v2_to_v22_converter.py`

```python
def _build_environment_enhanced(v2_scene: Dict[str, Any]) -> Dict[str, Any]:
    """增强版环境构建"""
    visual = v2_scene.get("visual_constraints", {})
    environment = visual.get("environment", "").strip()
    
    # 如果 environment 为空，从其他字段推断
    if not environment:
        notes = v2_scene.get("notes", "")
        narration = v2_scene.get("narration", {}).get("text", "")
        all_text = f"{notes} {narration}"
        
        # 环境关键词映射
        env_patterns = [
            (["沙地", "沙砾", "沙漠"], "gray-green desert wasteland, barren land"),
            (["天空", "日", "月"], "mystical sky, celestial scene"),
            (["回想", "回忆", "闪回"], "abstract memory space, soft blur"),
            (["怪鸟", "黑影"], "ominous sky with dark shadows"),
        ]
        
        for keywords, env_desc in env_patterns:
            if any(kw in all_text for kw in keywords):
                environment = env_desc
                break
        
        if not environment:
            environment = "mystical cultivation world"  # 默认值
    
    # 构建完整环境配置
    return {
        "location": environment,
        "time": visual.get("time_of_day", "day"),
        "weather": visual.get("weather", "clear"),
        "atmosphere": _build_atmosphere(v2_scene),
        "background_elements": visual.get("elements", [])
    }
```

---

## 人物一致性优化策略

### 问题：如何保持人物一致性，同时描述场景？

**核心矛盾**：身份约束强 → 环境表达弱；身份约束弱 → 人脸相似度低

### 解决方案：分层控制

1. **Prompt 层面**：
   - 分离"人物描述"和"环境描述"
   - 人物描述放在 Prompt 前面（优先级高）
   - 环境描述作为上下文补充

2. **LoRA 层面**：
   - 使用角色专属 LoRA 锚定人物特征（韩立的 hanli LoRA）
   - LoRA 权重建议：0.75-0.85（避免过度锁定）

3. **PuLID 层面**：
   - 参考强度根据镜头调整：
     - wide: 40-50%（环境优先）
     - medium: 60-70%（平衡）
     - close_up: 80-90%（人脸优先）

4. **后处理层面**：
   - 使用 InsightFace 验证人脸相似度
   - 相似度 < 0.65 时触发重试机制

### 推荐配置

```yaml
# config.yaml 建议值
image:
  pulid:
    default_reference_strength: 70  # 平衡点
    
  reference_strength_by_shot:
    wide: 45        # 降低，允许更多环境表达
    medium: 65      # 降低，平衡
    close_up: 85    # 保持，强锁脸
    
  lora_config:
    weight: 0.80    # 降低，避免过度锁定
    
  identity_verification:
    similarity_threshold: 0.65  # 降低阈值，允许更多变化
```

---

## 测试建议

修改完成后，运行以下测试验证：

```bash
# 1. 重新转换 V1 → V22（使用优化后的转换器）
python3 gen_video/utils/v1_to_v22_converter.py \
    lingjie/episode/1.json \
    --output-dir lingjie/v22/episode_1_v1_enhanced \
    --max-scenes 10

# 2. 验证转换质量
python3 gen_video/utils/conversion_validator.py \
    lingjie/v22/episode_1_v1_enhanced

# 3. 运行批量测试
proxychains4 python3 gen_video/test_lingjie_v22_batch.py \
    --scenes-dir lingjie/v22/episode_1_v1_enhanced \
    --max-scenes 10

# 4. 批量转换所有 V1 文件（可选）
for i in {1..55}; do
    python3 gen_video/utils/v1_to_v22_converter.py \
        lingjie/episode/${i}.json \
        --output-dir lingjie/v22/episode_${i} \
        --max-scenes 30
done
```

---

## 总结

| 问题 | 根因 | 优化方案 | 优先级 |
|------|------|----------|--------|
| 场景单调 | V1 `prompt` 字段被忽略 | 直接使用 V1 原始 prompt | ⭐⭐⭐ |
| 环境信息丢失 | `visual.composition` 未利用 | 融合 composition 和 environment | ⭐⭐⭐ |
| close_up 无背景 | 缺少背景暗示 | 从 description 推断背景 | ⭐⭐⭐ |
| 人脸相似度低 | LoRA 权重不可控 | 实现权重缩放（可选） | ⭐⭐ |

**核心优化点**：
1. **保留 V1 原始 prompt**：`"Han Li lying on gray-green sand,motionless,feeling the ground's dry heat."`
2. **补充角色锚定词**：`"hanli, young male cultivator with sharp composed eyes, wearing green daoist robe"`
3. **添加背景暗示**：`"with soft blurred desert background"`

**预期改进**：
- 人脸相似度从 ~0.4 提升到 ~0.7
- 场景丰富度显著提升（保留原始场景描述）
- 保持人物一致性的同时展示完整场景

