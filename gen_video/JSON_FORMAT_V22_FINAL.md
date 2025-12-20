# JSON格式 v2.2 Final - 直接包含所有决策信息

## 📋 总览

**核心原则**: JSON直接包含所有决策信息，无需LLM决策

**关键变化**:
- ❌ 不再需要v2→v2.1转换器
- ❌ 不再需要LLM做决策
- ✅ 所有参数直接写在JSON中
- ✅ LLM只做文案编辑（可选）

---

## 🎯 JSON结构

### 顶层结构

```json
{
  "version": "v2.2-final",
  "metadata": {...},
  "scene": {...}
}
```

### Scene结构

```json
{
  "scene": {
    "id": "scene_001",
    "duration_sec": 5.0,
    
    "intent": {...},           // 场景意图（仅描述）
    "shot": {...},             // Shot类型（直接指定）
    "pose": {...},             // Pose类型（直接指定）
    "model_route": {...},      // Model路由（直接指定）
    "character": {...},        // Character信息（完整）
    "environment": {...},      // 环境信息
    "prompt": {...},           // Prompt配置
    "generation_params": {...}, // 生成参数
    "validation": {...}        // 验证结果
  }
}
```

---

## 📝 字段详解

### 1. Intent（场景意图）

**作用**: 仅用于描述，不用于决策

```json
{
  "intent": {
    "type": "character_intro",
    "description": "韩立初次登场，展现其冷静内敛的气质"
  }
}
```

**注意**: `type`字段不再用于决策，仅用于文档和日志

---

### 2. Shot（镜头类型）

**作用**: 直接指定，无需LLM决策

```json
{
  "shot": {
    "type": "medium",
    "locked": true,
    "source": "intent_mapping",
    "description": "中景镜头，突出人物特征"
  }
}
```

**可选值**: `wide`, `medium`, `close_up`, `aerial`

**要求**: 必须指定，`locked: true`表示不可被覆盖

---

### 3. Pose（姿态类型）

**作用**: 直接指定，自动验证和修正

```json
{
  "pose": {
    "type": "stand",
    "locked": true,
    "validated_by": "shot_pose_rules",
    "auto_corrected": false,
    "description": "站立姿态，符合中景镜头要求"
  }
}
```

**可选值**: `stand`, `walk`, `sit`, `lying`, `kneel`, `face_only`

**自动修正**: 如果与Shot不兼容，自动修正并记录

---

### 4. Model Route（模型路由）

**作用**: 直接指定，无需LLM决策

```json
{
  "model_route": {
    "base_model": "flux",
    "identity_engine": "pulid",
    "locked": true,
    "decision_reason": "main_character -> flux + pulid",
    "character_role": "main"
  }
}
```

**base_model可选值**: `flux`, `sdxl`, `sdxl_turbo`

**identity_engine可选值**: `pulid`, `instantid`, `none`

**智能分流规则**:
- 主角 → `flux` + `pulid`
- NPC → `sdxl` + `instantid`
- 扩图任务 → `sdxl` + `none`
- 构图控制 → `sdxl` + `none`

---

### 5. Character（角色信息）

**作用**: 完整描述角色，包含所有锚定信息

```json
{
  "character": {
    "id": "hanli",
    "name": "韩立",
    "present": true,
    "role": "main",
    
    "identity": {
      "gender": "male",
      "age_range": "young_adult",
      "physique": "lean",
      "face_shape": "sharp"
    },
    
    "lora_config": {
      "type": "single",
      "lora_path": "path/to/HanLi_Face_v3.safetensors",
      "weight": 0.9,
      "trigger": "hanli"
    },
    
    "anchor_patches": {
      "temperament_anchor": "calm and restrained temperament, sharp but composed eyes",
      "explicit_lock_words": "wearing his iconic mid-late-stage green daoist robe",
      "face_detailer": {
        "enable": true,
        "trigger": "shot_scale >= medium",
        "denoise": 0.35
      }
    },
    
    "reference_image": "path/to/hanli_reference.jpg",
    "negative_gender_lock": [...]
  }
}
```

**关键字段**:
- `lora_config`: LoRA配置（单LoRA或LoRA Stack）
- `anchor_patches`: 运行时补丁（必须）
- `negative_gender_lock`: 性别负锁（必须）

---

### 6. Prompt（提示词配置）

**作用**: 模板填充，LLM只做文案编辑

```json
{
  "prompt": {
    "base_template": "{{character.name}}, {{character.anchor_patches.temperament_anchor}}, {{character.anchor_patches.explicit_lock_words}}, standing in {{environment.location}}, {{environment.atmosphere}}, cinematic lighting, high detail, epic atmosphere",
    
    "llm_enhancement": {
      "enable": true,
      "role": "copywriter",
      "tasks": [
        "enhance_scene_description",
        "add_atmosphere_details"
      ],
      "forbidden_tasks": [
        "decide_shot_type",
        "decide_pose_type",
        "decide_model_route"
      ]
    },
    
    "final": "HanLi, calm and restrained temperament, sharp but composed eyes, wearing his iconic mid-late-stage green daoist robe, standing in 黄枫谷, serene and mysterious atmosphere, cinematic lighting, high detail, epic atmosphere"
  }
}
```

**LLM角色**: `copywriter`（文案编辑）

**允许任务**:
- ✅ 增强场景描述
- ✅ 添加氛围细节
- ✅ 优化语言表达

**禁止任务**:
- ❌ 决定Shot类型
- ❌ 决定Pose类型
- ❌ 决定Model路由

---

## 🔧 使用方式

### 1. 直接使用新格式

**不再需要转换器**，直接使用新格式：

```python
# 旧方式（已废弃）
converter = JSONV2ToV21Converter()
scene_v21 = converter.convert_scene(scene_v2)

# 新方式（直接使用）
with open("scene_v22.json", "r") as f:
    scene = json.load(f)
    
executor = ExecutionExecutorV21(...)
result = executor.execute_scene(scene, output_dir)
```

### 2. 参数验证

**自动验证所有参数**：

```python
from utils.execution_validator import ExecutionValidator

validator = ExecutionValidator()
is_valid = validator.validate_scene(scene)

if not is_valid:
    errors = validator.get_errors()
    # 处理错误
```

### 3. 执行生成

**直接从JSON读取所有参数**：

```python
executor = ExecutionExecutorV21(...)

# 直接从JSON读取
shot_type = scene["scene"]["shot"]["type"]
pose_type = scene["scene"]["pose"]["type"]
model_route = scene["scene"]["model_route"]
character = scene["scene"]["character"]

# 执行生成（无需LLM决策）
result = executor.execute_scene(scene, output_dir)
```

---

## 📊 对比旧格式

### 旧格式（v2）

```json
{
  "version": "v2",
  "intent": {
    "type": "character_intro"  // 需要LLM决策
  },
  "character": {
    "pose": "standing"  // 需要LLM决策
  }
}
```

**问题**:
- ❌ 需要LLM决策
- ❌ 需要转换器
- ❌ 不稳定

### 新格式（v2.2-final）

```json
{
  "version": "v2.2-final",
  "scene": {
    "shot": {
      "type": "medium",  // 直接指定
      "locked": true
    },
    "pose": {
      "type": "stand",  // 直接指定
      "locked": true
    },
    "model_route": {
      "base_model": "flux",  // 直接指定
      "locked": true
    }
  }
}
```

**优势**:
- ✅ 无需LLM决策
- ✅ 无需转换器
- ✅ 稳定可预测

---

## 🎯 关键原则

1. **所有决策信息都在JSON中**
   - Shot类型（直接指定）
   - Pose类型（直接指定）
   - Model路由（直接指定）

2. **LLM只做文案编辑**
   - 场景描述优化
   - 氛围渲染
   - 不参与决策

3. **无需转换器**
   - 直接使用新格式
   - 旧格式直接废弃

---

## 🔗 相关文档

- `schemas/scene_v22_final.json` - JSON格式示例
- `WORK_PLAN_V22.md` - 工作计划追踪表
- `TECH_ARCHITECTURE_V2_1.md` - 技术架构文档

---

## 总结

**JSON格式v2.2-final的核心价值**:
- ✅ 直接包含所有决策信息
- ✅ 无需LLM决策
- ✅ 无需转换器
- ✅ 稳定可预测

**LLM的新角色**: 文案编辑，不是导演

