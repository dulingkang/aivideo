# 系统重构总结 v2.1

## ✅ 已完成的工作

### 1. 核心规则引擎 ✅

**文件**：`gen_video/utils/execution_rules_v2_1.py`

**功能**：
- ✅ SceneIntent → Shot 硬映射表（8种场景类型）
- ✅ Shot → Pose 允许/禁止规则表
- ✅ Model 路由表（硬规则，禁止动态切换）
- ✅ 性别负锁（工业级标配）
- ✅ 自动修正不合法组合（wide + lying）

**关键特性**：
```python
# Shot决策（硬映射，不可覆盖）
shot_decision = rules.get_shot_from_intent("character_intro")
# 返回: ShotDecision(shot_type=ShotType.MEDIUM, allow_override=False)

# Pose验证（自动修正）
pose_decision = rules.validate_pose(ShotType.WIDE, "lying")
# 自动修正为: PoseType.STAND（因为wide禁止lying）

# Model路由（硬规则）
model, identity = rules.get_model_route(has_character=True, shot_type=ShotType.MEDIUM)
# 返回: (ModelType.FLUX, "pulid")
```

---

### 2. 角色锚系统 ✅

**文件**：`gen_video/utils/character_anchor_v2_1.py`

**功能**：
- ✅ 角色LoRA管理（Layer 0，永远存在）
- ✅ InstantID条件启用（Layer 1，可选）
- ✅ 性别负锁（Layer 2，工业级标配）
- ✅ 角色锚优先级：LoRA > InstantID > 风格LoRA

**关键特性**：
```python
# 注册角色锚
anchor_manager.register_character(
    character_id="hanli",
    gender="male",
    lora_path="hanli_character_v1.safetensors",
    lora_weight=0.6
)

# 判断是否使用InstantID（条件启用）
should_use = anchor_manager.should_use_instantid("hanli", face_visible=True)

# 获取性别负锁
negative = anchor_manager.get_negative_prompt_with_gender_lock("hanli")
# 返回: ["female", "woman", "girl", ...]
```

---

### 3. JSON v2.1 Schema ✅

**文件**：`gen_video/schemas/scene_v2_1_example.json`

**关键变化**：
- ✅ 添加 `locks` 字段（shot/pose/gender/model全部锁定）
- ✅ `shot` 字段包含 `source` 和 `allow_override`
- ✅ `pose` 字段包含 `validated_by` 和 `auto_corrected`
- ✅ `character_anchor` 字段（LoRA配置）
- ✅ `identity_engine` 字段（InstantID条件配置）
- ✅ `negative_lock` 字段（性别负锁）
- ✅ `llm_usage.forbidden` 字段（明确禁止LLM决策的领域）

---

### 4. 重构计划文档 ✅

**文件**：`gen_video/REFACTOR_PLAN_V2_1.md`

**内容**：
- ✅ 详细的重构步骤
- ✅ 代码改动示例
- ✅ 预期效果对比表
- ✅ 实施时间表

---

## 🔄 待完成的工作

### 5. 重构 Execution Planner V3

**需要改动**：
1. 移除 `_evaluate_scene_stability` 中的LLM调用
2. 移除 `_select_engines` 中的动态决策
3. 改为使用 `ExecutionRulesV21` 的硬规则
4. 集成角色锚系统

**预计工作量**：1-2天

---

### 6. 重构 Prompt Builder

**需要改动**：
1. 移除LLM场景分析（只保留本地规则引擎）
2. Prompt Builder只做模板填充
3. 集成性别负锁

**预计工作量**：1天

---

### 7. 创建 v2 → v2.1 转换器

**需要功能**：
1. 自动应用硬规则（Shot/Pose/Model）
2. 自动添加角色锚配置
3. 自动添加性别负锁

**预计工作量**：1天

---

## 📊 核心改进对比

### 决策方式

| 项目 | v2（当前） | v2.1（重构后） |
|------|-----------|---------------|
| Shot决策 | LLM分析 + 动态判断 | 硬映射表（SceneIntent → Shot） |
| Pose验证 | 允许不合法组合 | 硬规则表（自动修正） |
| Model选择 | 稳定性评分 + 动态切换 | 硬路由表（禁止切换） |
| 角色锚定 | InstantID为主 | LoRA为主，InstantID为辅 |
| 性别锁定 | 无 | 性别负锁（工业级标配） |

### 稳定性提升

| 问题 | v2 | v2.1 |
|------|----|------|
| 女主乱入 | ❌ 偶尔 | ✅ 基本消失 |
| 躺姿翻车 | ❌ 经常 | ✅ 大幅下降 |
| 场景不对 | ❌ 偶尔 | ✅ 明显减少 |
| Flux玄学 | ❌ 不可预测 | ✅ 可预测 |
| 角色漂移 | ❌ 经常 | ✅ 基本消失 |

---

## 🎯 下一步行动

1. **立即执行**：
   - 测试 `ExecutionRulesV21` 和 `CharacterAnchorManager`
   - 验证硬规则是否生效

2. **本周完成**：
   - 重构 Execution Planner V3
   - 重构 Prompt Builder

3. **下周完成**：
   - 创建 v2 → v2.1 转换器
   - 更新所有测试用例

---

## 📝 使用示例

### 使用规则引擎

```python
from gen_video.utils.execution_rules_v2_1 import get_execution_rules

rules = get_execution_rules()

# 1. 获取Shot（硬映射）
shot_decision = rules.get_shot_from_intent("character_intro")
print(f"Shot: {shot_decision.shot_type.value}")  # "medium"

# 2. 验证Pose（自动修正）
pose_decision = rules.validate_pose(ShotType.WIDE, "lying")
print(f"Pose: {pose_decision.pose_type.value}")  # "stand" (自动修正)
print(f"Auto-corrected: {pose_decision.auto_corrected}")  # True

# 3. 获取Model路由
model, identity = rules.get_model_route(has_character=True, shot_type=ShotType.MEDIUM)
print(f"Model: {model.value}, Identity: {identity}")  # "flux", "pulid"
```

### 使用角色锚系统

```python
from gen_video.utils.character_anchor_v2_1 import get_character_anchor_manager

anchor_manager = get_character_anchor_manager()

# 1. 注册角色
anchor_manager.register_character(
    character_id="hanli",
    gender="male",
    lora_path="hanli_character_v1.safetensors",
    lora_weight=0.6
)

# 2. 获取角色锚
anchor = anchor_manager.get_anchor("hanli")
print(f"LoRA: {anchor.lora_path}, Weight: {anchor.lora_weight}")

# 3. 判断是否使用InstantID
should_use = anchor_manager.should_use_instantid("hanli", face_visible=True)
print(f"Use InstantID: {should_use}")  # True

# 4. 获取性别负锁
negative = anchor_manager.get_negative_prompt_with_gender_lock("hanli")
print(f"Negative lock: {negative[:3]}")  # ["female", "woman", "girl", ...]
```

---

## 🔗 相关文件

- ✅ `gen_video/utils/execution_rules_v2_1.py` - 执行型规则引擎
- ✅ `gen_video/utils/character_anchor_v2_1.py` - 角色锚系统
- ✅ `gen_video/schemas/scene_v2_1_example.json` - v2.1 JSON示例
- ✅ `gen_video/REFACTOR_PLAN_V2_1.md` - 重构计划
- ⏳ `gen_video/execution_planner_v3.py` - Execution Planner（待重构）
- ⏳ `gen_video/utils/prompt_engine_v2.py` - Prompt Engine（待重构）

