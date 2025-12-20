# 系统重构计划 v2.1 - 工业级稳定架构

## 📋 总览

**目标**：将系统从"研究型/规则型过度设计"拉回到"工业级稳定系统"

**核心原则**：
1. Shot/Pose/Gender/Model 全部"锁死"，不可被LLM覆盖
2. 角色身份来自LoRA，不来自prompt
3. 任何不合法组合 → 自动修正，而不是fallback
4. LLM只能补充描述，不能决策

---

## 🎯 重构步骤

### ✅ 第一步：创建核心规则引擎（已完成）

**文件**：
- `gen_video/utils/execution_rules_v2_1.py` - 执行型规则引擎
- `gen_video/utils/character_anchor_v2_1.py` - 角色锚系统

**功能**：
- SceneIntent → Shot 硬映射
- Shot → Pose 允许表
- Model 路由表
- 性别负锁

---

### 🔄 第二步：重构 Execution Planner V3

**文件**：`gen_video/execution_planner_v3.py`

**改动**：
1. **移除LLM决策**：
   - 删除 `_evaluate_scene_stability` 中的LLM调用
   - 删除 `_select_engines` 中的动态决策
   - 改为使用 `ExecutionRulesV21` 的硬规则

2. **Shot决策改为硬映射**：
   ```python
   # 旧代码（LLM决策）
   shot_type = self._normalize_shot_type(camera.get("shot", "medium"))
   
   # 新代码（硬映射）
   from utils.execution_rules_v2_1 import get_execution_rules
   rules = get_execution_rules()
   intent = scene.get("intent", {}).get("type", "character_intro")
   shot_decision = rules.get_shot_from_intent(intent)
   shot_type = shot_decision.shot_type
   ```

3. **Pose验证改为硬规则**：
   ```python
   # 旧代码（允许不合法组合）
   character_pose = character.get("pose", "standing")
   
   # 新代码（自动修正）
   pose_decision = rules.validate_pose(shot_type, character_pose)
   if pose_decision.auto_corrected:
       logger.warning(f"Pose已自动修正: {character_pose} -> {pose_decision.pose_type.value}")
   ```

4. **Model路由改为硬规则**：
   ```python
   # 旧代码（动态决策）
   if stability_score < 0.5:
       scene_engine = SceneEngine.SDXL
   else:
       scene_engine = SceneEngine.FLUX1
   
   # 新代码（硬规则）
   scene_model, identity_engine = rules.get_model_route(
       has_character=character_present,
       shot_type=shot_type
   )
   ```

---

### 🔄 第三步：重构 Prompt Builder

**文件**：`gen_video/utils/prompt_engine_v2.py` 和 `execution_planner_v3.py` 中的 `build_weighted_prompt`

**改动**：
1. **移除LLM场景分析**：
   - 删除 `analyze_scene` 调用（LLM模式）
   - 只保留本地规则引擎（如果需要）

2. **Prompt Builder只做模板填充**：
   ```python
   def build_prompt_v21(self, scene: Dict[str, Any]) -> str:
       """只做模板填充，不做决策"""
       # 1. 从scene中读取已锁定的shot/pose
       shot_type = scene["shot"]["type"]  # 已锁定
       pose_type = scene["pose"]["type"]  # 已锁定
       
       # 2. 填充模板
       shot_desc = self._get_shot_description(shot_type)
       pose_desc = self._get_pose_description(pose_type)
       scene_desc = scene["prompt"]["scene_description"]
       char_desc = scene["prompt"]["positive_core"]
       
       # 3. 合并（不进行LLM分析）
       return f"{shot_desc}, {pose_desc}, {scene_desc}, {char_desc}"
   ```

3. **添加性别负锁**：
   ```python
   from utils.character_anchor_v2_1 import get_character_anchor_manager
   
   anchor_manager = get_character_anchor_manager()
   character_id = scene["character"]["id"]
   negative_prompt = anchor_manager.get_negative_prompt_with_gender_lock(
       character_id=character_id,
       base_negative=scene.get("negative", [])
   )
   ```

---

### 🔄 第四步：集成角色锚系统

**文件**：所有使用角色身份的地方

**改动**：
1. **初始化角色锚管理器**：
   ```python
   from utils.character_anchor_v2_1 import get_character_anchor_manager
   
   # 在系统初始化时注册角色
   anchor_manager = get_character_anchor_manager(character_profiles)
   anchor_manager.register_character(
       character_id="hanli",
       gender="male",
       lora_path="hanli_character_v1.safetensors",
       lora_weight=0.6
   )
   ```

2. **使用角色锚**：
   ```python
   anchor = anchor_manager.get_anchor(character_id)
   
   # LoRA（永远存在）
   if anchor.lora_path:
       # 加载LoRA
       load_lora(anchor.lora_path, weight=anchor.lora_weight)
   
   # InstantID（条件启用）
   if anchor_manager.should_use_instantid(character_id, face_visible):
       # 使用InstantID
       use_instantid(anchor.instantid_strength)
   ```

---

### 🔄 第五步：更新JSON v2 → v2.1转换器

**文件**：创建 `gen_video/utils/convert_v2_to_v2_1.py`

**功能**：
- 将现有v2 JSON转换为v2.1格式
- 自动应用硬规则（Shot/Pose/Model）
- 自动添加角色锚配置
- 自动添加性别负锁

---

### 🔄 第六步：更新测试和验证

**文件**：更新所有测试文件

**改动**：
1. 更新测试用例以使用v2.1规则
2. 验证硬规则是否生效
3. 验证角色锚是否永不丢失
4. 验证不合法组合是否自动修正

---

## 📊 预期效果

### 稳定性提升

| 问题 | v2（当前） | v2.1（重构后） |
|------|-----------|---------------|
| 女主乱入 | ❌ 偶尔出现 | ✅ 基本消失（性别负锁） |
| 躺姿翻车 | ❌ 经常出现 | ✅ 大幅下降（硬规则修正） |
| 场景不对 | ❌ 偶尔出现 | ✅ 明显减少（Shot硬映射） |
| Flux玄学 | ❌ 不可预测 | ✅ 可预测（Model路由表） |
| 角色漂移 | ❌ 经常出现 | ✅ 基本消失（LoRA锚定） |

### 代码简化

- **删除**：~30% 的LLM决策代码
- **简化**：Prompt Builder从500行减少到~200行
- **稳定**：决策路径从"智能判断"改为"表驱动"

---

## 🚨 注意事项

1. **向后兼容**：
   - 保留v2格式支持（自动转换）
   - 逐步迁移到v2.1

2. **配置迁移**：
   - 更新 `config.yaml` 以支持v2.1规则
   - 添加角色LoRA路径配置

3. **测试覆盖**：
   - 确保所有场景类型都有测试用例
   - 验证硬规则是否生效

---

## 📅 实施时间表

1. **第1天**：创建核心规则引擎 ✅
2. **第2天**：重构Execution Planner
3. **第3天**：重构Prompt Builder
4. **第4天**：集成角色锚系统
5. **第5天**：更新转换器和测试

---

## 🔗 相关文件

- `gen_video/utils/execution_rules_v2_1.py` - 执行型规则引擎
- `gen_video/utils/character_anchor_v2_1.py` - 角色锚系统
- `gen_video/schemas/scene_v2_1_example.json` - v2.1 JSON示例
- `gen_video/execution_planner_v3.py` - Execution Planner（待重构）
- `gen_video/utils/prompt_engine_v2.py` - Prompt Engine（待重构）

