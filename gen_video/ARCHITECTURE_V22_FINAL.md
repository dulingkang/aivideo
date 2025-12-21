# v2.2-final 架构设计文档（最终版）

## 一、核心设计原则

### 1.1 规则驱动，非LLM决策
- **硬规则系统**：所有关键决策（Shot、Pose、Model）都通过硬规则表确定
- **JSON锁定**：v2.2-final JSON 中所有参数都已锁定，无需动态决策
- **LLM角色降级**：LLM 仅用于"修辞"（文本增强），不用于"决策"

### 1.2 执行型架构
- **Execution Executor V2.1**：纯执行器，不"计划"，只"执行已锁定结构"
- **完全确定性路径**：无LLM参与决策，只做模板填充和参数传递
- **失败重试策略**：同模型低风险重试，不改变决策

### 1.3 Execution Planner V3 的角色定位
- **已退休**：不再参与生成流程的决策
- **仅用于转换**：只负责 v2 → v2.1/v2.2 格式转换（向后兼容）
- **不参与执行**：v2.2-final 格式的执行完全由 Execution Executor V2.1 负责

---

## 二、系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    输入层 (Input Layer)                      │
│                                                              │
│  v2.2-final JSON (所有参数已锁定)                            │
│  - shot: {type: "medium", locked: true}                     │
│  - pose: {type: "sit", locked: true}                        │
│  - model_route: {base_model: "flux", identity_engine: "pulid"}│
│  - character: {lora_config, anchor_patches, ...}           │
│  - prompt: {final: "hanli, calm and restrained..."}         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              验证层 (Validation Layer)                        │
│                                                              │
│  ExecutionValidator                                          │
│  - 校验JSON结构完整性                                         │
│  - 校验参数合法性（shot/pose兼容性等）                        │
│  - 返回ValidationResult                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           执行层 (Execution Layer) - 核心                    │
│                                                              │
│  ExecutionExecutorV21                                        │
│  ├─ _normalize_scene_format()      # JSON格式规范化         │
│  ├─ _build_prompt()                 # 模板填充（无LLM）      │
│  ├─ _build_negative_prompt()        # 负面提示词构建         │
│  ├─ _execute_image_generation()    # 图像生成执行            │
│  │   ├─ 检测 Flux + PuLID → 触发增强模式                     │
│  │   ├─ 传递 scene 参数 → EnhancedImageGenerator            │
│  │   └─ 传递 character_lora → 强制启用LoRA                  │
│  └─ _execute_video_generation()    # 视频生成执行            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│        生成层 (Generation Layer) - 两种模式                   │
│                                                              │
│  ┌────────────────────────┐  ┌──────────────────────────┐  │
│  │  标准模式 (Standard)    │  │  增强模式 (Enhanced)      │  │
│  │                        │  │                          │  │
│  │  ImageGenerator        │  │  EnhancedImageGenerator  │  │
│  │  - Flux/SDXL/InstantID │  │  - PuLIDEngine           │  │
│  │  - LoRA支持            │  │  - DecoupledFusionEngine│  │
│  │  - 直接生成             │  │  - 不使用Planner决策     │  │
│  └────────────────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、关键组件说明

### 3.1 Execution Executor V2.1
**位置**: `utils/execution_executor_v21.py`

**职责**:
1. **JSON验证**：使用 `ExecutionValidator` 校验 v2.2-final JSON
2. **Prompt构建**：纯模板填充，从JSON中提取已锁定的参数
3. **参数传递**：将锁定的参数传递给生成器，不做任何决策
4. **执行协调**：协调图像、视频、音频生成

**关键方法**:
```python
def execute_scene(scene: Dict, output_dir: str) -> ExecutionResult:
    """执行场景生成（主入口）"""
    # 1. 规范化JSON格式
    scene = self._normalize_scene_format(scene)
    
    # 2. 验证JSON
    validation_result = self.validator.validate_scene(scene)
    
    # 3. 构建Prompt（模板填充）
    prompt = self._build_prompt(scene)
    negative_prompt = self._build_negative_prompt(scene)
    
    # 4. 执行图像生成
    image_result = self._execute_image_generation(...)
    
    # 5. 执行视频生成
    video_result = self._execute_video_generation(...)
```

### 3.2 EnhancedImageGenerator（增强模式）
**位置**: `enhanced_image_generator.py`

**职责**:
1. **PuLID引擎**：使用 `PuLIDEngine` 进行 Flux + PuLID 生成
2. **解耦融合**：使用 `DecoupledFusionEngine` 进行环境融合
3. **不使用Planner决策**：直接从JSON中读取锁定参数

**关键问题**:
- ❌ **当前问题**：`EnhancedImageGenerator` 中仍在使用 `ExecutionPlannerV3`
- ✅ **应该改为**：直接从JSON中读取锁定参数，不使用Planner的决策

**改造方向**:
```python
# 当前（错误）：
strategy = self.planner.plan_generation(scene)  # ❌ 使用Planner决策

# 应该改为（正确）：
# 直接从JSON中读取锁定参数
base_model = scene.get("model_route", {}).get("base_model", "flux")
identity_engine = scene.get("model_route", {}).get("identity_engine", "pulid")
shot_type = scene.get("shot", {}).get("type", "medium")
pose_type = scene.get("pose", {}).get("type", "stand")
# ✅ 不使用Planner，直接使用JSON中的锁定参数
```

### 3.3 Execution Planner V3（已退休）
**位置**: `utils/execution_planner_v3.py`

**当前状态**:
- ❌ **不应参与生成流程**：v2.2-final 格式的执行不需要Planner决策
- ✅ **仅用于转换**：只负责 v2 → v2.1/v2.2 格式转换（向后兼容）
- ⚠️ **遗留问题**：`EnhancedImageGenerator` 中仍在使用，需要移除

**保留场景**:
- 向后兼容：将旧的 v2 格式转换为 v2.2-final 格式
- 格式升级：帮助用户从旧格式迁移到新格式

**移除场景**:
- ❌ 生成流程中的决策（应该使用JSON中的锁定参数）
- ❌ 动态路由选择（应该使用JSON中的 `model_route`）
- ❌ Shot/Pose自动选择（应该使用JSON中的锁定值）

---

## 四、v2.2-final JSON 格式

### 4.1 核心字段（已锁定）
```json
{
  "version": "v2.2-final",
  "scene": {
    "shot": {
      "type": "medium",
      "locked": true,
      "source": "direct_specification"
    },
    "pose": {
      "type": "sit",
      "locked": true,
      "validated_by": "shot_pose_rules"
    },
    "model_route": {
      "base_model": "flux",
      "identity_engine": "pulid",
      "locked": true,
      "decision_reason": "main_character -> flux + pulid"
    },
    "character": {
      "lora_config": {
        "type": "single",
        "lora_path": "",
        "weight": 0.9,
        "trigger": "hanli"
      },
      "anchor_patches": {
        "temperament_anchor": "calm and restrained temperament...",
        "explicit_lock_words": "wearing his iconic green daoist robe..."
      }
    },
    "prompt": {
      "final": "hanli, calm and restrained temperament, ..."
    }
  }
}
```

### 4.2 执行流程
1. **读取锁定参数**：直接从JSON中读取所有参数
2. **验证兼容性**：检查 shot/pose 兼容性（硬规则）
3. **构建Prompt**：使用 `prompt.final` 或模板填充
4. **执行生成**：根据 `model_route` 选择生成模式
   - `flux + pulid` → 增强模式（EnhancedImageGenerator）
   - `flux` → 标准模式（ImageGenerator）
   - `sdxl + instantid` → 标准模式（ImageGenerator）

---

## 五、当前问题与改造计划

### 5.1 当前问题

#### 问题1：EnhancedImageGenerator 仍在使用 Planner
**位置**: `enhanced_image_generator.py`

**问题代码**:
```python
# 第94行
self.planner = ExecutionPlannerV3(self.config)  # ❌ 不应该初始化

# generate_scene 方法中
strategy = self.planner.plan_generation(scene)  # ❌ 不应该使用Planner决策
```

**影响**:
- Planner 可能会覆盖JSON中锁定的参数
- 违反了"规则驱动，非LLM决策"的原则
- 增加了不确定性和不稳定性

#### 问题2：Planner 的角色不清晰
**当前状态**:
- `ExecutionExecutorV21` 中已明确不使用Planner（✅ 正确）
- `EnhancedImageGenerator` 中仍在使用Planner（❌ 错误）

### 5.2 改造计划

#### 改造1：移除 EnhancedImageGenerator 中的 Planner
**步骤**:
1. 移除 `self.planner = ExecutionPlannerV3(self.config)` 初始化
2. 修改 `generate_scene` 方法，直接从JSON读取参数
3. 移除所有 `self.planner.plan_generation()` 调用

**改造后代码**:
```python
class EnhancedImageGenerator:
    def __init__(self, config_path: str):
        # ❌ 移除
        # self.planner = ExecutionPlannerV3(self.config)
        
        # ✅ 直接使用配置
        self.pulid_config = self.image_config.get("pulid", {})
        self.decoupled_config = self.image_config.get("decoupled_fusion", {})
    
    def generate_scene(self, scene: Dict, face_reference, original_prompt: str):
        # ❌ 移除
        # strategy = self.planner.plan_generation(scene)
        
        # ✅ 直接从JSON读取锁定参数
        base_model = scene.get("model_route", {}).get("base_model", "flux")
        identity_engine = scene.get("model_route", {}).get("identity_engine", "pulid")
        shot_type = scene.get("shot", {}).get("type", "medium")
        pose_type = scene.get("pose", {}).get("type", "stand")
        
        # 使用锁定参数进行生成
        ...
```

#### 改造2：明确 Planner 的保留用途
**保留场景**:
- 格式转换工具：`v2_to_v21_converter.py` 可以使用Planner
- 向后兼容：帮助用户从旧格式迁移

**移除场景**:
- 生成流程中的任何决策
- 动态路由选择
- Shot/Pose自动选择

---

## 六、执行流程对比

### 6.1 当前流程（错误）
```
v2.2-final JSON
    ↓
ExecutionExecutorV21
    ↓
EnhancedImageGenerator
    ↓
ExecutionPlannerV3.plan_generation()  ❌ 不应该使用
    ↓
PuLIDEngine.generate()
```

### 6.2 正确流程（改造后）
```
v2.2-final JSON
    ↓
ExecutionExecutorV21
    ├─ 读取锁定参数（shot, pose, model_route）
    ├─ 构建Prompt（模板填充）
    └─ 执行生成
        ↓
EnhancedImageGenerator
    ├─ 直接从JSON读取锁定参数  ✅ 不使用Planner
    ├─ 使用PuLIDEngine生成
    └─ 使用DecoupledFusionEngine融合
```

---

## 七、总结

### 7.1 核心原则
1. **规则驱动**：所有决策都通过硬规则表确定
2. **JSON锁定**：v2.2-final 中所有参数都已锁定
3. **纯执行**：Execution Executor V2.1 只执行，不决策
4. **Planner退休**：不再参与生成流程，仅用于格式转换

### 7.2 待改造项
1. ⚠️ **移除 EnhancedImageGenerator 中的 Planner**
2. ⚠️ **修改 generate_scene 方法，直接从JSON读取参数**
3. ✅ **ExecutionExecutorV21 已正确实现（不使用Planner）**

### 7.3 架构优势
1. **稳定性**：完全确定性路径，无LLM决策
2. **可追溯性**：所有决策都在JSON中明确记录
3. **可维护性**：规则驱动，易于调试和优化
4. **可扩展性**：新增规则只需更新规则表

---

## 八、具体改造点

### 8.1 EnhancedImageGenerator 中的 Planner 使用

#### 位置1：初始化（第94行）
```python
# ❌ 当前代码
self.planner = ExecutionPlannerV3(self.config)

# ✅ 应该移除
# 不再初始化 Planner
```

#### 位置2：generate_scene 方法（第815行）
```python
# ❌ 当前代码
strategy = self.planner.analyze_scene(scene)

# ✅ 应该改为
# 直接从JSON读取锁定参数
base_model = scene.get("model_route", {}).get("base_model", "flux")
identity_engine = scene.get("model_route", {}).get("identity_engine", "pulid")
shot_type = scene.get("shot", {}).get("type", "medium")
pose_type = scene.get("pose", {}).get("type", "stand")
```

#### 位置3：build_weighted_prompt（第856行）
```python
# ❌ 当前代码
prompt = self.planner.build_weighted_prompt(scene, strategy, original_prompt=original_prompt)

# ✅ 应该改为
# 直接使用JSON中的prompt.final，或使用original_prompt
prompt = scene.get("prompt", {}).get("final", original_prompt)
```

#### 位置4：清理方法（第1783行）
```python
# ❌ 当前代码
if self.planner is not None:
    if hasattr(self.planner, 'llm_client'):
        self.planner.llm_client = None

# ✅ 应该移除
# 不再需要清理 Planner
```

### 8.2 改造后的 generate_scene 方法结构

```python
def generate_scene(self, scene: Dict, face_reference, original_prompt: str):
    """
    生成场景图像（改造后：不使用Planner，直接从JSON读取参数）
    """
    # 1. 从JSON读取锁定参数（不使用Planner）
    model_route = scene.get("model_route", {})
    base_model = model_route.get("base_model", "flux")
    identity_engine = model_route.get("identity_engine", "pulid")
    
    shot_info = scene.get("shot", {})
    shot_type = shot_info.get("type", "medium")
    
    pose_info = scene.get("pose", {})
    pose_type = pose_info.get("type", "stand")
    
    # 2. 使用JSON中的prompt（不使用Planner构建）
    prompt_config = scene.get("prompt", {})
    prompt = prompt_config.get("final", original_prompt)
    
    # 3. 使用PuLID引擎生成（参数已锁定，无需决策）
    if base_model == "flux" and identity_engine == "pulid":
        result = self.pulid_engine.generate(
            prompt=prompt,
            face_reference=face_reference,
            # 使用JSON中的锁定参数
            shot_type=shot_type,
            pose_type=pose_type,
            ...
        )
    
    # 4. 使用解耦融合引擎（如果需要）
    if self.decoupled_config.get("enabled", False):
        result = self.fusion_engine.fuse(...)
    
    return result
```

---

## 九、下一步行动

1. **立即改造**：移除 `EnhancedImageGenerator` 中的 Planner 使用
   - 移除初始化（第94行）
   - 修改 `generate_scene` 方法（第815、856行）
   - 移除清理代码（第1783行）

2. **测试验证**：确保改造后功能正常
   - 测试 v2.2-final JSON 生成
   - 验证参数锁定是否生效
   - 检查生成质量是否受影响

3. **文档更新**：更新相关文档
   - 明确 Planner 的角色定位（仅用于转换）
   - 更新架构图
   - 更新使用指南

4. **代码清理**：移除不必要的 Planner 依赖
   - 检查其他文件是否仍在使用 Planner
   - 移除未使用的导入

