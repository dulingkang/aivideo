# 今日工作总结 - Execution Planner v2 集成

**日期**: 2025-01-XX  
**目标**: 实现 Scene JSON v2 格式 + Execution Planner（SDXL/Flux 自动切换）

---

## ✅ 已完成的工作

### 1. Scene JSON v2 格式设计

**核心思想**: 从"描述型 JSON"升级为"执行型 JSON"

- **v1 问题**: 字段不可执行、prompt 写死、模型选择混乱
- **v2 优势**: 意图与实现解耦、支持多模型演进、天然适配 LLM/非 LLM

**关键字段结构**:
```json
{
  "scene_id": 0,
  "version": "v2",
  "intent": { "type": "...", "emotion": "...", "tension_level": "..." },
  "visual_constraints": { "environment": "...", "elements": [...] },
  "camera": { "shot": "...", "angle": "...", "movement": "..." },
  "character": { "present": true/false, "visibility": "...", "face_visible": ... },
  "quality_target": { "style": "...", "detail_level": "..." },
  "generation_policy": { "image_model": "...", "video_model": "..." },
  "narration": { "text": "...", "voice_id": "..." }
}
```

**文档位置**: 
- 字段定义: 见对话记录中的"Scene JSON v2 字段 & 枚举表"
- 示例文件: `lingjie/episode/1.v2.json`

---

### 2. v1 → v2 自动转换脚本

**文件**: `lingjie/convert_scene_v1_to_v2.py`

**功能**:
- 自动将 v1 JSON 转换为 v2 格式
- 智能映射字段（mood → emotion, camera → camera.shot, etc.）
- 自动判断角色可见性和 InstantID 策略

**使用方法**:
```bash
cd /vepfs-dev/shawn/vid/fanren
python3 lingjie/convert_scene_v1_to_v2.py \
  --input lingjie/episode/1.json \
  --output lingjie/episode/1.v2.json
```

**转换结果**: 
- ✅ 已转换 `lingjie/episode/1.json` → `1.v2.json` (22 个场景)

---

### 3. Execution Planner v2 实现

**文件**: `gen_video/model_selector.py`

**核心方法**: `select_engine_for_scene_v2(scene) -> dict`

**决策规则**（优先级从高到低）:

1. **Rule 1: 有人物 + 近景/特写** → `instantid` (SDXL + InstantID，锁脸)
   - 条件: `character.present` + `face_visible=True` 或 `visibility in ["high", "mid"]` 或 `camera.shot in ["close_up", "medium"]`

2. **Rule 2: 有人物 + 远景** → `sdxl` (不锁脸)
   - 条件: `character.present` + `camera.shot == "wide"` 或 `visibility == "low"`

3. **Rule 3: 无人物 + 世界观镜头** → `flux1` / `flux2` (根据环境类型)
   - 条件: `!character.present` + `intent.type in ["title_reveal", "introduce_world", ...]`

4. **Fallback**: 默认 → `sdxl`

**返回结构**:
```python
{
    "engine": "instantid" | "sdxl" | "flux1" | "flux2",
    "mode": "instantid" | "normal" | "cinematic",
    "lock_face": bool,
    "task_type": "character" | "scene"
}
```

**测试结果** (基于 `1.v2.json` 的 22 个场景):
- `instantid`: 10 个场景 (45.5%) - 人物特写/中景
- `sdxl`: 6 个场景 (27.3%) - 人物远景
- `flux1`: 6 个场景 (27.3%) - 环境/世界观

---

### 4. 集成到图像生成流水线

**文件**: `gen_video/image_generator.py`

**修改位置**: `generate_from_script()` 方法

**集成逻辑**:
1. **自动检测 v2 格式**: 检查场景是否包含 `version: "v2"` 或 v2 特有字段
2. **初始化 Planner**: 如果是 v2，创建 `ModelSelector` 实例
3. **应用决策**: 对每个场景调用 `select_engine_for_scene_v2()`，获取引擎选择
4. **生成图像**: 调用 `generate_image()` 时传入 Planner 决策的 `model_engine` 和 `task_type`

**关键代码位置**:
- 检测逻辑: `image_generator.py` 第 6849-6869 行
- 应用决策: `image_generator.py` 第 7054-7073 行
- 调用生成: `image_generator.py` 第 7131-7143 行

---

### 5. 测试脚本

**文件**: 
- `gen_video/test_execution_planner_v2.py` - Planner 单独测试
- `gen_video/test_v2_integration.py` - 完整集成测试

**测试结果**: ✅ 全部通过

---

## 📁 关键文件清单

### 新增文件
1. `lingjie/convert_scene_v1_to_v2.py` - v1→v2 转换脚本
2. `lingjie/episode/1.v2.json` - 转换后的 v2 格式示例
3. `gen_video/test_execution_planner_v2.py` - Planner 测试
4. `gen_video/test_v2_integration.py` - 集成测试

### 修改文件
1. `gen_video/model_selector.py` - 新增 `select_engine_for_scene_v2()` 方法
2. `gen_video/image_generator.py` - 集成 Execution Planner v2

---

## 🎯 核心设计理念

### 1. 默认策略: SDXL 为主，Flux 为辅

**原因**:
- 小说推文是"人物驱动"内容
- SDXL + InstantID 在角色一致性上更稳定
- Flux 只在"世界/氛围"场景使用

### 2. 自动决策，无需手动指定

**优势**:
- JSON 中不写死模型名
- 模型升级时只需改 Planner 逻辑
- 支持多模型并行演进

### 3. 向后兼容

- v1 JSON 继续使用原有逻辑
- v2 JSON 自动启用 Planner
- 无需修改现有脚本

---

## 🚀 使用方式

### 方式 1: 使用 v2 JSON 生成图像

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
python main.py --script ../lingjie/episode/1.v2.json --output lingjie_ep1_v2
```

系统会自动:
1. 检测 v2 格式
2. 为每个场景选择最合适的引擎
3. 生成图像（SDXL+InstantID 用于人物，Flux 用于环境）

### 方式 2: 转换现有 v1 JSON

```bash
python3 lingjie/convert_scene_v1_to_v2.py \
  --input lingjie/episode/1.json \
  --output lingjie/episode/1.v2.json
```

---

## 📋 下一步工作（待完成）

### 1. 角色资产化（高优先级）

**目标**: 建立"韩立标准角色资产库"

**需要做**:
- [ ] 创建 `gen_video/prompt/character_assets.yaml` 或类似文件
- [ ] 定义韩立的"三层结构"（ID Core / Visual Canon / Negative Lock）
- [ ] 生成 3-5 张"标准韩立参考图"（专门用于 InstantID）
- [ ] 在 Prompt Engine 中集成角色资产读取逻辑

**参考**: 对话记录中的"《韩立 · 标准角色资产定义（Industrial Grade）》"

---

### 2. Prompt Engine 优化（中优先级）

**目标**: 配合 Execution Planner，优化 prompt 生成

**需要做**:
- [ ] 角色资产段只出现一次（不被 Rewriter 改写）
- [ ] 动作/环境同义词合并（避免 "lying on sand/ground/desert" 堆叠）
- [ ] 根据 Planner 决策调整 prompt 长度和内容

---

### 3. 实际生成测试（中优先级）

**目标**: 验证完整流水线

**需要做**:
- [ ] 使用 `1.v2.json` 实际生成一集图像
- [ ] 检查人物一致性（韩立是否稳定）
- [ ] 对比 v1 和 v2 的生成质量
- [ ] 根据结果微调 Planner 规则

---

### 4. 视频生成集成（低优先级）

**目标**: 在视频生成阶段也使用 v2 JSON

**需要做**:
- [ ] 在 `video_generator.py` 中检测 v2 格式
- [ ] 使用 `generation_policy.video_model` 选择视频模型
- [ ] 根据 `quality_target.motion_intensity` 调整视频参数

---

### 5. 文档完善（低优先级）

**需要做**:
- [ ] 编写 Scene JSON v2 Schema（JSON Schema 格式）
- [ ] 编写 Execution Planner 使用文档
- [ ] 更新 README，说明 v2 格式使用方法

---

## 🔍 关键决策记录

### 1. 为什么"SDXL 是默认，Flux 是特例"？

**原因**:
- 小说推文是"人物驱动"内容，不是"概念美术"
- SDXL + InstantID 在角色一致性上更稳定
- Flux 更适合"世界/氛围"场景，但人物稳定性不如 SDXL

**证据**: 测试结果显示 45.5% 的场景需要 `instantid`（人物特写），27.3% 需要 `sdxl`（人物远景），只有 27.3% 需要 `flux1`（环境）

---

### 2. 为什么"JSON 不写模型名"？

**原因**:
- 模型选择是"执行策略"，不是"内容描述"
- 未来模型升级时，只需改 Planner 逻辑，不需要重写所有 JSON
- 支持多模型并行演进（Flux 3.0、SDXL 2.0 等）

---

### 3. 为什么"角色资产要独立出来"？

**原因**:
- 角色信息（"他是谁"）不应该被场景 Prompt 改写
- 角色资产是"长期稳定、不参与博弈的锚点"
- 避免"角色被反复加入 → 又被反复压缩 → 语义破碎"

---

## 📝 注意事项

### 1. v2 JSON 格式已冻结

**重要**: 不要再往 v2 格式里加新字段，除非是"执行策略"相关的。内容描述应该通过 `intent` / `visual_constraints` 等现有字段表达。

### 2. Planner 规则可以调整

如果实际生成时发现决策不合理，可以修改 `model_selector.py` 中的 `select_engine_for_scene_v2()` 方法，不需要改 JSON。

### 3. 角色资产是下一步重点

目前 Execution Planner 已经能正确选择引擎，但"韩立不像"的问题还需要通过"角色资产化"来解决。

---

## 🎓 技术要点总结

### Execution Planner 决策流程

```
Scene JSON v2
    ↓
检测 v2 格式
    ↓
初始化 ModelSelector
    ↓
对每个场景调用 select_engine_for_scene_v2()
    ↓
根据 character / camera / intent 决策
    ↓
返回 { engine, mode, lock_face, task_type }
    ↓
调用 generate_image(..., model_engine=..., task_type=...)
```

### 关键判断逻辑

```python
# Rule 1: 人物 + 近景/特写 → instantid
if character.present and (face_visible or visibility in ["high", "mid"] or shot in ["close_up", "medium"]):
    return "instantid"  # 锁脸

# Rule 2: 人物 + 远景 → sdxl
if character.present and (shot == "wide" or visibility == "low"):
    return "sdxl"  # 不锁脸

# Rule 3: 无人物 + 世界观 → flux
if not character.present and intent.type in ["title_reveal", "introduce_world", ...]:
    return "flux1"  # 环境场景

# Fallback: 默认 sdxl
return "sdxl"
```

---

## ✅ 验证清单

- [x] v1 → v2 转换脚本正常工作
- [x] Execution Planner 能正确决策
- [x] 集成到 image_generator.py 成功
- [x] 测试脚本全部通过
- [ ] 实际生成测试（待明天）
- [ ] 角色资产化（待明天）
- [ ] Prompt Engine 优化（待明天）

---

## 📞 快速参考

### 测试 Execution Planner

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
python3 test_execution_planner_v2.py
```

### 测试完整集成

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
python3 test_v2_integration.py
```

### 转换 v1 → v2

```bash
cd /vepfs-dev/shawn/vid/fanren
python3 lingjie/convert_scene_v1_to_v2.py \
  --input lingjie/episode/1.json \
  --output lingjie/episode/1.v2.json
```

### 使用 v2 JSON 生成

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
python main.py --script ../lingjie/episode/1.v2.json --output lingjie_ep1_v2
```

---

---

## 🐛 Bug 修复记录

### 2025-01-XX: 修复 camera 字段类型错误

**问题**: 使用 v2 JSON 格式时，`camera` 字段是字典结构，但 `prompt/builder.py` 中代码假设它是字符串，导致 `AttributeError: 'dict' object has no attribute 'lower'`

**错误位置**: 
- `gen_video/prompt/builder.py` 第 146 行
- `gen_video/prompt/builder.py` 第 738 行

**修复方案**:
1. 新增 `_convert_camera_v2_to_string()` 方法，将 v2 格式的 camera 字典转换为字符串描述
2. 在 `build()` 方法中，检测 `camera_desc` 是否为字典，如果是则自动转换
3. 修复所有直接使用 `scene.get("camera")` 的地方，确保类型安全

**修复文件**: 
- `gen_video/prompt/builder.py` - 修复了 `build()` 方法中的 camera 字段处理
- `gen_video/image_generator.py` - 修复了多处直接使用 `scene.get("camera")` 的地方

**修复详情**:
1. 在 `prompt/builder.py` 中：
   - 添加了 `_convert_camera_v2_to_string()` 方法
   - 修复了 `build()` 方法第 131-146 行和第 737-738 行的 camera 字段处理

2. 在 `image_generator.py` 中：
   - 添加了 `_get_camera_string()` 辅助方法
   - 修复了以下位置的 camera 字段使用：
     - 第 2422 行：top-down 场景检测
     - 第 3651-3652 行：眼睛特写检测
     - 第 3709-3711 行：lying 姿势检测
     - 第 3883 行：场景类型检测
     - 第 3985 行：lying 姿势检测
     - 第 4447 行：眼睛/面部特写检测

**状态**: ✅ 已修复

---

### 2025-01-XX: 修复 v2 格式字段读取问题

**问题**: 使用 v2 JSON 格式时，prompt builder 没有正确读取 `visual_constraints.environment` 和 `character.pose` 字段，导致场景 1、3、7、8 的 prompt 生成不正确。

**错误场景**:
- 场景 1 (scene_002.png): top_down + lying + wide shot，环境描述缺失
- 场景 3 (scene_003.png): 三个太阳和四个月亮，环境描述缺失
- 场景 7 (scene_007.png): 转头动作，pose 描述缺失
- 场景 8 (scene_008.png): 太阳数量变化，环境描述缺失

**修复方案**:
1. 修复 `visual_constraints.environment` 字段读取：
   - 在 `build()` 方法中，优先从 `visual_constraints.environment` 读取（v2 格式）
   - 如果没有，则从 `visual.environment` 读取（v1 格式，向后兼容）
   - 同时修复了 `_build_scene_background_prompt_compact()` 方法

2. 修复 `character.pose` 字段读取：
   - 在 `build()` 方法中，优先从 `character.pose` 读取（v2 格式）
   - 将 v2 格式的 pose 值（如 "lying_motionless", "turning_head"）转换为可读描述
   - 如果没有，则从 `visual.character_pose` 读取（v1 格式，向后兼容）

**修复文件**: `gen_video/prompt/builder.py`

**修复位置**:
- 第 783-792 行：visual_constraints.environment 支持
- 第 551-555 行：character.pose 支持
- 第 884-890 行：character.pose 在动作描述中的支持
- 第 2166-2170 行：visual_constraints 在背景 prompt 中的支持

**状态**: ✅ 已修复

---

**文档版本**: v1.2  
**最后更新**: 2025-01-XX  
**状态**: ✅ Execution Planner v2 集成完成，camera 字段类型错误已修复，v2 格式字段读取问题已修复

