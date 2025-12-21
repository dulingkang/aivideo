# v2.2-final JSON 设计原则

## 核心设计理念

### 1. **配置分层原则**

```
config.yaml (全局默认值)
    ↓
JSON文件 (场景特定覆盖，可选)
    ↓
实际使用的参数
```

**原则**：
- 如果所有场景使用相同参数 → 只需在 `config.yaml` 中配置
- 如果某个场景需要特殊参数 → 在 JSON 中覆盖

### 2. **JSON 字段的可选性**

#### 必需字段（必须存在）
- `version`: "v2.2-final"
- `scene.id`: 场景ID
- `scene.shot`: 镜头类型
- `scene.pose`: 姿态类型
- `scene.character`: 角色信息
- `scene.prompt.final`: 最终提示词

#### 可选字段（如果不存在，从config.yaml读取）
- `scene.generation_params`: 生成参数
  - `width`: 宽度（默认：1536）
  - `height`: 高度（默认：1536）
  - `num_inference_steps`: 推理步数（默认：50）
  - `guidance_scale`: 引导强度（默认：7.5）
  - `seed`: 随机种子（默认：-1）

#### 示例

**场景1：使用默认参数（推荐）**
```json
{
  "version": "v2.2-final",
  "scene": {
    "id": "scene_001",
    "shot": {"type": "wide"},
    "pose": {"type": "stand"},
    "character": {...},
    "prompt": {"final": "..."}
    // 没有 generation_params，使用 config.yaml 中的默认值
  }
}
```

**场景2：需要特殊参数**
```json
{
  "version": "v2.2-final",
  "scene": {
    "id": "scene_002",
    "shot": {"type": "close_up"},
    "pose": {"type": "stand"},
    "character": {...},
    "prompt": {"final": "..."},
    "generation_params": {
      "num_inference_steps": 60,  // 特写需要更多步数
      "width": 1536,
      "height": 1536
      // 其他参数使用 config.yaml 默认值
    }
  }
}
```

## 代码实现

### 1. `enhanced_image_generator.py`

```python
# 优先使用JSON中的参数（如果存在），否则使用config.yaml中的默认值
gen_params = scene.get("generation_params", {})
width = gen_params.get("width") or self.pulid_config.get("width", 1536)
height = gen_params.get("height") or self.pulid_config.get("height", 1536)
num_steps = gen_params.get("num_inference_steps") or self.pulid_config.get("num_inference_steps", 50)
guidance = gen_params.get("guidance_scale") or self.pulid_config.get("guidance_scale", 7.5)
```

### 2. `execution_executor_v21.py`

```python
# 从config.yaml读取默认值
default_num_steps = 50  # 从 config.yaml 读取
num_steps = gen_params.get("num_inference_steps") or default_num_steps
```

### 3. `v2_to_v22_converter.py`

```python
# generation_params 现在是可选的，不再强制写入
# 如果所有场景使用相同参数，可以从JSON中删除此字段
```

## 优势

1. **减少冗余**：不需要在每个JSON文件中重复相同的配置
2. **统一管理**：所有默认值在 `config.yaml` 中统一管理
3. **易于维护**：修改默认值只需修改 `config.yaml`
4. **灵活性**：特殊场景仍可在JSON中覆盖

## 迁移指南

### 从旧格式迁移

**旧格式（每个JSON都有generation_params）**：
```json
{
  "scene": {
    "generation_params": {
      "width": 1536,
      "height": 1536,
      "num_inference_steps": 50,
      "guidance_scale": 7.5
    }
  }
}
```

**新格式（可选，如果使用默认值可以删除）**：
```json
{
  "scene": {
    // 如果使用默认值，可以删除 generation_params
    // 如果需要特殊参数，只写需要覆盖的字段
    "generation_params": {
      "num_inference_steps": 60  // 只覆盖需要的字段
    }
  }
}
```

## 验证工具

使用 `json_validator_v22.py` 验证JSON格式：

```bash
# 验证但不修复
python3 utils/json_validator_v22.py ../lingjie/v22

# 验证并自动修复（移除冗余的generation_params）
python3 utils/json_validator_v22.py ../lingjie/v22 --fix
```

## 最佳实践

1. **默认情况**：不在JSON中写 `generation_params`，使用 `config.yaml` 默认值
2. **特殊场景**：只在需要覆盖的字段中写 `generation_params`
3. **批量生成**：确保 `config.yaml` 中的默认值适合大多数场景
4. **验证**：使用验证工具确保JSON格式正确

