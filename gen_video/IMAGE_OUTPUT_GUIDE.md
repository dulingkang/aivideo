# 图片输出位置指南

## 📍 图片输出位置

### 1. 批量测试输出

**路径格式**: `outputs/batch_test_YYYYMMDD_HHMMSS/scene_XXX/novel_image.png`

**示例**:
```
outputs/batch_test_20251220_212014/scene_001/novel_image.png
outputs/batch_test_20251220_212014/scene_002/novel_image.png
```

### 2. v2.2测试输出

**路径格式**: `outputs/test_v22_YYYYMMDD_HHMMSS/scene_001/novel_image.png`

**示例**:
```
outputs/test_v22_20251221_091159/scene_001/novel_image.png
outputs/test_v22_full_20251221_091248/scene_001/novel_image.png
```

### 3. 完整生成输出

**路径格式**: `outputs/test_v22_full_YYYYMMDD_HHMMSS/scene_001/novel_image.png`

**包含文件**:
- `test_scene.json` - 测试场景JSON
- `generated_prompt.txt` - 生成的Prompt
- `scene_001/novel_image.png` - 生成的图片

---

## 🔍 查找最新图片

### 方法1: 使用find命令

```bash
# 查找最近1天生成的图片
find outputs -name "novel_image.png" -type f -mtime -1

# 查找最近修改的图片
find outputs -name "novel_image.png" -type f -mtime -1 | xargs ls -lt | head -5
```

### 方法2: 查看最新测试目录

```bash
# 查看最新的v2.2测试目录
ls -td outputs/test_v22* | head -1

# 查看最新的批量测试目录
ls -td outputs/batch_test_* | head -1
```

### 方法3: 查看具体场景

```bash
# 查看特定场景的图片
ls -lt outputs/batch_test_*/scene_*/novel_image.png | head -5
```

---

## 📊 输出目录结构

```
outputs/
├── batch_test_20251220_212014/
│   ├── scene_001/
│   │   └── novel_image.png
│   └── scene_002/
│       └── novel_image.png
├── test_v22_20251221_091159/
│   ├── test_scene.json
│   ├── generated_prompt.txt
│   └── scene_001/
│       └── novel_image.png
└── test_v22_full_20251221_091248/
    ├── test_scene.json
    ├── generated_prompt.txt
    └── scene_001/
        └── novel_image.png
```

---

## 🎯 在代码中获取输出路径

### ExecutionExecutorV21

```python
from utils.execution_executor_v21 import ExecutionExecutorV21

executor = ExecutionExecutorV21(...)
result = executor.execute_scene(scene, output_dir)

if result.success:
    print(f"图片路径: {result.image_path}")
    # 输出: outputs/test_v22_20251221_091159/scene_001/novel_image.png
```

### 路径构建逻辑

```python
# 如果output_dir已经包含scene_XXX，直接使用
if output_dir_path.name.startswith("scene_"):
    output_path = output_dir_path / "novel_image.png"
else:
    # 需要添加scene_XXX
    output_path = output_dir_path / f"scene_{scene_id:03d}" / "novel_image.png"
```

---

## 📝 注意事项

1. **路径格式**: 图片总是保存在 `scene_XXX/novel_image.png`
2. **自动创建**: 目录会自动创建，无需手动创建
3. **时间戳**: 测试目录包含时间戳，便于区分不同测试
4. **场景ID**: 场景ID从JSON中的`scene.id`或`scene_id`字段提取

---

## 🔗 相关文档

- `test_v22_full_generation.py` - 完整生成测试脚本
- `test_v22_image_generation.py` - 图像生成测试脚本
- `utils/execution_executor_v21.py` - 执行器实现

