# InstantID 使用指南

## 📁 人脸参考图片目录

### 目录位置

```
gen_video/models/face_references/
```

### 创建目录（如果不存在）

```bash
mkdir -p gen_video/models/face_references
```

## 📝 命名规范

### 标准命名（推荐）

| 文件名 | 用途 | 说明 |
|--------|------|------|
| `host_face.png` | 主持人正脸 | 科普主持人固定人设（推荐） |
| `character_face.png` | 角色人脸 | 通用角色人脸 |
| `realistic_face.png` | 真实感人脸 | 真实感人脸生成 |

### 自定义命名

格式：`{角色/用途}_{描述}.{扩展名}`

示例：
- `host_face_front.png` - 主持人正面照
- `host_face_side.png` - 主持人侧面照
- `character_face_male.png` - 男性角色人脸
- `character_face_female.png` - 女性角色人脸

### 命名规则

1. ✅ **使用下划线分隔**：`host_face.png`
2. ✅ **小写字母**：`host_face.png` 而不是 `Host_Face.png`
3. ✅ **描述性命名**：清楚说明图片用途
4. ✅ **支持格式**：`.png`, `.jpg`, `.jpeg`, `.webp`

## 🎯 使用方式

### 方式 1: 通过文件名指定（推荐）

#### API 调用

```bash
curl -X POST "http://localhost:8000/api/v1/images/generate" \
  -H "X-API-Key: test-key-123" \
  -F "prompt=科普主持人，专业形象，微笑" \
  -F "use_model_manager=true" \
  -F "task=host_face" \
  -F "face_image_name=host_face.png" \
  -F "width=1024" \
  -F "height=1024"
```

#### Python 代码

```python
from model_manager import ModelManager

manager = ModelManager()
image = manager.generate(
    task="host_face",
    prompt="科普主持人，专业形象，微笑",
    face_image_name="host_face.png",  # 从 face_references 目录加载
    width=1024,
    height=1024
)
```

### 方式 2: 自动查找（根据任务类型）

如果未指定 `face_image_name`，系统会根据任务类型自动查找：

- `task=host_face` → 自动查找 `host_face.png`
- `task=character_face` → 自动查找 `character_face.png`
- `task=realistic_face` → 自动查找 `realistic_face.png`

```bash
curl -X POST "http://localhost:8000/api/v1/images/generate" \
  -H "X-API-Key: test-key-123" \
  -F "prompt=科普主持人，专业形象" \
  -F "use_model_manager=true" \
  -F "task=host_face" \
  -F "width=1024" \
  -F "height=1024"
# 系统会自动查找 host_face.png
```

### 方式 3: 直接上传文件

```bash
curl -X POST "http://localhost:8000/api/v1/images/generate" \
  -H "X-API-Key: test-key-123" \
  -F "prompt=科普主持人，专业形象" \
  -F "use_model_manager=true" \
  -F "task=host_face" \
  -F "reference_image=@/path/to/face.jpg" \
  -F "reference_image_type=face" \
  -F "width=1024" \
  -F "height=1024"
```

**优先级**：上传的文件 > 指定的文件名 > 自动查找

## 📋 图片要求

### 推荐规格

- **分辨率**: 512x512 或更高（建议 1024x1024）
- **格式**: PNG 或 JPG
- **内容**: 清晰的正脸照片，面部完整可见
- **背景**: 简单背景效果更好

### 质量要求

- ✅ 人脸清晰，五官完整
- ✅ 正面或接近正面角度（推荐）
- ✅ 光线充足，无阴影遮挡
- ✅ 无眼镜、口罩等遮挡物（除非需要）
- ❌ 避免侧脸、模糊、低分辨率图片

## 🔍 查找顺序

系统按以下顺序查找人脸图片：

1. **上传的文件**（`reference_image` + `reference_image_type=face`）
2. **指定的文件名**（`face_image_name` 参数）
3. **自动查找**（根据 `task` 类型）
4. **默认图片**（如果配置了）

## 💡 使用建议

### 1. 主持人固定人设

```bash
# 1. 准备图片
cp /path/to/host_photo.png gen_video/models/face_references/host_face.png

# 2. 使用
curl -X POST "http://localhost:8000/api/v1/images/generate" \
  -H "X-API-Key: test-key-123" \
  -F "prompt=科普主持人，专业形象" \
  -F "use_model_manager=true" \
  -F "task=host_face" \
  -F "face_image_name=host_face.png"
```

### 2. 多角度支持

可以准备多个角度的图片：

```bash
gen_video/models/face_references/
  ├── host_face_front.png  # 正面
  ├── host_face_side.png   # 侧面
  └── host_face_45.png     # 45度角
```

使用时指定具体文件名：

```bash
-F "face_image_name=host_face_front.png"
```

### 3. 角色区分

不同角色使用不同的文件名：

```bash
gen_video/models/face_references/
  ├── host_face.png        # 主持人
  ├── character_face_1.png # 角色1
  └── character_face_2.png # 角色2
```

## 🔧 配置说明

### ModelManager 配置

人脸参考图片目录在 `ModelManager` 初始化时自动创建：

```python
self.face_references_dir = models_root / "face_references"
```

### 自动查找映射

任务类型到文件名的映射：

```python
task_to_filename = {
    "host_face": "host_face.png",
    "character_face": "character_face.png",
    "realistic_face": "realistic_face.png",
}
```

## ⚠️ 注意事项

1. **文件必须存在**: 如果指定的文件不存在，系统会尝试自动查找或跳过 InstantID
2. **图片质量**: 图片质量直接影响生成效果，建议使用高质量正脸照片
3. **文件格式**: 支持 PNG、JPG、JPEG、WEBP
4. **路径**: 使用相对文件名即可，系统会自动在 `face_references` 目录查找

## 📚 相关文档

- [InstantID集成指南.md](./InstantID集成指南.md)
- [完整视频生成流水线架构.md](./完整视频生成流水线架构.md)

