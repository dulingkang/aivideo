# InstantID 集成指南

## 📋 概述

InstantID 已集成到系统中，可以与 Flux 模型结合使用，实现固定人脸特征的图像生成。

## 🎯 功能特性

- ✅ **自动人脸特征提取**：使用 InsightFace 提取人脸嵌入向量
- ✅ **固定人脸生成**：生成时保持参考人脸的特征
- ✅ **与 LoRA 兼容**：可以同时使用 InstantID 和 LoRA
- ✅ **自动模式切换**：检测到人脸图像时自动使用 InstantID

## 📦 依赖要求

```bash
pip install insightface onnxruntime onnxruntime-gpu
```

## 🚀 使用方式

### 1. 通过 API 使用

#### 方式一：上传人脸参考图像

```bash
curl -X POST "http://localhost:8000/api/v1/images/generate" \
  -H "X-API-Key: test-key-123" \
  -F "prompt=科普主持人，专业形象，微笑" \
  -F "use_model_manager=true" \
  -F "task=host_face" \
  -F "reference_image=@face_reference.jpg" \
  -F "reference_image_type=face" \
  -F "width=1024" \
  -F "height=1024"
```

#### 方式二：使用前端界面

1. 在图像生成页面，上传参考图像
2. 选择"面部参考"（而不是"场景参考"）
3. 输入提示词，点击生成
4. 系统会自动检测并使用 InstantID

### 2. 通过 ModelManager 使用

```python
from model_manager import ModelManager
from PIL import Image

# 初始化 ModelManager
manager = ModelManager()

# 加载人脸参考图像
face_image = Image.open("face_reference.jpg")

# 生成图像（自动使用 InstantID）
image = manager.generate(
    task="host_face",  # 或 "host_face_instantid" 明确指定
    prompt="科普主持人，专业形象，微笑",
    face_image=face_image,  # 提供人脸图像
    face_strength=0.8,  # 人脸强度（0.0-1.0）
    width=1024,
    height=1024
)

image.save("output.png")
```

## ⚙️ 配置参数

### face_strength（人脸强度）

- **范围**：0.0 - 1.0
- **默认值**：0.8
- **说明**：
  - `0.0`：不使用 InstantID，仅使用提示词
  - `0.5`：中等强度，平衡人脸特征和提示词
  - `0.8`：高强度，优先保持人脸特征（推荐）
  - `1.0`：最高强度，完全固定人脸特征

### 任务类型

- `host_face`：默认模式，如果提供 `face_image` 会自动切换到 InstantID
- `host_face_instantid`：明确指定使用 InstantID
- `character_face_instantid`：角色人脸 + InstantID
- `realistic_face_instantid`：真实感人脸 + InstantID

## 🔧 模型路径配置

InstantID 相关模型路径在 `model_manager.py` 中配置：

```python
self.instantid_paths = {
    "instantid": "models/instantid",
    "controlnet": "models/instantid/ControlNet",
    "ip_adapter": "models/instantid/ip-adapter",
}
```

确保这些路径存在且包含相应的模型文件。

## 📝 工作流程

1. **上传人脸图像** → API 接收并保存
2. **检测任务类型** → 如果提供人脸图像，自动切换到 InstantID 模式
3. **提取人脸特征** → 使用 InsightFace 提取人脸嵌入向量
4. **加载模型** → Flux + InstantID Pipeline
5. **生成图像** → 结合提示词和人脸特征生成

## ⚠️ 注意事项

1. **模型文件**：确保 InstantID 模型文件已下载到 `models/instantid/` 目录
2. **依赖库**：确保 `insightface` 和 `onnxruntime` 已安装
3. **显存占用**：InstantID 会增加显存占用，建议至少 8GB 显存
4. **人脸质量**：参考人脸图像质量越高，生成效果越好
5. **完整实现**：当前是基础框架，InstantID 的完整实现可能需要根据实际库调整

## 🔍 调试

如果 InstantID 不工作，检查：

1. **依赖是否安装**：
   ```python
   import insightface
   print("✅ insightface 已安装")
   ```

2. **模型文件是否存在**：
   ```bash
   ls -la models/instantid/
   ```

3. **查看日志**：检查控制台输出，查看是否有错误信息

4. **测试人脸提取**：
   ```python
   from pipelines.flux_instantid_pipeline import FluxInstantIDPipeline
   pipeline = FluxInstantIDPipeline(...)
   pipeline.load()
   features = pipeline._extract_face_features(face_image)
   print(features)
   ```

## 📚 参考资源

- [InstantID 官方文档](https://huggingface.co/InstantX/InstantID)
- [InsightFace 文档](https://github.com/deepinsight/insightface)


