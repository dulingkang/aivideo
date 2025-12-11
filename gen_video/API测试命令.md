# API 测试命令

## ✅ 正确的 API 端点

**端点路径：** `POST /api/v1/images/generate`

---

## 🚀 使用 ModelManager + LoRA 生成

### 方法 1：使用 curl（需要 API Key）

```bash
curl -X POST "http://localhost:8000/api/v1/images/generate" \
  -H "X-API-Key: test-key-123" \
  -F "prompt=科普主持人，专业形象，微笑，正式着装" \
  -F "use_model_manager=true" \
  -F "task=host_face" \
  -F "width=1024" \
  -F "height=1024"
```

### 方法 2：使用 curl（带更多参数）

```bash
curl -X POST "http://localhost:8000/api/v1/images/generate" \
  -H "X-API-Key: test-key-123" \
  -F "prompt=科普主持人，专业形象，微笑，正式着装，演播室背景" \
  -F "use_model_manager=true" \
  -F "task=host_face" \
  -F "width=1024" \
  -F "height=1024" \
  -F "num_inference_steps=20" \
  -F "guidance_scale=3.5" \
  -F "seed=42"
```

---

## 🔑 API Key

默认的测试 API Key：
- `test-key-123` - 免费用户（10 张/天）
- `demo-key-456` - 付费用户（100 张/天）

---

## 📋 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `prompt` | string | 生成提示词（必需） | - |
| `use_model_manager` | bool | 是否使用 ModelManager | false |
| `task` | string | 任务类型（ModelManager模式） | - |
| `width` | int | 图像宽度（8的倍数） | 1024 |
| `height` | int | 图像高度（8的倍数） | 1024 |
| `num_inference_steps` | int | 推理步数 | 40 |
| `guidance_scale` | float | 引导强度 | 7.5 |
| `seed` | int | 随机种子（可选） | None |

### task 参数可选值

- `host_face` - 科普主持人脸（**会自动加载 LoRA**）
- `character_face` - 角色人脸（**会自动加载 LoRA**）
- `science_background` - 科学背景图
- `official_style` - 官方风格
- `fast_background` - 快速背景

---

## 🐍 Python 测试脚本

```python
import requests

url = "http://localhost:8000/api/v1/images/generate"
headers = {"X-API-Key": "test-key-123"}

data = {
    "prompt": "科普主持人，专业形象，微笑，正式着装",
    "use_model_manager": "true",
    "task": "host_face",
    "width": 1024,
    "height": 1024,
}

response = requests.post(url, headers=headers, data=data)
print(response.json())
```

---

## ✅ 验证 API 是否运行

```bash
# 检查健康状态
curl http://localhost:8000/api/v1/health

# 检查根路径
curl http://localhost:8000/
```

---

## 🔍 常见错误

### 404 Not Found

**原因：** 端点路径错误

**解决：** 使用正确的路径 `/api/v1/images/generate`

### 401 Unauthorized

**原因：** 缺少或错误的 API Key

**解决：** 添加 `-H "X-API-Key: test-key-123"`

### 429 Too Many Requests

**原因：** 配额已用完

**解决：** 等待第二天重置，或使用付费 API Key

---

## 📝 响应格式

成功响应：
```json
{
  "task_id": "uuid",
  "status": "completed",
  "image_url": "/api/v1/files/images/{filename}",
  "metadata": {
    "model_used": "flux1",
    "task": "host_face"
  }
}
```

错误响应：
```json
{
  "detail": "错误信息"
}
```

