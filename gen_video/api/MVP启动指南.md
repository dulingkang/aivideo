# MVP 启动指南

## 🎯 目标

快速验证市场需求，无需复杂的Redis/Celery配置，直接同步调用生成器。

## 📋 功能

### ✅ 已实现

1. **图像生成API**
   - 同步模式（无需Redis）
   - 基本配额管理（内存）
   - API Key认证

2. **配额管理**
   - 免费版：10张图/天，2个视频/天
   - 付费版：100张图/天，20个视频/天
   - 每天自动重置

3. **文件服务**
   - 生成的图像可通过URL访问
   - 自动保存到 `outputs/api/` 目录

### ⚠️ 待实现

1. **视频生成API**（需要根据实际接口实现）
2. **数据库持久化**（当前使用内存）
3. **支付系统**（当前手动管理API Key）

---

## 🚀 快速开始

### 1. 启动API服务

```bash
cd /vepfs-dev/shawn/vid/fanren
source /vepfs-dev/shawn/venv/py312/bin/activate  # 或你的conda环境

# 启动MVP API
python gen_video/api/mvp_main.py
```

服务将在 `http://localhost:8000` 启动。

### 2. 查看API文档

打开浏览器访问：
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. 测试API

```bash
# 运行测试脚本
python gen_video/api/test_mvp.py
```

或使用curl：

```bash
# 健康检查
curl http://localhost:8000/api/v1/health

# 查询配额
curl -H "X-API-Key: test-key-123" http://localhost:8000/api/v1/quota

# 生成图像
curl -X POST http://localhost:8000/api/v1/images/generate \
  -H "X-API-Key: test-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "一个美丽的风景，山峦起伏，云雾缭绕",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 40
  }'
```

---

## 🔑 API Key管理

### 默认测试Key

| API Key | 用户ID | 套餐 | 配额 |
|---------|--------|------|------|
| `test-key-123` | test_user | 免费版 | 10图/天, 2视频/天 |
| `demo-key-456` | demo_user | 付费版 | 100图/天, 20视频/天 |

### 添加新Key

编辑 `gen_video/api/mvp_main.py`，在 `API_KEYS` 字典中添加：

```python
API_KEYS = {
    "test-key-123": {"user_id": "test_user", "tier": "free"},
    "demo-key-456": {"user_id": "demo_user", "tier": "paid"},
    "your-new-key": {"user_id": "new_user", "tier": "paid"},  # 添加新Key
}
```

---

## 📊 配额管理

### 配额限制

| 套餐 | 图像/天 | 视频/天 |
|------|---------|---------|
| 免费版 | 10 | 2 |
| 付费版 | 100 | 20 |

### 配额重置

- 每天自动重置（基于日期）
- 可以通过 `/api/v1/quota` 查询剩余配额

### 配额存储

- **当前**：内存存储（重启后丢失）
- **未来**：迁移到数据库

---

## 📁 文件结构

```
gen_video/api/
├── mvp_main.py          # MVP主文件
├── test_mvp.py          # 测试脚本
├── MVP启动指南.md       # 本文档
└── ...

outputs/api/
├── images/              # 生成的图像
│   └── {task_id}.png
└── videos/              # 生成的视频（待实现）
    └── {task_id}.mp4
```

---

## 🔧 配置说明

### 环境要求

1. **Python环境**：已安装所有依赖
2. **模型文件**：已下载到 `gen_video/models/`
3. **配置文件**：`gen_video/config.yaml` 已配置

### 端口配置

默认端口：`8000`

修改端口（在 `mvp_main.py` 最后）：
```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # 修改端口号
```

---

## 🧪 测试示例

### Python示例

```python
import requests

BASE_URL = "http://localhost:8000"
API_KEY = "test-key-123"

# 生成图像
response = requests.post(
    f"{BASE_URL}/api/v1/images/generate",
    headers={"X-API-Key": API_KEY},
    json={
        "prompt": "一个美丽的风景",
        "width": 1024,
        "height": 1024,
    }
)

result = response.json()
print(f"任务ID: {result['task_id']}")
print(f"图像URL: {result['image_url']}")
```

### JavaScript示例

```javascript
const BASE_URL = "http://localhost:8000";
const API_KEY = "test-key-123";

// 生成图像
fetch(`${BASE_URL}/api/v1/images/generate`, {
  method: "POST",
  headers: {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    prompt: "一个美丽的风景",
    width: 1024,
    height: 1024,
  }),
})
.then(res => res.json())
.then(data => {
  console.log("任务ID:", data.task_id);
  console.log("图像URL:", data.image_url);
});
```

---

## ⚠️ 注意事项

### 1. 同步模式

- **当前**：同步执行，请求会阻塞直到完成
- **影响**：如果生成时间较长（30-60秒），客户端需要等待
- **未来**：迁移到异步模式（需要Redis/Celery）

### 2. 配额管理

- **当前**：内存存储，重启后丢失
- **未来**：迁移到数据库持久化

### 3. 错误处理

- 生成失败会返回500错误
- 配额用完会返回429错误
- 建议客户端实现重试机制

### 4. 性能

- 单线程处理（FastAPI默认）
- 并发请求会排队处理
- 建议使用 `uvicorn` 多进程模式：

```bash
uvicorn gen_video.api.mvp_main:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## 🚀 下一步

### 短期（1周内）

1. ✅ 完善视频生成API
2. ✅ 添加更多测试用例
3. ✅ 优化错误处理

### 中期（1个月内）

1. 迁移到数据库（配额持久化）
2. 实现异步任务队列（Redis/Celery）
3. 添加用户注册/登录
4. 实现支付系统

### 长期（3个月内）

1. 完善前端界面
2. 添加更多功能（模板、批量处理等）
3. 性能优化和监控
4. 部署到生产环境

---

## 📞 支持

如有问题，请查看：
- API文档：http://localhost:8000/docs
- 测试脚本：`gen_video/api/test_mvp.py`
- 日志输出：查看控制台输出

---

## 🚀 多模型组合方案（推荐）

针对**科普视频流水线 + 政府单**场景，我们推荐使用多模型组合方案：

| 用途 | 推荐模型 | 理由 |
|------|---------|------|
| **科普主持人固定人设** | **Flux 1-dev + InstantID** | 人脸最稳、LoRA 效果最好 |
| **中国式科教场景、太空、实验室** | **Hunyuan-DiT** | 中文理解最强 |
| **追求真实感和稳定性** | **Kolors** | 手部、光影优秀 |
| **极速出大量素材** | **SD3.x Turbo** | 一秒一张，高速流水线 |

**详细实施指南**：请查看 `多模型组合方案-实施指南.md`

**优势：**
- ✅ 固定人设：Flux + InstantID 确保 365 天不换脸
- ✅ 中文场景：Hunyuan-DiT 提供最强的中文理解能力
- ✅ 真实感：Kolors 提供优秀的手部和光影表现
- ✅ 高效率：SD3 Turbo 支持极速批量生成

---

**最后更新**：2024年

