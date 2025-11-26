# AI视频生成平台 - API服务

## 概述

这是一个通用的AI视频生成平台，提供RESTful API接口，支持：
- 图像生成（基于InstantID/SDXL）
- 视频生成（基于SVD）
- 异步任务处理
- 多用户支持
- 配额管理
- 资源管理

## 快速开始

### 1. 安装依赖

```bash
cd gen_video/api
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/videogen

# Redis配置（任务队列）
REDIS_URL=redis://localhost:6379/0

# JWT密钥
SECRET_KEY=your-secret-key-here

# API密钥
API_KEY=your-api-key-here
```

### 3. 启动服务

```bash
# 开发模式
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 4. 启动Celery Worker

```bash
# 启动worker处理任务
celery -A api.tasks worker --loglevel=info --concurrency=4
```

### 5. 访问API文档

打开浏览器访问：http://localhost:8000/docs

## API使用示例

### 生成图像

```bash
curl -X POST "http://localhost:8000/api/v1/images/generate" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "xianxia fantasy, Han Li, calm cultivator, medium shot",
    "width": 1536,
    "height": 864,
    "character_id": "hanli",
    "num_inference_steps": 40
  }'
```

响应：
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "estimated_time": 30,
  "created_at": "2024-01-01T12:00:00"
}
```

### 查询任务状态

```bash
curl -X GET "http://localhost:8000/api/v1/tasks/{task_id}" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

响应：
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "result": {
    "output_path": "https://example.com/outputs/image.png",
    "thumbnail": "https://example.com/outputs/thumb.png"
  },
  "error": null,
  "created_at": "2024-01-01T12:00:00",
  "updated_at": "2024-01-01T12:00:30"
}
```

### 生成视频

```bash
curl -X POST "http://localhost:8000/api/v1/videos/generate" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "scenes": [
      {
        "id": 0,
        "prompt": "xianxia fantasy, immortal realm sky, golden scroll",
        "duration": 5.0
      },
      {
        "id": 1,
        "prompt": "Han Li lying on sand, calm expression",
        "duration": 4.0
      }
    ],
    "video_config": {
      "fps": 24,
      "resolution": "1280x768",
      "upscale": true
    }
  }'
```

## 配置管理

### 系统级配置

`configs/system_config.yaml`:
```yaml
gpu_pool:
  enabled: true
  max_concurrent: 4
  gpu_ids: [0, 1, 2, 3]

model_cache:
  enabled: true
  max_size_gb: 100

storage:
  type: local
  base_path: /data/outputs
```

### 用户级配置

`configs/users/{user_id}.yaml`:
```yaml
quotas:
  images_per_day: 100
  videos_per_day: 10
  max_resolution: "1920x1080"

limits:
  max_duration: 60
  max_frames: 1440
```

### 项目级配置

`configs/projects/{project_id}.yaml`:
```yaml
character_profiles: "character_profiles.yaml"
scene_profiles: "scene_profiles.yaml"
style: "xianxia"
default_engine: "instantid"
```

## 部署

### Docker部署

```bash
# 构建镜像
docker build -t video-gen-api .

# 运行容器
docker run -d \
  --name video-gen-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/outputs:/app/outputs \
  -e DATABASE_URL=postgresql://... \
  -e REDIS_URL=redis://... \
  video-gen-api
```

### Docker Compose部署

```bash
docker-compose up -d
```

## 监控

### 健康检查

```bash
curl http://localhost:8000/api/v1/health
```

### Prometheus指标

访问：http://localhost:8000/metrics

## 安全建议

1. **生产环境**：
   - 使用HTTPS
   - 限制CORS来源
   - 使用强密码和JWT
   - 启用速率限制

2. **API密钥管理**：
   - 使用环境变量
   - 定期轮换密钥
   - 限制API密钥权限

3. **资源限制**：
   - 设置用户配额
   - 限制并发任务数
   - 监控资源使用

## 扩展开发

### 添加新的生成器

1. 在 `api/generators/` 创建新的生成器类
2. 实现 `generate()` 方法
3. 在 `api/tasks.py` 注册任务

### 添加新的配置项

1. 在 `api/config_manager.py` 添加配置加载逻辑
2. 在配置文件中添加新项
3. 在生成器中使用配置

## 故障排查

### 常见问题

1. **任务一直处于queued状态**
   - 检查Celery worker是否运行
   - 检查Redis连接
   - 查看worker日志

2. **GPU内存不足**
   - 减少并发任务数
   - 降低分辨率
   - 启用模型卸载

3. **生成失败**
   - 检查输入参数
   - 查看任务错误信息
   - 检查模型文件是否存在

## 许可证

MIT License

