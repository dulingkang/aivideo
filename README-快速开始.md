# 快速开始 - 立即测试API

## ✅ 环境已准备好

- ✅ 虚拟环境：`/vepfs-dev/shawn/venv/py312`（FastAPI已安装）
- ✅ 同步模式API：已创建（不依赖Redis）
- ✅ 配置文件：已存在

## 🚀 立即开始（2步）

### 步骤1：启动API服务器

```bash
cd /vepfs-dev/shawn/vid/fanren
./start_sync_api_with_venv.sh
```

看到以下信息表示启动成功：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 步骤2：测试API

**浏览器打开**：http://localhost:8000/docs

在Swagger UI中：
1. 找到 `/api/v1/images/generate`
2. 点击 "Try it out"
3. 填写测试数据（见下方）
4. 点击 "Execute"
5. 等待30-60秒查看结果

## 📝 测试数据示例

```json
{
  "prompt": "xianxia fantasy, Han Li, calm cultivator, medium shot, front view, facing camera",
  "width": 1536,
  "height": 864,
  "num_inference_steps": 40,
  "guidance_scale": 7.5
}
```

## ✅ 成功标志

返回结果包含：
- `"status": "completed"`
- `"image_path": "/path/to/image.png"`

生成的图像在：`outputs/api/images/{task_id}.png`

## 📚 更多信息

- [使用虚拟环境启动.md](./使用虚拟环境启动.md) - 详细启动说明
- [无Redis测试指南.md](./无Redis测试指南.md) - 同步模式说明
- [快速测试指南.md](./快速测试指南.md) - 完整测试指南

---

**现在就可以开始测试了** 🎉

