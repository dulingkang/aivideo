# Flux 模型选择说明

## 📋 需要下载的模型

**需要同时下载 Flux.1 和 Flux.2**，因为用途不同：

| 模型 | 用途 | 特点 | 大小 |
|------|------|------|------|
| **Flux.1** | 主持人脸 + FaceID<br>实验室/医学场景 | 最稳、更干净自然 | ~24GB |
| **Flux.2** | 科学背景图<br>太空/粒子/量子类 | 冲击力强、效果爆炸 | ~24GB |

---

## 🎯 使用场景

### Flux.1 使用场景

1. **主持人脸**（Flux.1 + FaceID）
   - 最稳的人脸生成
   - 固定人设，365天不换脸
   - 适合科普主持人、教师、医生、专家

2. **实验室/医学场景**
   - 更干净自然
   - 专业感强
   - 适合：实验室、医院、手术室、实验设备

### Flux.2 使用场景

1. **科学背景图**
   - 冲击力强
   - 视觉效果震撼
   - 适合：科技背景、科学展示

2. **太空/粒子/量子类**
   - 效果爆炸
   - 视觉冲击力强
   - 适合：太空场景、粒子效果、量子物理、能量光束

---

## 📥 下载方式

### 下载 Flux.1

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
source /vepfs-dev/shawn/venv/py312/bin/activate
proxychains4 python download_models.py --model flux1
```

### 下载 Flux.2

```bash
proxychains4 python download_models.py --model flux2
```

### 同时下载 Flux.1 和 Flux.2

```bash
# 方式1: 使用 flux 别名（自动下载 flux1 和 flux2）
proxychains4 python download_models.py --model flux

# 方式2: 分别下载
proxychains4 python download_models.py --model flux1
proxychains4 python download_models.py --model flux2
```

---

## ⚙️ 配置说明

### Flux.1 配置

```yaml
model_selection:
  character:
    engine: flux-instantid  # Flux.1 + InstantID（主持人脸）
    flux1_model_path: /vepfs-dev/shawn/vid/fanren/gen_video/models/flux1-dev
    flux1_base_model: black-forest-labs/FLUX.1-dev
  
  scene:
    flux1:
      model_path: /vepfs-dev/shawn/vid/fanren/gen_video/models/flux1-dev
      base_model: black-forest-labs/FLUX.1-dev
      # 实验室/医学场景，更干净自然
```

### Flux.2 配置

```yaml
model_selection:
  scene:
    flux2:
      model_path: /vepfs-dev/shawn/vid/fanren/gen_video/models/flux2-dev
      base_model: black-forest-labs/FLUX.1-schnell  # 注意：模型ID可能需要确认
      # 科学背景图、太空/粒子/量子类，冲击力强
```

---

## 🔍 自动选择逻辑

系统会根据提示词自动选择：

- **包含"太空"、"粒子"、"量子"、"科学背景"** → 使用 Flux.2
- **包含"实验室"、"医学"、"医疗"、"医院"** → 使用 Flux.1
- **包含"主持人"、"讲解员"** → 使用 Flux.1 + FaceID

---

## 📝 注意事项

1. **Flux.2 模型ID**: 当前配置为 `black-forest-labs/FLUX.1-schnell`，可能需要根据实际情况调整
2. **显存需求**: 每个模型约 24GB，两个模型共约 48GB
3. **存储空间**: 确保有足够的存储空间（约 50GB）

---

**最后更新**: 2024年12月



