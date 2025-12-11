# ✅ 切换到diffusers格式模型

> **更新时间**: 2025年12月10日

---

## 🎯 推荐模型

**`hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v`** ✅

**优势**:
- ✅ 标准diffusers格式
- ✅ 完整的pipeline（包含所有组件）
- ✅ 与当前代码完全兼容
- ✅ 可以直接使用，无需额外处理

---

## 📝 已完成的修改

1. ✅ **config.yaml**: 已更新为使用diffusers格式模型
2. ✅ **video_generator.py**: 已更新，支持自动识别HuggingFace模型ID

---

## 🚀 使用方法

### 方法1：使用HuggingFace自动下载（推荐）

模型会自动从HuggingFace下载，无需手动下载：

```yaml
# config.yaml
hunyuanvideo:
  model_path: hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v
```

### 方法2：下载到本地后使用

1. 下载模型到本地：
```bash
huggingface-cli download hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v \
    --local-dir /vepfs-dev/shawn/vid/fanren/gen_video/models/hunyuan-video-1.5-community
```

2. 更新config.yaml：
```yaml
hunyuanvideo:
  model_path: /vepfs-dev/shawn/vid/fanren/gen_video/models/hunyuan-video-1.5-community
```

---

## 🧪 测试

运行测试脚本：

```bash
python gen_video/test_hunyuanvideo_generation.py
```

---

## 📊 其他可选模型

| 模型 | 分辨率 | 类型 | 推荐度 |
|------|--------|------|--------|
| `hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v` | 480p | 图生视频 | ⭐⭐⭐⭐⭐ |
| `hunyuanvideo-community/HunyuanVideo-1.5-720p_i2v` | 720p | 图生视频 | ⭐⭐⭐⭐ |
| `hunyuanvideo-community/HunyuanVideo-1.5-480p_t2v` | 480p | 文生视频 | ⭐⭐⭐ |
| `hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v_distilled` | 480p | 蒸馏版 | ⭐⭐⭐⭐ |

---

## ✅ 预期效果

使用diffusers格式模型后：
- ✅ Pipeline可以完整加载
- ✅ 所有组件自动下载/加载
- ✅ 可以直接生成视频
- ✅ 无需手动处理组件

