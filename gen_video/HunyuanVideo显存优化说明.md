# HunyuanVideo 显存优化说明

## 问题分析

720p模型（`hunyuanvideo-community/HunyuanVideo-1.5-720p_i2v`）在transformer的attention计算时需要**21GB+显存**，即使：
- 降低输入分辨率到640x384
- 减少帧数到25帧
- 启用CPU offload
- 启用VAE tiling

仍然无法避免显存不足的问题。

## 根本原因

720p模型的transformer架构本身就需要大量显存，与输入分辨率关系不大。attention计算需要处理大量的tokens，导致显存需求很高。

## 解决方案

### 方案1：使用480p模型（推荐）

480p模型的显存需求更低，适合当前环境：

```yaml
hunyuanvideo:
  model_path: hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v  # 使用480p模型
  width: 640
  height: 480
  num_frames: 60  # 480p模型可以生成更多帧
```

**优势：**
- 显存需求低（约10-15GB）
- 生成速度快
- 质量仍然很好

**下载命令：**
```bash
proxychains4 huggingface-cli download hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v --local-dir /vepfs-dev/shawn/vid/fanren/gen_video/models/hunyuan-video-1.5-480p-i2v
```

### 方案2：等待其他进程释放显存

当前有其他进程占用了63GB显存，如果这些进程可以释放，720p模型可能可以运行。

检查占用显存的进程：
```bash
nvidia-smi
```

### 方案3：使用蒸馏版模型（更快，显存需求更低）

```yaml
hunyuanvideo:
  model_path: hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v_distilled
  width: 640
  height: 480
  num_frames: 60
```

## 当前配置（已优化但仍有问题）

```yaml
hunyuanvideo:
  model_path: /vepfs-dev/shawn/vid/fanren/gen_video/models/hunyuan-video-1.5-720p-i2v
  width: 640  # 已降低
  height: 384  # 已降低
  num_frames: 40  # 已降低
  force_cpu_offload: true  # 已启用
  enable_vae_tiling: true  # 已启用
```

即使这些优化都已应用，720p模型仍然需要21GB+显存用于attention计算。

## 建议

**立即行动：切换到480p模型**

1. 下载480p模型
2. 更新`config.yaml`中的`model_path`
3. 调整`width`和`height`为640x480
4. 增加`num_frames`到60（480p模型可以支持更多帧）

这样可以在当前显存限制下成功生成视频。

