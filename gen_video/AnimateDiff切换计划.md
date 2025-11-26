# AnimateDiff-SDXL 切换计划

## 📋 当前状态

- ✅ 使用 SVD 进行视频生成（已实现）
- ⚠️ 需要切换到 AnimateDiff-SDXL（文档建议）

## 🎯 切换目标

从 SVD 切换到 AnimateDiff-SDXL，以获得：
- 更长的视频（64 帧 vs 20 帧）
- 更好的动漫风格适配
- FreeInit 去闪烁功能
- 1080P 原生支持

## 📝 切换步骤

### 阶段 1：准备 AnimateDiff-SDXL 模型

#### 1.1 下载模型
```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
python download_stage1_models.py
# 这会下载 AnimateDiff-SDXL-1080P 模型到 models/animatediff-sdxl-1080p/
```

#### 1.2 验证模型
```bash
ls -lh models/animatediff-sdxl-1080p/
# 应该包含：
# - model_index.json
# - unet/ 目录
# - vae/ 目录
# - text_encoder/ 目录
```

### 阶段 2：实现 AnimateDiff 支持

#### 2.1 更新 config.yaml
```yaml
video:
  # 使用模型：svd, svd-xt, animatediff-sdxl
  model_type: animatediff-sdxl  # 从 svd-xt 改为 animatediff-sdxl
  model_path: /vepfs-dev/shawn/vid/fanren/gen_video/models/animatediff-sdxl-1080p
  # AnimateDiff 特定配置
  num_frames: 64  # 从 20 改为 64
  fps: 16  # 从 12 改为 16（64帧/4秒 = 16fps）
  width: 1920  # 从 1280 改为 1920
  height: 1080  # 从 720 改为 1080
  # AnimateDiff 参数
  animatediff:
    use_freeinit: true  # 启用 FreeInit 去闪烁
    freeinit_iter: 3  # FreeInit 迭代次数
    motion_bucket_id: 127  # 运动桶ID（AnimateDiff 使用）
    num_inference_steps: 50  # 推理步数
```

#### 2.2 更新 video_generator.py

需要添加 AnimateDiff 支持：

```python
def load_model(self):
    """加载视频生成模型"""
    model_type = self.video_config['model_type']
    model_path = self.video_config['model_path']
    
    if model_type == 'animatediff-sdxl':
        return self._load_animatediff(model_path)
    elif model_type in ['svd', 'svd-xt']:
        return self._load_svd(model_path)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def _load_animatediff(self, model_path):
    """加载 AnimateDiff-SDXL 模型"""
    from diffusers import AnimateDiffPipeline, DDIMScheduler
    from diffusers.utils import export_to_video
    
    # 加载 AnimateDiff pipeline
    pipe = AnimateDiffPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(self.device)
    
    # 启用内存优化
    if self.gpu_config['memory_efficient']:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()
    
    return pipe

def generate_video_animatediff(self, image_path, output_path, prompt=None):
    """使用 AnimateDiff 生成视频"""
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    image = image.resize((1920, 1080), Image.Resampling.LANCZOS)
    
    # 生成视频
    frames = self.pipe(
        image=image,
        prompt=prompt or "",
        num_frames=64,
        num_inference_steps=50,
        guidance_scale=7.5,
        motion_bucket_id=127,
    ).frames[0]
    
    # FreeInit 去闪烁（如果启用）
    if self.video_config.get('animatediff', {}).get('use_freeinit', False):
        frames = self._apply_freeinit(frames)
    
    # 导出视频
    export_to_video(frames, output_path, fps=16)
    return output_path

def _apply_freeinit(self, frames):
    """应用 FreeInit 去闪烁"""
    # TODO: 实现 FreeInit 算法
    # 参考: https://github.com/guoyww/AnimateDiff
    pass
```

#### 2.3 实现 FreeInit 去闪烁

FreeInit 是一个去闪烁算法，需要：
1. 安装 FreeInit 库（如果可用）
2. 或实现 FreeInit 算法

参考实现：
```python
def _apply_freeinit(self, frames):
    """应用 FreeInit 去闪烁"""
    import torch
    import numpy as np
    
    # 将 frames 转换为 tensor
    frames_tensor = torch.stack([torch.from_numpy(np.array(f)) for f in frames])
    
    # FreeInit 算法（简化版）
    # 1. 计算帧间差异
    # 2. 平滑过渡
    # 3. 减少闪烁
    
    # TODO: 完整实现
    return frames
```

### 阶段 3：测试和对比

#### 3.1 使用相同图像测试
```bash
# 使用 SVD 生成
python run_pipeline.py --output test_svd --max-scenes 1

# 切换到 AnimateDiff 后生成
python run_pipeline.py --output test_animatediff --max-scenes 1
```

#### 3.2 对比指标
- 视频长度（SVD: ~1.7秒, AnimateDiff: ~4秒）
- 帧数（SVD: 20帧, AnimateDiff: 64帧）
- 分辨率（SVD: 1280×720, AnimateDiff: 1920×1080）
- 闪烁程度（AnimateDiff + FreeInit 应该更少）
- 动漫风格适配度

### 阶段 4：优化和调整

#### 4.1 参数调优
- 调整 `motion_bucket_id`（控制运动幅度）
- 调整 `num_inference_steps`（平衡质量和速度）
- 调整 FreeInit 参数

#### 4.2 性能优化
- 启用 VAE tiling（减少显存）
- 启用 CPU offload（如果显存不足）
- 批量处理优化

## 📌 注意事项

1. **显存需求**
   - AnimateDiff-SDXL 需要更多显存（约 20-24GB）
   - 如果显存不足，需要启用 CPU offload

2. **生成时间**
   - AnimateDiff 生成 64 帧需要更长时间（约 2-3 分钟/视频）
   - SVD 生成 20 帧约 30-60 秒/视频

3. **模型兼容性**
   - 确保 AnimateDiff 模型与 SDXL 基础模型兼容
   - 可能需要调整 LoRA 加载方式

4. **FreeInit 实现**
   - FreeInit 可能没有现成的库
   - 可能需要从 AnimateDiff 官方仓库获取实现

## 🔗 参考资源

- AnimateDiff 官方仓库: https://github.com/guoyww/AnimateDiff
- AnimateDiff-SDXL: https://huggingface.co/guoyww/AnimateDiff-SDXL-1080P
- FreeInit 论文: https://arxiv.org/abs/2310.08569

## ✅ 检查清单

- [ ] 下载 AnimateDiff-SDXL 模型
- [ ] 验证模型完整性
- [ ] 更新 config.yaml
- [ ] 实现 AnimateDiff 加载逻辑
- [ ] 实现 AnimateDiff 生成逻辑
- [ ] 实现 FreeInit 去闪烁（可选）
- [ ] 测试生成功能
- [ ] 对比 SVD 和 AnimateDiff 效果
- [ ] 性能优化
- [ ] 文档更新

