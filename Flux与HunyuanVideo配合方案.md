# ✅ Flux + HunyuanVideo 配合方案

> **结论**: **完全可以配合，这是最佳组合之一！**  
> **工作流**: Flux生成图像 → HunyuanVideo将图像转为视频

---

## 🎯 一、为什么可以配合？

### 1.1 模型定位不同

| 模型 | 类型 | 功能 | 阶段 |
|------|------|------|------|
| **Flux 1.1** | 图像生成 | 文生图（Text-to-Image） | **第1阶段** |
| **HunyuanVideo** | 视频生成 | 图生视频（Image-to-Video） | **第2阶段** |

**结论**: 它们是**不同阶段**的模型，天然可以配合使用！

---

### 1.2 工作流配合

```
┌─────────────────┐
│  文本提示词      │
│  "科普主持人..." │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Flux 1.1       │  ← 第1阶段：生成高质量图像
│  生成图像        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HunyuanVideo    │  ← 第2阶段：将图像转为视频
│  图生视频        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  最终视频        │
└─────────────────┘
```

---

## 🚀 二、配合优势

### 2.1 技术优势

#### ✅ **图像质量最优**
- Flux 1.1 是2025年写实图像第一梯队
- 语义理解强，真人不畸变
- 角色一致性好（配合InstantID + LoRA）

#### ✅ **视频质量最优**
- HunyuanVideo 是腾讯开源的高质量视频模型
- 动画连续性强
- 能生成>5秒原生视频
- 支持消费级GPU运行

#### ✅ **完整工作流**
- 从文本到视频，全流程自动化
- 质量可控，每个环节都是最优选择

---

### 2.2 商业优势

#### ✅ **高端定位**
- 适合高端宣传片、科普视频
- 适合企业收费版、政府版
- 单价可以更高（¥99-199/条）

#### ✅ **差异化竞争**
- 市面上很少有"Flux + HunyuanVideo"的组合
- 你的技术栈完全匹配
- 可以建立技术壁垒

---

## 📋 三、技术实现方案

### 3.1 当前状态

#### ✅ 已有基础
- ✅ Flux 1.1 已集成（`gen_video/image_generator.py`）
- ✅ diffusers库已支持HunyuanVideo（你的diffusers版本已包含）
- ✅ 视频生成框架已搭建（`gen_video/video_generator.py`）

#### ⚠️ 需要补充
- ⚠️ HunyuanVideo模型需要下载
- ⚠️ 需要集成HunyuanVideo到video_generator.py
- ⚠️ 需要优化工作流

---

### 3.2 集成步骤

#### 步骤1: 下载HunyuanVideo模型

**模型信息**:
- **模型名称**: `Tencent-Hunyuan/HunyuanVideo` 或 `Tencent-Hunyuan/HunyuanVideo-ImageToVideo`
- **模型类型**: 图生视频（Image-to-Video）
- **模型大小**: 预计20-30GB

**下载方式**:
```bash
# 方式1: 使用HuggingFace CLI
huggingface-cli download Tencent-Hunyuan/HunyuanVideo-ImageToVideo \
    --local-dir /vepfs-dev/shawn/vid/fanren/gen_video/models/hunyuan-video

# 方式2: 使用git lfs
git lfs clone https://huggingface.co/Tencent-Hunyuan/HunyuanVideo-ImageToVideo \
    /vepfs-dev/shawn/vid/fanren/gen_video/models/hunyuan-video

# 方式3: 手动下载（如果提供下载链接）
```

**预计时间**: 2-4小时（取决于网络速度）

---

#### 步骤2: 修改配置文件

**修改`gen_video/config.yaml`**:
```yaml
video:
  # 模型类型：svd-xt, cogvideox-5b, hunyuanvideo
  model_type: hunyuanvideo  # 添加HunyuanVideo选项
  
  # HunyuanVideo配置
  hunyuanvideo:
    model_path: /vepfs-dev/shawn/vid/fanren/gen_video/models/hunyuan-video
    num_frames: 120  # 帧数（5秒@24fps）
    fps: 24
    width: 1280
    height: 768
    num_inference_steps: 50
    guidance_scale: 7.5
    # 其他参数根据官方文档
```

---

#### 步骤3: 集成到video_generator.py

**添加HunyuanVideo支持**:
```python
# 在gen_video/video_generator.py中添加

def _load_hunyuanvideo_model(self, model_path: str):
    """加载HunyuanVideo模型"""
    from diffusers import HunyuanVideoImageToVideoPipeline
    import torch
    
    print(f"加载HunyuanVideo模型: {model_path}")
    
    # 加载pipeline
    self.hunyuanvideo_pipeline = HunyuanVideoImageToVideoPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
    # 移动到GPU
    if torch.cuda.is_available():
        self.hunyuanvideo_pipeline = self.hunyuanvideo_pipeline.to("cuda")
        # 如果显存不足，使用CPU offload
        if torch.cuda.get_device_properties(0).total_memory < 24 * 1024**3:
            self.hunyuanvideo_pipeline.enable_model_cpu_offload()
    
    print("✅ HunyuanVideo模型加载完成")

def generate_with_hunyuanvideo(
    self,
    image: Image,
    prompt: str = "",
    num_frames: int = 120,
    fps: int = 24,
    width: int = 1280,
    height: int = 768,
    **kwargs
) -> str:
    """使用HunyuanVideo生成视频"""
    if self.hunyuanvideo_pipeline is None:
        model_path = self.video_config.get("hunyuanvideo", {}).get("model_path")
        self._load_hunyuanvideo_model(model_path)
    
    # 准备输入
    from PIL import Image
    if isinstance(image, str):
        image = Image.open(image)
    
    # 生成视频
    video = self.hunyuanvideo_pipeline(
        image=image,
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=kwargs.get("num_inference_steps", 50),
        guidance_scale=kwargs.get("guidance_scale", 7.5),
        num_frames=num_frames,
        generator=kwargs.get("generator"),
    ).frames[0]  # 获取第一帧序列
    
    # 保存视频
    output_path = self._get_output_path()
    self._save_video(video, output_path, fps=fps)
    
    return output_path
```

---

#### 步骤4: 修改主生成方法

**修改`generate`方法，支持HunyuanVideo**:
```python
def generate(
    self,
    image_path: str,
    prompt: str = "",
    model_type: Optional[str] = None,
    **kwargs
) -> str:
    """生成视频（支持多模型）"""
    # 确定使用的模型
    if model_type is None:
        model_type = self.video_config.get('model_type', 'svd-xt')
    
    # 根据模型类型选择生成方法
    if model_type == 'hunyuanvideo':
        return self.generate_with_hunyuanvideo(
            image=load_image(image_path),
            prompt=prompt,
            **kwargs
        )
    elif model_type in ['svd', 'svd-xt']:
        return self.generate_with_svd(
            image_path=image_path,
            prompt=prompt,
            **kwargs
        )
    # ...
```

---

### 3.3 完整工作流实现

#### 实现Flux → HunyuanVideo工作流

```python
# 创建新文件: gen_video/flux_hunyuan_workflow.py

from gen_video.image_generator import ImageGenerator
from gen_video.video_generator import VideoGenerator
from PIL import Image

class FluxHunyuanWorkflow:
    """Flux + HunyuanVideo完整工作流"""
    
    def __init__(self, config_path: str = "gen_video/config.yaml"):
        self.image_generator = ImageGenerator(config_path)
        self.video_generator = VideoGenerator(config_path)
    
    def generate_video_from_text(
        self,
        text_prompt: str,
        character_prompt: str = "",
        video_prompt: str = "",
        output_dir: str = "outputs/flux_hunyuan"
    ) -> str:
        """
        从文本生成完整视频
        
        Args:
            text_prompt: 文本提示词（用于生成图像）
            character_prompt: 角色提示词（可选，用于Flux生成角色）
            video_prompt: 视频动作提示词（可选，用于HunyuanVideo）
        
        Returns:
            最终视频路径
        """
        # 步骤1: 使用Flux生成图像
        print("步骤1: 使用Flux生成图像...")
        image_result = self.image_generator.generate(
            prompt=text_prompt if not character_prompt else character_prompt,
            task_type="character",  # 使用Flux + InstantID
            engine="flux-instantid"
        )
        image_path = image_result['image_path']
        print(f"✅ 图像生成完成: {image_path}")
        
        # 步骤2: 使用HunyuanVideo生成视频
        print("步骤2: 使用HunyuanVideo生成视频...")
        video_path = self.video_generator.generate(
            image_path=image_path,
            prompt=video_prompt if video_prompt else text_prompt,
            model_type="hunyuanvideo"
        )
        print(f"✅ 视频生成完成: {video_path}")
        
        return video_path
```

---

## 📊 四、性能与成本分析

### 4.1 性能要求

#### GPU显存需求

| 模型 | 显存需求 | 你的4090 |
|------|---------|---------|
| Flux 1.1 | ~12GB | ✅ 可用 |
| HunyuanVideo | ~18-24GB | ⚠️ 可能不足 |

**解决方案**:
1. **使用CPU offload**（速度慢但可用）
2. **使用模型量化**（INT8/FP16）
3. **使用云GPU**（推荐，成本可控）

---

#### 生成时间

| 阶段 | 模型 | 时间 | 总计 |
|------|------|------|------|
| 图像生成 | Flux 1.1 | ~30秒 | |
| 视频生成 | HunyuanVideo | ~3-5分钟 | |
| **总计** | | | **~4-6分钟** |

---

### 4.2 成本分析

#### 本地部署（4090）

**优势**:
- ✅ 无API调用费用
- ✅ 数据安全（本地处理）
- ✅ 可定制化

**劣势**:
- ⚠️ 显存可能不足（需要优化）
- ⚠️ 生成速度较慢
- ⚠️ 需要维护硬件

**成本**: ¥3,000-5,000/月（服务器成本）

---

#### 云GPU部署（推荐）

**优势**:
- ✅ 按需付费（成本可控）
- ✅ 显存充足（A100 80GB）
- ✅ 生成速度快
- ✅ 无需维护硬件

**劣势**:
- ⚠️ 需要网络传输
- ⚠️ 可能有API调用费用

**成本**: ¥0.5-1.5/分钟（按需计费）

**推荐方案**: 
- 本地4090跑Flux（图像生成）
- 云GPU跑HunyuanVideo（视频生成）

---

## 🎯 五、使用场景

### 5.1 适合场景

#### ✅ **高端宣传片**
- 科普视频（政府/企业）
- 产品宣传片
- 品牌广告

#### ✅ **高质量内容**
- 需要电影级画质
- 需要长视频（>5秒）
- 需要复杂运动

---

### 5.2 不适合场景

#### ❌ **批量生成**
- 成本太高（每个视频4-6分钟）
- 不适合短剧推文（量大）

#### ❌ **快速迭代**
- 生成时间太长
- 不适合需要快速反馈的场景

---

## 📋 六、实施建议

### 6.1 分阶段实施

#### **阶段1: 验证可行性（1周）**
- [ ] 下载HunyuanVideo模型
- [ ] 集成到video_generator.py
- [ ] 测试单视频生成
- [ ] 验证质量

#### **阶段2: 优化工作流（1周）**
- [ ] 实现Flux → HunyuanVideo工作流
- [ ] 优化参数
- [ ] 性能优化

#### **阶段3: 商业化（2周）**
- [ ] API接口适配
- [ ] 定价策略（高端收费）
- [ ] 上线测试

---

### 6.2 与CogVideoX的配合策略

**建议**: **双模型策略**

| 场景 | 模型 | 理由 |
|------|------|------|
| **大众量产** | CogVideoX-5B | 成本低、速度快、24G显卡可跑 |
| **高端收费** | HunyuanVideo | 质量最好、适合高价订单 |

**工作流**:
```
用户选择模型类型
    ↓
如果是"高端版" → Flux + HunyuanVideo
如果是"标准版" → Flux + CogVideoX-5B
```

---

## ✅ 七、总结

### 7.1 核心结论

1. ✅ **Flux + HunyuanVideo完全可以配合**
2. ✅ **这是最佳组合之一**（图像质量 + 视频质量）
3. ✅ **适合高端场景**（科普、宣传片、企业版）
4. ⚠️ **成本较高**（建议云GPU部署）

### 7.2 推荐方案

**双模型策略**:
- **标准版**: Flux + CogVideoX-5B（大众量产）
- **高端版**: Flux + HunyuanVideo（高端收费）

**这样既能覆盖大众市场，又能提供高端服务！**

---

## 🚀 下一步行动

1. ✅ 确认是否要集成HunyuanVideo
2. ✅ 如果确认，开始下载模型
3. ✅ 集成到video_generator.py
4. ✅ 测试验证

---

**文档版本**: v1.0  
**最后更新**: 2025年1月  
**参考**: 
- `最终决策方案-模型选择与业务路径.md`
- `技术实施指南-Flux切换与CogVideoX集成.md`

