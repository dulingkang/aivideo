# 🔧 技术实施指南：Flux切换 + CogVideoX集成

> **目标**: 将Flux 1.1切换为主力图像模型，集成CogVideoX-5B视频模型  
> **时间线**: 2-3周  
> **优先级**: P0（必须完成）

---

## 📋 第一部分：Flux 1.1切换为主力（第1周）

### 1.1 当前状态检查

#### 检查Flux是否已集成

```bash
# 检查Flux相关代码
grep -r "flux" gen_video/image_generator.py
grep -r "FLUX" gen_video/config.yaml
```

#### 检查Flux模型是否已下载

```bash
# 检查模型路径
ls -lh /vepfs-dev/shawn/vid/fanren/gen_video/models/flux1-dev
```

**如果模型不存在，需要先下载**:
```bash
# Flux 1.1 Dev模型下载（需要HuggingFace token）
# 参考: gen_video/Flux模型选择说明.md
```

---

### 1.2 修改配置文件

#### 步骤1: 修改`gen_video/config.yaml`

**当前配置**:
```yaml
image:
  engine: auto  # 当前是auto，需要改为flux-instantid
```

**修改为**:
```yaml
image:
  engine: flux-instantid  # 直接使用Flux + InstantID
  # 或者保持auto，但确保model_selection优先选择Flux
```

**修改model_selection配置**:
```yaml
image:
  model_selection:
    character:
      engine: flux-instantid  # 人物生成使用Flux
      # ... 其他配置保持不变
    scene:
      engine: flux-instantid  # 场景生成也使用Flux（可选，可以先测试）
```

---

#### 步骤2: 优化Flux参数

**当前Flux配置**（在`config.yaml`中）:
```yaml
model_selection:
  character:
    num_inference_steps: 28  # 已优化，保持
    guidance_scale: 7.5      # 已优化，保持
    width: 1536
    height: 864
```

**建议优化**:
- ✅ 保持当前参数（已经优化过）
- ✅ 如果质量不够，可以提高到30-35步
- ✅ 如果速度太慢，可以降低到25步

---

### 1.3 修改代码逻辑

#### 步骤1: 检查`gen_video/model_selector.py`

**确保Flux优先选择**:
```python
# 在model_selector.py中
def select_engine(self, task_type, ...):
    if task_type == TaskType.CHARACTER:
        return "flux-instantid"  # 确保返回Flux
    # ...
```

---

#### 步骤2: 检查`gen_video/image_generator.py`

**确保Flux pipeline正确加载**:
```python
# 检查是否有_load_flux_pipeline方法
# 检查Flux pipeline是否正确初始化
```

**如果缺少，需要添加**:
```python
def _load_flux_pipeline(self):
    """加载Flux pipeline"""
    from diffusers import DiffusionPipeline
    import torch
    
    flux_model_path = self.config.get("model_selection", {}).get("character", {}).get("flux1_model_path")
    
    self.flux_pipeline = DiffusionPipeline.from_pretrained(
        flux_model_path,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
    if torch.cuda.is_available():
        self.flux_pipeline = self.flux_pipeline.to("cuda")
```

---

### 1.4 测试验证

#### 测试1: 单图生成测试

```python
# 创建测试脚本: test_flux_switch.py
from gen_video.image_generator import ImageGenerator

generator = ImageGenerator("gen_video/config.yaml")

# 测试人物生成
result = generator.generate(
    prompt="一个年轻的科普主持人，站在科学实验室中",
    task_type="character"
)

print(f"生成成功: {result['image_path']}")
```

**验收标准**:
- ✅ 图像生成成功
- ✅ 角色一致性良好
- ✅ 生成时间 < 30秒
- ✅ 图像质量 > SDXL

---

#### 测试2: 批量生成测试

```python
# 测试批量生成
prompts = [
    "一个年轻的科普主持人，站在科学实验室中",
    "一个年轻的科普主持人，站在天文台",
    "一个年轻的科普主持人，站在森林中"
]

for prompt in prompts:
    result = generator.generate(prompt=prompt, task_type="character")
    print(f"生成成功: {result['image_path']}")
```

**验收标准**:
- ✅ 所有图像生成成功
- ✅ 角色一致性稳定
- ✅ 无内存泄漏

---

### 1.5 性能优化

#### 优化1: 模型预热

```python
# 在image_generator.py中添加
def warmup(self):
    """模型预热，减少首次生成延迟"""
    if self.flux_pipeline is None:
        self._load_flux_pipeline()
    
    # 生成一张小图预热
    dummy_prompt = "test"
    _ = self.flux_pipeline(
        prompt=dummy_prompt,
        num_inference_steps=1,
        height=512,
        width=512
    )
```

---

#### 优化2: 显存优化

```python
# 如果显存不足，使用CPU offload
if torch.cuda.get_device_properties(0).total_memory < 24 * 1024**3:  # < 24GB
    self.flux_pipeline.enable_model_cpu_offload()
else:
    self.flux_pipeline = self.flux_pipeline.to("cuda")
```

---

## 📋 第二部分：CogVideoX-5B集成（第2-3周）

### 2.1 模型调研和下载

#### 步骤1: 查找CogVideoX-5B模型

**可能的来源**:
1. HuggingFace: `THUDM/CogVideoX-5b`（需要确认）
2. 官方GitHub: https://github.com/THUDM/CogVideoX
3. 模型社区: ModelScope, OpenXLab

**调研任务**:
- [ ] 确认模型下载地址
- [ ] 确认模型大小（预计20-30GB）
- [ ] 确认使用文档
- [ ] 确认依赖要求

---

#### 步骤2: 下载模型

```bash
# 方式1: 使用HuggingFace CLI（如果有）
huggingface-cli download THUDM/CogVideoX-5b --local-dir /vepfs-dev/shawn/vid/fanren/gen_video/models/cogvideox-5b

# 方式2: 使用git lfs（如果模型在GitHub）
git lfs clone https://huggingface.co/THUDM/CogVideoX-5b /vepfs-dev/shawn/vid/fanren/gen_video/models/cogvideox-5b

# 方式3: 手动下载（如果提供下载链接）
# 下载后解压到指定目录
```

**预计时间**: 2-4小时（取决于网络速度）

---

### 2.2 安装依赖

#### 检查CogVideoX依赖

```bash
# 查看CogVideoX官方文档，确认依赖
# 可能需要:
# - transformers
# - diffusers (特定版本)
# - torch (特定版本)
# - 其他依赖
```

**安装依赖**:
```bash
# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 安装依赖（根据官方文档）
pip install transformers diffusers torch
# 或其他依赖
```

---

### 2.3 代码集成

#### 步骤1: 修改`gen_video/video_generator.py`

**添加CogVideoX支持**:
```python
def _load_cogvideox_model(self, model_path: str):
    """加载CogVideoX-5B模型"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # 或使用diffusers（根据官方文档）
        # from diffusers import CogVideoXPipeline
        
        # 加载模型（根据官方文档调整）
        self.cogvideox_pipeline = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if torch.cuda.is_available():
            self.cogvideox_pipeline = self.cogvideox_pipeline.to("cuda")
            
    except Exception as e:
        print(f"CogVideoX加载失败: {e}")
        raise
```

---

#### 步骤2: 添加模型选择逻辑

**修改`load_model`方法**:
```python
def load_model(self):
    """加载视频生成模型（支持SVD、AnimateDiff、CogVideoX）"""
    if self.model_loaded:
        return
    
    model_type = self.video_config.get('model_type', 'svd-xt')
    model_path = self.video_config.get('model_path')
    
    if model_type == 'cogvideox-5b':
        self._load_cogvideox_model(model_path)
    elif model_type in ['svd', 'svd-xt']:
        self._load_svd_model(model_path)
    # ... 其他模型
```

---

#### 步骤3: 实现CogVideoX生成方法

**添加生成方法**:
```python
def generate_with_cogvideox(
    self,
    image: Image,
    prompt: str = "",
    num_frames: int = 120,
    fps: int = 24,
    **kwargs
) -> str:
    """使用CogVideoX生成视频"""
    # 根据官方文档实现
    # 注意: CogVideoX可能需要特定的输入格式
    
    output_path = self._get_output_path()
    
    # 调用CogVideoX pipeline
    video = self.cogvideox_pipeline(
        image=image,
        prompt=prompt,
        num_frames=num_frames,
        fps=fps,
        **kwargs
    )
    
    # 保存视频
    video.save(output_path)
    
    return output_path
```

---

#### 步骤4: 修改主生成方法

**修改`generate`方法，支持模型选择**:
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
    if model_type == 'cogvideox-5b':
        return self.generate_with_cogvideox(
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

### 2.4 配置文件更新

#### 修改`gen_video/config.yaml`

**添加CogVideoX配置**:
```yaml
video:
  # 模型类型：svd-xt, cogvideox-5b, hunyuanvideo
  model_type: cogvideox-5b  # 切换为CogVideoX（测试后）
  # 或保持svd-xt，通过API参数选择
  
  # CogVideoX配置
  cogvideox:
    model_path: /vepfs-dev/shawn/vid/fanren/gen_video/models/cogvideox-5b
    num_frames: 120
    fps: 24
    width: 1280
    height: 768
    num_inference_steps: 50
    guidance_scale: 7.5
    # 其他参数根据官方文档
```

---

### 2.5 测试验证

#### 测试1: 基础生成测试

```python
# 创建测试脚本: test_cogvideox.py
from gen_video.video_generator import VideoGenerator
from PIL import Image

generator = VideoGenerator("gen_video/config.yaml")
generator.load_model()

# 测试图像转视频
image_path = "test_image.png"
result = generator.generate(
    image_path=image_path,
    prompt="一个科普主持人在讲解",
    model_type="cogvideox-5b"
)

print(f"生成成功: {result}")
```

**验收标准**:
- ✅ 视频生成成功
- ✅ 视频质量 > SVD-XT（写实场景）
- ✅ 脸部稳定性好
- ✅ 生成时间 < 5分钟

---

#### 测试2: 性能测试

```python
# 测试显存使用
import torch

torch.cuda.empty_cache()
before = torch.cuda.memory_allocated()

# 生成视频
result = generator.generate(...)

after = torch.cuda.memory_allocated()
print(f"显存使用: {(after - before) / 1024**3:.2f} GB")
```

**验收标准**:
- ✅ 显存使用 < 24GB（4090可用）
- ✅ 无内存泄漏
- ✅ 可以连续生成多个视频

---

#### 测试3: 质量对比测试

```python
# 对比SVD-XT和CogVideoX
test_cases = [
    {"image": "test1.png", "prompt": "科普主持人讲解"},
    {"image": "test2.png", "prompt": "科普主持人站在实验室"},
]

for case in test_cases:
    # 使用SVD-XT生成
    svd_result = generator.generate(
        image_path=case["image"],
        prompt=case["prompt"],
        model_type="svd-xt"
    )
    
    # 使用CogVideoX生成
    cogvideox_result = generator.generate(
        image_path=case["image"],
        prompt=case["prompt"],
        model_type="cogvideox-5b"
    )
    
    # 对比质量（人工评估）
    print(f"测试案例: {case['prompt']}")
    print(f"SVD-XT: {svd_result}")
    print(f"CogVideoX: {cogvideox_result}")
```

**验收标准**:
- ✅ CogVideoX质量 > SVD-XT（写实场景）
- ✅ 脸部稳定性更好
- ✅ 运动更自然

---

### 2.6 API接口适配

#### 修改`gen_video/api/mvp_main.py`

**添加模型选择参数**:
```python
@app.post("/api/v1/videos/generate")
async def generate_video(
    request: VideoGenerateRequest,
    api_key: str = Header(..., alias="X-API-Key")
):
    """生成视频（支持多模型）"""
    # 验证API Key
    user = verify_api_key(api_key)
    
    # 检查配额
    if not check_quota(user, "video"):
        raise HTTPException(status_code=403, detail="配额不足")
    
    # 确定使用的模型
    model_type = request.model_type or "cogvideox-5b"  # 默认CogVideoX
    
    # 生成视频
    generator = VideoGenerator()
    result = generator.generate(
        image_path=request.image_path,
        prompt=request.prompt,
        model_type=model_type,
        **request.params
    )
    
    return {
        "task_id": generate_task_id(),
        "status": "completed",
        "video_url": f"/api/v1/files/videos/{result}"
    }
```

---

## 📋 第三部分：工作流优化（第4周）

### 3.1 短剧推文工作流设计

#### 工作流步骤

```
1. 原文输入
   ↓
2. LLM分镜（自动拆分场景）
   ↓
3. Flux生成角色底图 + 场景图
   ↓
4. CogVideoX生成视频片段
   ↓
5. CosyVoice生成配音
   ↓
6. 字幕生成和合成
   ↓
7. 视频拼接和导出
```

---

### 3.2 实现自动分镜

#### 使用LLM进行分镜

```python
# 创建新文件: gen_video/script_splitter.py
from openai import OpenAI  # 或其他LLM API

class ScriptSplitter:
    """脚本分镜器"""
    
    def __init__(self, llm_api_key: str):
        self.client = OpenAI(api_key=llm_api_key)
    
    def split_script(self, script: str) -> List[Dict]:
        """将脚本拆分为场景"""
        prompt = f"""
请将以下小说/推文内容拆分为多个场景，每个场景包含：
1. 场景描述（用于生成图像）
2. 角色动作（用于生成视频）
3. 旁白文本（用于配音）

原文：
{script}

请以JSON格式返回，格式如下：
[
  {{
    "scene_id": 1,
    "description": "场景描述",
    "action": "角色动作",
    "narration": "旁白文本",
    "duration": 5
  }}
]
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # 解析JSON
        scenes = json.loads(response.choices[0].message.content)
        
        return scenes
```

---

### 3.3 批量生成优化

#### 实现批量生成

```python
# 在video_generator.py中添加
def generate_batch(
    self,
    scenes: List[Dict],
    model_type: str = "cogvideox-5b"
) -> List[str]:
    """批量生成视频片段"""
    results = []
    
    for i, scene in enumerate(scenes):
        print(f"生成场景 {i+1}/{len(scenes)}: {scene['description']}")
        
        # 生成图像
        image_generator = ImageGenerator()
        image_path = image_generator.generate(
            prompt=scene['description'],
            task_type="character"
        )
        
        # 生成视频
        video_path = self.generate(
            image_path=image_path,
            prompt=scene['action'],
            model_type=model_type
        )
        
        results.append({
            "scene_id": scene['scene_id'],
            "video_path": video_path,
            "narration": scene['narration']
        })
    
    return results
```

---

## 📋 第四部分：问题排查

### 4.1 常见问题

#### 问题1: Flux模型加载失败

**症状**: `FileNotFoundError` 或 `Model not found`

**解决方案**:
1. 检查模型路径是否正确
2. 检查模型是否完整下载
3. 检查HuggingFace token是否有效

---

#### 问题2: CogVideoX显存不足

**症状**: `CUDA out of memory`

**解决方案**:
1. 使用模型量化（INT8/FP16）
2. 使用CPU offload
3. 降低分辨率或帧数
4. 使用梯度检查点（如果支持）

---

#### 问题3: 视频质量不佳

**症状**: 视频模糊、抖动、不连贯

**解决方案**:
1. 增加推理步数
2. 优化提示词
3. 调整运动参数
4. 使用插帧（RIFE）

---

## 📋 第五部分：验收标准

### 5.1 Flux切换验收

- [ ] Flux生成质量 > SDXL
- [ ] 角色一致性稳定
- [ ] 单图生成时间 < 30秒
- [ ] 无内存泄漏
- [ ] API接口正常

---

### 5.2 CogVideoX集成验收

- [ ] CogVideoX可正常生成视频
- [ ] 视频质量 > SVD-XT（写实场景）
- [ ] 脸部稳定性好
- [ ] 单视频生成时间 < 5分钟
- [ ] 显存使用 < 24GB
- [ ] API接口正常

---

### 5.3 工作流验收

- [ ] 全流程自动化
- [ ] 可以批量生成
- [ ] 视频质量稳定
- [ ] 用户体验良好

---

## 🚀 下一步

1. ✅ 开始执行Day 1任务（Flux切换）
2. ✅ 准备CogVideoX调研
3. ✅ 准备测试数据

---

**文档版本**: v1.0  
**最后更新**: 2025年1月  
**参考文档**: 
- `最终决策方案-模型选择与业务路径.md`
- `Flux模型选择说明.md`
- CogVideoX官方文档

