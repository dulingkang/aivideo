# Prompt Engine V2 使用指南

## 📋 概述

Prompt Engine V2 是基于架构设计文档的工程化实现，提供了**可扩展、可观测、可调优**的Prompt工程系统。

## 🏗️ 架构特点

1. **模块化设计**：每个模块可独立部署和测试
2. **数据驱动**：使用 `PromptPackage` 作为核心数据结构
3. **模型适配**：支持 Flux / CogVideoX / HunyuanVideo 等不同模型
4. **缓存机制**：支持内存缓存和Redis缓存（可扩展）
5. **指标监控**：内置指标收集和日志记录
6. **风格模板**：支持YAML配置文件，便于在线编辑

## 🚀 快速开始

### 基础使用

```python
from utils.prompt_engine_v2 import PromptEngine, UserRequest

# 创建引擎
engine = PromptEngine()

# 创建用户请求
req = UserRequest(
    text="那夜他手握长剑，踏入断桥",
    scene_type="novel",
    style="xianxia_v2",
    user_tier="professional"
)

# 执行处理
pkg = engine.run(req)

# 获取结果
print(f"最终Prompt: {pkg.final_prompt}")
print(f"Negative Prompt: {pkg.negative}")
print(f"目标模型: {pkg.model_target}")
```

### 与现有系统集成

在 `video_generator.py` 中集成：

```python
from utils.prompt_engine_v2 import PromptEngine, UserRequest

class VideoGenerator:
    def __init__(self, ...):
        # 初始化Prompt Engine V2
        self.prompt_engine_v2 = PromptEngine()
    
    def _build_prompt_v2(self, scene: Dict, model_type: str) -> Tuple[str, str]:
        """使用Prompt Engine V2构建prompt"""
        req = UserRequest(
            text=scene.get("description", ""),
            scene_type=scene.get("type", "general"),
            style=scene.get("style"),
            target_model=model_type,
            params=scene.get("params", {})
        )
        
        pkg = self.prompt_engine_v2.run(req)
        return pkg.final_prompt, pkg.negative
```

## 📦 核心组件

### 1. PromptPackage（数据结构）

模块间传递的核心对象，包含：
- `raw_text`: 原始文本
- `rewritten_text`: 重写后的文本
- `scene_struct`: 场景结构（镜头、角色）
- `style`: 风格模板
- `camera`: 相机配置
- `negative`: 负面提示词
- `final_prompt`: 最终prompt
- `model_target`: 目标模型
- `metadata`: 元数据（处理时间、QA评分等）

### 2. Model Adapter（模型适配器）

每个模型都有专门的适配器：

- **CogVideoXAdapter**: 固定分辨率720x480，8fps
- **HunyuanVideoAdapter**: 支持多种分辨率，自动对齐8的倍数
- **FluxAdapter**: 简洁清晰的prompt格式

### 3. 风格模板系统

风格模板存储在 `style_templates.yaml`，支持：
- `pre_prompt`: 前缀提示词
- `post_prompt`: 后缀提示词
- `negative_list`: 负面词列表
- `preferred_camera`: 偏好相机配置

### 4. 缓存系统

支持内存缓存和Redis缓存（可扩展）：

```python
from utils.prompt_engine_v2 import MemoryCache, PromptEngine

# 使用内存缓存
cache = MemoryCache()
engine = PromptEngine(cache=cache)

# 或使用Redis缓存（需要实现RedisCache类）
# cache = RedisCache(host='localhost', port=6379)
# engine = PromptEngine(cache=cache)
```

## 🔧 配置

### 风格模板配置

编辑 `utils/style_templates.yaml`：

```yaml
xianxia_v2:
  template_id: xianxia_v2
  pre_prompt: "ethereal, mist, soft rim light"
  post_prompt: "film grain low, 35mm lens"
  negative_list:
    - "oversaturated"
    - "watermark"
  preferred_camera:
    shot: "medium"
    motion: "slow"
    lens: "35mm"
```

### LLM客户端配置

实现 `LLMClient` 接口以使用真实LLM：

```python
from utils.prompt_engine_v2 import LLMClient

class OpenAILLMClient(LLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def rewrite_prompt(self, text: str, scene: str) -> str:
        # 调用OpenAI API
        response = openai.ChatCompletion.create(...)
        return response.choices[0].message.content
    
    def decompose_shots(self, text: str) -> SceneStruct:
        # 调用OpenAI API分解镜头
        ...

# 使用
llm_client = OpenAILLMClient(api_key="...")
engine = PromptEngine(llm_client=llm_client)
```

## 📊 指标监控

获取处理指标：

```python
metrics = engine.get_metrics()
print(f"总请求数: {metrics['total_requests']}")
print(f"缓存命中: {metrics['cache_hits']}")
print(f"QA修复: {metrics['qa_fixes']}")
print(f"错误数: {metrics['errors']}")
```

## 🔒 安全检查

Prompt QA模块会自动检查：
- 必需字段完整性
- 敏感词检测
- Prompt长度验证
- 安全性评分

## 🧪 测试

运行测试脚本：

```bash
python3 test_prompt_engine_v2.py
```

## 📈 性能优化

1. **缓存策略**：重写结果缓存24小时
2. **并行处理**：各模块可并行执行（未来版本）
3. **批量处理**：支持批量请求（未来版本）

## 🔄 迁移指南

从 V1 迁移到 V2：

1. 替换导入：
   ```python
   # 旧版本
   from utils.prompt_engine import PromptEngine
   
   # 新版本
   from utils.prompt_engine_v2 import PromptEngine, UserRequest
   ```

2. 更新调用方式：
   ```python
   # 旧版本
   result = engine.process(user_input="...", scene_type="novel")
   
   # 新版本
   req = UserRequest(text="...", scene_type="novel")
   pkg = engine.run(req)
   ```

3. 获取结果：
   ```python
   # 旧版本
   prompt = result["prompt"]
   negative = result["negative_prompt"]
   
   # 新版本
   prompt = pkg.final_prompt
   negative = pkg.negative
   ```

## 🛠️ 扩展开发

### 添加新的模型适配器

```python
from utils.prompt_engine_v2 import ModelAdapter, PromptPackage

class MyModelAdapter(ModelAdapter):
    def build_prompt(self, pkg: PromptPackage) -> str:
        # 实现模型特定的prompt构建逻辑
        return f"{pkg.rewritten_text}, {pkg.style.pre_prompt}"
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # 验证和调整参数
        return params

# 注册适配器
engine.adapters["mymodel"] = MyModelAdapter()
```

### 添加新的风格模板

在 `style_templates.yaml` 中添加：

```yaml
my_style:
  template_id: my_style
  pre_prompt: "..."
  post_prompt: "..."
  negative_list: [...]
  preferred_camera: {...}
```

## 📝 最佳实践

1. **使用缓存**：对于重复的文本，启用缓存可显著提升性能
2. **风格选择**：根据场景类型选择合适的风格模板
3. **参数验证**：使用适配器的 `validate_params` 确保参数正确
4. **监控指标**：定期检查指标，优化性能
5. **模板管理**：使用Git管理风格模板，支持版本控制

## 🐛 故障排查

### 问题：缓存不生效

检查缓存实现是否正确，确保 `CacheInterface` 方法已实现。

### 问题：LLM调用失败

检查 `LLMClient` 实现，确保API调用正确。

### 问题：风格模板未加载

检查 `style_templates.yaml` 文件路径和格式是否正确。

## 📚 参考文档

- 架构设计文档：见用户提供的设计文档
- 风格模板配置：`utils/style_templates.yaml`
- 测试用例：`test_prompt_engine_v2.py`

