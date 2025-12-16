# Prompt Engine V2 - 本地模式使用指南

## ✅ 完全本地运行，无需LLM API

Prompt Engine V2 **默认使用本地规则引擎**，可以在**完全离线**的环境下运行，**不需要任何外部LLM API**。

## 🎯 本地模式特点

1. **零依赖外部服务**：不需要OpenAI、Claude等LLM API
2. **完全离线运行**：可以在内网环境运行
3. **规则引擎**：使用智能规则进行文本处理和场景分解
4. **高性能**：本地处理速度快，毫秒级响应
5. **可扩展**：可以轻松添加更多规则和关键词

## 🚀 快速开始

### 基础使用（完全本地）

```python
from utils.prompt_engine_v2 import PromptEngine, UserRequest

# 创建引擎（默认就是本地模式）
engine = PromptEngine()  # 无需任何配置，默认使用SimpleLLMClient

# 创建请求
req = UserRequest(
    text="那夜他手握长剑，踏入断桥",
    scene_type="novel",
    style="xianxia_v2"
)

# 执行处理（完全本地）
pkg = engine.run(req)

print(f"Prompt: {pkg.final_prompt}")
print(f"Negative: {pkg.negative}")
```

## 📋 本地规则引擎功能

### 1. 文本重写（Prompt Rewriter）

本地规则引擎会自动：
- ✅ 保留原始文本核心内容
- ✅ 根据场景类型添加风格关键词
- ✅ 自动添加构图、光线、质量增强词
- ✅ 检测并增强动作描述
- ✅ 检测并增强环境描述

**示例**：
```
输入: "那夜他手握长剑，踏入断桥"
输出: "那夜他手握长剑，踏入断桥, cinematic, dramatic, rule of thirds, rim light, high-resolution, cinematic, detailed, in moonlight"
```

### 2. 场景分解（Scene Decomposer）

本地规则引擎会自动：
- ✅ 按句号/逗号智能分割文本
- ✅ 为每个片段创建镜头
- ✅ 自动检测镜头类型（wide/medium/close-up）
- ✅ 自动检测动作和情绪
- ✅ 估算镜头时长

**示例**：
```
输入: "那夜他手握长剑，踏入断桥，月光如水洒在剑身上"
输出: 3个镜头
  镜头1: "那夜他手握长剑，踏入断桥" (2.0秒)
  镜头2: "月光如水洒在剑身上" (1.5秒)
```

### 3. 关键词库

本地规则引擎内置了丰富的关键词库：

- **场景关键词**：novel, drama, scientific, government, enterprise
- **质量增强词**：high-resolution, cinematic, detailed, professional
- **构图关键词**：rule of thirds, symmetrical composition, leading lines
- **光线关键词**：rim light, soft light, dramatic lighting
- **动作关键词**：walking, running, sitting, standing, looking, moving
- **环境关键词**：山谷→mountain valley, 瀑布→waterfall, 彩虹→rainbow

## 🔧 自定义规则

### 添加新的环境关键词

编辑 `SimpleLLMClient` 的 `environment_keywords`：

```python
self.environment_keywords = {
    "山谷": "mountain valley",
    "瀑布": "waterfall",
    "你的新词": "your new translation"
}
```

### 添加新的动作关键词

编辑 `SimpleLLMClient` 的 `action_keywords`：

```python
self.action_keywords = {
    "walking": "walking slowly",
    "你的新动作": "your new action description"
}
```

## 📊 性能对比

| 模式 | 响应时间 | 依赖 | 成本 |
|------|---------|------|------|
| **本地模式** | < 10ms | 无 | 免费 |
| LLM API模式 | 500-2000ms | 需要API | 按token收费 |

## 🆚 本地模式 vs LLM模式

### 本地模式（默认）

**优点**：
- ✅ 完全免费
- ✅ 响应速度快
- ✅ 无需网络
- ✅ 数据隐私安全
- ✅ 可完全离线运行

**适用场景**：
- 批量处理
- 内网环境
- 对成本敏感
- 对响应速度要求高

### LLM模式（可选）

如果需要更智能的处理，可以实现 `LLMClient` 接口：

```python
class OpenAILLMClient(LLMClient):
    def rewrite_prompt(self, text: str, scene: str) -> str:
        # 调用OpenAI API
        response = openai.ChatCompletion.create(...)
        return response.choices[0].message.content

# 使用LLM模式
llm_client = OpenAILLMClient(api_key="...")
engine = PromptEngine(llm_client=llm_client)
```

## 🧪 测试本地模式

运行测试脚本：

```bash
python3 test_local_prompt_engine.py
```

## 📝 使用示例

### 示例1：小说推文

```python
from utils.prompt_engine_v2 import PromptEngine, UserRequest

engine = PromptEngine()

req = UserRequest(
    text="那夜他手握长剑，踏入断桥，月光如水洒在剑身上",
    scene_type="novel",
    style="xianxia_v2"
)

pkg = engine.run(req)
# 完全本地处理，无需API
```

### 示例2：科学场景

```python
req = UserRequest(
    text="黑洞在太空中旋转，周围有吸积盘",
    scene_type="scientific",
    target_model="hunyuanvideo"
)

pkg = engine.run(req)
# 自动选择HunyuanVideo适配器
```

### 示例3：风景场景

```python
req = UserRequest(
    text="一个美丽的山谷，有瀑布和彩虹",
    scene_type="general"
)

pkg = engine.run(req)
# 使用通用风格模板
```

## 🔍 验证本地模式

检查是否使用本地模式：

```python
engine = PromptEngine()
print(type(engine.llm))  # 应该显示: <class 'SimpleLLMClient'>
```

## 💡 最佳实践

1. **默认使用本地模式**：对于大多数场景，本地规则引擎已经足够
2. **缓存利用**：重复请求会自动使用缓存，提升性能
3. **关键词扩展**：根据业务需求扩展关键词库
4. **风格模板**：使用YAML配置文件管理风格模板

## ❓ 常见问题

**Q: 本地模式的效果如何？**

A: 本地模式使用智能规则引擎，对于常见场景（小说、科学、风景等）效果很好。如果需要更复杂的语义理解，可以考虑LLM模式。

**Q: 可以混合使用吗？**

A: 可以！对于简单场景使用本地模式，复杂场景使用LLM模式。

**Q: 如何提升本地模式的效果？**

A: 
1. 扩展关键词库
2. 优化规则逻辑
3. 调整风格模板
4. 使用缓存提升重复请求性能

## 📚 相关文档

- 完整使用指南：`PROMPT_ENGINE_V2_README.md`
- 测试脚本：`test_local_prompt_engine.py`
- 风格模板：`style_templates.yaml`

