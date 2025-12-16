# 下一步开发计划

## ✅ 刚完成的功能

### 1. 视频质量分析器
- ✅ `utils/video_quality_analyzer.py` - 自动分析视频质量
  - 色彩分析（饱和度、亮度、对比度）
  - 帧间一致性（闪烁检测）
  - 清晰度评估
  - 运动流畅度分析
  - 自动生成优化建议

### 2. 批量处理工具
- ✅ `utils/batch_processor.py` - 批量生成和分析
  - 批量视频生成
  - 批量质量分析
  - 进度跟踪和日志记录

### 3. Prompt缓存管理器
- ✅ `utils/prompt_cache.py` - 持久化缓存
  - 文件系统缓存
  - TTL过期管理
  - 自动清理过期缓存

### 4. HunyuanVideo色彩修复
- ✅ 增加推理步数（30→40）
- ✅ 调整色彩参数（降低饱和度）
- ✅ 插帧后统一色彩调整

## 🚀 下一步开发建议

### 1. 视频质量自动优化（高优先级）

**目标**：根据质量分析结果自动调整参数

```python
# 自动优化流程
analyzer = VideoQualityAnalyzer()
results = analyzer.analyze(video_path)

# 根据分析结果自动调整配置
if results['color_analysis']['saturation']['mean'] > 150:
    # 自动降低饱和度
    config['saturation_factor'] = 0.7
    # 重新生成
```

**实现要点**：
- 集成质量分析器到生成流程
- 实现自动参数调整逻辑
- 支持迭代优化（生成→分析→调整→重新生成）

### 2. 批量生成优化（高优先级）

**目标**：支持大规模批量生成，提升效率

**功能**：
- 任务队列管理
- 失败重试机制
- 进度监控和通知
- 资源使用优化

### 3. 性能监控和告警（中优先级）

**目标**：实时监控系统性能，及时发现问题

**功能**：
- GPU显存监控
- 生成时间统计
- 错误率统计
- 自动告警（显存不足、错误率过高等）

### 4. API服务化（中优先级）

**目标**：将系统封装为RESTful API服务

**功能**：
- FastAPI/Flask接口
- 异步任务处理
- 任务状态查询
- 结果下载接口

### 5. 配置管理优化（低优先级）

**目标**：支持多环境配置、配置热更新

**功能**：
- 环境变量支持
- 配置版本管理
- 配置验证
- 配置热重载

## 📊 当前系统能力

### 已实现
- ✅ 图像生成（Flux 1/2）
- ✅ 视频生成（HunyuanVideo 1.5）
- ✅ Prompt Engine V2（完全本地）
- ✅ 视频质量分析
- ✅ 批量处理工具
- ✅ 缓存系统

### 待优化
- ⏳ 自动质量优化
- ⏳ 批量生成效率
- ⏳ 性能监控
- ⏳ API服务化

## 🎯 推荐开发顺序

1. **视频质量自动优化**（最重要，直接影响用户体验）
2. **批量生成优化**（提升生产效率）
3. **性能监控**（保障系统稳定）
4. **API服务化**（便于集成和部署）

## 📝 使用示例

### 视频质量分析
```python
from utils.video_quality_analyzer import VideoQualityAnalyzer

analyzer = VideoQualityAnalyzer()
results = analyzer.analyze("outputs/test_novel/novel_video.mp4")
report = analyzer.generate_report(results)
print(report)
```

### 批量生成
```python
from utils.batch_processor import BatchProcessor
from generate_novel_video import NovelVideoGenerator

processor = BatchProcessor()
generator = NovelVideoGenerator()

prompts = ["提示词1", "提示词2", "提示词3"]
results = processor.process_videos(
    prompts,
    Path("outputs/batch"),
    lambda p, d: generator.generate(prompt=p, output_dir=d)
)
```

