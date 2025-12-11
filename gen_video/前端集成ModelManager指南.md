# 前端集成 ModelManager 指南

## ✅ 后端已集成 ModelManager

API 端点 `/api/v1/images/generate` 已支持 ModelManager，新增参数：

- `use_model_manager`: `true/false` - 是否使用 ModelManager
- `task`: 任务类型（ModelManager 模式）

## 🎯 前端更新建议

### 1. 添加任务类型选择器

在 `ImageGen.vue` 中添加任务类型选择：

```vue
<el-form-item label="任务类型（ModelManager）">
  <el-radio-group v-model="form.useModelManager">
    <el-radio :label="false">使用原有系统</el-radio>
    <el-radio :label="true">使用 ModelManager（推荐）</el-radio>
  </el-radio-group>
</el-form-item>

<el-form-item v-if="form.useModelManager" label="选择任务类型">
  <el-select v-model="form.task" placeholder="自动选择或手动指定">
    <el-option label="自动选择（推荐）" value="" />
    <el-option label="科普主持人脸" value="host_face" />
    <el-option label="科学背景" value="science_background" />
    <el-option label="实验室场景" value="lab_scene" />
    <el-option label="官方风格" value="official_style" />
    <el-option label="快速背景" value="fast_background" />
  </el-select>
  <div class="form-tip">
    自动选择会根据提示词自动选择最优模型
  </div>
</el-form-item>
```

### 2. 更新表单数据

```typescript
const form = reactive({
  // ... 现有字段
  useModelManager: false,  // 是否使用 ModelManager
  task: '',                // 任务类型（ModelManager 模式）
})
```

### 3. 更新 API 调用

```typescript
const handleGenerate = async () => {
  // ... 现有代码
  
  const formData = new FormData()
  formData.append('prompt', form.prompt)
  // ... 其他字段
  
  // 添加 ModelManager 参数
  if (form.useModelManager) {
    formData.append('use_model_manager', 'true')
    if (form.task) {
      formData.append('task', form.task)
    }
  }
  
  // ... 调用 API
}
```

### 4. 显示使用的模型

在结果中显示使用的模型信息：

```vue
<div v-if="result.metadata?.model_used" class="model-info">
  <el-tag type="success">使用的模型: {{ result.metadata.model_used }}</el-tag>
  <el-tag v-if="result.metadata.task" type="info">
    任务类型: {{ result.metadata.task }}
  </el-tag>
</div>
```

## 📋 任务类型说明

| 任务类型 | 使用的模型 | 说明 |
|---------|-----------|------|
| `host_face` | Kolors | 科普主持人脸 |
| `science_background` | Flux.2 | 科学背景（冲击力强） |
| `lab_scene` | Flux.1 | 实验室场景（更干净自然） |
| `official_style` | Hunyuan-DiT | 官方感科教宣传图 |
| `fast_background` | SD3 Turbo | 快速背景（批量生成） |

## 🧪 测试步骤

1. **启动 API 服务**:
   ```bash
   cd /vepfs-dev/shawn/vid/fanren/gen_video
   source /vepfs-dev/shawn/venv/py312/bin/activate
   python api/mvp_main.py
   ```

2. **启动前端**:
   ```bash
   cd gen_video/api/web_ui_vue
   npm run dev
   ```

3. **测试生成**:
   - 选择"使用 ModelManager"
   - 选择任务类型或留空（自动选择）
   - 输入提示词
   - 点击生成

## 📝 API 调用示例

### 使用 ModelManager

```bash
curl -X POST "http://localhost:8000/api/v1/images/generate" \
  -H "X-API-Key: test-key-123" \
  -F "prompt=科普主持人，专业形象" \
  -F "task=host_face" \
  -F "use_model_manager=true" \
  -F "width=1024" \
  -F "height=1024"
```

### 使用原有系统

```bash
curl -X POST "http://localhost:8000/api/v1/images/generate" \
  -H "X-API-Key: test-key-123" \
  -F "prompt=测试提示词" \
  -F "use_model_manager=false" \
  -F "width=1024" \
  -F "height=1024"
```

## ✅ 优势

1. **自动选择最优模型**: 根据任务类型自动选择
2. **统一接口**: 所有模型通过统一接口调用
3. **延迟加载**: 节省显存，只在需要时加载
4. **易于扩展**: 添加新模型只需实现 Pipeline 接口

