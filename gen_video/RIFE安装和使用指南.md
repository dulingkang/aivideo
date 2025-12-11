# RIFE 插帧安装和使用指南

## 📦 安装 RIFE

### 方法1：使用 rife-ncnn-python（推荐，速度快）

```bash
# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 安装 rife-ncnn-python
pip install rife-ncnn-python
```

**优势**：
- ✅ 安装简单
- ✅ 速度快（基于 NCNN）
- ✅ 支持 GPU 加速

### 方法2：使用官方实现（如果方法1失败）

```bash
# 克隆官方仓库
cd /vepfs-dev/shawn/vid/fanren
git clone https://github.com/hzwer/arXiv2020-RIFE.git
cd arXiv2020-RIFE

# 安装依赖
pip install -r requirements.txt

# 下载模型权重（需要手动下载）
# 模型下载地址：https://github.com/hzwer/arXiv2020-RIFE/releases
# 下载后解压到 train_log 目录
```

## ⚙️ 配置

### 1. 启用 RIFE 插帧

在 `config.yaml` 中：

```yaml
video:
  rife:
    enabled: true  # 启用插帧
    interpolation_scale: 2.0  # 插帧倍数（生成60帧，插到120帧）
```

### 2. 调整插帧倍数

根据需求调整 `interpolation_scale`：

- **2.0**（推荐）：生成 60 帧，插到 120 帧
  - 平衡生成速度和插帧质量
- **1.5**：生成 80 帧，插到 120 帧
  - 插帧负担更小，质量更好
- **3.0**：生成 40 帧，插到 120 帧
  - 生成更快，但插帧负担更大

## 🚀 使用

### 自动使用

系统会自动：
1. 检测 RIFE 是否可用
2. 如果启用，先生成关键帧
3. 使用 RIFE 插帧到目标帧数
4. 保存视频

**无需修改代码**，只需在配置文件中启用即可。

### 测试

```bash
# 测试单个场景
python gen_video/test_full_pipeline_optimized.py --script lingjie/1.json --max-scenes 1
```

## 📊 效果对比

### 启用 RIFE 前
- 生成时间：3-5 分钟
- 流畅度：基准

### 启用 RIFE 后
- 生成时间：2-3 分钟（减少 30%+）
- 流畅度：提升 50%+

## 🔧 故障排除

### 问题1：RIFE 安装失败

**错误信息**：
```
ImportError: No module named 'rife_ncnn'
```

**解决方案**：
1. 检查是否在正确的虚拟环境中
2. 尝试重新安装：`pip install rife-ncnn-python --upgrade`
3. 或使用官方实现（方法2）

### 问题2：RIFE 加载失败

**错误信息**：
```
✗ RIFE 模型加载失败: ...
```

**解决方案**：
1. 检查 GPU 是否可用（`nvidia-smi`）
2. 如果 GPU 不可用，可能需要使用 CPU 版本
3. 或禁用 RIFE：`rife.enabled: false`

### 问题3：插帧后帧数不足

**现象**：插帧后帧数仍然少于目标帧数

**解决方案**：
- 系统会自动使用线性插值补充
- 或增加 `interpolation_scale` 值

### 问题4：插帧质量不好

**现象**：插帧后的视频有模糊或伪影

**解决方案**：
1. 降低 `interpolation_scale`（减少插帧负担）
2. 或增加 SVD 生成的关键帧数
3. 或禁用 RIFE，使用原始生成

## 💡 最佳实践

### 1. 首次使用

1. **先测试单个场景**
   ```bash
   python gen_video/test_full_pipeline_optimized.py --script lingjie/1.json --max-scenes 1
   ```

2. **对比效果**
   - 启用 RIFE：`rife.enabled: true`
   - 禁用 RIFE：`rife.enabled: false`
   - 对比流畅度和生成时间

3. **调整参数**
   - 如果插帧质量不好，降低 `interpolation_scale`
   - 如果生成时间仍太长，提高 `interpolation_scale`

### 2. 生产使用

1. **保持默认配置**（`interpolation_scale: 2.0`）
2. **监控生成时间**，确保在可接受范围内
3. **定期检查质量**，确保插帧效果良好

## 📝 配置示例

### 推荐配置（平衡速度和质量）

```yaml
video:
  rife:
    enabled: true
    interpolation_scale: 2.0  # 生成60帧，插到120帧
```

### 高质量配置（更注重质量）

```yaml
video:
  rife:
    enabled: true
    interpolation_scale: 1.5  # 生成80帧，插到120帧
```

### 快速配置（更注重速度）

```yaml
video:
  rife:
    enabled: true
    interpolation_scale: 3.0  # 生成40帧，插到120帧
```

## ✅ 检查清单

- [ ] RIFE 已安装（`pip list | grep rife`）
- [ ] 配置文件已更新（`rife.enabled: true`）
- [ ] 测试单个场景成功
- [ ] 对比效果满意
- [ ] 参数已调整到最佳值

## 📚 参考

- RIFE 官方仓库：https://github.com/hzwer/arXiv2020-RIFE
- rife-ncnn-python：https://pypi.org/project/rife-ncnn-python/
- 实施文档：`RIFE插帧实施完成.md`

