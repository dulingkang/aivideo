# 双环境设置指南

## 问题
- CosyVoice 需要 transformers 4.51.3
- 其他流水线需要更新版本的 transformers（支持 Flux.1 等）
- 两个版本不兼容，无法在同一环境中共存

## 解决方案：双环境架构

### 环境1：主流水线环境（主环境）
- **用途**: 图像生成、视频生成、字幕生成等
- **Python 版本**: 3.12
- **关键依赖**:
  - transformers >= 4.57.0（支持 Flux.1）
  - torch, torchaudio
  - diffusers
  - 其他流水线依赖

### 环境2：CosyVoice 环境（子环境）
- **用途**: 仅用于 CosyVoice TTS
- **Python 版本**: 3.10 或 3.12
- **关键依赖**:
  - transformers == 4.51.3
  - torch, torchaudio
  - CosyVoice 相关依赖

## 实施步骤

### 1. 创建 CosyVoice 独立环境

```bash
# 创建新的虚拟环境（推荐使用 Python 3.10）
cd /vepfs-dev/shawn/vid/fanren
python3.10 -m venv venv_cosyvoice

# 激活环境
source venv_cosyvoice/bin/activate

# 安装 CosyVoice 依赖
cd CosyVoice
pip install -r requirements.txt

# 安装特定版本的 transformers
pip install transformers==4.51.3 --force-reinstall

# 验证安装
python -c "import transformers; print(transformers.__version__)"
# 应该输出: 4.51.3
```

### 2. 修改 TTS Generator 使用子进程

修改 `gen_video/tts_generator.py`，添加子进程调用选项：

```python
# 在 config.yaml 中添加配置
# tts:
#   cosyvoice:
#     use_subprocess: true  # 使用子进程模式
#     subprocess_python: /vepfs-dev/shawn/vid/fanren/venv_cosyvoice/bin/python
#     subprocess_script: gen_video/tools/cosyvoice_subprocess_wrapper.py
```

### 3. 测试子进程调用

```bash
# 在主环境中测试
cd /vepfs-dev/shawn/vid/fanren/gen_video
python tools/cosyvoice_subprocess_wrapper.py \
  --text "测试文本" \
  --output test_output.wav \
  --prompt-speech assets/prompts/haoran_prompt_5s.wav \
  --prompt-text "欢迎来到本期知识探索，我们将继续介绍黑洞的基本概念" \
  --mode zero_shot
```

## 优势

1. **版本隔离**: 两个环境完全独立，不会相互影响
2. **稳定性**: CosyVoice 环境专门优化，不会因为其他依赖更新而破坏
3. **灵活性**: 可以独立更新两个环境的依赖
4. **可维护性**: 问题隔离，更容易调试

## 注意事项

1. **性能**: 子进程调用会有轻微的性能开销（启动时间），但通常可以忽略
2. **路径**: 确保所有路径都是绝对路径，避免相对路径问题
3. **错误处理**: 子进程的错误需要正确传递回主进程
4. **资源**: 两个环境会占用更多磁盘空间

## 验证

运行测试脚本验证两个环境都正常工作：

```bash
# 测试主环境
python gen_video/test_stage1.py

# 测试 CosyVoice 环境（通过子进程）
python gen_video/tools/test_cosyvoice_subprocess.py
```

