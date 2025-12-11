# 双环境配置完整指南

## 环境配置

### 主流程环境（主环境）
- **路径**: `/vepfs-dev/shawn/venv/py312`
- **用途**: 图像生成、视频生成、字幕生成等
- **Python 版本**: 3.12
- **关键依赖**: 
  - transformers >= 4.57.0（支持 Flux.1）
  - torch, torchaudio
  - diffusers
  - 其他流水线依赖

### CosyVoice 环境（子环境）
- **路径**: `/vepfs-dev/shawn/venv/cosyvoice`
- **用途**: 仅用于 CosyVoice TTS
- **Python 版本**: 3.10 或 3.12
- **关键依赖**:
  - transformers == 4.51.3
  - torch, torchaudio
  - CosyVoice 相关依赖

## 配置文件

### config.yaml 配置

```yaml
tts:
  engine: cosyvoice
  cosyvoice:
    # 启用子进程模式
    use_subprocess: true
    subprocess_python: /vepfs-dev/shawn/venv/cosyvoice/bin/python
    subprocess_script: gen_video/tools/cosyvoice_subprocess_wrapper.py
    
    # 其他 CosyVoice 配置
    model_name: CosyVoice2-0.5B
    model_path: /vepfs-dev/shawn/vid/fanren/CosyVoice/pretrained_models/CosyVoice2-0.5B
    use_cosyvoice2: true
    mode: zero_shot
    prompt_speech: /vepfs-dev/shawn/vid/fanren/gen_video/assets/prompts/haoran_prompt_5s.wav
    prompt_text: "欢迎来到本期知识探索，我们将继续介绍黑洞的基本概念"
    text_frontend: false
```

## 工作流程

### 1. 主流程启动（使用主环境）

```bash
# 激活主环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 运行主流程
cd /vepfs-dev/shawn/vid/fanren/gen_video
python main.py --script script.json --output test_video
```

### 2. TTS 生成（自动调用 CosyVoice 子环境）

当需要生成 TTS 时：
1. `TTSGenerator` 检测到 `use_subprocess=true`
2. 调用 `synthesize_cosyvoice_subprocess` 方法
3. 使用 `subprocess.run` 调用 CosyVoice 环境的 Python
4. 执行 `cosyvoice_subprocess_wrapper.py`
5. 在 CosyVoice 环境中加载模型并生成语音
6. 返回生成的音频文件路径

### 3. 流程继续（主环境）

- 主流程继续使用主环境
- 音频文件已生成，可以继续后续处理

## 验证配置

### 检查环境

```bash
# 检查主环境
source /vepfs-dev/shawn/venv/py312/bin/activate
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
# 应该显示: transformers: 4.57.1 或更高版本

# 检查 CosyVoice 环境
source /vepfs-dev/shawn/venv/cosyvoice/bin/activate
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
# 应该显示: transformers: 4.51.3
```

### 测试子进程调用

```bash
# 在主环境中测试
source /vepfs-dev/shawn/venv/py312/bin/activate
cd /vepfs-dev/shawn/vid/fanren/gen_video

python -c "
from tts_generator import TTSGenerator
tts = TTSGenerator('config.yaml')
tts.generate('测试文本', 'outputs/test_subprocess.wav')
print('✅ 子进程调用成功')
"
```

## 常见问题

### Q: 子进程调用会阻塞主流程吗？
A: 不会。`subprocess.run` 会等待子进程完成，但这是正常的同步调用。如果需要异步，可以后续优化。

### Q: 性能影响大吗？
A: 很小。子进程启动时间 < 1 秒，模型加载时间与直接调用相同。

### Q: 如何调试子进程问题？
A: 子进程的错误会通过 `stderr` 传递回主进程，可以在主进程的日志中看到。

### Q: 可以同时运行多个 TTS 任务吗？
A: 可以，但需要注意 GPU 内存。每个子进程都会加载模型，可能需要足够的 GPU 内存。

## 环境变量设置

### CosyVoice 环境（解决磁盘空间问题）

在 CosyVoice 环境的 `activate` 脚本中添加：

```bash
# 修复磁盘空间问题
export PIP_CACHE_DIR="/vepfs-dev/shawn/.pip_cache"
export TMPDIR="/vepfs-dev/shawn/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$TMPDIR"
```

运行永久修复脚本：
```bash
bash gen_video/tools/permanent_fix_pip_space.sh
```

## 完整测试流程

```bash
# 1. 激活主环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 2. 测试 TTS（会自动调用 CosyVoice 子环境）
cd /vepfs-dev/shawn/vid/fanren/gen_video
python -c "
from tts_generator import TTSGenerator
tts = TTSGenerator('config.yaml')
tts.generate('大家好，我是科普哥哥。今天我们来聊聊科学的奥秘。', 'outputs/test_dual_env.wav')
print('✅ TTS 生成成功')
"

# 3. 测试完整流程
python main.py --script outputs/test/script.json --output test_dual_env
```

