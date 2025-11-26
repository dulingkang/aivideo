# AI视频生成系统

基于 Stability-AI generative-models 的完整本地AI视频生成流水线

## 功能特性

- 🎬 **视频生成**: 使用 Stable Video Diffusion (SVD) 将图像转换为视频
- 🎙️ **配音生成**: 支持 ChatTTS、OpenVoice、Coqui TTS
- 📝 **字幕生成**: 使用 WhisperX 进行语音识别和字幕对齐
- 🎞️ **视频合成**: 自动拼接视频片段、添加音频、字幕、背景音乐
- 📋 **脚本解析**: 支持 Markdown 格式的分镜脚本

## 安装步骤

### 1. 安装依赖

```bash
cd gen_video
pip install -r requirements.txt
```

### 2. 安装 Stability-AI generative-models

```bash
# 克隆仓库
cd ..
git clone https://github.com/Stability-AI/generative-models.git
cd generative-models

# 安装依赖
pip install -r requirements/pt2.txt
pip install .
```

### 3. 下载模型

#### SVD 模型（视频生成）

```bash
# 使用 huggingface-cli 下载
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt
```

或者从 [HuggingFace](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) 手动下载。

#### WhisperX 模型（字幕生成）

模型会在首次运行时自动下载。

#### ChatTTS 模型（配音生成）

模型会在首次运行时自动下载。

### 4. 安装 FFmpeg（视频合成）

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# 或使用 conda
conda install -c conda-forge ffmpeg
```

## 配置

编辑 `config.yaml` 文件，配置路径、模型参数等。

主要配置项：
- `paths`: 输入/输出目录路径
- `video`: 视频生成参数（模型、帧数、分辨率等）
- `tts`: TTS 引擎和参数
- `subtitle`: 字幕生成参数
- `composition`: 视频合成参数

## 使用方法

### 方法1: 使用运行脚本（推荐）

```bash
cd gen_video
python run_pipeline.py \
    --markdown ../灵界/2.md \
    --image-dir ../灵界/img2/jpgsrc \
    --output lingjie_ep2
```

### 方法2: 分步执行

#### 1. 解析脚本

```bash
python script_parser.py \
    --markdown ../灵界/2.md \
    --image-dir ../灵界/img2/jpgsrc \
    --output temp/script.json
```

#### 2. 生成视频片段

```bash
python video_generator.py \
    --input ../灵界/img2/jpgsrc \
    --output outputs/videos
```

#### 3. 生成配音

```bash
python tts_generator.py \
    --text "你的旁白文本" \
    --output outputs/audio.wav
```

#### 4. 生成字幕

```bash
python subtitle_generator.py \
    --audio outputs/audio.wav \
    --output outputs/subtitle.srt
```

#### 5. 合成最终视频

```bash
python video_composer.py \
    --videos outputs/videos/*.mp4 \
    --audio outputs/audio.wav \
    --subtitle outputs/subtitle.srt \
    --output outputs/final.mp4
```

### 方法3: 使用主程序

```bash
python main.py \
    --markdown ../灵界/2.md \
    --image-dir ../灵界/img2/jpgsrc \
    --output lingjie_ep2
```

## 脚本格式

系统支持 Markdown 格式的分镜脚本，包含：
- 场景描述表格
- 旁白表格
- 开场白和结束语

示例格式见 `../灵界/2.md`

## 输出结构

```
outputs/
├── lingjie_ep2/
│   ├── videos/          # 视频片段
│   │   ├── scene_001.mp4
│   │   ├── scene_002.mp4
│   │   └── ...
│   ├── audio.wav        # 配音
│   ├── subtitle.srt     # 字幕
│   └── lingjie_ep2.mp4  # 最终视频
└── ...
```

## 常见问题

### 1. 模型加载失败

确保已正确安装 generative-models 并设置环境变量：
```bash
export GENERATIVE_MODELS_PATH=/path/to/generative-models
```

### 2. CUDA 内存不足

在 `config.yaml` 中调整：
- 减小 `batch_size`
- 启用 `memory_efficient: true`
- 使用 `mixed_precision: fp16`

### 3. FFmpeg 错误

确保已安装 FFmpeg 并可在命令行中访问：
```bash
ffmpeg -version
```

### 4. 字幕显示问题

检查字幕文件格式和字体配置：
- 确保字幕文件是 UTF-8 编码
- 检查系统是否有所需字体（如 SimHei）

## 性能优化

### GPU 加速

- 确保使用 CUDA 设备
- 在 `config.yaml` 中设置 `device_id: 0`

### 批量处理

- 调整 `batch_size` 参数
- 使用多进程处理（`num_workers`）

### 内存优化

- 启用 `memory_efficient: true`
- 使用 `mixed_precision: fp16`
- 分批处理大量场景

## 许可证

本项目使用 MIT 许可证。模型使用需遵循各自的许可证：
- Stability AI models: [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)
- WhisperX: MIT License
- ChatTTS: MIT License

## 参考链接

- [Stability-AI generative-models](https://github.com/Stability-AI/generative-models)
- [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)
- [WhisperX](https://github.com/m-bain/whisperX)
- [ChatTTS](https://github.com/2noise/ChatTTS)

