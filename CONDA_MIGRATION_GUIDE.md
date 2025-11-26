# 从 Pip 虚拟环境迁移到 Conda 环境指南

## 当前环境信息

- **Python 版本**: 3.12
- **虚拟环境路径**: `/vepfs-dev/shawn/venv/py312/bin/activate`
- **PyTorch 版本**: 2.8.0+cu126 (CUDA 12.6)
- **FAISS**: 当前使用 faiss-cpu，需要迁移到 faiss-gpu

## 迁移步骤

### 1. 创建 Conda 环境

```bash
# 使用 environment.yml 创建环境
conda env create -f environment.yml

# 或者手动创建（如果遇到依赖冲突）
conda create -n fanren python=3.12
conda activate fanren
```

### 2. 安装 FAISS GPU（关键步骤）

FAISS GPU 在 conda 中更容易安装：

```bash
# 激活环境
conda activate fanren

# 安装 faiss-gpu（conda-forge 提供）
conda install -c conda-forge faiss-gpu=1.7.4

# 验证安装
python -c "import faiss; print(faiss.get_num_gpus())"
```

**注意**: 
- conda-forge 的 faiss-gpu 版本可能与你需要的版本不同
- 如果 1.7.4 不可用，可以尝试: `conda install -c conda-forge faiss-gpu`
- 或者使用 pip 安装: `pip install faiss-gpu`（但 conda 方式更推荐）

### 3. 安装其他依赖

```bash
# 激活环境
conda activate fanren

# 安装 pip 依赖
pip install -r gen_video/requirements.txt

# 如果需要 CosyVoice 的依赖
pip install -r CosyVoice/requirements.txt
```

### 4. 验证环境

```bash
# 检查 Python 版本
python --version  # 应该是 3.12.x

# 检查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查 FAISS GPU
python -c "import faiss; print(f'FAISS GPU devices: {faiss.get_num_gpus()}')"

# 检查其他关键包
python -c "import transformers, diffusers, whisperx, modelscope; print('All packages imported successfully')"
```

## FAISS GPU 安装说明

### 为什么使用 Conda 安装 FAISS GPU？

1. **依赖管理更好**: conda 会自动处理 CUDA、cuDNN 等依赖
2. **编译问题更少**: conda 提供预编译的二进制包
3. **版本兼容性**: conda 会确保与 CUDA 版本匹配

### 如果 conda 安装失败

如果 conda 安装 faiss-gpu 遇到问题，可以尝试：

```bash
# 方法1: 使用 pip（需要确保 CUDA 环境正确）
pip install faiss-gpu

# 方法2: 从源码编译（最可靠但最复杂）
# 参考: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
```

### FAISS 版本选择

- **faiss-gpu 1.7.4**: conda-forge 提供的稳定版本
- **faiss-gpu 最新版**: 可能需要从源码编译
- **faiss-cpu**: 如果 GPU 版本有问题，可以先用 CPU 版本测试

## 环境对比

| 项目 | Pip 环境 | Conda 环境 |
|------|---------|-----------|
| Python | 3.12 | 3.12 |
| PyTorch | 2.8.0+cu126 | 2.8.0 |
| FAISS | faiss-cpu 1.13.0 | faiss-gpu 1.7.4 |
| CUDA | 12.6 | 12.6 |

## 常见问题

### Q1: conda 安装 faiss-gpu 时提示找不到包

```bash
# 更新 conda
conda update conda

# 添加 conda-forge 通道
conda config --add channels conda-forge

# 搜索可用版本
conda search faiss-gpu -c conda-forge
```

### Q2: FAISS GPU 无法检测到 GPU

```bash
# 检查 CUDA 版本
nvidia-smi

# 检查 PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

# 确保 FAISS 和 PyTorch 使用相同的 CUDA 版本
```

### Q3: 依赖冲突

如果遇到依赖冲突：

```bash
# 方法1: 使用 mamba（更快的依赖解析器）
conda install mamba -c conda-forge
mamba env create -f environment.yml

# 方法2: 分步安装
conda install python=3.12 faiss-gpu -c conda-forge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r gen_video/requirements.txt
```

## 迁移后测试

```bash
# 测试 FAISS GPU
python -c "
import faiss
import numpy as np

# 创建测试数据
d = 64
nb = 1000
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')

# 创建 GPU 索引
res = faiss.StandardGpuResources()
index = faiss.IndexFlatL2(d)
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

# 添加数据
gpu_index.add(xb)

print('✓ FAISS GPU 测试成功')
"

# 测试项目代码
cd /vepfs-dev/shawn/vid/fanren
python tools/video_processing/build_index.py --help
```

## 备份和恢复

### 导出当前环境（迁移前）

```bash
# 导出 pip 环境
source /vepfs-dev/shawn/venv/py312/bin/activate
pip freeze > requirements_backup.txt
```

### 导出 conda 环境（迁移后）

```bash
conda activate fanren
conda env export > environment_backup.yml
```

## 下一步

1. ✅ 创建 conda 环境
2. ✅ 安装 faiss-gpu
3. ✅ 安装其他依赖
4. ✅ 测试环境
5. ✅ 更新项目脚本中的激活路径

## 更新脚本中的环境路径

如果项目中有脚本硬编码了虚拟环境路径，需要更新：

```bash
# 查找需要更新的文件
grep -r "/vepfs-dev/shawn/venv/py312" .

# 替换为 conda 环境
# 旧: source /vepfs-dev/shawn/venv/py312/bin/activate
# 新: conda activate fanren
```

