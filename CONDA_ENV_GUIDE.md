# Conda 环境管理指南

## 使用 Conda 的优势

✅ **易于迁移和恢复**：通过 `environment.yml` 文件可以完整恢复环境  
✅ **跨平台兼容**：Linux、Windows、macOS 都可以使用  
✅ **依赖管理更可靠**：conda 会处理复杂的依赖关系  
✅ **版本锁定**：可以精确控制所有包的版本  

## 创建和导出环境

### 1. 创建新环境（如果还没有）

```bash
# 使用 environment.yml 创建环境
conda env create -f environment.yml

# 或者手动创建并安装
conda create -n fanren python=3.10
conda activate fanren
pip install -r gen_video/requirements.txt
```

### 2. 导出当前环境配置

```bash
# 激活环境
conda activate fanren

# 导出完整环境配置（推荐，包含所有依赖）
conda env export > environment.yml

# 或者只导出手动安装的包（更简洁）
conda env export --from-history > environment.yml
```

### 3. 导出时排除构建信息（可选，使文件更简洁）

```bash
conda env export | grep -v "^prefix: " > environment.yml
```

## 在新环境中恢复

### 方法1：使用 environment.yml（推荐）

```bash
# 在新机器上直接创建环境
conda env create -f environment.yml

# 激活环境
conda activate fanren
```

### 方法2：如果遇到依赖冲突

```bash
# 先创建基础环境
conda create -n fanren python=3.10

# 激活环境
conda activate fanren

# 然后安装依赖
pip install -r gen_video/requirements.txt
```

## 更新环境

```bash
# 激活环境
conda activate fanren

# 安装新包后，更新 environment.yml
conda env export > environment.yml
```

## 常用命令

```bash
# 查看所有环境
conda env list

# 删除环境
conda env remove -n fanren

# 克隆环境（用于备份）
conda create --name fanren_backup --clone fanren

# 查看环境中安装的包
conda list

# 导出为 requirements.txt（如果需要）
pip freeze > requirements_conda.txt
```

## 注意事项

1. **CUDA 版本**：根据你的 GPU 和 CUDA 版本调整 `cudatoolkit` 版本
2. **Python 版本**：建议固定 Python 版本（如 3.10）以确保兼容性
3. **通道优先级**：`channels` 的顺序很重要，conda 会按顺序查找包
4. **pip 包**：某些包只能通过 pip 安装，放在 `pip:` 下

## 迁移检查清单

- [ ] 导出 `environment.yml` 文件
- [ ] 检查 Python 版本是否匹配
- [ ] 确认 CUDA 版本兼容性
- [ ] 备份自定义配置文件
- [ ] 记录特殊安装步骤（如从 GitHub 安装的包）

