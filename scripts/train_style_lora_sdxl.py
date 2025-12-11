#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练风格LoRA脚本
直接使用train_dreambooth_lora_sdxl.py，只是作为入口点
"""

import sys
from pathlib import Path

# 直接执行原始训练脚本
if __name__ == "__main__":
    # 获取原始训练脚本路径
    scripts_dir = Path(__file__).parent
    original_script = scripts_dir / "train_dreambooth_lora_sdxl.py"
    
    # 直接执行原始脚本，传递所有参数
    import subprocess
    import sys
    
    # 使用当前Python解释器执行原始脚本
    cmd = [sys.executable, str(original_script)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))

