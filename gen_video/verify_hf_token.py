#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 HuggingFace token 是否有效"""

import os
import sys
from pathlib import Path

# 尝试导入 huggingface_hub
try:
    from huggingface_hub import whoami, HfApi
except ImportError:
    print("✗ 未安装 huggingface_hub")
    print("  请运行: pip install huggingface_hub")
    sys.exit(1)

def get_token():
    """获取 HuggingFace token"""
    # 方法1: 从环境变量
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token
    
    # 方法2: 从 token 文件
    token_paths = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]
    
    for token_path in token_paths:
        if token_path.exists():
            try:
                token = token_path.read_text().strip()
                if token:
                    return token
            except Exception:
                continue
    
    return None

def verify_token(token):
    """验证 token 是否有效"""
    try:
        api = HfApi(token=token)
        user = api.whoami()
        return True, user
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("验证 HuggingFace Token")
    print("=" * 60)
    
    # 获取 token
    token = get_token()
    if not token:
        print("\n✗ 未找到 HuggingFace token")
        print("\n请设置 token:")
        print("  方法1: 设置环境变量")
        print("    export HF_TOKEN='your_token_here'")
        print("  方法2: 使用登录命令")
        print("    huggingface-cli login")
        print("    或")
        print("    python3 -c \"from huggingface_hub import login; login()\"")
        sys.exit(1)
    
    print(f"\n找到 token: {token[:10]}...{token[-4:]}")
    
    # 验证 token
    print("\n正在验证 token...")
    is_valid, result = verify_token(token)
    
    if is_valid:
        user = result
        print("\n✓ Token 有效！")
        print(f"  用户名: {user.get('name', 'unknown')}")
        print(f"  邮箱: {user.get('email', 'unknown')}")
        print(f"  组织: {', '.join(user.get('orgs', [])) if user.get('orgs') else '无'}")
        
        # 测试访问受限模型
        print("\n测试访问受限模型 pyannote/segmentation-3.0...")
        try:
            api = HfApi(token=token)
            # 尝试获取模型信息
            model_info = api.model_info("pyannote/segmentation-3.0")
            print("✓ 可以访问受限模型")
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "gated" in error_msg.lower():
                print("⚠ 需要接受使用条款")
                print(f"  请访问: https://huggingface.co/pyannote/segmentation-3.0")
                print(f"  点击 'Agree and access repository' 接受使用条款")
            else:
                print(f"⚠ 访问受限模型时出错: {e}")
        
        sys.exit(0)
    else:
        print(f"\n✗ Token 无效或已过期")
        print(f"  错误: {result}")
        print("\n请重新登录:")
        print("  huggingface-cli login")
        print("  或")
        print("  python3 -c \"from huggingface_hub import login; login()\"")
        sys.exit(1)

if __name__ == "__main__":
    main()

