#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace 登录和验证工具
支持通过代理登录
"""

import os
import sys
from pathlib import Path

# 检查代理配置
HTTP_PROXY = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
HTTPS_PROXY = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')

if HTTP_PROXY:
    os.environ['HTTP_PROXY'] = HTTP_PROXY
    os.environ['http_proxy'] = HTTP_PROXY
if HTTPS_PROXY:
    os.environ['HTTPS_PROXY'] = HTTPS_PROXY
    os.environ['https_proxy'] = HTTPS_PROXY

try:
    from huggingface_hub import login, whoami, HfApi
except ImportError:
    print("✗ 未安装 huggingface_hub")
    print("  请运行: pip install huggingface_hub")
    sys.exit(1)

def check_login():
    """检查当前登录状态"""
    try:
        user = whoami()
        print("✓ 已登录 HuggingFace")
        print(f"  用户名: {user.get('name', 'unknown')}")
        print(f"  邮箱: {user.get('email', 'unknown')}")
        return True, user
    except Exception as e:
        print(f"✗ 未登录或登录已过期: {e}")
        return False, None

def do_login(token=None):
    """执行登录"""
    if token:
        # 使用提供的 token
        try:
            login(token=token, add_to_git_credential=True)
            print("✓ 登录成功！")
            return True
        except Exception as e:
            print(f"✗ 登录失败: {e}")
            return False
    else:
        # 交互式登录
        try:
            print("\n请输入您的 HuggingFace token")
            print("  获取 token: https://huggingface.co/settings/tokens")
            print("  或直接粘贴 token (以 hf_ 开头):")
            token = input("Token: ").strip()
            
            if not token:
                print("✗ 未输入 token")
                return False
            
            login(token=token, add_to_git_credential=True)
            print("✓ 登录成功！")
            return True
        except KeyboardInterrupt:
            print("\n\n✗ 登录被取消")
            return False
        except Exception as e:
            print(f"✗ 登录失败: {e}")
            return False

def main():
    print("=" * 60)
    print("HuggingFace 登录和验证工具")
    print("=" * 60)
    
    if HTTP_PROXY or HTTPS_PROXY:
        print(f"\n使用代理配置:")
        if HTTP_PROXY:
            print(f"  HTTP_PROXY={HTTP_PROXY}")
        if HTTPS_PROXY:
            print(f"  HTTPS_PROXY={HTTPS_PROXY}")
    
    # 检查当前登录状态
    print("\n[1] 检查当前登录状态...")
    is_logged_in, user = check_login()
    
    if is_logged_in:
        # 测试访问受限模型
        print("\n[2] 测试访问受限模型 pyannote/segmentation-3.0...")
        try:
            api = HfApi()
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
        
        print("\n✓ 验证完成，当前登录状态正常")
        sys.exit(0)
    
    # 需要登录
    print("\n[2] 需要登录")
    
    # 检查是否有命令行参数（token）
    if len(sys.argv) > 1:
        token = sys.argv[1]
        print(f"使用命令行提供的 token: {token[:10]}...{token[-4:]}")
        if do_login(token=token):
            # 再次检查
            check_login()
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # 交互式登录
        if do_login():
            # 再次检查
            check_login()
            sys.exit(0)
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()

