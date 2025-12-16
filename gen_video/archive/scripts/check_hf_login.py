#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 HuggingFace 登录状态"""

from huggingface_hub import whoami
import sys

try:
    user = whoami()
    print("✓ 已登录 HuggingFace")
    print(f"  用户名: {user.get('name', 'unknown')}")
    print(f"  邮箱: {user.get('email', 'unknown')}")
    print(f"  组织: {user.get('orgs', [])}")
    sys.exit(0)
except Exception as e:
    print(f"✗ 未登录或登录已过期")
    print(f"  错误: {e}")
    print("\n需要登录，请运行:")
    print("  python3 -c \"from huggingface_hub import login; login()\"")
    print("  或")
    print("  huggingface-cli login")
    sys.exit(1)

