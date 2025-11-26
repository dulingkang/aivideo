#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt模块重构静态验证

不依赖运行环境，只检查代码结构和语法。
"""

import ast
import sys
from pathlib import Path


def check_syntax(file_path):
    """检查文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def check_imports(file_path):
    """检查导入语句"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return True, imports
    except Exception as e:
        return False, str(e)


def check_class_methods(file_path, class_name, method_name):
    """检查类是否有指定方法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                return method_name in methods, methods
        return False, []
    except Exception as e:
        return False, str(e)


def check_method_delegation(file_path, class_name, method_name, target):
    """检查方法是否委托给目标"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        # 检查方法体中是否包含目标字符串
                        method_source = ast.get_source_segment(content, item)
                        if method_source and target in method_source:
                            return True, method_source[:200]
        return False, None
    except Exception as e:
        return False, str(e)


def main():
    """运行静态验证"""
    print("=" * 60)
    print("Prompt模块重构静态验证")
    print("=" * 60)
    
    base_path = Path(__file__).parent
    results = []
    
    # 1. 检查 prompt 模块文件语法
    print("\n1. 检查 prompt 模块文件语法...")
    prompt_files = [
        "prompt/__init__.py",
        "prompt/token_estimator.py",
        "prompt/parser.py",
        "prompt/optimizer.py",
        "prompt/builder.py"
    ]
    
    for file_path in prompt_files:
        full_path = base_path / file_path
        if full_path.exists():
            ok, error = check_syntax(full_path)
            if ok:
                print(f"  ✓ {file_path}: 语法正确")
                results.append((f"{file_path} 语法", True))
            else:
                print(f"  ✗ {file_path}: 语法错误 - {error}")
                results.append((f"{file_path} 语法", False))
        else:
            print(f"  ✗ {file_path}: 文件不存在")
            results.append((f"{file_path} 存在", False))
    
    # 2. 检查 ImageGenerator 语法
    print("\n2. 检查 ImageGenerator 语法...")
    image_gen_path = base_path / "image_generator.py"
    if image_gen_path.exists():
        ok, error = check_syntax(image_gen_path)
        if ok:
            print(f"  ✓ image_generator.py: 语法正确")
            results.append(("image_generator.py 语法", True))
        else:
            print(f"  ✗ image_generator.py: 语法错误 - {error}")
            results.append(("image_generator.py 语法", False))
    else:
        print(f"  ✗ image_generator.py: 文件不存在")
        results.append(("image_generator.py 存在", False))
    
    # 3. 检查 prompt 模块导入
    print("\n3. 检查 prompt 模块导入...")
    init_path = base_path / "prompt" / "__init__.py"
    if init_path.exists():
        ok, imports = check_imports(init_path)
        if ok:
            print(f"  ✓ prompt/__init__.py: 导入正确")
            print(f"    导出: {', '.join(imports)}")
            results.append(("prompt 模块导入", True))
        else:
            print(f"  ✗ prompt/__init__.py: 导入错误 - {imports}")
            results.append(("prompt 模块导入", False))
    
    # 4. 检查 ImageGenerator.build_prompt 方法
    print("\n4. 检查 ImageGenerator.build_prompt 方法...")
    if image_gen_path.exists():
        has_method, methods = check_class_methods(image_gen_path, "ImageGenerator", "build_prompt")
        if has_method:
            print(f"  ✓ build_prompt 方法存在")
            results.append(("build_prompt 方法存在", True))
            
            # 检查是否委托给 PromptBuilder
            delegated, source = check_method_delegation(
                image_gen_path, "ImageGenerator", "build_prompt", "self.prompt_builder.build"
            )
            if delegated:
                print(f"  ✓ build_prompt 已委托给 PromptBuilder")
                results.append(("build_prompt 委托", True))
            else:
                print(f"  ✗ build_prompt 未委托给 PromptBuilder")
                results.append(("build_prompt 委托", False))
        else:
            print(f"  ✗ build_prompt 方法不存在")
            results.append(("build_prompt 方法存在", False))
    
    # 5. 检查 PromptBuilder.build 方法
    print("\n5. 检查 PromptBuilder.build 方法...")
    builder_path = base_path / "prompt" / "builder.py"
    if builder_path.exists():
        has_method, methods = check_class_methods(builder_path, "PromptBuilder", "build")
        if has_method:
            print(f"  ✓ PromptBuilder.build 方法存在")
            results.append(("PromptBuilder.build 方法", True))
        else:
            print(f"  ✗ PromptBuilder.build 方法不存在")
            results.append(("PromptBuilder.build 方法", False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计: {passed}/{total} 验证通过")
    
    if passed == total:
        print("\n🎉 所有静态验证通过！代码结构正确！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个验证失败，需要修复")
        return 1


if __name__ == "__main__":
    sys.exit(main())








