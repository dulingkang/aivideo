#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查GPU显存使用情况
"""

import subprocess
import re

def check_gpu_memory():
    """检查GPU显存使用情况"""
    print("=" * 60)
    print("GPU显存使用情况检查")
    print("=" * 60)
    
    # 运行nvidia-smi
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("无法运行nvidia-smi")
        return
    
    lines = result.stdout.strip().split('\n')
    
    total_memory = 0
    processes = []
    
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(',')
        if len(parts) >= 3:
            pid = parts[0].strip()
            process_name = parts[1].strip()
            memory_str = parts[2].strip()
            
            # 解析内存大小
            memory_match = re.search(r'(\d+\.?\d*)\s*(MiB|GiB)', memory_str)
            if memory_match:
                memory_value = float(memory_match.group(1))
                memory_unit = memory_match.group(2)
                
                # 转换为MiB
                if memory_unit == 'GiB':
                    memory_mib = memory_value * 1024
                else:
                    memory_mib = memory_value
                
                total_memory += memory_mib
                processes.append({
                    'pid': pid,
                    'name': process_name,
                    'memory_mib': memory_mib,
                    'memory_gib': memory_mib / 1024
                })
    
    # 获取GPU总显存
    result2 = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader'], 
                           capture_output=True, text=True)
    if result2.returncode == 0:
        total_gpu_memory = result2.stdout.strip()
        print(f"\nGPU总显存: {total_gpu_memory}")
    
    print(f"\n占用显存的进程:")
    print(f"{'PID':<10} {'进程名':<30} {'显存占用':<15}")
    print("-" * 60)
    
    # 按显存占用排序
    processes.sort(key=lambda x: x['memory_mib'], reverse=True)
    
    for proc in processes:
        print(f"{proc['pid']:<10} {proc['name'][:28]:<30} {proc['memory_gib']:.2f} GiB")
    
    print("-" * 60)
    print(f"{'总计':<10} {'':<30} {total_memory/1024:.2f} GiB")
    
    # 检查是否有足够的显存
    if total_memory > 0:
        # 假设总显存是95GB
        total_gpu_gb = 95.22
        used_gb = total_memory / 1024
        free_gb = total_gpu_gb - used_gb
        
        print(f"\n显存使用情况:")
        print(f"  已使用: {used_gb:.2f} GB")
        print(f"  可用: {free_gb:.2f} GB")
        print(f"  使用率: {used_gb/total_gpu_gb*100:.1f}%")
        
        if free_gb < 20:
            print(f"\n⚠️  警告: 可用显存不足20GB，HunyuanVideo可能需要更多显存")
            print(f"   建议:")
            print(f"   1. 等待其他进程完成")
            print(f"   2. 停止占用显存的进程（如果可能）")
            print(f"   3. 使用480p模型而不是720p模型")
    
    print()

if __name__ == "__main__":
    check_gpu_memory()

