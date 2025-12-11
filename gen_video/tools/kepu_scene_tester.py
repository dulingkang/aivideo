#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科普场景测试脚本
用于批量生成测试场景，验证场景提示词库的质量
"""

import os
import sys
import yaml
import argparse
import gc
from pathlib import Path
from typing import List, Dict, Optional
import json
from tqdm import tqdm
import torch

# 添加gen_video路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_generator import ImageGenerator


class KepuSceneTester:
    """科普场景测试器"""
    
    def __init__(self, config_path: str = "config.yaml", model_engine: Optional[str] = None):
        """初始化测试器"""
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (Path(__file__).parent.parent / self.config_path).resolve()
        
        # 加载场景提示词库
        scenes_yaml_path = Path(__file__).parent.parent / "prompt" / "kepu_scenes.yaml"
        with open(scenes_yaml_path, 'r', encoding='utf-8') as f:
            self.scenes_config = yaml.safe_load(f)
        
        self.kepu_scenes = self.scenes_config.get('kepu_scenes', {})
        self.kepu_style = self.scenes_config.get('kepu_style', {})
        
        # 保存模型引擎选择
        self.model_engine = model_engine
        
        # 初始化图像生成器
        print("初始化图像生成器...")
        self.image_generator = ImageGenerator(str(self.config_path))
        
        # 如果指定了模型引擎，修改配置
        if self.model_engine:
            print(f"使用指定模型引擎: {self.model_engine}")
            # 修改配置中的引擎
            import yaml as yaml_module
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml_module.safe_load(f)
            config['image']['engine'] = self.model_engine
            # 重新初始化（如果需要）
            self.image_generator.engine = self.model_engine
        
        # 创建输出目录
        self.output_dir = Path(__file__).parent.parent / "outputs" / "kepu_scene_tests"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def cleanup_memory(self):
        """清理显存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    def get_memory_info(self) -> str:
        """获取显存使用信息"""
        if not torch.cuda.is_available():
            return "CUDA不可用"
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        
        return f"已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB"
    
    def test_all_scenes(self, num_examples: int = 2):
        """测试所有场景类型"""
        print(f"\n开始测试所有场景类型，每个类型生成 {num_examples} 个示例...")
        if self.model_engine:
            print(f"使用模型引擎: {self.model_engine}")
        print(f"当前显存状态: {self.get_memory_info()}")
        
        results = {}
        
        for scene_type, scene_config in self.kepu_scenes.items():
            print(f"\n{'='*60}")
            print(f"测试场景类型: {scene_config['name']} ({scene_type})")
            print(f"{'='*60}")
            print(f"显存状态: {self.get_memory_info()}")
            
            scene_results = self.test_scene_type(scene_type, scene_config, num_examples)
            results[scene_type] = scene_results
            
            # 每个场景类型测试完成后清理显存
            print(f"  清理显存...")
            self.cleanup_memory()
            print(f"  清理后显存状态: {self.get_memory_info()}")
        
        # 生成测试报告
        self.generate_test_report(results)
        
        return results
    
    def test_scene_type(self, scene_type: str, scene_config: Dict, num_examples: int = 2) -> List[Dict]:
        """测试单个场景类型"""
        scene_name = scene_config['name']
        examples = scene_config.get('examples', [])
        
        if not examples:
            print(f"  ⚠ 场景类型 {scene_name} 没有示例提示词")
            return []
        
        # 选择要测试的示例
        test_examples = examples[:num_examples] if len(examples) >= num_examples else examples
        
        results = []
        
        for idx, example_prompt in enumerate(test_examples, 1):
            print(f"\n  测试示例 {idx}/{len(test_examples)}: {example_prompt[:50]}...")
            
            # 构建完整提示词
            base_prompt = scene_config.get('base_prompt', '')
            full_prompt = f"{base_prompt}, {example_prompt}" if base_prompt else example_prompt
            
            # 添加通用风格
            base_style = self.kepu_style.get('base_style', '')
            if base_style:
                full_prompt = f"{full_prompt}, {base_style}"
            
            # 构建负面提示词
            negative_prompt = scene_config.get('negative_prompt', '')
            common_negative = self.kepu_style.get('common_negative', '')
            full_negative = f"{negative_prompt}, {common_negative}" if negative_prompt else common_negative
            
            # 生成图像
            output_filename = f"{scene_type}_example_{idx}.png"
            output_path = self.output_dir / scene_type / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                print(f"    生成图像: {output_path.name}")
                
                # 生成参数
                generate_kwargs = {
                    'prompt': full_prompt,
                    'output_path': output_path,
                    'negative_prompt': full_negative,
                    'num_inference_steps': 20,
                    'guidance_scale': 7.5,
                    'task_type': "scene"
                }
                
                # 如果指定了模型引擎，使用它
                if self.model_engine:
                    generate_kwargs['model_engine'] = self.model_engine
                
                generated_path = self.image_generator.generate_image(**generate_kwargs)
                
                result = {
                    'scene_type': scene_type,
                    'scene_name': scene_name,
                    'example_index': idx,
                    'prompt': example_prompt,
                    'full_prompt': full_prompt,
                    'negative_prompt': full_negative,
                    'output_path': str(generated_path),
                    'status': 'success'
                }
                
                print(f"    ✅ 生成成功: {generated_path}")
                results.append(result)
                
                # 清理显存
                self.cleanup_memory()
                
            except Exception as e:
                print(f"    ❌ 生成失败: {e}")
                result = {
                    'scene_type': scene_type,
                    'scene_name': scene_name,
                    'example_index': idx,
                    'prompt': example_prompt,
                    'full_prompt': full_prompt,
                    'negative_prompt': full_negative,
                    'output_path': None,
                    'status': 'failed',
                    'error': str(e)
                }
                results.append(result)
                
                # 清理显存（即使失败也要清理）
                self.cleanup_memory()
        
        return results
    
    def test_specific_scene(self, scene_type: str, example_index: int = 0):
        """测试特定场景的特定示例"""
        if scene_type not in self.kepu_scenes:
            print(f"❌ 场景类型 {scene_type} 不存在")
            return None
        
        scene_config = self.kepu_scenes[scene_type]
        examples = scene_config.get('examples', [])
        
        if example_index >= len(examples):
            print(f"❌ 示例索引 {example_index} 超出范围（共 {len(examples)} 个示例）")
            return None
        
        example_prompt = examples[example_index]
        print(f"\n测试场景: {scene_config['name']} - 示例 {example_index + 1}")
        print(f"提示词: {example_prompt}")
        
        results = self.test_scene_type(scene_type, scene_config, num_examples=1)
        return results[0] if results else None
    
    def generate_test_report(self, results: Dict):
        """生成测试报告"""
        report_path = self.output_dir / "test_report.json"
        
        # 统计信息
        total_tests = 0
        successful_tests = 0
        failed_tests = 0
        
        for scene_type, scene_results in results.items():
            total_tests += len(scene_results)
            for result in scene_results:
                if result['status'] == 'success':
                    successful_tests += 1
                else:
                    failed_tests += 1
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'successful': successful_tests,
                'failed': failed_tests,
                'success_rate': f"{(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
            },
            'results': results
        }
        
        # 保存JSON报告
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print("测试报告")
        print(f"{'='*60}")
        print(f"总测试数: {total_tests}")
        print(f"成功: {successful_tests}")
        print(f"失败: {failed_tests}")
        print(f"成功率: {report['summary']['success_rate']}")
        print(f"\n详细报告已保存到: {report_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description='科普场景测试脚本')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--scene-type', type=str, default=None,
                       help='测试特定场景类型（如: universe, quantum, earth等）')
    parser.add_argument('--example-index', type=int, default=0,
                       help='测试特定示例索引（仅当指定--scene-type时有效）')
    parser.add_argument('--num-examples', type=int, default=2,
                       help='每个场景类型测试的示例数量（默认: 2）')
    parser.add_argument('--all', action='store_true',
                       help='测试所有场景类型')
    parser.add_argument('--model-engine', type=str, default=None,
                       choices=['sdxl', 'kolors', 'sd3-turbo', 'flux1', 'flux2', 'hunyuan-dit', 'auto'],
                       help='指定模型引擎（默认: auto，显存不足时建议使用 sdxl 或 kolors）')
    
    args = parser.parse_args()
    
    tester = KepuSceneTester(config_path=args.config, model_engine=args.model_engine)
    
    if args.scene_type:
        # 测试特定场景
        result = tester.test_specific_scene(args.scene_type, args.example_index)
        if result:
            print(f"\n测试完成: {result['status']}")
    elif args.all:
        # 测试所有场景
        tester.test_all_scenes(num_examples=args.num_examples)
    else:
        # 默认测试所有场景
        tester.test_all_scenes(num_examples=args.num_examples)


if __name__ == '__main__':
    main()

