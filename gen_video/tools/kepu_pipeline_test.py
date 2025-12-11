#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科普视频完整流程测试工具
Day 8-9: 完整流程测试

功能：
1. 选择3-5个不同主题进行测试
2. 使用科普哥哥和未来姐姐分别生成
3. 记录生成时间和问题
4. 生成测试报告
"""

import os
import sys
import yaml
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# 添加gen_video路径
_script_dir = Path(__file__).parent
_gen_video_dir = _script_dir.parent
sys.path.insert(0, str(_gen_video_dir))

from tools.kepu_quick_generate import KepuQuickGenerator
from main import AIVideoPipeline


class KepuPipelineTester:
    """科普视频流程测试器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化测试器"""
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (Path(__file__).parent.parent / self.config_path).resolve()
        
        self.quick_generator = KepuQuickGenerator(str(self.config_path))
        self.test_results = []
        
    def select_test_topics(self, count: int = 5) -> List[Dict]:
        """选择测试选题"""
        # 从知识库中提取所有选题
        all_topics = []
        for category_data in self.quick_generator.kepu_topics:
            category_name = category_data.get('category', '')
            for topic in category_data.get('topics', []):
                topic_copy = topic.copy()
                topic_copy['category'] = category_name
                all_topics.append(topic_copy)
        
        # 按类别分组
        topics_by_category = {}
        for topic in all_topics:
            category = topic.get('category', 'other')
            if category not in topics_by_category:
                topics_by_category[category] = []
            topics_by_category[category].append(topic)
        
        # 从每个类别选择1-2个选题
        selected = []
        categories = list(topics_by_category.keys())
        
        for i in range(count):
            category = categories[i % len(categories)]
            if topics_by_category[category]:
                topic = topics_by_category[category].pop(0)
                selected.append(topic)
        
        return selected
    
    def test_single_topic(
        self,
        topic: Dict,
        ip_character: str = "kepu_gege",
        output_dir: Optional[Path] = None
    ) -> Dict:
        """测试单个选题"""
        topic_name = topic.get('title', '未知选题')
        category = topic.get('category', 'other')
        description = topic.get('description', '')
        
        print("\n" + "="*60)
        print(f"测试选题: {topic_name}")
        print(f"类别: {category}")
        print(f"IP角色: {ip_character}")
        print("="*60)
        
        # 生成输出名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"test_{category}_{ip_character}_{timestamp}"
        
        if output_dir:
            output_path = output_dir / output_name
        else:
            output_path = Path(self.quick_generator.config_path.parent) / "outputs" / "test" / output_name
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 记录开始时间
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # 1. 生成脚本JSON
            print(f"\n[1/4] 生成脚本JSON...")
            # 构建 topic 字典（匹配 generate_script 的参数格式）
            topic_dict = {
                'title': topic_name,
                'description': description,
                'category_name': category,
                'scene_type': topic.get('scene_type', category.lower()),
                'duration': topic.get('duration', 60),
                'difficulty': topic.get('difficulty', '初级'),
                'script_template': topic.get('script_template', f'{category.lower()}_template.json'),
                'keywords': topic.get('keywords', [])
            }
            script_json = self.quick_generator.generate_script(topic_dict, ip_character)
            
            script_path = output_path / "script.json"
            with open(script_path, 'w', encoding='utf-8') as f:
                json.dump(script_json, f, ensure_ascii=False, indent=2)
            print(f"✓ 脚本已保存: {script_path}")
            
            # 2. 初始化流水线
            print(f"\n[2/4] 初始化流水线...")
            pipeline = AIVideoPipeline(
                str(self.config_path),
                load_image=True,
                load_video=True,
                load_tts=True,
                load_subtitle=True,
                load_composer=True
            )
            print("✓ 流水线初始化成功")
            
            # 3. 生成完整视频
            print(f"\n[3/4] 生成完整视频...")
            pipeline.process_script(str(script_path), output_name=output_name)
            print("✓ 视频生成完成")
            
            # 4. 检查输出文件
            print(f"\n[4/4] 检查输出文件...")
            output_video = Path(pipeline.paths['output_dir']) / output_name / f"{output_name}_final.mp4"
            if output_video.exists():
                file_size = output_video.stat().st_size / (1024 * 1024)  # MB
                print(f"✓ 视频文件存在: {output_video}")
                print(f"  文件大小: {file_size:.2f} MB")
            else:
                warnings.append(f"视频文件不存在: {output_video}")
                print(f"⚠ 视频文件不存在: {output_video}")
            
            # 计算生成时间
            elapsed_time = time.time() - start_time
            elapsed_minutes = elapsed_time / 60
            
            result = {
                'topic_name': topic_name,
                'category': category,
                'ip_character': ip_character,
                'output_name': output_name,
                'output_path': str(output_path),
                'video_path': str(output_video) if output_video.exists() else None,
                'elapsed_time': elapsed_time,
                'elapsed_minutes': elapsed_minutes,
                'file_size_mb': file_size if output_video.exists() else 0,
                'status': 'success',
                'errors': errors,
                'warnings': warnings,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"\n✅ 测试完成")
            print(f"  生成时间: {elapsed_minutes:.2f} 分钟")
            if output_video.exists():
                print(f"  文件大小: {file_size:.2f} MB")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            errors.append(str(e))
            
            result = {
                'topic_name': topic_name,
                'category': category,
                'ip_character': ip_character,
                'output_name': output_name,
                'output_path': str(output_path),
                'video_path': None,
                'elapsed_time': elapsed_time,
                'elapsed_minutes': elapsed_time / 60,
                'file_size_mb': 0,
                'status': 'failed',
                'errors': errors,
                'warnings': warnings,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def run_tests(
        self,
        topic_count: int = 5,
        ip_characters: List[str] = None,
        output_dir: Optional[Path] = None
    ) -> List[Dict]:
        """运行完整测试"""
        if ip_characters is None:
            ip_characters = ["kepu_gege", "weilai_jiejie"]
        
        print("="*60)
        print("科普视频完整流程测试")
        print("="*60)
        print(f"测试选题数: {topic_count}")
        print(f"IP角色: {', '.join(ip_characters)}")
        print()
        
        # 选择测试选题
        test_topics = self.select_test_topics(topic_count)
        
        print(f"选择的测试选题:")
        for i, topic in enumerate(test_topics, 1):
            print(f"  {i}. {topic.get('title')} ({topic.get('category')})")
        print()
        
        # 运行测试
        all_results = []
        for ip_char in ip_characters:
            for topic in test_topics:
                result = self.test_single_topic(topic, ip_char, output_dir)
                all_results.append(result)
                self.test_results.append(result)
        
        return all_results
    
    def generate_report(self, output_path: Optional[Path] = None) -> Path:
        """生成测试报告"""
        if output_path is None:
            output_path = Path(self.quick_generator.config_path.parent) / "outputs" / "test" / "test_report.md"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        total_tests = len(self.test_results)
        success_count = sum(1 for r in self.test_results if r['status'] == 'success')
        failed_count = total_tests - success_count
        
        total_time = sum(r['elapsed_minutes'] for r in self.test_results)
        avg_time = total_time / total_tests if total_tests > 0 else 0
        
        total_size = sum(r['file_size_mb'] for r in self.test_results)
        
        # 生成报告
        report_lines = [
            "# 科普视频完整流程测试报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 测试概览",
            "",
            f"- **总测试数**: {total_tests}",
            f"- **成功**: {success_count}",
            f"- **失败**: {failed_count}",
            f"- **成功率**: {success_count/total_tests*100:.1f}%" if total_tests > 0 else "-",
            f"- **总生成时间**: {total_time:.2f} 分钟",
            f"- **平均生成时间**: {avg_time:.2f} 分钟",
            f"- **总文件大小**: {total_size:.2f} MB",
            "",
            "## 详细结果",
            "",
        ]
        
        # 按IP角色分组
        results_by_ip = {}
        for result in self.test_results:
            ip_char = result['ip_character']
            if ip_char not in results_by_ip:
                results_by_ip[ip_char] = []
            results_by_ip[ip_char].append(result)
        
        for ip_char, results in results_by_ip.items():
            report_lines.append(f"### {ip_char}")
            report_lines.append("")
            report_lines.append("| 选题 | 类别 | 状态 | 生成时间(分钟) | 文件大小(MB) | 输出路径 |")
            report_lines.append("|------|------|------|----------------|--------------|----------|")
            
            for result in results:
                status_icon = "✅" if result['status'] == 'success' else "❌"
                report_lines.append(
                    f"| {result['topic_name']} | {result['category']} | {status_icon} | "
                    f"{result['elapsed_minutes']:.2f} | {result['file_size_mb']:.2f} | "
                    f"{result['output_name']} |"
                )
            
            report_lines.append("")
        
        # 错误和警告
        all_errors = []
        all_warnings = []
        for result in self.test_results:
            all_errors.extend(result.get('errors', []))
            all_warnings.extend(result.get('warnings', []))
        
        if all_errors:
            report_lines.append("## 错误列表")
            report_lines.append("")
            for i, error in enumerate(all_errors, 1):
                report_lines.append(f"{i}. {error}")
            report_lines.append("")
        
        if all_warnings:
            report_lines.append("## 警告列表")
            report_lines.append("")
            for i, warning in enumerate(all_warnings, 1):
                report_lines.append(f"{i}. {warning}")
            report_lines.append("")
        
        # 建议
        report_lines.extend([
            "## 优化建议",
            "",
        ])
        
        if avg_time > 120:  # 2小时
            report_lines.append("- ⚠️ 平均生成时间超过2小时，需要优化生成速度")
        
        if failed_count > 0:
            report_lines.append(f"- ⚠️ 有 {failed_count} 个测试失败，需要检查错误原因")
        
        if success_count == total_tests:
            report_lines.append("- ✅ 所有测试通过，流程运行正常")
        
        report_lines.append("")
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n✓ 测试报告已保存: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="科普视频完整流程测试工具")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--count", type=int, default=3, help="测试选题数量（默认3个）")
    parser.add_argument("--ip", type=str, nargs='+', default=["kepu_gege", "weilai_jiejie"], 
                       help="IP角色列表（默认：kepu_gege weilai_jiejie）")
    parser.add_argument("--output-dir", type=str, help="输出目录（可选）")
    
    args = parser.parse_args()
    
    # 初始化测试器
    tester = KepuPipelineTester(args.config)
    
    # 设置输出目录
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行测试
    results = tester.run_tests(
        topic_count=args.count,
        ip_characters=args.ip,
        output_dir=output_dir
    )
    
    # 生成报告
    report_path = tester.generate_report()
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print(f"总测试数: {len(results)}")
    print(f"成功: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"失败: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"测试报告: {report_path}")


if __name__ == "__main__":
    main()

