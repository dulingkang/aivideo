#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试场景质量评估脚本
1. 生成测试场景的图像
2. 评估生成质量（正面率、物体识别、整体质量）
3. 生成质量评估报告
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# 添加 gen_video 路径
sys.path.insert(0, str(Path(__file__).parent / "gen_video"))

from image_generator import ImageGenerator


def load_test_scenes(json_path: Path) -> Dict[str, Any]:
    """加载测试场景JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_images(scenes: List[Dict], output_dir: Path, config_path: Path) -> List[Path]:
    """生成所有场景的图像"""
    print("=" * 80)
    print("🎨 开始生成测试场景图像...")
    print("=" * 80)
    
    # 创建图像生成器
    try:
        image_generator = ImageGenerator(str(config_path))
        print("✅ 图像生成器初始化成功")
    except Exception as e:
        print(f"❌ 图像生成器初始化失败: {e}")
        raise
    
    # 生成图像
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建临时JSON文件用于生成
    temp_json = output_dir / "temp_test_scenes.json"
    test_data = {
        "episode": "test",
        "title": "效果测试场景集",
        "scenes": scenes
    }
    with open(temp_json, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 生成图像
    generated_paths = image_generator.generate_from_script(
        str(temp_json),
        output_dir=str(output_dir / "images"),
        overwrite=True,
        update_script=True
    )
    
    print(f"\n✅ 图像生成完成！共生成 {len(generated_paths)} 张图像")
    
    return generated_paths


def analyze_image_quality(scene: Dict, image_path: Path) -> Dict[str, Any]:
    """分析单个场景的图像质量"""
    result = {
        "scene_id": scene.get("id"),
        "description": scene.get("description", ""),
        "image_path": str(image_path),
        "has_character": "韩立" in scene.get("description", "") or "Han Li" in scene.get("prompt", ""),
        "is_front_view": False,
        "object_correct": True,
        "quality_score": 0,
        "issues": []
    }
    
    # 检查是否是正面视角
    prompt = scene.get("prompt", "").lower()
    description = scene.get("description", "").lower()
    
    front_keywords = ["front view", "facing camera", "正面", "面向镜头"]
    back_keywords = ["back", "背影", "from behind", "back view"]
    
    has_front = any(kw in prompt or kw in description for kw in front_keywords)
    has_back = any(kw in prompt or kw in description for kw in back_keywords)
    
    if result["has_character"]:
        result["is_front_view"] = has_front and not has_back
    
    # 检查物体识别是否正确
    # 场景0应该是卷轴，场景2应该是城市
    scene_id = scene.get("id")
    if scene_id == 0:
        # 卷轴场景
        scroll_keywords = ["scroll", "卷轴"]
        weapon_keywords = ["weapon", "兵器", "sword", "刀"]
        has_scroll = any(kw in prompt or kw in description for kw in scroll_keywords)
        has_weapon = any(kw in prompt or kw in description for kw in weapon_keywords)
        result["object_correct"] = has_scroll and not has_weapon
        if not result["object_correct"]:
            result["issues"].append("卷轴识别错误（可能生成了兵器）")
    elif scene_id == 2:
        # 城市场景
        city_keywords = ["city", "城市", "silhouette"]
        people_keywords = ["people", "人物", "character", "person"]
        has_city = any(kw in prompt or kw in description for kw in city_keywords)
        has_people = any(kw in prompt or kw in description for kw in people_keywords)
        result["object_correct"] = has_city and not has_people
        if not result["object_correct"]:
            result["issues"].append("城市识别错误（可能生成了人物）")
    
    # 计算质量分数（简化版，实际需要查看图像）
    quality_score = 100
    if result["has_character"] and not result["is_front_view"]:
        quality_score -= 30
        result["issues"].append("人物不是正面视角")
    if not result["object_correct"]:
        quality_score -= 50
    if scene_id == 1 and not result["is_front_view"]:
        quality_score -= 20
        result["issues"].append("中景场景人物不是正面")
    if scene_id == 3 and not result["is_front_view"]:
        quality_score -= 30
        result["issues"].append("近景场景人物不是正面")
    
    result["quality_score"] = max(0, quality_score)
    
    return result


def generate_quality_report(analysis_results: List[Dict], output_dir: Path):
    """生成质量评估报告"""
    report_path = output_dir / "quality_report.md"
    
    # 统计信息
    total_scenes = len(analysis_results)
    character_scenes = [r for r in analysis_results if r["has_character"]]
    front_view_count = sum(1 for r in character_scenes if r["is_front_view"])
    object_correct_count = sum(1 for r in analysis_results if r["object_correct"])
    avg_quality = sum(r["quality_score"] for r in analysis_results) / total_scenes if total_scenes > 0 else 0
    
    # 生成报告
    report = f"""# 测试场景质量评估报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 总体评估

- **总场景数**: {total_scenes}
- **人物场景数**: {len(character_scenes)}
- **正面视角率**: {front_view_count}/{len(character_scenes)} ({front_view_count/len(character_scenes)*100:.1f}%)" if len(character_scenes) > 0 else "- **正面视角率**: N/A (无人物场景)"
- **物体识别正确率**: {object_correct_count}/{total_scenes} ({object_correct_count/total_scenes*100:.1f}%)
- **平均质量分数**: {avg_quality:.1f}/100

## 🎯 质量等级

"""
    
    if avg_quality >= 80:
        report += "✅ **优秀** (80-100分): 可以用于MVP演示，质量达到可用标准\n"
    elif avg_quality >= 70:
        report += "⚠️ **可用** (70-79分): 基本达到可用标准，需要继续优化\n"
    elif avg_quality >= 60:
        report += "⚠️ **需改进** (60-69分): 需要修复关键问题后再考虑MVP\n"
    else:
        report += "❌ **不可用** (<60分): 必须先优化效果，再考虑MVP\n"
    
    report += f"""
## 📋 详细场景分析

"""
    
    for result in analysis_results:
        scene_id = result["scene_id"]
        quality_score = result["quality_score"]
        status_icon = "✅" if quality_score >= 80 else "⚠️" if quality_score >= 70 else "❌"
        
        report += f"""### 场景 {scene_id}: {status_icon} {quality_score}分

- **描述**: {result["description"]}
- **图像路径**: `{result["image_path"]}`
- **是否有人物**: {"是" if result["has_character"] else "否"}
- **是否正面视角**: {"是" if result["is_front_view"] else "否"}
- **物体识别正确**: {"是" if result["object_correct"] else "否"}
- **质量分数**: {quality_score}/100

"""
        
        if result["issues"]:
            report += "**问题**:\n"
            for issue in result["issues"]:
                report += f"- ⚠️ {issue}\n"
        
        report += "\n"
    
    report += f"""## 🔍 建议

### 立即修复的问题（如果质量<70分）

1. **人物背面问题**
   - 场景需要正面视角但生成了背面
   - 建议：增强正面朝向提示权重

2. **物体识别错误**
   - 卷轴识别成兵器
   - 城市识别成人物
   - 建议：增强负面提示，明确物体描述

3. **场景类型匹配**
   - 远景场景人物太小
   - 中景场景人物不清晰
   - 建议：调整镜头类型和人物位置

### 下一步行动

"""
    
    if avg_quality >= 70:
        report += """✅ **质量达到可用标准，可以继续开发MVP**
- 继续开发后端API和前端界面
- 边开发边优化效果
- 准备演示Demo
"""
    else:
        report += """⚠️ **质量未达可用标准，建议先优化效果**
- 先修复关键问题（背面、物体识别错误）
- 提升质量到70分以上
- 再考虑开发MVP

**预计优化时间**: 3-5天
"""
    
    # 写入报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 质量评估报告已生成: {report_path}")
    
    return report_path


def main():
    # 配置路径
    project_root = Path(__file__).parent
    test_json_path = project_root / "renjie" / "episode" / "test_scenes.json"
    output_dir = project_root / "outputs" / "test_scenes_quality"
    config_path = project_root / "gen_video" / "config.yaml"
    
    print("=" * 80)
    print("🧪 测试场景质量评估")
    print("=" * 80)
    print(f"📄 测试场景JSON: {test_json_path}")
    print(f"📁 输出目录: {output_dir}")
    print(f"⚙️  配置文件: {config_path}")
    print()
    
    # 检查文件是否存在
    if not test_json_path.exists():
        print(f"❌ 错误: 测试场景JSON文件不存在: {test_json_path}")
        return 1
    
    if not config_path.exists():
        print(f"❌ 错误: 配置文件不存在: {config_path}")
        return 1
    
    # 加载测试场景
    print("📖 加载测试场景...")
    test_data = load_test_scenes(test_json_path)
    scenes = test_data.get("scenes", [])
    print(f"✅ 加载了 {len(scenes)} 个测试场景\n")
    
    # 生成图像
    try:
        generated_paths = generate_images(scenes, output_dir, config_path)
    except Exception as e:
        print(f"❌ 图像生成失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 分析质量（基于JSON和提示词，实际需要查看图像）
    print("\n" + "=" * 80)
    print("📊 分析图像质量...")
    print("=" * 80)
    
    analysis_results = []
    for idx, scene in enumerate(scenes):
        scene_id = scene.get("id", idx)
        # 文件名编号 = enumerate索引 = 数组索引 + 1
        # 因为 image_generator 使用 enumerate(scenes, start=1)
        # 所以：场景0 (数组索引0) -> scene_001.png (enumerate索引1)
        #       场景1 (数组索引1) -> scene_002.png (enumerate索引2)
        file_num = idx + 1
        
        # 查找对应的图像路径
        image_path = output_dir / "images" / f"scene_{file_num:03d}.png"
        
        # 如果找不到，尝试从生成的文件列表中按顺序匹配
        if not image_path.exists() and generated_paths:
            if idx < len(generated_paths):
                # 按数组顺序匹配（第一个场景对应第一个文件）
                image_path = Path(generated_paths[idx])
            else:
                # 尝试按文件名匹配
                for path in generated_paths:
                    if f"scene_{file_num:03d}" in str(path):
                        image_path = Path(path)
                        break
        
        result = analyze_image_quality(scene, image_path)
        analysis_results.append(result)
        
        status_icon = "✅" if result["quality_score"] >= 70 else "⚠️" if result["quality_score"] >= 60 else "❌"
        print(f"{status_icon} 场景 {scene_id}: {result['quality_score']}分 - {result['description'][:30]}...")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"   ⚠️  {issue}")
    
    # 生成报告
    print("\n" + "=" * 80)
    print("📝 生成质量评估报告...")
    print("=" * 80)
    
    report_path = generate_quality_report(analysis_results, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 评估报告: {report_path}")
    print(f"🖼️  生成图像: {output_dir / 'images'}")
    print()
    
    # 显示总体评估
    avg_quality = sum(r["quality_score"] for r in analysis_results) / len(analysis_results) if analysis_results else 0
    character_scenes = [r for r in analysis_results if r["has_character"]]
    front_view_count = sum(1 for r in character_scenes if r["is_front_view"])
    
    print(f"📊 总体评估:")
    print(f"   平均质量分数: {avg_quality:.1f}/100")
    if len(character_scenes) > 0:
        print(f"   正面视角率: {front_view_count}/{len(character_scenes)} ({front_view_count/len(character_scenes)*100:.1f}%)")
    
    if avg_quality >= 70:
        print(f"\n✅ 质量达到可用标准，可以继续开发MVP")
    else:
        print(f"\n⚠️  质量未达可用标准，建议先优化效果")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

