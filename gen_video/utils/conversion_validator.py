"""
转换后 JSON 验证工具
检测转换后的 v2.2-final JSON 文件是否有问题
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class IssueLevel(Enum):
    """问题级别"""
    ERROR = "error"      # 错误：会导致生成失败或严重问题
    WARNING = "warning"  # 警告：可能影响生成质量
    INFO = "info"        # 信息：建议改进


@dataclass
class ValidationIssue:
    """验证问题"""
    level: IssueLevel
    field: str
    message: str
    suggestion: Optional[str] = None


class ConversionValidator:
    """转换后 JSON 验证器"""
    
    # Pose 类型与描述的关键词映射
    POSE_KEYWORDS = {
        "stand": ["standing", "stand", "upright", "on feet"],
        "sit": ["sitting", "sit", "seated"],
        "lying": ["lying", "laying", "on the ground", "motionless", "prone"],
        "walk": ["walking", "walk", "stride"],
        "kneel": ["kneeling", "kneel", "on knees"],
        "face_only": ["face", "close-up face", "portrait"]
    }
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
    
    def validate_file(self, json_path: Path) -> List[ValidationIssue]:
        """验证单个 JSON 文件"""
        self.issues = []
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self.issues.append(ValidationIssue(
                level=IssueLevel.ERROR,
                field="file",
                message=f"无法读取文件: {e}",
                suggestion="检查文件格式是否正确"
            ))
            return self.issues
        
        # 验证顶层结构
        if "version" not in data:
            self.issues.append(ValidationIssue(
                level=IssueLevel.ERROR,
                field="version",
                message="缺少顶层 version 字段",
                suggestion="添加 \"version\": \"v2.2-final\""
            ))
        
        if "scene" not in data:
            self.issues.append(ValidationIssue(
                level=IssueLevel.ERROR,
                field="scene",
                message="缺少 scene 字段",
                suggestion="检查 JSON 结构"
            ))
            return self.issues
        
        scene = data.get("scene", {})
        
        # 验证 scene 内部 version
        if "version" not in scene:
            self.issues.append(ValidationIssue(
                level=IssueLevel.ERROR,
                field="scene.version",
                message="scene 内部缺少 version 字段",
                suggestion="添加 \"version\": \"v2.2-final\" 到 scene 对象中"
            ))
        
        # 验证 shot+pose 兼容性
        self._validate_shot_pose_compatibility(scene)
        
        # 验证 prompt 与 pose 一致性
        self._validate_prompt_pose_consistency(scene)
        
        # 验证必需字段
        self._validate_required_fields(scene)
        
        # 验证 generation_params
        self._validate_generation_params(scene)
        
        # 验证 character 配置
        self._validate_character_config(scene)
        
        return self.issues
    
    def _validate_shot_pose_compatibility(self, scene: Dict[str, Any]):
        """验证 shot+pose 兼容性"""
        shot = scene.get("shot", {})
        pose = scene.get("pose", {})
        
        shot_type = shot.get("type")
        pose_type = pose.get("type")
        
        if not shot_type or not pose_type:
            return
        
        # 已知的不合法组合
        incompatible_combinations = [
            ("wide", "lying"),
            ("aerial", "lying"),
            ("close_up", "stand"),
            ("close_up", "walk"),
            ("close_up", "sit"),
            ("close_up", "kneel"),
        ]
        
        if (shot_type, pose_type) in incompatible_combinations:
            self.issues.append(ValidationIssue(
                level=IssueLevel.ERROR,
                field="shot+pose",
                message=f"不合法组合: {shot_type} + {pose_type}",
                suggestion=f"使用规则引擎自动修正，或手动修改为合法组合"
            ))
        
        # 检查是否被自动修正但未标记
        if pose.get("auto_corrected") is False:
            # 如果原本应该被修正但没有标记，可能是转换工具的问题
            pass
    
    def _validate_prompt_pose_consistency(self, scene: Dict[str, Any]):
        """验证 prompt 与 pose 一致性"""
        pose = scene.get("pose", {})
        prompt = scene.get("prompt", {})
        
        pose_type = pose.get("type")
        prompt_final = prompt.get("final", "")
        
        if not pose_type or not prompt_final:
            return
        
        # 检查 prompt 中是否包含与 pose 不一致的描述
        expected_keywords = self.POSE_KEYWORDS.get(pose_type, [])
        conflicting_keywords = []
        
        # 检查其他 pose 类型的关键词是否出现在 prompt 中
        for other_pose, keywords in self.POSE_KEYWORDS.items():
            if other_pose == pose_type:
                continue
            
            for keyword in keywords:
                if keyword.lower() in prompt_final.lower():
                    conflicting_keywords.append(keyword)
        
        if conflicting_keywords:
            # 检查是否包含正确的关键词
            has_correct_keyword = any(
                keyword.lower() in prompt_final.lower() 
                for keyword in expected_keywords
            )
            
            if not has_correct_keyword:
                self.issues.append(ValidationIssue(
                    level=IssueLevel.ERROR,
                    field="prompt.pose_consistency",
                    message=f"Prompt 与 Pose 不一致: pose.type={pose_type}，但 prompt 包含冲突关键词: {', '.join(conflicting_keywords[:3])}",
                    suggestion=f"更新 prompt.final，使用与 pose.type='{pose_type}' 一致的描述（应包含: {', '.join(expected_keywords[:2])}）"
                ))
            else:
                # 有正确的关键词，但也有冲突的，可能是警告
                self.issues.append(ValidationIssue(
                    level=IssueLevel.WARNING,
                    field="prompt.pose_consistency",
                    message=f"Prompt 包含冲突的 pose 描述: {', '.join(conflicting_keywords[:2])}，但 pose.type={pose_type}",
                    suggestion="清理 prompt 中的冲突描述，确保只包含与 pose.type 一致的描述"
                ))
    
    def _validate_required_fields(self, scene: Dict[str, Any]):
        """验证必需字段"""
        required_fields = [
            ("id", "scene.id"),
            ("shot", "scene.shot"),
            ("pose", "scene.pose"),
            ("character", "scene.character"),
        ]
        
        for field, full_path in required_fields:
            if field not in scene:
                self.issues.append(ValidationIssue(
                    level=IssueLevel.ERROR,
                    field=full_path,
                    message=f"缺少必需字段: {field}",
                    suggestion=f"添加 {full_path} 字段"
                ))
        
        # 验证 prompt.final
        prompt = scene.get("prompt", {})
        if "final" not in prompt or not prompt.get("final"):
            self.issues.append(ValidationIssue(
                level=IssueLevel.ERROR,
                field="prompt.final",
                message="缺少或为空: prompt.final",
                suggestion="确保 prompt.final 包含完整的提示词"
            ))
    
    def _validate_generation_params(self, scene: Dict[str, Any]):
        """验证 generation_params"""
        gen_params = scene.get("generation_params", {})
        
        if not gen_params:
            self.issues.append(ValidationIssue(
                level=IssueLevel.WARNING,
                field="generation_params",
                message="缺少 generation_params，将使用 config.yaml 默认值",
                suggestion="如果场景需要特殊参数，建议添加 generation_params"
            ))
            return
        
        # 检查关键参数
        steps = gen_params.get("num_inference_steps")
        if steps and steps < 20:
            self.issues.append(ValidationIssue(
                level=IssueLevel.WARNING,
                field="generation_params.num_inference_steps",
                message=f"推理步数过少: {steps}，可能影响生成质量",
                suggestion="建议使用 40-50 步"
            ))
    
    def _validate_character_config(self, scene: Dict[str, Any]):
        """验证 character 配置"""
        character = scene.get("character", {})
        
        if not character:
            return
        
        present = character.get("present", False)
        
        # 如果角色存在，检查必需配置
        if present:
            if "lora_config" not in character:
                self.issues.append(ValidationIssue(
                    level=IssueLevel.WARNING,
                    field="character.lora_config",
                    message="角色存在但缺少 lora_config",
                    suggestion="如果角色需要身份保持，应添加 lora_config"
                ))
            
            if "reference_image" not in character:
                self.issues.append(ValidationIssue(
                    level=IssueLevel.WARNING,
                    field="character.reference_image",
                    message="角色存在但缺少 reference_image",
                    suggestion="添加 reference_image 路径"
                ))


def validate_converted_json(json_path: Path, verbose: bool = True) -> bool:
    """验证转换后的 JSON 文件
    
    Args:
        json_path: JSON 文件路径
        verbose: 是否显示详细信息
    
    Returns:
        True 如果验证通过，False 如果有错误
    """
    validator = ConversionValidator()
    issues = validator.validate_file(json_path)
    
    if not issues:
        if verbose:
            print(f"✓ {json_path.name}: 验证通过")
        return True
    
    # 按级别分组
    errors = [i for i in issues if i.level == IssueLevel.ERROR]
    warnings = [i for i in issues if i.level == IssueLevel.WARNING]
    infos = [i for i in issues if i.level == IssueLevel.INFO]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"验证结果: {json_path.name}")
        print(f"{'='*60}")
        
        if errors:
            print(f"\n✗ 错误 ({len(errors)} 个):")
            for issue in errors:
                print(f"  [{issue.field}] {issue.message}")
                if issue.suggestion:
                    print(f"    建议: {issue.suggestion}")
        
        if warnings:
            print(f"\n⚠ 警告 ({len(warnings)} 个):")
            for issue in warnings:
                print(f"  [{issue.field}] {issue.message}")
                if issue.suggestion:
                    print(f"    建议: {issue.suggestion}")
        
        if infos:
            print(f"\nℹ 信息 ({len(infos)} 个):")
            for issue in infos:
                print(f"  [{issue.field}] {issue.message}")
    
    # 如果有错误，返回 False
    return len(errors) == 0


def validate_directory(json_dir: Path, pattern: str = "scene_*_v22.json", verbose: bool = True) -> Dict[str, bool]:
    """验证目录中的所有 JSON 文件
    
    Args:
        json_dir: JSON 文件目录
        pattern: 文件匹配模式
        verbose: 是否显示详细信息
    
    Returns:
        文件路径 -> 验证结果的字典
    """
    json_files = sorted(json_dir.glob(pattern))
    
    if not json_files:
        if verbose:
            print(f"✗ 未找到匹配的文件: {json_dir / pattern}")
        return {}
    
    results = {}
    
    if verbose:
        print(f"验证 {len(json_files)} 个文件...\n")
    
    for json_file in json_files:
        is_valid = validate_converted_json(json_file, verbose=verbose)
        results[str(json_file)] = is_valid
    
    # 统计
    if verbose:
        valid_count = sum(1 for v in results.values() if v)
        invalid_count = len(results) - valid_count
        
        print(f"\n{'='*60}")
        print(f"验证完成")
        print(f"{'='*60}")
        print(f"总计: {len(results)}")
        print(f"✓ 通过: {valid_count}")
        print(f"✗ 失败: {invalid_count}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证转换后的 v2.2-final JSON 文件")
    parser.add_argument(
        "path",
        type=str,
        help="JSON 文件路径或目录路径"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="scene_*_v22.json",
        help="文件匹配模式（仅目录模式，默认: scene_*_v22.json）"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，只显示结果"
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"✗ 路径不存在: {path}")
        exit(1)
    
    if path.is_file():
        # 验证单个文件
        is_valid = validate_converted_json(path, verbose=not args.quiet)
        exit(0 if is_valid else 1)
    elif path.is_dir():
        # 验证目录
        results = validate_directory(path, pattern=args.pattern, verbose=not args.quiet)
        all_valid = all(results.values())
        exit(0 if all_valid else 1)
    else:
        print(f"✗ 无效路径: {path}")
        exit(1)

