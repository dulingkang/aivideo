#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execution Validator - 校验JSON是否可执行

功能：
1. 校验v2.1-exec JSON的完整性和合法性
2. 检查所有必需字段
3. 验证硬规则约束
4. 生成校验报告
"""

import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from utils.execution_rules_v2_1 import get_execution_rules, ShotType, PoseType
from utils.character_anchor_v2_1 import get_character_anchor_manager

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """校验级别"""
    ERROR = "error"      # 错误：不可执行
    WARNING = "warning"  # 警告：可执行但可能有问题
    INFO = "info"        # 信息：建议优化


@dataclass
class ValidationIssue:
    """校验问题"""
    level: ValidationLevel
    field: str
    message: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """校验结果"""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int = 0
    errors_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors_count,
            "warnings": self.warnings_count,
            "issues": [
                {
                    "level": issue.level.value,
                    "field": issue.field,
                    "message": issue.message,
                    "suggestion": issue.suggestion
                }
                for issue in self.issues
            ]
        }


class ExecutionValidator:
    """执行型JSON校验器"""
    
    def __init__(self):
        """初始化校验器"""
        self.rules = get_execution_rules()
        self.anchor_manager = get_character_anchor_manager()
        logger.info("Execution Validator 初始化完成")
    
    def validate_scene(self, scene: Dict[str, Any]) -> ValidationResult:
        """
        校验单个场景
        
        Args:
            scene: v2.1-exec格式的场景JSON
            
        Returns:
            ValidationResult: 校验结果
        """
        issues: List[ValidationIssue] = []
        
        # 1. 检查版本
        version = scene.get("version", "")
        if not version.startswith("v2.1"):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field="version",
                message=f"版本不匹配: {version}，需要 v2.1-exec",
                suggestion="使用 JSONV2ToV21Converter 转换"
            ))
        
        # 2. 检查Shot（必须锁定）
        shot = scene.get("shot", {})
        if not shot.get("locked"):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field="shot.locked",
                message="Shot未锁定，无法执行",
                suggestion="设置 shot.locked = true"
            ))
        
        shot_type_str = shot.get("type", "")
        try:
            shot_type = ShotType(shot_type_str)
        except ValueError:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field="shot.type",
                message=f"无效的Shot类型: {shot_type_str}",
                suggestion="使用: wide, medium, close_up, aerial"
            ))
        
        # 3. 检查Pose（必须锁定）
        pose = scene.get("pose", {})
        if not pose.get("locked"):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field="pose.locked",
                message="Pose未锁定，无法执行",
                suggestion="设置 pose.locked = true"
            ))
        
        pose_type_str = pose.get("type", "")
        try:
            pose_type = PoseType(pose_type_str)
        except ValueError:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field="pose.type",
                message=f"无效的Pose类型: {pose_type_str}",
                suggestion="使用: stand, walk, sit, lying, kneel, face_only"
            ))
        
        # 4. 验证Shot → Pose组合合法性
        if shot_type and pose_type:
            is_forbidden = self.rules.check_forbidden_combinations(shot_type, pose_type)
            if is_forbidden:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    field="shot+pose",
                    message=f"不合法组合: {shot_type.value} + {pose_type.value}",
                    suggestion="使用规则引擎自动修正"
                ))
        
        # 5. 检查Model路由（必须锁定）
        model_route = scene.get("model_route", {})
        if model_route.get("allow_fallback", True):
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field="model_route.allow_fallback",
                message="允许fallback会降低稳定性",
                suggestion="设置 allow_fallback = false"
            ))
        
        if not model_route.get("decision_reason"):
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field="model_route.decision_reason",
                message="缺少决策原因（可解释性）",
                suggestion="添加 decision_reason 字段"
            ))
        
        # 6. 检查Character配置
        character = scene.get("character", {})
        if character.get("present"):
            character_id = character.get("id")
            if not character_id:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    field="character.id",
                    message="有角色但缺少character.id",
                    suggestion="添加 character.id 字段"
                ))
            
            # 检查角色锚
            if character_id:
                anchor = self.anchor_manager.get_anchor(character_id)
                if not anchor:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        field="character_anchor",
                        message=f"角色 {character_id} 未注册角色锚",
                        suggestion="使用 CharacterAnchorManager.register_character 注册"
                    ))
        
        # 7. 检查Prompt
        prompt = scene.get("prompt", {})
        if not prompt.get("positive_core") and character.get("present"):
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field="prompt.positive_core",
                message="缺少角色核心描述",
                suggestion="添加 prompt.positive_core 字段"
            ))
        
        if not prompt.get("scene_description"):
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field="prompt.scene_description",
                message="缺少场景描述",
                suggestion="添加 prompt.scene_description 字段"
            ))
        
        # 8. 检查性别负锁
        negative_lock = scene.get("negative_lock", {})
        if character.get("present") and not negative_lock.get("gender"):
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field="negative_lock.gender",
                message="缺少性别负锁（可能导致性别错误）",
                suggestion="设置 negative_lock.gender = true"
            ))
        
        # 统计错误和警告
        errors_count = sum(1 for issue in issues if issue.level == ValidationLevel.ERROR)
        warnings_count = sum(1 for issue in issues if issue.level == ValidationLevel.WARNING)
        
        return ValidationResult(
            is_valid=(errors_count == 0),
            issues=issues,
            warnings_count=warnings_count,
            errors_count=errors_count
        )
    
    def validate_episode(self, episode: Dict[str, Any]) -> Dict[int, ValidationResult]:
        """
        校验整集JSON
        
        Args:
            episode: v2.1-exec格式的episode JSON
            
        Returns:
            Dict[scene_id, ValidationResult]: 每个场景的校验结果
        """
        scenes = episode.get("scenes", [])
        results = {}
        
        for scene in scenes:
            scene_id = scene.get("scene_id", 0)
            result = self.validate_scene(scene)
            results[scene_id] = result
        
        return results
    
    def generate_report(
        self,
        validation_result: ValidationResult,
        scene_id: Optional[int] = None
    ) -> str:
        """
        生成校验报告
        
        Args:
            validation_result: 校验结果
            scene_id: 场景ID（可选）
            
        Returns:
            str: 校验报告文本
        """
        lines = []
        
        if scene_id is not None:
            lines.append(f"场景 {scene_id} 校验报告")
        else:
            lines.append("校验报告")
        
        lines.append("=" * 60)
        
        if validation_result.is_valid:
            lines.append("✅ 校验通过（可执行）")
        else:
            lines.append("❌ 校验失败（不可执行）")
        
        lines.append(f"错误: {validation_result.errors_count}")
        lines.append(f"警告: {validation_result.warnings_count}")
        lines.append("")
        
        # 按级别分组显示问题
        errors = [issue for issue in validation_result.issues if issue.level == ValidationLevel.ERROR]
        warnings = [issue for issue in validation_result.issues if issue.level == ValidationLevel.WARNING]
        infos = [issue for issue in validation_result.issues if issue.level == ValidationLevel.INFO]
        
        if errors:
            lines.append("❌ 错误:")
            for issue in errors:
                lines.append(f"  - [{issue.field}] {issue.message}")
                if issue.suggestion:
                    lines.append(f"    建议: {issue.suggestion}")
            lines.append("")
        
        if warnings:
            lines.append("⚠️  警告:")
            for issue in warnings:
                lines.append(f"  - [{issue.field}] {issue.message}")
                if issue.suggestion:
                    lines.append(f"    建议: {issue.suggestion}")
            lines.append("")
        
        if infos:
            lines.append("ℹ️  信息:")
            for issue in infos:
                lines.append(f"  - [{issue.field}] {issue.message}")
                if issue.suggestion:
                    lines.append(f"    建议: {issue.suggestion}")
        
        return "\n".join(lines)


def validate_json_file(json_path: str) -> Tuple[bool, str]:
    """
    校验JSON文件
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        Tuple[bool, str]: (是否有效, 报告文本)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        validator = ExecutionValidator()
        
        if "scenes" in data:
            # 整集JSON
            results = validator.validate_episode(data)
            
            # 生成总报告
            all_valid = all(r.is_valid for r in results.values())
            total_errors = sum(r.errors_count for r in results.values())
            total_warnings = sum(r.warnings_count for r in results.values())
            
            report_lines = [
                "整集校验报告",
                "=" * 60,
                f"总场景数: {len(results)}",
                f"有效场景: {sum(1 for r in results.values() if r.is_valid)}",
                f"总错误数: {total_errors}",
                f"总警告数: {total_warnings}",
                "",
            ]
            
            for scene_id, result in results.items():
                if not result.is_valid or result.warnings_count > 0:
                    report_lines.append(validator.generate_report(result, scene_id))
                    report_lines.append("")
            
            report = "\n".join(report_lines)
        else:
            # 单个场景JSON
            result = validator.validate_scene(data)
            report = validator.generate_report(result)
        
        return all_valid if "scenes" in data else result.is_valid, report
        
    except Exception as e:
        logger.error(f"校验失败: {e}")
        import traceback
        return False, f"校验异常: {e}\n{traceback.format_exc()}"


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python execution_validator.py <scene.json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    is_valid, report = validate_json_file(json_path)
    
    print(report)
    sys.exit(0 if is_valid else 1)

