"""
配置管理器 - 支持多层级配置（系统级、用户级、项目级）
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy

class ConfigManager:
    """配置管理器，支持多层级配置合并"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.system_config: Dict[str, Any] = {}
        self.user_configs: Dict[str, Dict[str, Any]] = {}
        self.project_configs: Dict[str, Dict[str, Any]] = {}
        self._load_system_config()
    
    def _load_system_config(self):
        """加载系统级配置"""
        system_config_path = self.config_dir / "system_config.yaml"
        if system_config_path.exists():
            with open(system_config_path, 'r', encoding='utf-8') as f:
                self.system_config = yaml.safe_load(f) or {}
        else:
            self.system_config = self._get_default_system_config()
    
    def load_user_config(self, user_id: str):
        """加载用户级配置"""
        user_config_path = self.config_dir / "users" / f"{user_id}.yaml"
        if user_config_path.exists():
            with open(user_config_path, 'r', encoding='utf-8') as f:
                self.user_configs[user_id] = yaml.safe_load(f) or {}
        else:
            # 使用默认用户配置
            self.user_configs[user_id] = self._get_default_user_config()
    
    def load_project_config(self, project_id: str):
        """加载项目级配置"""
        project_config_path = self.config_dir / "projects" / f"{project_id}.yaml"
        if project_config_path.exists():
            with open(project_config_path, 'r', encoding='utf-8') as f:
                self.project_configs[project_id] = yaml.safe_load(f) or {}
    
    def get_config(self, user_id: str, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取合并后的配置
        
        Args:
            user_id: 用户ID
            project_id: 项目ID（可选）
        
        Returns:
            合并后的配置字典
        """
        # 从系统配置开始
        config = deepcopy(self.system_config)
        
        # 合并用户配置
        if user_id not in self.user_configs:
            self.load_user_config(user_id)
        if user_id in self.user_configs:
            config = self._merge_config(config, self.user_configs[user_id])
        
        # 合并项目配置
        if project_id:
            if project_id not in self.project_configs:
                self.load_project_config(project_id)
            if project_id in self.project_configs:
                config = self._merge_config(config, self.project_configs[project_id])
        
        return config
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并配置"""
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def _get_default_system_config(self) -> Dict[str, Any]:
        """默认系统配置"""
        return {
            "gpu_pool": {
                "enabled": True,
                "max_concurrent": 4,
                "gpu_ids": [0, 1, 2, 3]
            },
            "model_cache": {
                "enabled": True,
                "max_size_gb": 100
            },
            "storage": {
                "type": "local",
                "base_path": "/data/outputs"
            },
            "task_queue": {
                "broker": "redis://localhost:6379/0",
                "result_backend": "redis://localhost:6379/0"
            }
        }
    
    def _get_default_user_config(self) -> Dict[str, Any]:
        """默认用户配置"""
        return {
            "quotas": {
                "images_per_day": 100,
                "videos_per_day": 10,
                "max_resolution": "1920x1080"
            },
            "limits": {
                "max_duration": 60,
                "max_frames": 1440,
                "max_scenes": 100
            }
        }
    
    def update_user_config(self, user_id: str, config: Dict[str, Any]):
        """更新用户配置"""
        self.user_configs[user_id] = config
        # 保存到文件
        user_config_path = self.config_dir / "users" / f"{user_id}.yaml"
        user_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(user_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

