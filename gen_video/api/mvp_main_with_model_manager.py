#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVP API - 集成 ModelManager 多模型协调系统
支持统一接口调用所有模型
"""

from fastapi import FastAPI, Depends, HTTPException, status, Header, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime
from pathlib import Path
import json
import shutil
from collections import defaultdict
from PIL import Image

# 导入 ModelManager
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  警告: 无法导入 ModelManager: {e}")
    MODEL_MANAGER_AVAILABLE = False
    ModelManager = None

app = FastAPI(
    title="AI Video Generation MVP - Multi-Model",
    description="多模型协调系统 API",
    version="0.2.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ModelManager 单例 ====================

_model_manager: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    """获取 ModelManager（单例）"""
    global _model_manager
    if _model_manager is None:
        if not MODEL_MANAGER_AVAILABLE:
            raise RuntimeError("ModelManager 不可用，请检查依赖")
        print("🔧 初始化 ModelManager...")
        models_root = Path(__file__).parent.parent / "models"
        _model_manager = ModelManager(models_root=str(models_root), lazy_load=True)
        print("✅ ModelManager 初始化完成")
    return _model_manager

# ==================== 简单的配额管理 ====================

user_quotas = defaultdict(lambda: {
    "daily_images": 0,
    "daily_videos": 0,
    "last_reset": datetime.now().date(),
})

DEFAULT_QUOTAS = {
    "free": {"images": 10, "videos": 2},
    "paid": {"images": 100, "videos": 20},
}

API_KEYS = {
    "test-key-123": {"user_id": "test_user", "tier": "free"},
    "demo-key-456": {"user_id": "demo_user", "tier": "paid"},
}

def verify_api_key(x_api_key: Optional[str] = Header(None)) -> dict:
    """验证 API Key"""
    if x_api_key is None:
        # MVP 阶段允许无 key 访问
        return {"user_id": "anonymous", "tier": "free"}
    
    if x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API Key"
        )
    
    return API_KEYS[x_api_key]

def check_quota(user_id: str, tier: str, resource_type: str) -> bool:
    """检查配额"""
    quota = user_quotas[user_id]
    today = datetime.now().date()
    
    if quota["last_reset"] < today:
        quota["daily_images"] = 0
        quota["daily_videos"] = 0
        quota["last_reset"] = today
    
    if resource_type == "image":
        limit = DEFAULT_QUOTAS[tier]["images"]
        if quota["daily_images"] >= limit:
            return False
        quota["daily_images"] += 1
    elif resource_type == "video":
        limit = DEFAULT_QUOTAS[tier]["videos"]
        if quota["daily_videos"] >= limit:
            return False
        quota["daily_videos"] += 1
    
    return True

# ==================== 数据模型 ====================

class ImageResponse(BaseModel):
    """图像生成响应"""
    task_id: str
    image_url: str
    model_used: str
    task_type: str
    generation_time: float
    metadata: Dict[str, Any]

# ==================== API端点 ====================

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "AI Video Generation MVP API - Multi-Model",
        "version": "0.2.0",
        "docs": "/docs",
        "status": "running",
        "model_manager": MODEL_MANAGER_AVAILABLE
    }

@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    manager_status = {}
    if MODEL_MANAGER_AVAILABLE:
        try:
            manager = get_model_manager()
            models_status = manager.list_models()
            manager_status = {
                "available": True,
                "models": {name: info["exists"] for name, info in models_status.items()}
            }
        except Exception as e:
            manager_status = {"available": False, "error": str(e)}
    else:
        manager_status = {"available": False}
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_manager": manager_status
    }

@app.get("/api/v1/models/status")
async def get_models_status(current_user: dict = Depends(verify_api_key)):
    """获取所有模型状态"""
    if not MODEL_MANAGER_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ModelManager 不可用"
        )
    
    manager = get_model_manager()
    status = manager.list_models()
    
    return {
        "models": {
            name: {
                "exists": info["exists"],
                "loaded": info["loaded"],
                "path": info["path"]
            }
            for name, info in status.items()
        }
    }

@app.get("/api/v1/models/routing")
async def get_routing_table(current_user: dict = Depends(verify_api_key)):
    """获取任务路由表"""
    if not MODEL_MANAGER_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ModelManager 不可用"
        )
    
    manager = get_model_manager()
    
    return {
        "routing_table": manager.routing_table,
        "available_tasks": list(manager.routing_table.keys())
    }

@app.post("/api/v1/images/generate", response_model=ImageResponse)
async def generate_image(
    prompt: str = Form(..., description="生成提示词"),
    task: Optional[str] = Form(None, description="任务类型（可选）：host_face, science_background, official_style, fast_background 等。如果未指定，将根据提示词自动选择"),
    negative_prompt: Optional[str] = Form(None, description="负面提示词"),
    width: int = Form(1024, ge=512, le=2048, description="图像宽度（必须是8的倍数）"),
    height: int = Form(1024, ge=512, le=2048, description="图像高度（必须是8的倍数）"),
    num_inference_steps: Optional[int] = Form(None, description="推理步数（可选，使用模型默认值）"),
    guidance_scale: Optional[float] = Form(None, description="引导强度（可选，使用模型默认值）"),
    seed: Optional[int] = Form(None, description="随机种子"),
    current_user: dict = Depends(verify_api_key)
):
    """
    生成图像（使用 ModelManager 多模型协调系统）
    
    - **prompt**: 生成提示词
    - **task**: 任务类型（可选）
      - `host_face`: 科普主持人脸 → Kolors
      - `science_background`: 科学背景 → Flux.2
      - `lab_scene`: 实验室场景 → Flux.1
      - `official_style`: 官方风格 → Hunyuan-DiT
      - `fast_background`: 快速背景 → SD3 Turbo
      - 如果未指定，将根据提示词自动选择
    - **width/height**: 图像分辨率（必须是8的倍数）
    - **num_inference_steps**: 推理步数（可选，使用模型默认值）
    - **guidance_scale**: 引导强度（可选，使用模型默认值）
    - **seed**: 随机种子
    
    注意：
    - 系统会根据任务类型自动选择最优模型
    - 首次使用某个模型时会自动加载（延迟加载）
    - 生成时间取决于选择的模型（SD3 Turbo 最快，Flux 质量最高）
    """
    if not MODEL_MANAGER_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ModelManager 不可用，请检查依赖"
        )
    
    user_id = current_user["user_id"]
    tier = current_user["tier"]
    
    # 验证分辨率
    if width % 8 != 0 or height % 8 != 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="分辨率必须是8的倍数"
        )
    
    # 检查配额
    if not check_quota(user_id, tier, "image"):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"今日图像生成配额已用完（{DEFAULT_QUOTAS[tier]['images']}张/天）"
        )
    
    task_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # 获取 ModelManager
        manager = get_model_manager()
        
        # 如果没有指定 task，根据提示词自动选择
        if task is None:
            # 简单的关键词检测
            prompt_lower = prompt.lower()
            if any(kw in prompt_lower for kw in ["主持人", "人脸", "角色", "人物", "face", "character"]):
                task = "host_face"
            elif any(kw in prompt_lower for kw in ["实验室", "医学", "医疗", "lab", "medical"]):
                task = "lab_scene"
            elif any(kw in prompt_lower for kw in ["量子", "粒子", "太空", "宇宙", "quantum", "particle", "space"]):
                task = "science_background"
            elif any(kw in prompt_lower for kw in ["中国", "官方", "宣传", "教育", "chinese", "official"]):
                task = "official_style"
            else:
                task = "science_background"  # 默认使用科学背景
        
        print(f"\n{'='*80}")
        print(f"生成图像请求")
        print(f"{'='*80}")
        print(f"  任务ID: {task_id}")
        print(f"  提示词: {prompt}")
        print(f"  任务类型: {task}")
        print(f"  分辨率: {width}x{height}")
        
        # 准备输出路径
        output_dir = Path(__file__).parent.parent.parent / "outputs" / "api" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task_id}.png"
        
        # 使用 ModelManager 生成图像
        image = manager.generate(
            task=task,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        # 保存图像
        image.save(output_path)
        
        # 获取使用的模型
        routed_model = manager.route(task)
        
        # 计算生成时间
        generation_time = (datetime.now() - start_time).total_seconds()
        
        print(f"  ✅ 生成成功")
        print(f"  使用模型: {routed_model}")
        print(f"  生成时间: {generation_time:.2f} 秒")
        print(f"  保存路径: {output_path}")
        
        # 返回响应
        return ImageResponse(
            task_id=task_id,
            image_url=f"/api/v1/images/{task_id}",
            model_used=routed_model,
            task_type=task,
            generation_time=generation_time,
            metadata={
                "width": width,
                "height": height,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
            }
        )
        
    except Exception as e:
        print(f"  ❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图像生成失败: {str(e)}"
        )

@app.get("/api/v1/images/{task_id}")
async def get_image(task_id: str):
    """获取生成的图像"""
    image_path = Path(__file__).parent.parent.parent / "outputs" / "api" / "images" / f"{task_id}.png"
    
    if not image_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="图像不存在"
        )
    
    return FileResponse(
        image_path,
        media_type="image/png",
        filename=f"{task_id}.png"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

