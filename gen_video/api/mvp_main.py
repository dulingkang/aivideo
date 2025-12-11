#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小MVP - AI视频生成平台API
快速验证市场需求，无需Redis/Celery
"""

from fastapi import FastAPI, Depends, HTTPException, status, Header, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import shutil
from collections import defaultdict
from PIL import Image

# 导入生成器
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from image_generator import ImageGenerator
    from video_generator import VideoGenerator
except ImportError:
    print("⚠️  警告: 无法导入生成器，请确保在正确的环境中运行")
    ImageGenerator = None
    VideoGenerator = None

# 导入 ModelManager（多模型协调系统）
try:
    from model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  警告: 无法导入 ModelManager: {e}")
    MODEL_MANAGER_AVAILABLE = False
    ModelManager = None

# 导入 ModelManager（多模型协调系统）
try:
    from model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  警告: 无法导入 ModelManager: {e}")
    MODEL_MANAGER_AVAILABLE = False
    ModelManager = None

app = FastAPI(
    title="AI Video Generation MVP",
    description="最小可行产品 - AI视频生成平台API",
    version="0.1.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP阶段允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 简单的配额管理（内存）====================

# 用户配额（内存存储，重启后丢失）
user_quotas = defaultdict(lambda: {
    "daily_images": 0,
    "daily_videos": 0,
    "last_reset": datetime.now().date(),
    "api_key": None,
})

# 默认配额
DEFAULT_QUOTAS = {
    "free": {"images": 10, "videos": 2},
    "paid": {"images": 100, "videos": 20},
}

# 简单的API Key管理（生产环境应使用数据库）
API_KEYS = {
    "test-key-123": {"user_id": "test_user", "tier": "free"},
    "demo-key-456": {"user_id": "demo_user", "tier": "paid"},
}

# ==================== 数据模型 ====================

class ImageRequest(BaseModel):
    """图像生成请求（JSON部分）"""
    prompt: str = Field(..., min_length=1, max_length=500, description="生成提示词")
    negative_prompt: Optional[str] = Field(None, max_length=500, description="负面提示词")
    width: int = Field(1024, ge=512, le=2048, description="图像宽度（必须是8的倍数）")
    height: int = Field(1024, ge=512, le=2048, description="图像高度（必须是8的倍数）")
    num_inference_steps: int = Field(40, ge=10, le=100, description="推理步数")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="引导尺度")
    seed: Optional[int] = Field(None, description="随机种子")
    use_reference_image: bool = Field(False, description="是否使用参考图像")
    reference_image_type: Optional[str] = Field("scene", description="参考图像类型：scene（场景）或face（面部）")
    
    @validator('width', 'height')
    def validate_resolution(cls, v):
        if v % 8 != 0:
            raise ValueError('分辨率必须是8的倍数')
        return v

class VideoScene(BaseModel):
    """视频场景"""
    prompt: str = Field(..., min_length=1, max_length=500, description="场景提示词")
    duration: float = Field(5.0, ge=1.0, le=30.0, description="场景时长（秒）")
    image_path: Optional[str] = Field(None, description="预生成的图像路径（可选）")
    # 风格配置（可选，支持在API中指定）
    style: Optional[str] = Field(None, description="视频风格（可选）：scientific(科普), commercial(产品广告), dramatic(戏剧), realistic(写实), xianxia(仙侠)")
    visual: Optional[Dict[str, Any]] = Field(None, description="视觉配置（可选），包含style和composition等字段")

class VideoRequest(BaseModel):
    """视频生成请求"""
    scenes: List[VideoScene] = Field(..., min_items=1, max_items=10, description="场景列表")
    fps: int = Field(24, ge=15, le=30, description="帧率")
    width: int = Field(1280, ge=512, le=1920, description="视频宽度")
    height: int = Field(768, ge=512, le=1080, description="视频高度")
    # 运动参数（可选，用于调整动画自然度）
    motion_bucket_id: Optional[float] = Field(None, ge=1.0, le=2.0, description="运动幅度参数（1.0-2.0，越小越自然，推荐1.3-1.5）")
    noise_aug_strength: Optional[float] = Field(None, ge=0.0001, le=0.0005, description="运动平滑度参数（0.0001-0.0005，越小越平滑，推荐0.0002-0.00025）")

class ImageResponse(BaseModel):
    """图像生成响应"""
    task_id: str
    status: str
    image_url: Optional[str] = None
    image_path: Optional[str] = None
    width: int
    height: int
    file_size: int
    quota_remaining: Dict[str, int]
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None  # 新增：元数据（包含使用的模型等信息）

class VideoResponse(BaseModel):
    """视频生成响应"""
    task_id: str
    status: str
    video_url: Optional[str] = None
    video_path: Optional[str] = None
    duration: float
    file_size: int
    quota_remaining: Dict[str, int]
    created_at: datetime

# ==================== 认证（简化版）====================

async def verify_api_key(x_api_key: str = Header(None)):
    """验证API Key"""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少API Key，请在Header中添加: X-API-Key"
        )
    
    if x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的API Key"
        )
    
    user_info = API_KEYS[x_api_key]
    return {
        "user_id": user_info["user_id"],
        "tier": user_info["tier"],
        "api_key": x_api_key
    }

def check_quota(user_id: str, tier: str, resource: str) -> bool:
    """检查配额"""
    quota = user_quotas[user_id]
    
    # 检查是否需要重置（每天重置）
    if quota["last_reset"] < datetime.now().date():
        quota["daily_images"] = 0
        quota["daily_videos"] = 0
        quota["last_reset"] = datetime.now().date()
    
    # 获取配额限制
    limits = DEFAULT_QUOTAS.get(tier, DEFAULT_QUOTAS["free"])
    
    if resource == "image":
        if quota["daily_images"] >= limits["images"]:
            return False
        quota["daily_images"] += 1
    elif resource == "video":
        if quota["daily_videos"] >= limits["videos"]:
            return False
        quota["daily_videos"] += 1
    
    return True

def get_quota_remaining(user_id: str, tier: str) -> Dict[str, int]:
    """获取剩余配额"""
    quota = user_quotas[user_id]
    limits = DEFAULT_QUOTAS.get(tier, DEFAULT_QUOTAS["free"])
    
    # 检查是否需要重置
    if quota["last_reset"] < datetime.now().date():
        quota["daily_images"] = 0
        quota["daily_videos"] = 0
        quota["last_reset"] = datetime.now().date()
    
    return {
        "images": max(0, limits["images"] - quota["daily_images"]),
        "videos": max(0, limits["videos"] - quota["daily_videos"]),
    }

# ==================== 生成器单例 ====================

_image_generator = None
_video_generator = None
_model_manager = None

def get_image_generator():
    """获取图像生成器（单例，延迟加载，不占用启动显存）"""
    global _image_generator
    if _image_generator is None:
        if ImageGenerator is None:
            raise RuntimeError("图像生成器未安装")
        config_path = Path(__file__).parent.parent / "config.yaml"
        print("🔧 初始化图像生成器（延迟加载，不占用启动显存）...")
        # 只创建实例，不加载任何模型（延迟加载）
        _image_generator = ImageGenerator(config_path=str(config_path))
        # 注意：pipeline 会在首次使用时自动加载（延迟加载，节省启动时间和显存）
        print("ℹ️  Pipeline将在首次生成时自动加载（不占用启动显存）")
    return _image_generator

def get_video_generator():
    """获取视频生成器（单例，延迟加载，不占用启动显存）"""
    global _video_generator
    if _video_generator is None:
        if VideoGenerator is None:
            raise RuntimeError("视频生成器未安装")
        config_path = Path(__file__).parent.parent / "config.yaml"
        print("🔧 初始化视频生成器（延迟加载，不占用启动显存）...")
        # 只创建实例，不加载任何模型（延迟加载）
        _video_generator = VideoGenerator(config_path=str(config_path))
        print("ℹ️  视频模型将在首次生成时自动加载（不占用启动显存）")
    return _video_generator

def get_model_manager() -> Optional[ModelManager]:
    """获取 ModelManager（单例，可选，延迟加载，不占用启动显存）"""
    global _model_manager
    if _model_manager is None and MODEL_MANAGER_AVAILABLE:
        models_root = Path(__file__).parent.parent / "models"
        config_path = Path(__file__).parent.parent / "config.yaml"
        print("🔧 初始化 ModelManager（延迟加载，不占用启动显存）...")
        # lazy_load=True 确保不预加载模型
        _model_manager = ModelManager(models_root=str(models_root), lazy_load=True, config_path=str(config_path))
        print("✅ ModelManager 初始化完成（模型将在首次使用时加载）")
    return _model_manager

# ==================== 工具函数 ====================

def get_available_loras():
    """获取可用的 LoRA 列表"""
    lora_dir = Path(__file__).parent.parent / "models" / "lora"
    available_loras = {
        "character": [],
        "style": []
    }
    
    if not lora_dir.exists():
        return available_loras
    
    for item in lora_dir.iterdir():
        if item.is_dir():
            # 检查是否有 safetensors 文件
            safetensors = list(item.glob("*.safetensors"))
            if safetensors:
                lora_name = item.name
                # 判断类型
                if "host" in lora_name.lower() or "person" in lora_name.lower():
                    lora_type = "character"
                    description = "主持人/角色 LoRA"
                elif "anime" in lora_name.lower() or "style" in lora_name.lower():
                    lora_type = "style"
                    description = "风格 LoRA"
                else:
                    lora_type = "character"
                    description = "角色 LoRA"
                
                available_loras[lora_type].append({
                    "name": lora_name,
                    "description": description
                })
    
    return available_loras

# ==================== API端点 ====================

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "AI Video Generation MVP API",
        "version": "0.1.0",
        "docs": "/docs",
        "status": "running",
        "note": "这是最小MVP版本，用于快速验证市场需求"
    }

@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "generators": {
            "image": ImageGenerator is not None,
            "video": VideoGenerator is not None,
        }
    }

@app.get("/api/v1/styles")
async def list_styles(current_user: dict = Depends(verify_api_key)):
    """
    获取可用的视频风格列表
    
    返回所有在配置文件中定义的风格模板，包括：
    - scientific: 科普/教育风格
    - commercial: 产品广告风格
    - dramatic: 戏剧/情感风格
    - realistic: 写实风格（默认）
    - xianxia: 仙侠风格
    """
    try:
        from gen_video.utils.style_validator import StyleValidator
        from pathlib import Path
        
        config_path = Path(__file__).parent.parent / "config.yaml"
        validator = StyleValidator(str(config_path))
        
        styles = {}
        for style_name in validator.get_available_styles():
            style_info = validator.get_style_template(style_name)
            if style_info:
                styles[style_name] = {
                    "keywords": style_info.get('keywords', []),
                    "description": style_info.get('description', ''),
                    "negative_keywords": style_info.get('negative_keywords', [])
                }
        
        return {
            "styles": styles,
            "default_style": validator.get_default_style(),
            "count": len(styles)
        }
    except Exception as e:
        return {
            "error": str(e),
            "styles": {},
            "default_style": "realistic",
            "count": 0
        }

@app.get("/api/v1/loras")
async def list_loras():
    """获取可用的 LoRA 列表"""
    loras = get_available_loras()
    return {
        "character": loras["character"],
        "style": loras["style"]
    }

@app.post("/api/v1/images/generate", response_model=ImageResponse)
async def generate_image(
    prompt: str = Form(..., description="生成提示词"),
    negative_prompt: Optional[str] = Form(None, description="负面提示词"),
    width: int = Form(1024, ge=512, le=2048, description="图像宽度（必须是8的倍数）"),
    height: int = Form(1024, ge=512, le=2048, description="图像高度（必须是8的倍数）"),
    num_inference_steps: int = Form(40, ge=10, le=100, description="推理步数"),
    guidance_scale: float = Form(7.5, ge=1.0, le=20.0, description="引导尺度"),
    seed: Optional[int] = Form(None, description="随机种子"),
    reference_image: Optional[UploadFile] = File(None, description="参考图像（可选）"),
    reference_image_type: Optional[str] = Form("scene", description="参考图像类型：scene（场景）或face（面部）"),
    character_lora: Optional[str] = Form(None, description="角色LoRA适配器名称（可选，None表示不使用LoRA仅使用参考图，空字符串表示禁用，非空字符串表示使用指定LoRA）"),
    style_lora: Optional[str] = Form(None, description="风格LoRA适配器名称（可选，None表示不使用LoRA仅使用参考图，空字符串表示禁用，非空字符串表示使用指定LoRA）"),
    model_engine: Optional[str] = Form("auto", description="指定模型引擎（可选）：flux-instantid（人物生成）, hunyuan-dit（中文场景）, kolors（真实感场景）, sd3-turbo（批量生成）, auto（自动选择，推荐）"),
    task_type: Optional[str] = Form(None, description="任务类型（可选）：character（人物生成）, scene（场景生成）, batch（批量生成）。如果未指定，将自动检测"),
    use_model_manager: bool = Form(True, description="是否使用 ModelManager 多模型协调系统（默认启用，推荐）"),
    task: Optional[str] = Form(None, description="任务类型（ModelManager模式）：host_face（主持人脸）, science_background（科学背景）, official_style（官方风格）, fast_background（快速背景）等"),
    face_image_name: Optional[str] = Form(None, description="人脸参考图片文件名（从 models/face_references/ 目录加载，如：host_face.png）。如果未指定，会根据任务类型自动查找"),
    current_user: dict = Depends(verify_api_key)
):
    """
    生成图像（同步模式，支持多模型自动选择）
    
    - **prompt**: 生成提示词
    - **width/height**: 图像分辨率（必须是8的倍数）
    - **num_inference_steps**: 推理步数（越多质量越好，但越慢）
    - **reference_image**: 参考图像文件（可选，支持场景参考或面部参考）
    - **reference_image_type**: 参考图像类型（scene=场景参考，face=面部参考）
    - **character_lora**: 角色LoRA适配器名称
      - None: 不使用角色LoRA，仅使用参考图生成正常图像（不会使用默认的hanli）
      - "": 明确禁用角色LoRA
      - "lora_name": 使用指定的角色LoRA
    - **style_lora**: 风格LoRA适配器名称
      - None: 不使用风格LoRA，仅使用参考图生成正常图像（不会使用默认的anime_style）
      - "": 明确禁用风格LoRA
      - "lora_name": 使用指定的风格LoRA
    - **model_engine**: 指定模型引擎（可选）
      - "auto": 自动选择（推荐，根据任务类型和提示词自动选择最适合的模型）
      - "flux-instantid": 人物生成（主持人固定人设）
      - "hunyuan-dit": 中文场景（中国式科教场景）
      - "kolors": 真实感场景（手部、光影优秀）
      - "sd3-turbo": 批量生成（极速出大量素材）
    - **task_type**: 任务类型（可选）
      - "character": 人物生成
      - "scene": 场景生成
      - "batch": 批量生成
      - 如果未指定，将根据提示词和参考图像自动检测
    
    注意：
    - 当character_lora和style_lora都是None时，系统不会使用任何LoRA，仅使用参考图生成正常图像
    - 此模式会同步执行，可能需要30-60秒
    - 推荐使用 model_engine="auto" 让系统自动选择最适合的模型
    """
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
    
    try:
        # 准备输出路径
        output_dir = Path(__file__).parent.parent.parent / "outputs" / "api" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task_id}.png"
        
        # 准备参考图像路径
        reference_image_path = None
        face_reference_image_path = None
        
        # 处理上传的参考图像
        if reference_image:
            # 保存上传的参考图像
            upload_dir = Path(__file__).parent.parent.parent / "outputs" / "api" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取文件扩展名
            file_ext = Path(reference_image.filename).suffix if reference_image.filename else ".png"
            if file_ext not in [".png", ".jpg", ".jpeg", ".webp"]:
                file_ext = ".png"
            
            # 保存文件
            ref_image_path = upload_dir / f"{task_id}_reference{file_ext}"
            with open(ref_image_path, "wb") as buffer:
                shutil.copyfileobj(reference_image.file, buffer)
            
            # 验证图像文件
            try:
                # 打开并验证图像
                img = Image.open(ref_image_path)
                img.verify()  # 验证文件完整性
                # verify后需要重新打开才能获取size
                img = Image.open(ref_image_path)
                width, height = img.size
                print(f"  ✓ 参考图像已上传: {reference_image.filename} ({width}x{height})")
            except Exception as e:
                if ref_image_path.exists():
                    ref_image_path.unlink()  # 删除无效文件
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"无效的图像文件: {str(e)}"
                )
            
            # 根据类型设置参考图像路径
            if reference_image_type == "face":
                face_reference_image_path = ref_image_path
                print(f"  ℹ 使用面部参考图像: {ref_image_path.name}")
            else:
                reference_image_path = ref_image_path
                print(f"  ℹ 使用场景参考图像: {ref_image_path.name}")
        
        print(f"🎨 开始生成图像 (任务: {task_id})...")
        print(f"   用户: {user_id} ({tier})")
        print(f"   提示词: {prompt[:50]}...")
        print(f"   模型引擎: {model_engine or 'auto（自动选择）'}")
        print(f"   任务类型: {task_type or 'auto（自动检测）'}")
        print(f"   使用 ModelManager: {use_model_manager}")
        if use_model_manager:
            print(f"   ModelManager 任务: {task or 'auto（自动选择）'}")
        print(f"   LoRA设置: character_lora={character_lora}, style_lora={style_lora}")
        if reference_image_path:
            print(f"   场景参考图像: {reference_image_path.name}")
        if face_reference_image_path:
            print(f"   面部参考图像: {face_reference_image_path.name}")
        
        start_time = time.time()
        
        # 如果使用 ModelManager（默认启用）
        if use_model_manager:
            if MODEL_MANAGER_AVAILABLE:
                try:
                    manager = get_model_manager()
                    
                    # 如果没有指定 task，根据提示词自动选择
                    task_for_manager = task
                    if not task_for_manager:
                        prompt_lower = prompt.lower()
                        if any(kw in prompt_lower for kw in ["主持人", "人脸", "角色", "人物", "face", "character"]):
                            task_for_manager = "host_face"
                        elif any(kw in prompt_lower for kw in ["实验室", "医学", "医疗", "lab", "medical"]):
                            task_for_manager = "lab_scene"
                        elif any(kw in prompt_lower for kw in ["量子", "粒子", "太空", "宇宙", "quantum", "particle", "space"]):
                            task_for_manager = "science_background"
                        elif any(kw in prompt_lower for kw in ["中国", "官方", "宣传", "教育", "chinese", "official"]):
                            task_for_manager = "official_style"
                        else:
                            task_for_manager = "science_background"
                    
                    print(f"  🎯 使用 ModelManager，任务类型: {task_for_manager}")
                    
                    # 处理人脸参考图像（用于 InstantID）
                    face_image = None
                    if face_reference_image_path:
                        try:
                            face_image = Image.open(face_reference_image_path)
                            print(f"  ✅ 已加载上传的人脸参考图像，将使用 InstantID")
                        except Exception as e:
                            print(f"  ⚠️  人脸图像加载失败: {e}")
                    
                    # 使用 ModelManager 生成
                    # 如果 num_inference_steps 或 guidance_scale 为 None，让 ModelManager 从配置读取
                    image = manager.generate(
                        task=task_for_manager,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps if num_inference_steps is not None else None,  # None 表示从配置读取
                        guidance_scale=guidance_scale if guidance_scale is not None else None,  # None 表示从配置读取
                        seed=seed,
                        face_image=face_image,  # 上传的图片（优先）
                        face_image_name=face_image_name,  # 从目录加载的图片名（备选）
                        face_strength=0.8  # InstantID 强度，可配置
                    )
                    
                    # 保存图像
                    image.save(output_path)
                    
                    # 获取使用的模型
                    routed_model = manager.route(task_for_manager)
                    
                    elapsed_time = time.time() - start_time
                    
                    print(f"  ✅ ModelManager 生成成功")
                    print(f"  使用模型: {routed_model}")
                    print(f"  生成时间: {elapsed_time:.1f} 秒")
                    
                    # 返回响应
                    file_size = output_path.stat().st_size if output_path.exists() else 0
                    
                    return ImageResponse(
                        task_id=task_id,
                        status="completed",
                        image_url=f"/api/v1/files/images/{task_id}.png",
                        image_path=str(output_path),
                        width=width,
                        height=height,
                        file_size=file_size,
                        quota_remaining=get_quota_remaining(user_id, tier),
                        created_at=datetime.now(),
                        metadata={"model_used": routed_model, "task": task_for_manager, "generation_time": elapsed_time}
                    )
                except Exception as e:
                    print(f"  ⚠️  ModelManager 生成失败，回退到 ImageGenerator: {e}")
                    import traceback
                    traceback.print_exc()
                    # 继续使用原来的 ImageGenerator
            else:
                print(f"  ⚠️  ModelManager 不可用，使用 ImageGenerator")
                # 继续使用原来的 ImageGenerator
        
        # 获取生成器并生成图像（原有逻辑，当 ModelManager 未启用或失败时使用）
        generator = get_image_generator()
        
        # 准备场景信息
        # 通用提示词优化：只做必要的替换，不添加额外内容
        optimized_prompt = prompt
        
        # 1. 替换"高空拍摄"为"鸟瞰视角"（避免被误解为飞机视角）
        if "高空拍摄" in optimized_prompt or "高空" in optimized_prompt:
            optimized_prompt = optimized_prompt.replace("高空拍摄", "鸟瞰视角")
            optimized_prompt = optimized_prompt.replace("高空", "鸟瞰")
            print(f"  ℹ 提示词优化: 将'高空拍摄'替换为'鸟瞰视角'")
        
        # 2. 检查并警告提示词长度（不自动增强，保持通用性）
        try:
            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            current_tokens = len(tokenizer(optimized_prompt, truncation=False, return_tensors="pt").input_ids[0])
            
            if current_tokens > 77:
                print(f"  ⚠ 警告: 提示词长度 ({current_tokens} tokens) 超过 77 tokens 限制，将被 CLIP 自动截断")
                print(f"  ⚠ 建议精简提示词以避免关键信息丢失")
            elif current_tokens > 70:
                print(f"  ℹ 提示词长度: {current_tokens} tokens (接近77 tokens限制)")
            else:
                print(f"  ℹ 提示词长度: {current_tokens} tokens")
        except Exception:
            # 如果无法加载tokenizer，使用简单估算
            chinese_chars = sum(1 for c in optimized_prompt if ord(c) > 127)
            english_words = len([w for w in optimized_prompt.split() if not any(ord(c) > 127 for c in w)])
            estimated_tokens = int(chinese_chars * 1.5 + english_words * 1.3)
            if estimated_tokens > 77:
                print(f"  ⚠ 警告: 提示词可能超过 77 tokens 限制（估算: {estimated_tokens} tokens）")
        
        # 3. 负面提示词处理（保持通用性，不添加默认项）
        # 注意：为了保持通用性，不自动添加任何默认负面提示词
        # 用户可以根据需要自己提供负面提示词
        enhanced_negative = negative_prompt or ""
        if enhanced_negative:
            print(f"  ℹ 使用用户提供的负面提示词")
        else:
            print(f"  ℹ 未提供负面提示词，使用空字符串（保持通用性）")
        
        print(f"  ℹ 优化后的提示词: {optimized_prompt[:100]}...")
        print(f"  ℹ 负面提示词: {enhanced_negative[:100]}...")
        
        scene = {
            "prompt": optimized_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "negative_prompt": enhanced_negative,
        }
        
        # 同步调用生成器（原有逻辑）
        generated_image_path = generator.generate_image(
            prompt=optimized_prompt,
            output_path=output_path,
            negative_prompt=enhanced_negative,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            reference_image_path=reference_image_path,
            face_reference_image_path=face_reference_image_path,
            use_lora=None,
            character_lora=character_lora,
            style_lora=style_lora,
            scene=scene,
            model_engine=model_engine if model_engine != "auto" else None,  # auto 时传 None，让系统自动选择
            task_type=task_type,
        )
        elapsed_time = time.time() - start_time
        
        # 获取文件信息
        file_size = generated_image_path.stat().st_size if generated_image_path.exists() else 0
        
        print(f"✅ 图像生成完成 (任务: {task_id}, 耗时: {elapsed_time:.1f}秒)")
        print(f"   输出路径: {generated_image_path}")
        
        # 构建URL（相对路径）
        image_url = f"/api/v1/files/images/{task_id}.png"
        
        return ImageResponse(
            task_id=task_id,
            status="completed",
            image_url=image_url,
            image_path=str(generated_image_path),
            width=width,
            height=height,
            file_size=file_size,
            quota_remaining=get_quota_remaining(user_id, tier),
            created_at=datetime.now()
        )
        
    except Exception as e:
        print(f"❌ 图像生成失败 (任务: {task_id}): {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图像生成失败: {str(e)}"
        )

@app.post("/api/v1/videos/generate", response_model=VideoResponse)
async def generate_video(
    request: VideoRequest,
    current_user: dict = Depends(verify_api_key)
):
    """
    生成视频（同步模式）
    
    - **scenes**: 场景列表（至少1个，最多10个）
    - **fps**: 帧率（15-30）
    - **width/height**: 视频分辨率
    
    注意：此模式会同步执行，可能需要几分钟到十几分钟
    """
    user_id = current_user["user_id"]
    tier = current_user["tier"]
    
    # 检查配额
    if not check_quota(user_id, tier, "video"):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"今日视频生成配额已用完（{DEFAULT_QUOTAS[tier]['videos']}个/天）"
        )
    
    task_id = str(uuid.uuid4())
    
    try:
        # 准备输出路径
        output_dir = Path(__file__).parent.parent.parent / "outputs" / "api" / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task_id}.mp4"
        
        print(f"🎬 开始生成视频 (任务: {task_id})...")
        print(f"   用户: {user_id} ({tier})")
        print(f"   场景数: {len(request.scenes)}")
        print(f"   分辨率: {request.width}x{request.height}")
        print(f"   帧率: {request.fps} fps")
        
        total_start_time = time.time()
        
        # 获取生成器
        image_generator = get_image_generator()
        video_generator = get_video_generator()
        
        # 准备场景数据
        scenes_data = []
        image_paths = []
        
        # 第一步：为每个场景生成或获取图像
        print(f"📸 步骤1: 生成/获取场景图像...")
        image_output_dir = output_dir / "images"
        image_output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, scene in enumerate(request.scenes):
            scene_id = i + 1
            print(f"  处理场景 {scene_id}/{len(request.scenes)}: {scene.prompt[:50]}...")
            
            # 如果场景已有图像路径，直接使用
            if scene.image_path and Path(scene.image_path).exists():
                image_path = Path(scene.image_path)
                print(f"  ✓ 使用已有图像: {image_path.name}")
            else:
                # 需要生成图像
                print(f"  🎨 生成图像...")
                scene_image_path = image_output_dir / f"scene_{scene_id:03d}.png"
                
                # 调用图像生成API
                # 注意：width和height通过scene字典传递，不是直接参数
                # 使用优化后的参数：None 表示从配置读取（28步，7.5引导强度）
                generated_image_path = image_generator.generate_image(
                    prompt=scene.prompt,
                    output_path=scene_image_path,
                    negative_prompt=None,
                    num_inference_steps=None,  # None 表示从配置读取（优化后的28步）
                    guidance_scale=None,  # None 表示从配置读取（7.5）
                    seed=None,
                    reference_image_path=None,
                    face_reference_image_path=None,
                    use_lora=None,
                    character_lora=None,
                    style_lora=None,
                    scene={
                        "id": scene_id,
                        "prompt": scene.prompt,
                        "width": request.width,
                        "height": request.height,
                    },
                    model_engine="auto",  # 自动选择模型
                    task_type="scene"  # 场景生成
                )
                image_path = Path(generated_image_path)
                print(f"  ✓ 图像生成完成: {image_path.name}")
            
            image_paths.append(image_path)
            # 构建场景数据，包含风格信息
            scene_data = {
                "id": scene_id,
                "prompt": scene.prompt,
                "description": scene.prompt,  # 使用prompt作为description
                "duration": scene.duration,
                "image_path": str(image_path),
            }
            # 添加风格配置（如果提供）
            if scene.style:
                scene_data["style"] = scene.style
            if scene.visual:
                scene_data["visual"] = scene.visual
            elif scene.style:
                # 如果只有style，构建visual对象
                scene_data["visual"] = {"style": scene.style}
            
            scenes_data.append(scene_data)
        
        # 第二步：为每个图像生成视频片段
        print(f"🎬 步骤2: 生成视频片段...")
        video_segments = []
        segments_dir = output_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (scene_data, image_path) in enumerate(zip(scenes_data, image_paths)):
            scene_id = scene_data["id"]
            duration = scene_data["duration"]
            
            # 计算需要的帧数
            num_frames = int(duration * request.fps)
            if num_frames < 14:  # SVD最少需要14帧
                num_frames = 14
                print(f"  ⚠ 场景 {scene_id} 时长过短，调整为最少帧数: {num_frames}")
            
            video_segment_path = segments_dir / f"scene_{scene_id:03d}.mp4"
            
            print(f"  生成场景 {scene_id}/{len(scenes_data)} 视频片段 ({duration}秒, {num_frames}帧)...")
            
            # 调用视频生成器
            # 使用用户提供的参数，或使用更保守的默认值（更自然的动画）
            motion_bucket_id = request.motion_bucket_id if request.motion_bucket_id is not None else 1.3  # 默认1.3，更自然
            noise_aug_strength = request.noise_aug_strength if request.noise_aug_strength is not None else 0.0002  # 默认0.0002，更平滑
            
            print(f"  ℹ 运动参数: motion_bucket_id={motion_bucket_id}, noise_aug_strength={noise_aug_strength}")
            
            generated_video_path = video_generator.generate_video(
                image_path=str(image_path),
                output_path=str(video_segment_path),
                num_frames=num_frames,
                fps=request.fps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                scene=scene_data,
            )
            
            video_segments.append(Path(generated_video_path))
            print(f"  ✓ 视频片段生成完成: {Path(generated_video_path).name}")
        
        # 第三步：拼接所有视频片段
        print(f"🔗 步骤3: 拼接视频片段...")
        
        if len(video_segments) > 1:
            # 使用VideoComposer拼接视频
            try:
                from video_composer import VideoComposer
                config_path = Path(__file__).parent.parent / "config.yaml"
                composer = VideoComposer(config_path=str(config_path))
                
                video_segment_paths = [str(vp) for vp in video_segments]
                final_video_path = composer.concat_videos_ffmpeg(
                    video_segment_paths,
                    str(output_path)
                )
                print(f"  ✓ 视频拼接完成: {Path(final_video_path).name}")
            except Exception as e:
                print(f"  ⚠ VideoComposer拼接失败: {e}，尝试使用FFmpeg直接拼接")
                # 备用方案：直接使用FFmpeg拼接
                import subprocess
                concat_file = output_dir / "concat_list.txt"
                with open(concat_file, 'w', encoding='utf-8') as f:
                    for video_path in video_segments:
                        f.write(f"file '{video_path.absolute()}'\n")
                
                subprocess.run([
                    'ffmpeg', '-f', 'concat', '-safe', '0',
                    '-i', str(concat_file),
                    '-c', 'copy',
                    '-y', str(output_path)
                ], check=True, capture_output=True)
                print(f"  ✓ 视频拼接完成（使用FFmpeg）")
        else:
            # 只有一个片段，直接复制
            shutil.copy(video_segments[0], output_path)
            print(f"  ✓ 单场景视频完成")
        
        elapsed_time = time.time() - total_start_time
        
        # 检查文件大小
        file_size = output_path.stat().st_size if output_path.exists() else 0
        
        if not output_path.exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="视频生成失败：输出文件不存在"
            )
        
        # 注意：配额已在 check_quota() 调用时自动使用，无需再次调用
        
        total_duration = sum(s.duration for s in request.scenes)
        
        print(f"✅ 视频生成完成 (任务: {task_id}, 耗时: {elapsed_time:.1f}秒)")
        print(f"   输出路径: {output_path}")
        print(f"   文件大小: {file_size / 1024 / 1024:.2f} MB")
        print(f"   总时长: {total_duration:.1f}秒")
        
        # 生成视频URL
        video_url = f"/api/v1/files/videos/{task_id}.mp4"
        
        # 获取剩余配额
        quota_remaining = get_quota_remaining(user_id, tier)
        
        # 返回响应
        return VideoResponse(
            task_id=task_id,
            status="completed",
            video_url=video_url,
            video_path=str(output_path),
            duration=total_duration,
            file_size=file_size,
            quota_remaining=quota_remaining,
            created_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 视频生成失败 (任务: {task_id}): {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"视频生成失败: {str(e)}"
        )

@app.get("/api/v1/quota")
async def get_quota(current_user: dict = Depends(verify_api_key)):
    """获取当前配额信息"""
    user_id = current_user["user_id"]
    tier = current_user["tier"]
    
    quota_remaining = get_quota_remaining(user_id, tier)
    limits = DEFAULT_QUOTAS.get(tier, DEFAULT_QUOTAS["free"])
    
    return {
        "user_id": user_id,
        "tier": tier,
        "limits": limits,
        "remaining": quota_remaining,
        "reset_at": (datetime.now() + timedelta(days=1)).date().isoformat(),
    }

@app.get("/api/v1/files/images/{filename}")
async def get_image(filename: str):
    """获取生成的图像文件"""
    image_path = Path(__file__).parent.parent.parent / "outputs" / "api" / "images" / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="图像文件不存在")
    return FileResponse(image_path)

@app.get("/api/v1/files/videos/{filename}")
async def get_video(filename: str):
    """获取生成的视频文件"""
    video_path = Path(__file__).parent.parent.parent / "outputs" / "api" / "videos" / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="视频文件不存在")
    return FileResponse(video_path)

# ==================== 科普视频生成 ====================

class KepuVideoRequest(BaseModel):
    """科普视频生成请求"""
    topic: str = Field(..., min_length=1, max_length=200, description="选题标题（如：什么是黑洞？）")
    ip_character: str = Field("kepu_gege", description="IP角色：kepu_gege（科普哥哥）或 weilai_jiejie（未来姐姐）")
    duration: Optional[int] = Field(None, ge=30, le=300, description="视频时长（秒），可选，默认从知识库读取")

@app.post("/api/v1/kepu/generate", response_model=VideoResponse)
async def generate_kepu_video(
    request: KepuVideoRequest,
    current_user: dict = Depends(verify_api_key)
):
    """
    生成科普视频（完整流水线）
    
    - **topic**: 选题标题（如：什么是黑洞？）
    - **ip_character**: IP角色（kepu_gege 或 weilai_jiejie）
    - **duration**: 视频时长（秒），可选
    
    注意：此接口会生成完整的科普视频，包括：
    1. 从知识库读取选题信息
    2. 生成脚本JSON
    3. 生成场景图像（使用科普主持人）
    4. 生成配音
    5. 生成视频片段
    6. 合成最终视频
    
    预计耗时：3-10分钟（取决于场景数量）
    """
    user_id = current_user["user_id"]
    tier = current_user["tier"]
    
    # 检查配额
    if not check_quota(user_id, tier, "video"):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"今日视频生成配额已用完（{DEFAULT_QUOTAS[tier]['videos']}个/天）"
        )
    
    task_id = str(uuid.uuid4())
    
    try:
        # 导入科普视频生成器
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from tools.kepu_quick_generate import KepuQuickGenerator
        
        # 准备输出路径
        output_dir = Path(__file__).parent.parent.parent / "outputs" / "api" / "kepu_videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎬 开始生成科普视频 (任务: {task_id})...")
        print(f"   用户: {user_id} ({tier})")
        print(f"   选题: {request.topic}")
        print(f"   IP角色: {request.ip_character}")
        
        total_start_time = time.time()
        
        # 初始化科普视频生成器（延迟加载，不占用启动显存）
        config_path = Path(__file__).parent.parent / "config.yaml"
        generator = KepuQuickGenerator(config_path=str(config_path))
        
        # 生成视频
        output_video = generator.generate_video(
            topic_title=request.topic,
            ip_character=request.ip_character
        )
        
        elapsed_time = time.time() - total_start_time
        
        # 检查文件是否存在
        if not output_video.exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="视频生成失败：输出文件不存在"
            )
        
        # 获取文件信息
        file_size = output_video.stat().st_size if output_video.exists() else 0
        
        # 复制到API输出目录
        api_output_path = output_dir / f"{task_id}.mp4"
        shutil.copy2(output_video, api_output_path)
        
        # 获取视频时长
        try:
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(api_output_path)],
                capture_output=True,
                text=True
            )
            duration = float(result.stdout.strip()) if result.stdout.strip() else 60.0
        except:
            duration = 60.0  # 默认值
        
        print(f"✅ 科普视频生成完成 (任务: {task_id}, 耗时: {elapsed_time:.1f}秒)")
        print(f"   输出路径: {api_output_path}")
        print(f"   文件大小: {file_size / 1024 / 1024:.2f} MB")
        print(f"   时长: {duration:.1f}秒")
        
        # 生成视频URL
        video_url = f"/api/v1/files/kepu_videos/{task_id}.mp4"
        
        # 获取剩余配额
        quota_remaining = get_quota_remaining(user_id, tier)
        
        # 返回响应
        return VideoResponse(
            task_id=task_id,
            status="completed",
            video_url=video_url,
            video_path=str(api_output_path),
            duration=duration,
            file_size=file_size,
            quota_remaining=quota_remaining,
            created_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 科普视频生成失败 (任务: {task_id}): {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"科普视频生成失败: {str(e)}"
        )

@app.get("/api/v1/kepu/topics")
async def list_kepu_topics(current_user: dict = Depends(verify_api_key)):
    """列出所有可用的科普选题"""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from tools.kepu_quick_generate import KepuQuickGenerator
        
        config_path = Path(__file__).parent.parent / "config.yaml"
        generator = KepuQuickGenerator(config_path=str(config_path))
        
        topics = generator.list_topics()
        
        return {
            "topics": topics,
            "total": len(topics)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取选题列表失败: {str(e)}"
        )

@app.get("/api/v1/files/kepu_videos/{filename}")
async def get_kepu_video(filename: str):
    """获取生成的科普视频文件"""
    video_path = Path(__file__).parent.parent.parent / "outputs" / "api" / "kepu_videos" / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="视频文件不存在")
    return FileResponse(video_path)

# ==================== 启动 ====================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 启动AI视频生成平台MVP API")
    print("=" * 60)
    print(f"📖 API文档: http://localhost:8000/docs")
    print(f"🔑 测试API Key: test-key-123 (免费版)")
    print(f"🔑 演示API Key: demo-key-456 (付费版)")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)

