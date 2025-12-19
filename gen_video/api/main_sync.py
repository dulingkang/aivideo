"""
AI视频生成平台 - API主入口（同步版本，不依赖Redis）
用于测试和开发环境，直接调用生成器，不经过任务队列
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime
from pathlib import Path
import sys
import json

# 添加路径以便导入生成器
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = FastAPI(
    title="AI Video Generation Platform (Sync Mode)",
    description="通用AI视频生成平台API（同步模式，不依赖Redis）",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2配置（简化版）
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ==================== 数据模型 ====================

class ImageRequest(BaseModel):
    """图像生成请求"""
    prompt: str = Field(..., min_length=1, max_length=500, description="生成提示词")
    negative_prompt: Optional[str] = Field(None, max_length=500, description="负面提示词")
    width: int = Field(1536, ge=512, le=2048, description="图像宽度（必须是8的倍数）")
    height: int = Field(864, ge=512, le=2048, description="图像高度（必须是8的倍数）")
    num_inference_steps: int = Field(40, ge=10, le=100, description="推理步数")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="引导尺度")
    seed: Optional[int] = Field(None, description="随机种子")
    character_id: Optional[str] = Field(None, description="角色ID")
    scene_config: Optional[Dict[str, Any]] = Field(None, description="场景配置")
    style: Optional[str] = Field("xianxia", description="风格")
    output_format: str = Field("png", pattern="^(png|jpg|jpeg)$", description="输出格式")
    
    @validator('width', 'height')
    def validate_resolution(cls, v):
        if v % 8 != 0:
            raise ValueError('Resolution must be multiple of 8')
        return v

class VideoScene(BaseModel):
    """视频场景"""
    id: int = Field(..., description="场景ID")
    prompt: str = Field(..., min_length=1, max_length=500, description="场景提示词")
    duration: float = Field(5.0, ge=1.0, le=60.0, description="场景时长（秒）")
    image_path: Optional[str] = Field(None, description="预生成的图像路径（可选）")

class VideoRequest(BaseModel):
    """视频生成请求"""
    scenes: List[VideoScene] = Field(..., min_items=1, max_items=100, description="场景列表")
    video_config: Optional[Dict[str, Any]] = Field(None, description="视频配置")
    output_format: str = Field("mp4", pattern="^(mp4|avi|mov)$", description="输出格式")

class M6VideoRequest(BaseModel):
    """M6 单段视频生成请求（Anchor I2V + 身份验证）"""
    prompt: str = Field(..., min_length=1, max_length=800, description="视频提示词（建议包含动作/场景描述）")
    input_image_path: str = Field(..., description="Anchor 输入图路径（服务器本地路径）")
    reference_image_path: Optional[str] = Field(None, description="参考图路径（不传则使用 input_image_path）")
    shot_type: str = Field("medium", pattern="^(wide|medium|medium_close|close|extreme_close)$", description="镜头类型")
    motion_intensity: str = Field("moderate", pattern="^(gentle|moderate|dynamic)$", description="运动强度")
    quick: bool = Field(False, description="快速模式（更少步数/更少重试，适合冒烟测试）")
    num_frames: Optional[int] = Field(None, ge=8, le=240, description="覆盖 HunyuanVideo num_frames（可选）")
    num_inference_steps: Optional[int] = Field(None, ge=1, le=60, description="覆盖 HunyuanVideo 推理步数（可选）")
    max_retries: Optional[int] = Field(None, ge=0, le=10, description="覆盖验证失败后的最大重试次数（可选）")

class M6VideoResponse(BaseModel):
    """M6 单段视频生成响应（含身份验证指标）"""
    task_id: str
    status: str
    video_path: Optional[str] = None
    report_path: Optional[str] = None
    passed: Optional[bool] = None
    avg_similarity: Optional[float] = None
    min_similarity: Optional[float] = None
    drift_ratio: Optional[float] = None
    face_detect_ratio: Optional[float] = None
    issues: Optional[List[str]] = None
    created_at: datetime

class ImageResponse(BaseModel):
    """图像生成响应"""
    task_id: str
    status: str
    image_path: Optional[str] = None
    thumbnail: Optional[str] = None
    width: int
    height: int
    file_size: Optional[int] = None
    created_at: datetime

# ==================== 认证（简化版）====================

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """验证用户token（简化版）"""
    return {"user_id": "default", "api_key": token}

# ==================== 生成器初始化 ====================

_image_generator = None
_video_generator = None
_m6_video_generator = None
_config_path = None

def get_image_generator():
    """获取图像生成器（单例）"""
    global _image_generator, _config_path
    if _image_generator is None:
        if _config_path is None:
            _config_path = str(Path(__file__).parent.parent / "config.yaml")
        print("🔧 初始化图像生成器...")
        from image_generator import ImageGenerator
        _image_generator = ImageGenerator(_config_path)
    return _image_generator

def get_video_generator():
    """获取视频生成器（单例）"""
    global _video_generator, _config_path
    if _video_generator is None:
        if _config_path is None:
            _config_path = str(Path(__file__).parent.parent / "config.yaml")
        print("🔧 初始化视频生成器...")
        from video_generator import VideoGenerator
        _video_generator = VideoGenerator(_config_path)
    return _video_generator

def get_m6_video_generator():
    """获取 M6 视频生成器（单例）"""
    global _m6_video_generator, _config_path
    if _m6_video_generator is None:
        if _config_path is None:
            _config_path = str(Path(__file__).parent.parent / "config.yaml")
        print("🔧 初始化 M6 视频生成器（EnhancedVideoGeneratorM6）...")
        from enhanced_video_generator_m6 import EnhancedVideoGeneratorM6
        _m6_video_generator = EnhancedVideoGeneratorM6(_config_path)
    return _m6_video_generator

# ==================== API端点 ====================

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "AI Video Generation Platform API (Sync Mode)",
        "version": "1.0.0",
        "mode": "sync",
        "docs": "/docs",
        "note": "此模式不依赖Redis，直接同步调用生成器"
    }

@app.post("/api/v1/images/generate", response_model=ImageResponse)
async def generate_image(
    request: ImageRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    生成图像（同步模式）
    
    - **prompt**: 生成提示词
    - **width/height**: 图像分辨率（必须是8的倍数）
    - **character_id**: 角色ID（如果使用角色模板）
    - **scene_config**: 场景配置（相机、光照、情绪等）
    
    注意：此模式会同步执行，可能需要较长时间（30-60秒）
    """
    task_id = str(uuid.uuid4())
    
    try:
        # 准备输出路径
        output_dir = Path(__file__).parent.parent.parent / "outputs" / "api" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task_id}.png"
        
        # 准备场景信息
        scene = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "character_id": request.character_id,
            "scene_config": request.scene_config,
            "style": request.style,
            "negative_prompt": request.negative_prompt,
        }
        
        # 获取生成器并生成图像
        generator = get_image_generator()
        
        print(f"🎨 开始生成图像 (任务: {task_id})...")
        print(f"   提示词: {request.prompt[:50]}...")
        
        # 同步调用生成器
        generated_image_path = generator.generate_image(
            prompt=request.prompt,
            output_path=output_path,
            negative_prompt=request.negative_prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed,
            reference_image_path=None,
            face_reference_image_path=None,
            use_lora=None,
            scene=scene,
        )
        
        # 获取文件信息
        file_size = generated_image_path.stat().st_size if generated_image_path.exists() else 0
        
        print(f"✅ 图像生成完成 (任务: {task_id})")
        print(f"   输出路径: {generated_image_path}")
        
        return ImageResponse(
            task_id=task_id,
            status="completed",
            image_path=str(generated_image_path),
            thumbnail=str(generated_image_path),
            width=request.width,
            height=request.height,
            file_size=file_size,
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

@app.post("/api/v1/videos/generate")
async def generate_video(
    request: VideoRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    生成视频（同步模式）
    
    注意：此模式会同步执行，可能需要很长时间（几分钟到十几分钟）
    建议先测试图像生成，视频生成功能待Redis环境准备好后再测试
    """
    task_id = str(uuid.uuid4())
    
    return {
        "task_id": task_id,
        "status": "not_implemented",
        "message": "视频生成功能在同步模式下暂未实现，请使用异步模式（需要Redis）",
        "note": "视频生成耗时较长，建议使用异步任务队列"
    }

@app.post("/api/v1/m6/videos/generate", response_model=M6VideoResponse)
async def generate_m6_video(
    request: M6VideoRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    生成 M6 单段视频（Anchor I2V）并输出身份验证指标（同步模式）。
    - 输入为服务器本地的 anchor/reference 图片路径
    - 输出为视频路径 + verifier report JSON 路径 + 关键指标（passed/avg/min/drift/face）
    """
    task_id = str(uuid.uuid4())

    input_image = Path(request.input_image_path)
    if not input_image.exists():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"input_image_path 不存在: {input_image}")

    reference_image = Path(request.reference_image_path) if request.reference_image_path else input_image
    if not reference_image.exists():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"reference_image_path 不存在: {reference_image}")

    # 输出目录
    out_dir = Path(__file__).parent.parent.parent / "outputs" / "api" / "m6"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{task_id}.mp4"
    report_path = out_dir / f"{task_id}.json"

    generator = get_m6_video_generator()
    try:
        # 覆盖 hunyuanvideo 参数（按需）
        generator.video_config.setdefault("hunyuanvideo", {})
        hv = generator.video_config["hunyuanvideo"]

        # quick 默认（可被显式参数覆盖）
        if request.quick:
            hv["num_frames"] = int(request.num_frames or 24)
            hv["num_inference_steps"] = int(request.num_inference_steps or 8)
            if request.max_retries is None:
                max_retries = 0
            else:
                max_retries = int(request.max_retries)
        else:
            if request.num_frames is not None:
                hv["num_frames"] = int(request.num_frames)
            if request.num_inference_steps is not None:
                hv["num_inference_steps"] = int(request.num_inference_steps)
            max_retries = int(request.max_retries) if request.max_retries is not None else None

        scene = {
            "prompt": request.prompt,
            "motion_intensity": request.motion_intensity,
        }

        vp, result = generator.generate_video_with_identity_check(
            image_path=str(input_image),
            output_path=str(video_path),
            reference_image=str(reference_image),
            scene=scene,
            shot_type=request.shot_type,
            enable_verification=True,
            max_retries=max_retries,
        )

        payload: Dict[str, Any] = {
            "task_id": task_id,
            "input_image": str(input_image),
            "reference_image": str(reference_image),
            "video_path": str(vp) if vp else "",
            "shot_type": request.shot_type,
            "motion_intensity": request.motion_intensity,
            "passed": False,
            "avg_similarity": 0.0,
            "min_similarity": 0.0,
            "drift_ratio": 1.0,
            "face_detect_ratio": 0.0,
            "issues": [],
        }
        if result is not None:
            payload.update(
                {
                    "passed": bool(result.passed),
                    "avg_similarity": float(result.avg_similarity),
                    "min_similarity": float(result.min_similarity),
                    "drift_ratio": float(result.drift_ratio),
                    "face_detect_ratio": float(result.face_detect_ratio),
                    "issues": list(result.issues or []),
                }
            )
        else:
            payload["issues"] = ["无验证结果（result=None）"]

        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return M6VideoResponse(
            task_id=task_id,
            status="completed",
            video_path=str(vp) if vp else str(video_path),
            report_path=str(report_path),
            passed=bool(payload.get("passed")),
            avg_similarity=float(payload.get("avg_similarity", 0.0)),
            min_similarity=float(payload.get("min_similarity", 0.0)),
            drift_ratio=float(payload.get("drift_ratio", 1.0)),
            face_detect_ratio=float(payload.get("face_detect_ratio", 0.0)),
            issues=list(payload.get("issues") or []),
            created_at=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"M6 视频生成失败: {str(e)}")

@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "mode": "sync",
        "timestamp": datetime.now().isoformat(),
        "note": "同步模式，不依赖Redis"
    }

# ==================== 启动 ====================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 启动API服务器（同步模式）")
    print("=" * 60)
    print("⚠️  注意：此模式不依赖Redis，直接同步调用生成器")
    print("   图像生成可能需要30-60秒，请耐心等待")
    print("=" * 60)
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)

