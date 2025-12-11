"""
Celery任务定义
处理异步任务：图像生成、视频生成等
"""
from celery import Task
from .celery_app import celery_app
import sys
from pathlib import Path
import traceback
from typing import Dict, Any, Optional
import json

# 添加父目录到路径，以便导入生成器模块
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class CallbackTask(Task):
    """带有回调的任务基类"""
    def on_success(self, retval, task_id, args, kwargs):
        print(f"✅ 任务 {task_id} 成功完成")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        print(f"❌ 任务 {task_id} 失败: {exc}")
        print(f"错误信息: {traceback.format_exc()}")

@celery_app.task(
    name="generate_image",
    base=CallbackTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60
)
def generate_image_task(
    self,
    task_id: str,
    prompt: str,
    user_id: str,
    config_path: Optional[str] = None,
    **kwargs
):
    """
    异步图像生成任务
    
    Args:
        task_id: 任务ID
        prompt: 生成提示词
        user_id: 用户ID
        config_path: 配置文件路径
        **kwargs: 其他参数（width, height, num_inference_steps等）
    
    Returns:
        生成的图像路径
    """
    try:
        # 更新任务状态为processing
        update_task_status(task_id, "processing", progress=10)
        
        # 导入图像生成器（延迟导入，避免启动时加载模型）
        from image_generator import ImageGenerator
        
        # 确定配置文件路径
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config.yaml")
        
        # 创建图像生成器
        print(f"🔧 初始化图像生成器 (任务: {task_id})...")
        generator = ImageGenerator(config_path)
        
        update_task_status(task_id, "processing", progress=30)
        
        # 生成图像
        print(f"🎨 开始生成图像 (任务: {task_id})...")
        print(f"   提示词: {prompt[:50]}...")
        
        # 构建场景字典（用于生成器）
        scene = {
            "prompt": prompt,
            "width": kwargs.get("width", 1536),
            "height": kwargs.get("height", 864),
            "num_inference_steps": kwargs.get("num_inference_steps", 40),
            "guidance_scale": kwargs.get("guidance_scale", 7.5),
            "seed": kwargs.get("seed"),
            "character_id": kwargs.get("character_id"),
            "scene_config": kwargs.get("scene_config"),
            "style": kwargs.get("style", "xianxia"),
        }
        
        update_task_status(task_id, "processing", progress=50)
        
        # 准备输出路径
        output_dir = Path(__file__).parent.parent.parent / "outputs" / "api" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task_id}.png"
        
        # 调用真实的图像生成器
        print(f"🎨 调用图像生成器生成图像...")
        generated_image_path = generator.generate_image(
            prompt=prompt,
            output_path=output_path,
            negative_prompt=kwargs.get("negative_prompt"),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            num_inference_steps=kwargs.get("num_inference_steps", 40),
            seed=kwargs.get("seed"),
            reference_image_path=None,  # 可以根据character_id加载参考图
            face_reference_image_path=None,  # 可以根据character_id加载人脸参考图
            use_lora=None,  # 使用默认配置
            scene=scene,  # 传递完整场景信息
        )
        
        update_task_status(task_id, "processing", progress=90)
        
        # 准备返回结果
        result = {
            "image_path": str(generated_image_path),
            "thumbnail": str(generated_image_path),  # 可以后续生成缩略图
            "width": scene["width"],
            "height": scene["height"],
            "file_size": generated_image_path.stat().st_size if generated_image_path.exists() else 0,
        }
        
        update_task_status(task_id, "completed", progress=100, result=result)
        
        print(f"✅ 图像生成完成 (任务: {task_id})")
        return result
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"❌ 图像生成失败 (任务: {task_id}): {error_msg}")
        
        update_task_status(task_id, "failed", error=error_msg)
        
        # 重试机制
        if self.request.retries < self.max_retries:
            print(f"🔄 重试任务 {task_id} ({self.request.retries + 1}/{self.max_retries})")
            raise self.retry(exc=e, countdown=60)
        else:
            raise e

@celery_app.task(
    name="generate_video",
    base=CallbackTask,
    bind=True,
    max_retries=2,  # 视频生成任务重试次数较少（耗时更长）
    default_retry_delay=120
)
def generate_video_task(
    self,
    task_id: str,
    scenes: list,
    user_id: str,
    config_path: Optional[str] = None,
    **kwargs
):
    """
    异步视频生成任务
    
    Args:
        task_id: 任务ID
        scenes: 场景列表
        user_id: 用户ID
        config_path: 配置文件路径
        **kwargs: 其他参数（video_config等）
    
    Returns:
        生成的视频路径
    """
    try:
        update_task_status(task_id, "processing", progress=5)
        
        # 导入视频生成器
        from video_generator import VideoGenerator
        
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config.yaml")
        
        print(f"🔧 初始化视频生成器 (任务: {task_id})...")
        generator = VideoGenerator(config_path)
        
        update_task_status(task_id, "processing", progress=20)
        
        print(f"🎬 开始生成视频 (任务: {task_id})...")
        print(f"   场景数: {len(scenes)}")
        
        # 准备输出目录
        output_dir = Path(__file__).parent.parent.parent / "outputs" / "api" / "videos" / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        update_task_status(task_id, "processing", progress=30)
        
        # 第一步：为每个场景生成图像（如果还没有）
        # 导入图像生成器
        from image_generator import ImageGenerator
        image_gen = ImageGenerator(config_path)
        
        image_paths = []
        for idx, scene in enumerate(scenes):
            scene_image_path = scene.get("image_path")
            
            # 如果场景已有图像路径，直接使用
            if scene_image_path and Path(scene_image_path).exists():
                print(f"  ✓ 场景 {idx+1} 使用已有图像: {scene_image_path}")
                image_paths.append(Path(scene_image_path))
            else:
                # 需要先生成图像
                print(f"  🎨 为场景 {idx+1} 生成图像...")
                scene_image_path = output_dir / "images" / f"scene_{idx+1:03d}.png"
                scene_image_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 生成图像
                generated_path = image_gen.generate_image(
                    prompt=scene.get("prompt", ""),
                    output_path=scene_image_path,
                    scene=scene,
                    negative_prompt=scene.get("negative_prompt"),
                    guidance_scale=scene.get("guidance_scale", 7.5),
                    num_inference_steps=scene.get("num_inference_steps", 40),
                    seed=scene.get("seed"),
                )
                image_paths.append(generated_path)
        
        update_task_status(task_id, "processing", progress=60)
        
        # 第二步：为每个图像生成视频片段
        video_segments = []
        total_duration = 0
        
        for idx, (scene, image_path) in enumerate(zip(scenes, image_paths)):
            print(f"  🎬 为场景 {idx+1} 生成视频...")
            
            # 计算帧数（根据duration）
            duration = scene.get("duration", 5.0)
            fps = kwargs.get("video_config", {}).get("fps", 24)
            num_frames = int(duration * fps)
            
            video_output_path = output_dir / "segments" / f"scene_{idx+1:03d}.mp4"
            video_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 生成视频
            generated_video = generator.generate_video(
                image_path=str(image_path),
                output_path=str(video_output_path),
                num_frames=num_frames,
                fps=fps,
                scene=scene,
            )
            video_segments.append(Path(generated_video))
            total_duration += duration
        
        update_task_status(task_id, "processing", progress=80)
        
        # 第三步：拼接所有视频片段（如果需要）
        if len(video_segments) > 1:
            print(f"  🔗 拼接 {len(video_segments)} 个视频片段...")
            final_output_path = output_dir / f"{task_id}.mp4"
            
            # 使用ffmpeg拼接视频
            import subprocess
            concat_file = output_dir / "concat_list.txt"
            with open(concat_file, 'w') as f:
                for video_path in video_segments:
                    f.write(f"file '{video_path.absolute()}'\n")
            
            subprocess.run([
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                '-y', str(final_output_path)
            ], check=True, capture_output=True)
            
            output_path = final_output_path
        else:
            output_path = video_segments[0]
        
        update_task_status(task_id, "processing", progress=95)
        
        # 准备返回结果
        result = {
            "video_path": str(output_path),
            "thumbnail": str(output_path),  # 可以后续生成缩略图
            "duration": total_duration,
            "scenes_count": len(scenes),
            "segments": [str(p) for p in video_segments],
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ 视频生成失败 (任务: {task_id}): {error_msg}")
        
        update_task_status(task_id, "failed", error=error_msg)
        
        if self.request.retries < self.max_retries:
            print(f"🔄 重试任务 {task_id} ({self.request.retries + 1}/{self.max_retries})")
            raise self.retry(exc=e, countdown=120)
        else:
            raise e

# 任务状态管理（简化版，后续可以改为数据库）
_task_storage = {}

def update_task_status(
    task_id: str,
    status: str,
    progress: Optional[int] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
):
    """
    更新任务状态（简化版，使用内存存储）
    后续应该改为数据库存储
    """
    from datetime import datetime
    
    if task_id not in _task_storage:
        _task_storage[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "progress": 0,
            "result": None,
            "error": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
    
    task_info = _task_storage[task_id]
    task_info["status"] = status
    task_info["updated_at"] = datetime.now()
    
    if progress is not None:
        task_info["progress"] = progress
    if result is not None:
        task_info["result"] = result
    if error is not None:
        task_info["error"] = error
    
    return task_info

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """获取任务状态"""
    return _task_storage.get(task_id)

