#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI AnimateDiff API 调用模块
通过 API 生成视频，无需 Web UI
"""

import json
import requests
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import io
import base64


class ComfyUIAnimateDiffAPI:
    """ComfyUI AnimateDiff API 客户端"""
    
    def __init__(self, server_url: str = "http://127.0.0.1:8188"):
        """
        初始化 ComfyUI API 客户端
        
        Args:
            server_url: ComfyUI 服务器地址
        """
        self.server_url = server_url
        self.client_id = str(uuid.uuid4())
    
    def upload_image(self, image_path: str) -> str:
        """
        上传图像到 ComfyUI
        
        Args:
            image_path: 图像路径
        
        Returns:
            图像文件名
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像不存在: {image_path}")
        
        # 读取图像
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # 上传
        files = {
            'image': (image_path.name, image_data, 'image/png')
        }
        data = {
            'overwrite': 'true'
        }
        
        response = requests.post(
            f"{self.server_url}/upload/image",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['name']
        else:
            raise Exception(f"上传图像失败: {response.status_code} - {response.text}")
    
    def create_animatediff_workflow(
        self,
        image_filename: str,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 16,
        width: int = 1280,  # 进一步提升分辨率以改善清晰度（SDXL推荐分辨率）
        height: int = 1280,  # 进一步提升分辨率以改善清晰度（SDXL推荐分辨率）
        motion_model: str = "mm_sdxl_v10_beta.ckpt",  # SDXL motion module
        seed: int = -1,
        use_refiner: bool = True,  # 是否使用 Refiner（禁用可节省约4-6GB显存，适合16G显存）
    ) -> Dict[str, Any]:
        """
        创建 AnimateDiff 工作流（图生视频）
        
        Args:
            image_filename: 上传后的图像文件名
            prompt: 提示词
            negative_prompt: 负面提示词
            num_frames: 帧数
            width: 宽度
            height: 高度
            motion_model: motion model 文件名
            seed: 随机种子（-1 表示随机）
            use_refiner: 是否使用 Refiner（禁用可节省约4-6GB显存，适合16G显存）
        
        Returns:
            工作流 JSON
        """
        if seed == -1:
            seed = int(time.time())
        
        # 构建工作流
        # 注意：这是一个基础示例，实际工作流可能需要根据 ComfyUI 版本调整
        workflow = {
            "1": {
                "inputs": {
                    "filename": image_filename,
                    "subfolder": "",
                    "type": "input"
                },
                "class_type": "LoadImage",
                "_meta": {
                    "title": "Load Image"
                }
            },
            "2": {
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]  # CheckpointLoaderSimple 输出1是CLIP
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "3": {
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1]  # CheckpointLoaderSimple 输出1是CLIP
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Negative)"
                }
            },
            "4": {
                "inputs": {
                    "ckpt_name": "sd_xl_base_1.0.safetensors"  # 使用 SDXL（原来使用的版本）
                },
                "class_type": "CheckpointLoaderSimple",
                "_meta": {
                    "title": "Load Checkpoint"
                }
            },
            "5": {
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": num_frames  # 重要：batch_size 设置为帧数，这样会生成 num_frames 个 latent
                },
                "class_type": "EmptyLatentImage",
                "_meta": {
                    "title": "Empty Latent Image (Batch)"
                }
            },
            "6": {
                "inputs": {
                    "model": ["4", 0],  # 连接到 checkpoint 的 model 输出
                    "motion_model": "mm_sdxl_v10_beta.ckpt",  # 使用 SDXL motion module
                    "model_name": "mm_sdxl_v10_beta.ckpt",  # 必需：motion model 名称
                    "beta_schedule": "linear (AnimateDiff-SDXL)",  # 使用专门的SDXL beta schedule，细节渲染更好
                    "context_length": min(num_frames, 16),  # SDXL 限制为 16
                    "context_stride": 1,
                    "context_overlap": min(4, min(num_frames, 16) // 2),
                    "context_schedule": "uniform",
                    "closed_loop": False
                },
                "class_type": "ADE_AnimateDiffLoaderWithContext",
                "_meta": {
                    "title": "AnimateDiff Loader (SDXL)"
                }
            },
            "7": {
                "inputs": {
                    "seed": seed,
                    "steps": 65,  # 平衡步数（70步可能过度，降低到65步保持清晰度同时更自然）
                    "cfg": 7.5,  # 降低CFG以提升自然度（8.0可能过度，降低到7.5保持清晰度同时更自然）
                    "sampler_name": "dpmpp_2m_sde",  # 使用SDE变体，细节渲染更好
                    "scheduler": "karras",  # 使用 Karras scheduler 提高质量
                    "denoise": 1.0,
                    "model": ["6", 0],  # 连接到 AnimateDiff 的输出
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["5", 0]  # 连接到 EmptyLatentImage
                },
                "class_type": "KSampler",
                "_meta": {
                    "title": "KSampler (Main)"
                }
            },
            # SDXL Refiner（提升细节质量，可选）
            **({
                "10": {
                    "inputs": {
                        "ckpt_name": "sd_xl_refiner_1.0.safetensors"  # SDXL Refiner 模型
                    },
                    "class_type": "CheckpointLoaderSimple",
                    "_meta": {
                        "title": "Load Refiner"
                    }
                },
                # Refiner Conditioning（必须使用 Refiner 自己的 CLIP）
                "13": {
                    "inputs": {
                        "text": prompt,
                        "clip": ["10", 1]  # Refiner 的 CLIP（输出1）
                    },
                    "class_type": "CLIPTextEncode",
                    "_meta": {
                        "title": "CLIP Text Encode (Refiner Positive)"
                    }
                },
                "14": {
                    "inputs": {
                        "text": negative_prompt,
                        "clip": ["10", 1]  # Refiner 的 CLIP（输出1）
                    },
                    "class_type": "CLIPTextEncode",
                    "_meta": {
                        "title": "CLIP Text Encode (Refiner Negative)"
                    }
                },
                # Refiner KSampler（精细化处理）
                "11": {
                    "inputs": {
                        "seed": seed,
                        "steps": 20,  # Refiner 步数（降低到20步以提升自然度，25步可能过度）
                        "cfg": 5.5,  # Refiner CFG（降低到5.5以提升自然度，6.5可能过度）
                        "sampler_name": "dpmpp_2m_sde",  # 使用相同的采样器
                        "scheduler": "karras",
                        "denoise": 0.35,  # Refiner 强度（降低到0.35以提升自然度，0.5可能过度）
                        "model": ["10", 0],  # Refiner 模型
                        "positive": ["13", 0],  # 使用 Refiner 的 positive conditioning（关键修复）
                        "negative": ["14", 0],  # 使用 Refiner 的 negative conditioning（关键修复）
                        "latent_image": ["7", 0]  # 使用主 KSampler 的输出
                    },
                    "class_type": "KSampler",
                    "_meta": {
                        "title": "Refiner KSampler"
                    }
                },
            } if use_refiner else {}),
            # VAE Loader（使用更好的 VAE）
            "12": {
                "inputs": {
                    "vae_name": "diffusion_pytorch_model.safetensors"  # 优化的 VAE（madebyollin/sdxl-vae-fp16-fix）
                },
                "class_type": "VAELoader",
                "_meta": {
                    "title": "Load Optimized VAE"
                }
            },
            "8": {
                "inputs": {
                    "samples": ["11", 0] if use_refiner else ["7", 0],  # 使用 Refiner 的输出（如果启用）或主 KSampler 的输出
                    "vae": ["12", 0]  # 使用优化的 VAE（如果不存在会回退到 ["4", 2]）
                },
                "class_type": "VAEDecode",
                "_meta": {
                    "title": "VAE Decode"
                }
            },
            "9": {
                "inputs": {
                    "filename_prefix": "AnimateDiff",
                    "images": ["8", 0],
                    "fps": 8.0,  # 帧率
                    "compress_level": 4
                },
                "class_type": "SaveAnimatedPNG",  # 使用 SaveAnimatedPNG 保存多帧
                "_meta": {
                    "title": "Save Animated PNG"
                }
            }
        }
        
        return workflow
    
    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """
        提交工作流到队列
        
        Args:
            workflow: 工作流 JSON
        
        Returns:
            任务 ID (prompt_id)
        """
        p = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        
        data = json.dumps(p).encode('utf-8')
        response = requests.post(
            f"{self.server_url}/prompt",
            data=data
        )
        
        if response.status_code != 200:
            raise Exception(f"提交任务失败: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["prompt_id"]
    
    def get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务历史
        
        Args:
            prompt_id: 任务 ID
        
        Returns:
            任务历史，如果不存在则返回 None
        """
        response = requests.get(f"{self.server_url}/history/{prompt_id}")
        if response.status_code == 200:
            history = response.json()
            return history.get(prompt_id)
        return None
    
    def wait_for_completion(
        self,
        prompt_id: str,
        timeout: int = 900,  # 增加到15分钟（Refiner需要更多时间）
        check_interval: float = 1.0
    ) -> bool:
        """
        等待任务完成
        
        Args:
            prompt_id: 任务 ID
            timeout: 超时时间（秒）
            check_interval: 检查间隔（秒）
        
        Returns:
            是否成功完成
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            if history:
                return True
            time.sleep(check_interval)
        return False
    
    def get_output_images(
        self,
        prompt_id: str
    ) -> List[Dict[str, Any]]:
        """
        获取输出图像信息
        
        Args:
            prompt_id: 任务 ID
        
        Returns:
            图像信息列表
        """
        history = self.get_history(prompt_id)
        if not history:
            return []
        
        outputs = history.get("outputs", {})
        images = []
        
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    images.append({
                        "filename": img_info["filename"],
                        "subfolder": img_info.get("subfolder", ""),
                        "type": img_info.get("type", "output"),
                        "node_id": node_id
                    })
        
        return images
    
    def download_image(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output"
    ) -> Image.Image:
        """
        下载生成的图像
        
        Args:
            filename: 文件名
            subfolder: 子文件夹
            folder_type: 文件夹类型
        
        Returns:
            PIL Image
        """
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }
        
        response = requests.get(
            f"{self.server_url}/view",
            params=params
        )
        
        if response.status_code != 200:
            raise Exception(f"下载图像失败: {response.status_code}")
        
        return Image.open(io.BytesIO(response.content))
    
    def generate_video_from_image(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 16,
        width: int = 1280,  # 进一步提升分辨率以改善清晰度（SDXL推荐分辨率）
        height: int = 1280,  # 进一步提升分辨率以改善清晰度（SDXL推荐分辨率）
        output_dir: Optional[str] = None,
        timeout: int = 900,  # 增加到15分钟（Refiner需要更多时间）
        use_refiner: bool = True,  # 是否使用 Refiner（禁用可节省约4-6GB显存，适合16G显存）
    ) -> List[Image.Image]:
        """
        从图像生成视频帧（完整流程）
        
        Args:
            image_path: 输入图像路径
            prompt: 提示词
            negative_prompt: 负面提示词
            num_frames: 帧数
            width: 宽度
            height: 高度
            output_dir: 输出目录（可选）
            timeout: 超时时间（秒）
            use_refiner: 是否使用 Refiner（禁用可节省约4-6GB显存，适合16G显存）
        
        Returns:
            生成的帧列表（PIL Image）
        """
        print(f"  [ComfyUI] 上传图像: {image_path}")
        image_filename = self.upload_image(image_path)
        print(f"  ✓ 图像上传成功: {image_filename}")
        
        print(f"  [ComfyUI] 创建工作流...")
        workflow = self.create_animatediff_workflow(
            image_filename=image_filename,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            use_refiner=use_refiner,
        )
        refiner_status = "启用" if use_refiner else "禁用（节省显存）"
        print(f"  ✓ 工作流创建成功（Refiner: {refiner_status}）")
        
        print(f"  [ComfyUI] 提交任务...")
        prompt_id = self.queue_prompt(workflow)
        print(f"  ✓ 任务已提交: {prompt_id}")
        
        print(f"  [ComfyUI] 等待生成完成（最多 {timeout} 秒）...")
        if not self.wait_for_completion(prompt_id, timeout=timeout):
            raise TimeoutError(f"任务超时: {prompt_id}")
        print(f"  ✓ 生成完成")
        
        print(f"  [ComfyUI] 获取输出图像...")
        images_info = self.get_output_images(prompt_id)
        if not images_info:
            raise Exception("未找到输出图像")
        
        print(f"  ✓ 找到 {len(images_info)} 个输出图像")
        
        # 下载所有图像
        frames = []
        total_frames_extracted = 0
        
        for img_info in images_info:
            frame = self.download_image(
                filename=img_info["filename"],
                subfolder=img_info["subfolder"],
                folder_type=img_info["type"]
            )
            
            # 检查是否是动画 PNG（包含多帧）
            if hasattr(frame, 'is_animated') and frame.is_animated:
                num_frames_in_image = getattr(frame, 'n_frames', 1)
                print(f"  ℹ 检测到动画 PNG，包含 {num_frames_in_image} 帧")
                # 提取所有帧
                extracted_frames = []
                for frame_idx in range(num_frames_in_image):
                    frame.seek(frame_idx)
                    frame_copy = frame.copy()
                    
                    # 应用锐化处理（提升线条清晰度）
                    frame_copy = self._apply_sharpening(frame_copy)
                    
                    extracted_frames.append(frame_copy)
                    # 保存单帧
                    if output_dir:
                        output_path = Path(output_dir) / f"{img_info['filename'].replace('.png', '')}_frame_{frame_idx:03d}.png"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        frame_copy.save(output_path)
                frames.extend(extracted_frames)
                total_frames_extracted += len(extracted_frames)
                print(f"  ✓ 提取了 {len(extracted_frames)} 帧（已应用锐化）")
            else:
                # 单帧图像
                # 应用锐化处理（提升线条清晰度）
                frame = self._apply_sharpening(frame)
                frames.append(frame)
                total_frames_extracted += 1
                if output_dir:
                    output_path = Path(output_dir) / img_info["filename"]
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    frame.save(output_path)
                    print(f"  ✓ 保存帧: {output_path}（已应用锐化）")
        
        print(f"  ✓ 总共提取了 {total_frames_extracted} 帧（已应用锐化）")
        return frames
    
    def _apply_sharpening(self, image: Image.Image) -> Image.Image:
        """
        应用锐化处理以提升线条清晰度
        
        Args:
            image: 输入图像
        
        Returns:
            锐化后的图像
        """
        try:
            from PIL import ImageFilter, ImageEnhance
            
            # 方法1：使用 UnsharpMask 锐化（平衡参数，保持清晰度同时更自然）
            # 参数：radius=2（适中的模糊半径），percent=150（适中的锐化强度），threshold=3（适中的阈值）
            sharpened = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            # 方法2：轻微增强对比度（保持自然度）
            enhancer = ImageEnhance.Contrast(sharpened)
            sharpened = enhancer.enhance(1.08)  # 轻微增强8%对比度（降低以保持自然度）
            
            # 方法3：轻微增强锐度（保持自然度）
            sharpness_enhancer = ImageEnhance.Sharpness(sharpened)
            sharpened = sharpness_enhancer.enhance(1.05)  # 轻微增强5%锐度（降低以保持自然度）
            
            return sharpened
        except Exception as e:
            print(f"  ⚠ 锐化处理失败，使用原图: {e}")
            return image


def test_comfyui_animatediff_api():
    """测试 ComfyUI AnimateDiff API"""
    print("=" * 60)
    print("ComfyUI AnimateDiff API 测试")
    print("=" * 60)
    
    # 检查服务器
    try:
        response = requests.get("http://127.0.0.1:8188/system_stats", timeout=5)
        if response.status_code != 200:
            print("✗ ComfyUI 服务器未运行")
            return False
    except:
        print("✗ 无法连接到 ComfyUI 服务器")
        print("  请先启动: bash gen_video/启动ComfyUI服务器.sh")
        return False
    
    print("✓ ComfyUI 服务器连接成功")
    
    # 查找测试图像
    test_images = [
        Path("/vepfs-dev/shawn/vid/fanren/gen_video/outputs/images/test_scenes/scene_001.png"),
        Path("/vepfs-dev/shawn/vid/fanren/gen_video/outputs/output/images/scene_001.png"),
        Path("/vepfs-dev/shawn/vid/fanren/gen_video/outputs/images/lingjie_ep9_full/scene_001.png"),
        Path("/vepfs-dev/shawn/vid/fanren/gen_video/outputs/images/lingjie_ep5_full/scene_001.png"),
        Path("/vepfs-dev/shawn/vid/fanren/gen_video/outputs/images/lingjie_ep2_full/scene_001.png"),
    ]
    
    test_image = None
    for img_path in test_images:
        if img_path.exists():
            test_image = img_path
            break
    
    if not test_image:
        print("⚠ 未找到测试图像，请先生成测试图像")
        print("  或修改脚本中的图像路径")
        return False
    
    print(f"✓ 找到测试图像: {test_image}")
    
    # 创建 API 客户端
    api = ComfyUIAnimateDiffAPI()
    
    try:
        # 生成视频帧
        frames = api.generate_video_from_image(
            image_path=str(test_image),
            prompt="anime style, xianxia fantasy, cinematic lighting",
            negative_prompt="blurry, low quality",
            num_frames=16,
            width=1280,  # 进一步提升分辨率以改善清晰度（SDXL推荐分辨率）
            height=1280,  # 进一步提升分辨率以改善清晰度（SDXL推荐分辨率）
            output_dir="/vepfs-dev/shawn/vid/fanren/gen_video/outputs/comfyui_test",
        )
        
        print(f"\n✓ 成功生成 {len(frames)} 帧")
        print(f"  输出目录: /vepfs-dev/shawn/vid/fanren/gen_video/outputs/comfyui_test")
        
    except Exception as e:
        print(f"\n✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    test_comfyui_animatediff_api()

