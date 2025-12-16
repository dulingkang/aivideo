# 临时修复：在 image_generator.py 中添加 _generate_image_flux_simple 方法
# 这个方法应该添加到 _generate_image_instantid 方法之后

def _generate_image_flux_simple(
    self,
    prompt: str,
    output_path: Path,
    negative_prompt: Optional[str] = None,
    guidance_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    seed: Optional[int] = None,
    scene: Optional[Dict[str, Any]] = None,
) -> Path:
    """使用 Flux pipeline 生成图像（简化版，不处理 LoRA）"""
    if self.pipeline is None:
        raise RuntimeError("Flux pipeline 未加载")
    
    import torch
    from PIL import Image
    
    generator = None
    if seed is not None:
        generator = torch.Generator(device=self.device).manual_seed(seed)
    
    # 使用配置的默认值
    guidance = guidance_scale or 3.5  # Flux 默认引导强度较低
    steps = num_inference_steps or 18  # Flux 默认步数较少
    
    print(f"  🎨 使用 Flux pipeline 生成图像")
    print(f"  提示词: {prompt[:50]}...")
    print(f"  引导强度: {guidance}")
    print(f"  推理步数: {steps}")
    
    # 从 scene 获取尺寸（如果有）
    width = self.width
    height = self.height
    if scene and isinstance(scene, dict):
        width = scene.get("width", width)
        height = scene.get("height", height)
    
    try:
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        
        image = result.images[0]
        image.save(output_path)
        print(f"  ✅ Flux 图像生成成功: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  ❌ Flux 图像生成失败: {e}")
        import traceback
        traceback.print_exc()
        raise

