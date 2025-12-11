#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI视频生成平台 - Web界面（Gradio）
提供完整的Web界面，类似正式网站
"""

import gradio as gr
import requests
import json
import time
from pathlib import Path
from typing import Optional, Tuple
import uuid
from datetime import datetime

# API配置
API_BASE_URL = "http://localhost:8000"
API_KEY = "test-key-123"  # 默认API Key

# 输出目录
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "web_ui"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== API调用函数 ====================

def call_api(endpoint: str, method: str = "GET", data: Optional[dict] = None, api_key: str = API_KEY) -> dict:
    """调用API"""
    headers = {"X-API-Key": api_key}
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            headers["Content-Type"] = "application/json"
            response = requests.post(url, headers=headers, json=data, timeout=300)
        else:
            return {"error": f"不支持的HTTP方法: {method}"}
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API错误 ({response.status_code}): {response.text}"}
    except Exception as e:
        return {"error": f"请求失败: {str(e)}"}

def get_quota_info(api_key: str) -> dict:
    """获取配额信息"""
    result = call_api("/api/v1/quota", api_key=api_key)
    return result

def generate_image_api(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_steps: int,
    guidance_scale: float,
    seed: Optional[int],
    api_key: str,
    reference_image: Optional[str] = None,
    reference_image_type: str = "scene"
) -> Tuple[Optional[str], str]:
    """生成图像（API调用，支持参考图像）"""
    if not prompt.strip():
        return None, "❌ 请输入提示词"
    
    # 准备请求数据（使用FormData支持文件上传）
    form_data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt if negative_prompt else None,
        "width": width,
        "height": height,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "seed": seed if seed else None,
        "reference_image_type": reference_image_type,
    }
    
    # 如果有参考图像，添加到FormData
    files = None
    if reference_image and Path(reference_image).exists():
        try:
            with open(reference_image, 'rb') as f:
                files = {'reference_image': (Path(reference_image).name, f, 'image/png')}
            print(f"  ℹ 使用参考图像: {Path(reference_image).name}")
        except Exception as e:
            return None, f"❌ 无法读取参考图像: {str(e)}"
    
    # 调用API
    headers = {"X-API-Key": api_key}
    url = f"{API_BASE_URL}/api/v1/images/generate"
    
    try:
        if files:
            # 使用multipart/form-data上传文件
            response = requests.post(url, headers=headers, data=form_data, files=files, timeout=300)
        else:
            # 使用JSON方式（兼容旧代码）
            response = requests.post(url, headers=headers, json=form_data, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
        else:
            return None, f"❌ API错误 ({response.status_code}): {response.text}"
    except Exception as e:
        return None, f"❌ 请求失败: {str(e)}"
    
    if "error" in result:
        return None, f"❌ {result['error']}"
    
    if result.get("status") == "completed":
        # 获取图像URL
        image_url = result.get("image_url")
        if image_url:
            full_url = f"{API_BASE_URL}{image_url}"
            # 下载图像到本地
            try:
                img_response = requests.get(full_url, timeout=30)
                if img_response.status_code == 200:
                    local_path = OUTPUT_DIR / f"{result['task_id']}.png"
                    local_path.write_bytes(img_response.content)
                    
                    quota_info = result.get("quota_remaining", {})
                    message = f"✅ 生成成功！\n"
                    message += f"📊 剩余配额: 图像 {quota_info.get('images', 0)} 张, 视频 {quota_info.get('videos', 0)} 个"
                    return str(local_path), message
            except Exception as e:
                return None, f"❌ 下载图像失败: {str(e)}"
        
        return None, "✅ 生成成功，但无法获取图像"
    else:
        return None, f"⚠️ 状态: {result.get('status', 'unknown')}"

# ==================== Gradio界面 ====================

def create_web_ui():
    """创建Web界面"""
    
    with gr.Blocks(
        title="AI视频生成平台",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        """
    ) as demo:
        
        # 头部
        gr.HTML("""
        <div class="header">
            <h1>🎬 AI视频生成平台</h1>
            <p>专业的AI图像和视频生成服务</p>
        </div>
        """)
        
        # API Key输入
        with gr.Row():
            api_key_input = gr.Textbox(
                label="API Key",
                value=API_KEY,
                type="password",
                placeholder="请输入您的API Key",
                info="默认测试Key: test-key-123"
            )
            quota_btn = gr.Button("查询配额", variant="secondary")
        
        # 配额显示
        quota_display = gr.JSON(label="配额信息", visible=False)
        
        # 主标签页
        with gr.Tabs() as tabs:
            
            # ========== 图像生成标签页 ==========
            with gr.Tab("🎨 图像生成"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="描述您想要生成的图像，例如：一个美丽的风景，山峦起伏，云雾缭绕",
                            lines=3,
                            info="详细描述有助于生成更好的图像"
                        )
                        negative_prompt = gr.Textbox(
                            label="负面提示词（可选）",
                            placeholder="描述不想要的内容，例如：模糊，低质量，变形",
                            lines=2
                        )
                        
                        with gr.Row():
                            image_width = gr.Slider(
                                label="宽度",
                                minimum=512,
                                maximum=2048,
                                value=1024,
                                step=64,
                                info="必须是8的倍数"
                            )
                            image_height = gr.Slider(
                                label="高度",
                                minimum=512,
                                maximum=2048,
                                value=1024,
                                step=64,
                                info="必须是8的倍数"
                            )
                        
                        with gr.Row():
                            num_steps = gr.Slider(
                                label="推理步数",
                                minimum=10,
                                maximum=100,
                                value=40,
                                step=5,
                                info="更多步数=更好质量，但更慢"
                            )
                            guidance_scale = gr.Slider(
                                label="引导尺度",
                                minimum=1.0,
                                maximum=20.0,
                                value=7.5,
                                step=0.5
                            )
                        
                        seed_input = gr.Number(
                            label="随机种子（可选）",
                            value=None,
                            precision=0,
                            info="相同种子会生成相同图像"
                        )
                        
                        reference_image_input = gr.File(
                            label="参考图像（可选）",
                            file_types=["image"],
                            type="filepath",
                            info="上传参考图像，用于控制生成风格或角色特征"
                        )
                        
                        reference_image_type = gr.Radio(
                            label="参考图像类型",
                            choices=["scene", "face"],
                            value="scene",
                            info="scene=场景参考（控制整体风格），face=面部参考（控制角色特征，需要InstantID）"
                        )
                        
                        image_generate_btn = gr.Button("生成图像", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        image_output = gr.Image(
                            label="生成的图像",
                            type="filepath",
                            height=600
                        )
                        image_status = gr.Textbox(
                            label="状态",
                            interactive=False,
                            lines=3
                        )
            
            # ========== 视频生成标签页 ==========
            with gr.Tab("🎬 视频生成"):
                gr.Markdown("### 视频生成功能正在开发中...")
                gr.Markdown("""
                **即将推出：**
                - 图生视频
                - 脚本驱动生成
                - 批量处理
                """)
            
            # ========== 任务历史标签页 ==========
            with gr.Tab("📋 任务历史"):
                gr.Markdown("### 任务历史功能正在开发中...")
                gr.Markdown("""
                **即将推出：**
                - 查看历史任务
                - 下载生成结果
                - 任务管理
                """)
            
            # ========== 使用指南标签页 ==========
            with gr.Tab("📖 使用指南"):
                gr.Markdown("""
                ## 使用指南
                
                ### 1. 获取API Key
                - 默认测试Key: `test-key-123` (免费版，10图/天)
                - 演示Key: `demo-key-456` (付费版，100图/天)
                - 联系管理员获取正式API Key
                
                ### 2. 生成图像
                1. 在"图像生成"标签页输入提示词
                2. 调整参数（可选）
                3. 点击"生成图像"按钮
                4. 等待30-60秒
                5. 查看生成的图像
                
                ### 3. 提示词技巧
                - **详细描述**：越详细越好，包括风格、颜色、构图等
                - **负面提示词**：描述不想要的内容
                - **参数调整**：
                  - 推理步数：40-50步通常效果最好
                  - 引导尺度：7-8适合大多数场景
                
                ### 4. 配额说明
                - 免费版：10张图/天，2个视频/天
                - 付费版：100张图/天，20个视频/天
                - 配额每天自动重置
                
                ### 5. 常见问题
                - **生成失败**：检查提示词是否为空，API Key是否正确
                - **配额用完**：等待第二天重置，或升级到付费版
                - **图像质量**：尝试增加推理步数，优化提示词
                
                ### 6. 技术支持
                - API文档：http://localhost:8000/docs
                - 问题反馈：联系管理员
                """)
        
        # ========== 事件绑定 ==========
        
        # 查询配额
        def query_quota(api_key: str):
            if not api_key:
                return gr.update(value={"error": "请输入API Key"}, visible=True)
            result = get_quota_info(api_key)
            return gr.update(value=result, visible=True)
        
        quota_btn.click(
            fn=query_quota,
            inputs=[api_key_input],
            outputs=[quota_display]
        )
        
        # 生成图像
        image_generate_btn.click(
            fn=generate_image_api,
            inputs=[
                image_prompt,
                negative_prompt,
                image_width,
                image_height,
                num_steps,
                guidance_scale,
                seed_input,
                api_key_input,
                reference_image_input,
                reference_image_type,
            ],
            outputs=[image_output, image_status]
        )
    
    return demo

# ==================== 启动 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 启动AI视频生成平台Web界面")
    print("=" * 60)
    print(f"🌐 Web界面: http://localhost:7860")
    print(f"📖 API文档: http://localhost:8000/docs")
    print(f"🔑 默认API Key: {API_KEY}")
    print("=" * 60)
    print()
    
    demo = create_web_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 设置为True可以生成公网链接
        show_error=True
    )

