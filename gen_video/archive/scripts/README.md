# 已归档的脚本文件

本目录包含已归档的一次性修复脚本、过时的测试文件和诊断工具。

## 归档原则

1. **一次性修复脚本**：问题已解决，脚本不再需要
2. **过时的测试文件**：已被新的测试文件替代
3. **临时诊断工具**：问题已解决，工具不再需要

## 文件分类

### Fix 脚本（一次性修复）

- `fix_scene2_issues.py` - 修复scene_002生成问题
- `fix_stuck_download.py` - 修复下载卡住问题
- `fix_mixed_action_camera.py` - 修复混合action和camera问题
- `fix_all_mixed.py` - 修复所有混合问题
- `fix_action_and_visual.py` - 修复action和visual问题
- `fix_remaining_mixed.py` - 修复剩余混合问题
- `fix_remaining_english.py` - 修复剩余英文问题
- `fix_mixed_translations.py` - 修复混合翻译问题
- `fix_visual_fields.py` - 修复visual字段问题
- `fix_2json_visual.py` - 修复v2 JSON visual问题
- `fix_cosyvoice_model.sh` - 修复CosyVoice模型问题
- `image_generator_flux_fix.py` - Flux临时修复文件

### 诊断和分析工具

- `diagnose.py` - 诊断工具
- `diagnose_scenes.py` - 场景诊断工具
- `analyze_face_consistency.py` - 人脸一致性分析
- `analyze_face_simple.py` - 简单人脸分析
- `analyze_image_generation.py` - 图像生成分析
- `analyze_images.py` - 图像分析
- `analyze_motion_parsing.py` - 运动解析分析
- `analyze_scene_issues.py` - 场景问题分析
- `analyze_visual_fields.py` - visual字段分析
- `check_all_visual_fields.py` - 检查所有visual字段
- `check_audio_content.py` - 检查音频内容
- `check_camera_motion.py` - 检查相机运动
- `check_config.py` - 检查配置
- `check_download_status.py` - 检查下载状态
- `check_flux_model_id.py` - 检查Flux模型ID
- `check_full_pipeline.py` - 检查完整流水线
- `check_gpu_memory.py` - 检查GPU内存
- `check_hf_login.py` - 检查HuggingFace登录
- `check_models_and_modules.py` - 检查模型和模块
- `check_prompt_speech.py` - 检查prompt和语音
- `check_sync.py` - 检查同步
- `scene_intent_analyzer.py` - 场景意图分析器
- `scene_motion_analyzer.py` - 场景运动分析器

### 过时的测试文件

- `test_5_json.py` - 测试5个JSON
- `test_checkpoint_500.py` - 测试checkpoint 500
- `test_comfyui.py` - ComfyUI测试（已过时）
- `test_comfyui_animatediff.py` - ComfyUI AnimateDiff测试
- `test_comfyui_multiple_scenes.py` - ComfyUI多场景测试
- `test_animatediff_text2video.py` - AnimateDiff文本转视频测试
- `test_cogvideox.py` - CogVideoX测试（待集成）
- `test_host_person.py` - 主持人测试
- `test_hunyuan_dit.py` - HunyuanDiT测试
- `test_hunyuanvideo_simple.py` - HunyuanVideo简单测试
- `test_hunyuanvideo_generation.py` - HunyuanVideo生成测试
- `test_model_loading.py` - 模型加载测试
- `test_optimized_pipeline.py` - 优化流水线测试
- `test_prompt_refactor.py` - Prompt重构测试
- `test_script.py` - 脚本测试
- `test_stage1.py` - Stage1测试
- `test_path.py` - 路径测试
- `test_svd_rife_experiments.py` - SVD RIFE实验
- `test.py` - 通用测试
- `generate_test_scene_v2.py` - 生成测试场景v2
- `image_analyzer.py` - 图像分析器

## 保留的核心测试文件

以下测试文件仍然有用，保留在根目录：

- `test_execution_planner_v2.py` - Execution Planner v2测试
- `test_image_generation.py` - 图像生成测试
- `test_lingjie_scenes.py` - 灵界场景测试
- `test_novel_video.py` - 小说推文测试
- `test_video_generation.py` - 视频生成测试
- `test_video_quality.py` - 视频质量测试
- `test_complete_pipeline.py` - 完整流水线测试
- `test_complete_pipeline_commercial.py` - 商业化流水线测试
- `test_hunyuanvideo.py` - HunyuanVideo测试
- `test_hunyuanvideo_1.5.py` - HunyuanVideo 1.5测试
- `test_prompt_engine.py` - Prompt Engine测试
- `test_prompt_engine_v2.py` - Prompt Engine v2测试
- `test_v2_integration.py` - v2集成测试
- `test_local_prompt_engine.py` - 本地Prompt Engine测试
- `test_pipeline_image_to_video.py` - 图生视频流水线测试
- `test_tts.py` - TTS测试
- `test_cosyvoice_simple.py` - CosyVoice简单测试
- `test_new_json_fields.py` - 新JSON字段测试
- `test_model_manager.py` - 模型管理器测试

## 归档时间

2025-01-XX

