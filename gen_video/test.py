import json
from pathlib import Path
from main import AIVideoPipeline

config_path = 'config.yaml'
output_name = 'lingjie_ep5_auto_v1'

pipeline = AIVideoPipeline(
    config_path,
    load_image=False,
    load_video=False,
    load_tts=False,
    load_subtitle=False,
    load_composer=True,
)

base_output = Path(pipeline.paths['output_dir']) / output_name
video_dir = base_output / 'videos'
video_clips = sorted(str(p.resolve()) for p in video_dir.glob('*.mp4'))
print('视频目录', video_dir)
print('找到视频片段', len(video_clips))
if not video_clips:
    raise SystemExit('未找到视频片段')

audio_path = base_output / 'audio.wav'
print('音频存在?', audio_path.exists())
if not audio_path.exists():
    raise SystemExit('未找到配音音频')

subtitle_path = base_output / 'subtitle.srt'
print('字幕存在?', subtitle_path.exists())
if not subtitle_path.exists():
    subtitle_path = None

script_path = Path('temp') / f'{output_name}_script.json'
if script_path.exists():
    with open(script_path, 'r', encoding='utf-8') as f:
        script = json.load(f)
    scenes = script.get('scenes')
else:
    scenes = None

print('开始合成...')
final_video = pipeline.compose_final_video(
    video_clips,
    str(audio_path),
    str(subtitle_path) if subtitle_path else None,
    output_name,
    scenes=scenes,
)
print('完成:', final_video)