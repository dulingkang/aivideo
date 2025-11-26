#!/bin/bash

# proxychains4 python3 tools/video_processing/pipeline.py \
#   --episode 142 \
#   --input gen_video/raw_videos/142_4K.mp4 \
#   --crop 3840:1960:0:0 \
#   --delogo 3300:80:535:300 \
#   --auto-aspect \
#   --trim-start 151 \
#   --trim-end 135 \
#   --scale 1742:980 \
#   --subtitle-method ocr

# # 处理151集
# proxychains4 python3 tools/video_processing/pipeline.py \
#   --episode 151 \
#   --input gen_video/raw_videos/151_4K.mp4 \
#   --crop 3840:1960:0:0 \
#   --delogo 3300:80:535:300 \
#   --auto-aspect \
#   --trim-start 162 \
#   --trim-end 88 \
#   --scale 1742:980 \
#   --subtitle-method ocr

# # 处理165集
# proxychains4 python3 tools/video_processing/pipeline.py \
#   --episode 165 \
#   --input gen_video/raw_videos/165_4K.mp4 \
#   --crop 3840:1960:0:0 \
#   --delogo 3300:80:535:300 \
#   --auto-aspect \
#   --trim-start 156 \
#   --trim-end 131 \
#   --scale 1742:980 \
#   --subtitle-method ocr

# proxychains4 python3 tools/video_processing/pipeline.py \
#   --episode 170 \
#   --input gen_video/raw_videos/170_1080P.mp4 \
#   --crop 1920:980:0:0 \
#   --delogo 1650:40:200:150 \
#   --auto-aspect \
#   --trim-start 154 \
#   --trim-end 1236 \
#   --scale 1742:980 \
#   --subtitle-method ocr


# proxychains4 python3 tools/video_processing/pipeline.py \
#   --episode 150 \
#   --input gen_video/raw_videos/150_4K.mp4 \
#   --crop 3840:1960:0:0 \
#   --delogo 3300:80:535:300 \
#   --auto-aspect \
#   --trim-start 163 \
#   --trim-end 92 \
#   --scale 1742:980 \
#   --subtitle-method ocr \
#   --mute

# proxychains4 python3 tools/video_processing/pipeline.py \
#   --episode 158 \
#   --input gen_video/raw_videos/158_4K.mp4 \
#   --crop 3840:1960:0:0 \
#   --delogo 3300:80:535:300 \
#   --auto-aspect \
#   --trim-start 169 \
#   --trim-end 111 \
#   --scale 1742:980 \
#   --subtitle-method ocr \
#   --mute

# proxychains4 python3 tools/video_processing/pipeline.py \
#   --episode 162 \
#   --input gen_video/raw_videos/162_4K.mp4 \
#   --crop 3840:1960:0:0 \
#   --delogo 3300:80:535:300 \
#   --auto-aspect \
#   --trim-start 155 \
#   --trim-end 96 \
#   --scale 1742:980 \
#   --subtitle-method ocr \
#   --mute

proxychains4 python3 tools/video_processing/pipeline.py \
  --episode 144 \
  --input gen_video/raw_videos/144_4K.mp4 \
  --crop 3840:1960:0:0 \
  --delogo 3300:80:535:300 \
  --auto-aspect \
  --trim-start 147 \
  --trim-end 141 \
  --scale 1742:980 \
  --subtitle-method ocr \
  --mute

proxychains4 python3 tools/video_processing/pipeline.py \
  --episode 147 \
  --input gen_video/raw_videos/147_4K.mp4 \
  --crop 3840:1960:0:0 \
  --delogo 3300:80:535:300 \
  --auto-aspect \
  --trim-start 148 \
  --trim-end 90 \
  --scale 1742:980 \
  --subtitle-method ocr \
  --mute

proxychains4 python3 tools/video_processing/pipeline.py \
  --episode 149 \
  --input gen_video/raw_videos/149_4K.mp4 \
  --crop 3840:1960:0:0 \
  --delogo 3300:80:535:300 \
  --auto-aspect \
  --trim-start 163 \
  --trim-end 91 \
  --scale 1742:980 \
  --subtitle-method ocr \
  --mute