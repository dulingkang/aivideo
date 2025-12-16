import json
from pathlib import Path


def map_scene_role(scene_id: int, total_scenes: int) -> str:
    if scene_id == 0:
        return "opening"
    if scene_id == 999 or scene_id == total_scenes - 1:
        return "ending"
    return "key_moment"


def map_scene_type(episode_title: str) -> str:
    # 目前全部按小说推文/动画解说归类为 novel，后续可细分
    return "novel"


def map_emotion(mood: str) -> str:
    m = (mood or "").lower()
    if m in ["serene", "calm"]:
        return "serene"
    if m in ["solemn", "tense", "perilous", "fierce", "brutal"]:
        return "tense"
    if m in ["agony"]:
        return "sad"
    if m in ["alert"]:
        return "alert"
    if m in ["mysterious", "contemplative"]:
        return "mysterious"
    return "neutral"


def map_tension_level(mood: str) -> str:
    m = (mood or "").lower()
    if m in ["serene", "calm", "contemplative"]:
        return "low"
    if m in ["solemn", "mysterious"]:
        return "medium"
    return "high"


def map_intent_type(action: str, scene_id: int) -> str:
    a = (action or "").lower()
    if "title" in a:
        return "title_reveal"
    if "end-credit" in a or "ending" in a:
        return "emotional_beat"
    if "recall" in a or "remember" in a:
        return "flashback"
    if "casting" in a or "spell" in a or "attack" in a or "hitting" in a:
        return "conflict"
    if "observing" in a or "watching" in a or "sensing" in a:
        return "introduce_world"
    if scene_id == 0:
        return "title_reveal"
    return "emotional_beat"


def map_time_of_day(lighting: str) -> str:
    l = (lighting or "").lower()
    if "day" in l:
        return "day"
    if "sunset" in l or "dusk" in l:
        return "dusk"
    if "night" in l:
        return "night"
    if "mystical" in l:
        return "day"
    return "day"


def map_lighting_style(lighting: str) -> str:
    l = (lighting or "").lower()
    if "soft" in l:
        return "soft"
    if "mystical" in l:
        return "soft_cinematic"
    if "dramatic" in l:
        return "dramatic_contrast"
    if "day" in l or "bright" in l:
        return "soft_cinematic"
    if "sunset" in l or "dusk" in l:
        return "soft_cinematic"
    if "night" in l:
        return "dramatic_contrast"
    return "soft_cinematic"


def map_camera_shot(camera: str) -> str:
    c = (camera or "").lower()
    if "extreme" in c and "eye" in c:
        return "extreme_close"
    if "extreme" in c and "close" in c:
        return "extreme_close"
    if "close-up" in c or "close up" in c:
        return "close_up"
    if "medium" in c:
        return "medium"
    if "wide" in c:
        return "wide"
    return "medium"


def map_camera_angle(camera: str) -> str:
    c = (camera or "").lower()
    if "top-down" in c or "top down" in c:
        return "top_down"
    if "low-angle" in c or "low angle" in c:
        return "low_angle"
    if "high-angle" in c or "high angle" in c:
        return "high_angle"
    return "eye_level"


def map_camera_movement(camera: str, motion: dict) -> str:
    # 优先用 visual.motion.type
    t = (motion or {}).get("type", "") or ""
    t = t.lower()
    if t in ["static"]:
        return "static"
    if t in ["pan"]:
        return "pan"
    if t in ["pull_out"]:
        return "dolly_out"
    if t in ["push_in"]:
        return "dolly_in"
    # fallback 根据 camera 字符串猜
    c = (camera or "").lower()
    if "pan" in c:
        return "pan"
    if "time-lapse" in c:
        return "static"
    return "static"


def has_hanli(scene: dict) -> bool:
    text_fields = [
        scene.get("description", ""),
        scene.get("prompt", ""),
        (scene.get("visual") or {}).get("composition", ""),
    ]
    joined = " ".join(text_fields).lower()
    return "han li" in joined or "hanli" in joined


def map_visibility(camera: str) -> str:
    c = (camera or "").lower()
    if "extreme eye" in c or "face" in c or "close-up" in c or "close up" in c:
        return "high"
    if "medium" in c:
        return "mid"
    if "wide" in c:
        return "low"
    return "mid"


def map_body_coverage(camera: str) -> str:
    c = (camera or "").lower()
    if "close-up" in c or "close up" in c or "eye" in c or "face" in c or "mouth" in c:
        return "head_only"
    if "medium" in c:
        return "half_body"
    if "wide" in c:
        return "full_body"
    return "half_body"


def infer_face_visible(scene: dict) -> bool:
    camera = (scene.get("camera") or "").lower()
    comp = (scene.get("visual") or {}).get("composition", "").lower()
    prompt = (scene.get("prompt") or "").lower()
    text = " ".join([camera, comp, prompt])
    return any(k in text for k in ["face", "eyes", "close-up", "close up"])


def map_motion_intensity(motion: dict) -> str:
    t = (motion or {}).get("type", "") or ""
    s = (motion or {}).get("speed", "") or ""
    t, s = t.lower(), s.lower()
    if t == "static":
        return "static"
    if s in ["slow", ""]:
        return "gentle"
    if s in ["medium"]:
        return "moderate"
    return "dynamic"


def convert_file(input_path: str, output_path: str | None = None) -> None:
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".v2.json")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    episode = data.get("episode", 1)
    title = data.get("title", "")
    scenes_v1 = data.get("scenes", [])

    episode_id = f"lingjie_ep{episode}"
    target_fps = 24

    scenes_v2 = []
    total_scenes = len(scenes_v1)

    for idx, s in enumerate(scenes_v1):
        scene_id = s.get("id", idx)
        duration = float(s.get("duration", 3))
        mood = s.get("mood", "")
        lighting = s.get("lighting", "")
        action = s.get("action", "")
        camera = s.get("camera", "")
        visual = s.get("visual") or {}
        motion = visual.get("motion") or {}
        narration_text = s.get("narration", "")
        image_path = s.get("image_path", "")

        # 基础
        scene_role = map_scene_role(scene_id, total_scenes)
        scene_type = map_scene_type(title)

        # intent
        intent = {
            "type": map_intent_type(action, scene_id),
            "emotion": map_emotion(mood),
            "tension_level": map_tension_level(mood),
            "narrative_function": "setup" if scene_role in ["opening", "title_card"] else "exposition",
        }

        # visual_constraints
        environment = visual.get("environment", "")
        composition = visual.get("composition", "")
        fx = visual.get("fx", "")
        visual_constraints = {
            "environment": environment,
            "time_of_day": map_time_of_day(lighting),
            "weather": "mist" if "mist" in environment.lower() else "clear",
            "location_type": "outdoor",
            "elements": [],
            "color_palette": "cool_with_gold" if "gold" in (environment + composition).lower() else "neutral",
            "dominant_colors": [],
        }
        # 简单从 fx / composition 中抽 elements
        lower_all = (composition + " " + fx).lower()
        if "scroll" in lower_all:
            visual_constraints["elements"].append("golden_scroll")
        if "spirit" in lower_all:
            visual_constraints["elements"].append("spirit_particles")

        # camera
        camera_block = {
            "shot": map_camera_shot(camera),
            "angle": map_camera_angle(camera),
            "movement": map_camera_movement(camera, motion),
            "focal_length_mm": 35,
            "depth_of_field": "shallow",
        }

        # character
        present = has_hanli(s)
        character_block: dict = {"present": present}
        if present:
            visibility = map_visibility(camera)
            character_block.update(
                {
                    "id": "hanli",
                    "importance": "primary",
                    "visibility": visibility,
                    "body_coverage": map_body_coverage(camera),
                    "pose": action.lower().replace(" ", "_") if action else "",
                    "face_visible": infer_face_visible(s),
                    "instantid_policy": "auto",
                }
            )

        # quality_target
        quality_target = {
            "style": "xianxia_anime",
            "detail_level": "high",
            "lighting_style": map_lighting_style(lighting),
            "motion_intensity": map_motion_intensity(motion),
            "camera_stability": "stable",
        }

        # generation_policy
        if present:
            vis = character_block.get("visibility", "low")
            if vis in ["high", "mid"]:
                image_model = "flux_instantid"
            else:
                image_model = "flux"
            allow_face_lock = vis in ["high", "mid"]
        else:
            image_model = "flux"
            allow_face_lock = False

        generation_policy = {
            "image_model": image_model,
            "video_model": "hunyuan_i2v",
            "priority": "normal",
            "allow_face_lock": allow_face_lock,
            "allow_upscale": True,
        }

        # narration
        narration = {
            "text": narration_text,
            "voice_id": "yunjuan_xianyin",
            "emotion_hint": map_emotion(mood),
            "timing_policy": "fit_scene",
        }

        # assets
        assets = {
            "image_output": image_path,
            "video_output": "",
            "reference_images": [],
        }

        scene_v2 = {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "version": "v2",
            "duration_sec": duration,
            "target_fps": target_fps,
            "scene_role": scene_role,
            "scene_type": scene_type,
            "intent": intent,
            "visual_constraints": visual_constraints,
            "camera": camera_block,
            "character": character_block,
            "quality_target": quality_target,
            "generation_policy": generation_policy,
            "narration": narration,
            "assets": assets,
            "notes": s.get("description", ""),
        }
        scenes_v2.append(scene_v2)

    out_data = {
        "episode": episode,
        "title": title,
        "scenes": scenes_v2,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Converted to v2: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert lingjie episode scene JSON from v1 to v2 schema.")
    parser.add_argument(
        "--input",
        default="lingjie/episode/1.json",
        help="v1 episode json path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output v2 json path (default: input with .v2.json)",
    )
    args = parser.parse_args()

    convert_file(args.input, args.output)


