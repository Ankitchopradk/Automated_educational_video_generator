from typing import TypedDict


class VideoState(TypedDict, total=False):
    user_request: str

    # Planner outputs
    topic: str
    platform: str
    duration: int
    language: str            # Kokoro code, e.g. "hi", "en-us"
    language_name: str       # human name, e.g. "Hindi"
    voice: str
    aspect_ratio: str

    # Script & storyboard
    selected_script: str
    storyboard: list

    # Manim code generation & rendering
    manim_code: str
    video_path: str          # raw visual video from Manim (.mp4)
    video_duration: float    # exact video duration in seconds

    # Timeline (NEW - extracted from rendered video)
    timeline_path: str       # timeline.json from Manim
    timeline_data: dict      # parsed timeline data
    timeline_scenes: list    # scene-level timeline segments

    # Narration (NEW - generated FROM video timeline)
    narration_path: str      # narration.json
    narration_data: dict     # timeline-aware narration segments

    # Audio (NEW - segment-by-segment generation)
    audio_segments_dir: str  # directory with segment_001.wav, segment_002.wav, etc.
    audio_alignment: dict    # audio alignment report
    audio_path: str          # final_audio.wav (assembled timeline audio)

    # FFmpeg merge
    final_video_path: str    # merged video + audio (.mp4)

    # QA (NEW - comprehensive validation)
    qa_result: dict
    qa_path: str             # qa_report.json