"""
Timeline extraction from Manim code and rendered video.

This module instruments Manim code to capture the actual animation timeline,
then generates frame-accurate narration segments.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any


def inject_timeline_tracking(manim_code: str) -> str:
    """
    Inject timeline tracking into Manim code so every animation records
    its start/end time and frame range.

    This modifies the GeneratedScene class to track all animations automatically.
    """

    # Timeline tracking mixin that will be injected
    tracking_code = '''
import json
from pathlib import Path

class _TimelineTracker:
    """Mixin to track animation timeline without interfering with rendering."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timeline_events = []
        self._current_scene_id = 0
        self._scene_start_time = 0
        self._scene_start_frame = 0

    def begin_scene(self):
        """Mark the start of a new scene block."""
        self._current_scene_id += 1
        self._scene_start_time = self.renderer.time
        self._scene_start_frame = int(self.renderer.time * self.camera.frame_rate)

        # Record scene start event
        self._timeline_events.append({
            "type": "scene_start",
            "scene_id": self._current_scene_id,
            "time": self._scene_start_time,
            "frame": self._scene_start_frame,
        })

    def end_scene(self, duration):
        """Mark the end of a scene block and pad to target duration."""
        elapsed = self.renderer.time - self._scene_start_time
        remaining = float(duration) - elapsed

        if remaining > 0.05:
            self.wait(remaining)

        # Record scene end event
        end_time = self.renderer.time
        end_frame = int(end_time * self.camera.frame_rate)

        self._timeline_events.append({
            "type": "scene_end",
            "scene_id": self._current_scene_id,
            "time": end_time,
            "frame": end_frame,
            "duration": end_time - self._scene_start_time,
        })

        self.clear()

    def tear_down(self):
        """Called after rendering completes - save timeline to JSON."""
        super().tear_down()

        # Calculate final video properties
        final_time = self.renderer.time
        final_frame = int(final_time * self.camera.frame_rate)
        frame_rate = self.camera.frame_rate

        timeline = {
            "video_duration": final_time,
            "total_frames": final_frame,
            "frame_rate": frame_rate,
            "scenes": self._timeline_events,
        }

        # Save to output directory
        output_path = Path("timeline.json")
        output_path.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
        print(f"[Timeline] Saved timeline to {output_path}")
'''

    # Check if timeline tracking already exists
    if "_TimelineTracker" in manim_code:
        return manim_code

    # Find the GeneratedScene class definition
    scene_match = re.search(
        r"(?m)^class\s+GeneratedScene\s*\(([^)]*)\)\s*:",
        manim_code
    )

    if not scene_match:
        raise ValueError("No GeneratedScene class found in Manim code")

    # Parse existing base classes
    bases = [b.strip() for b in scene_match.group(1).split(",") if b.strip()]

    # Add _TimelineTracker as first base (must come before Scene for MRO)
    if "_TimelineTracker" not in bases:
        bases.insert(0, "_TimelineTracker")

    # Reconstruct class header
    new_header = f"class GeneratedScene({', '.join(bases)}):"

    # Replace the class header
    manim_code = (
        manim_code[:scene_match.start()]
        + new_header
        + manim_code[scene_match.end():]
    )

    # Inject the tracking code after imports
    import_marker = "from manim import *"
    if import_marker in manim_code:
        manim_code = manim_code.replace(
            import_marker,
            import_marker + "\n" + tracking_code,
            1
        )
    else:
        # Fallback: prepend to file
        manim_code = tracking_code + "\n" + manim_code

    return manim_code


def extract_video_duration(video_path: str) -> float:
    """
    Use ffprobe to get exact video duration in seconds.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to extract video duration: {e}")


def load_timeline(timeline_path: str) -> dict[str, Any]:
    """
    Load the timeline.json file generated during Manim rendering.
    """
    path = Path(timeline_path)
    if not path.exists():
        raise FileNotFoundError(f"Timeline file not found: {timeline_path}")

    try:
        timeline = json.loads(path.read_text(encoding="utf-8"))
        return timeline
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid timeline JSON: {e}")


def build_scene_timeline(timeline: dict, storyboard: list) -> list[dict]:
    """
    Build scene-level timeline by matching timeline events with storyboard.

    Returns a list of scene segments with timing information.
    """
    scenes = []
    scene_events = [e for e in timeline["scenes"] if e["type"] == "scene_start"]
    end_events = [e for e in timeline["scenes"] if e["type"] == "scene_end"]

    for i, (start_event, end_event) in enumerate(zip(scene_events, end_events)):
        # Match with storyboard if available
        storyboard_scene = storyboard[i] if i < len(storyboard) else {}

        scene = {
            "segment_id": i + 1,
            "scene_id": start_event["scene_id"],
            "start_time": start_event["time"],
            "end_time": end_event["time"],
            "duration": end_event["duration"],
            "start_frame": start_event["frame"],
            "end_frame": end_event["frame"],
            "visual_description": storyboard_scene.get("visual_description", ""),
            "animation": storyboard_scene.get("animation", ""),
            "on_screen_text": storyboard_scene.get("on_screen_text", ""),
        }
        scenes.append(scene)

    return scenes


def validate_timeline(timeline: dict, video_duration: float, tolerance: float = 0.1) -> bool:
    """
    Validate that the timeline matches the actual video duration.
    """
    timeline_duration = timeline.get("video_duration", 0)
    diff = abs(timeline_duration - video_duration)

    if diff > tolerance:
        print(f"[Timeline] WARNING: Timeline duration {timeline_duration:.2f}s "
              f"differs from video duration {video_duration:.2f}s by {diff:.2f}s")
        return False

    return True
