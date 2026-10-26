"""
Timeline-aware narration generation.

Generates narration segments based on the actual video timeline,
ensuring perfect synchronization between visuals and voice.
"""

import json
from typing import Any

from langchain_openai import ChatOpenAI


async def generate_timeline_narration(
    timeline_scenes: list[dict],
    user_request: str,
    llm: ChatOpenAI,
) -> dict:
    """
    Generate narration for each scene based on its actual timeline and visual content.

    This is the KEY synchronization point: narration is generated FROM the video,
    not the other way around.

    Args:
        timeline_scenes: List of scene dictionaries with timing info
        user_request: Original user request for context
        llm: Language model for narration generation

    Returns:
        Narration dictionary with segments aligned to timeline
    """

    # Build context about the video structure
    video_duration = sum(s["duration"] for s in timeline_scenes)
    scene_count = len(timeline_scenes)

    # Create a structured description of the timeline
    timeline_description = []
    for scene in timeline_scenes:
        desc = (
            f"Scene {scene['segment_id']}: "
            f"{scene['start_time']:.2f}s - {scene['end_time']:.2f}s "
            f"({scene['duration']:.2f}s duration)\n"
            f"  Visual: {scene.get('visual_description', 'N/A')}\n"
            f"  Animation: {scene.get('animation', 'N/A')}\n"
            f"  On-screen: {scene.get('on_screen_text', 'N/A')}"
        )
        timeline_description.append(desc)

    timeline_text = "\n\n".join(timeline_description)

    prompt = f"""You are an expert educational video narrator.

A video has been rendered with the following timeline structure.
Your task is to write narration for EACH scene that:

1. Describes what is visually happening during that EXACT time interval
2. Fits naturally within the scene's duration (do NOT exceed it)
3. Matches the educational content being shown
4. Uses clear, natural language suitable for text-to-speech
5. Maintains continuity across scenes

CRITICAL RULES:
- Each narration segment must describe ONLY what happens in its time interval
- Keep narration concise - aim for ~2-3 words per second of scene duration
- Use simple, spoken language (no complex sentences)
- Do NOT narrate things that aren't visible yet
- Do NOT exceed the scene's time allocation
- Write naturally for voice synthesis (avoid special characters)
- If a scene is very short (<2s), keep narration minimal or empty

USER REQUEST (for context):
{user_request}

VIDEO TIMELINE:
Total duration: {video_duration:.2f} seconds
Total scenes: {scene_count}

{timeline_text}

Return ONLY valid JSON in this exact format:
{{
  "video_duration": {video_duration},
  "segments": [
    {{
      "id": 1,
      "start": 0.00,
      "end": 4.20,
      "start_frame": 0,
      "end_frame": 252,
      "text": "In this video, we will explore the basic concepts..."
    }},
    {{
      "id": 2,
      "start": 4.20,
      "end": 8.50,
      "start_frame": 252,
      "end_frame": 510,
      "text": "First, let's look at how this process begins..."
    }}
  ]
}}

Generate narration for ALL {scene_count} scenes now.
"""

    print(f"[Narration] Generating timeline-aware narration for {scene_count} scenes...")

    response = await llm.ainvoke(prompt)
    content = response.content.strip()

    # Parse response
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    try:
        narration_data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse narration JSON: {e}\n\nResponse:\n{content}")

    # Validate structure
    if "segments" not in narration_data:
        raise ValueError("Narration response missing 'segments' key")

    segments = narration_data["segments"]

    if len(segments) != len(timeline_scenes):
        print(f"[Narration] WARNING: Generated {len(segments)} segments "
              f"but expected {len(timeline_scenes)}")

    # Validate each segment has required fields
    for seg in segments:
        required = ["id", "start", "end", "text"]
        missing = [f for f in required if f not in seg]
        if missing:
            raise ValueError(f"Segment {seg.get('id', '?')} missing fields: {missing}")

    # Fill in frame numbers if missing
    for seg, scene in zip(segments, timeline_scenes):
        if "start_frame" not in seg:
            seg["start_frame"] = scene["start_frame"]
        if "end_frame" not in seg:
            seg["end_frame"] = scene["end_frame"]

    print(f"[Narration] Generated {len(segments)} narration segments successfully")

    return narration_data


def validate_narration_timing(
    narration: dict,
    video_duration: float,
    tolerance: float = 0.1
) -> list[str]:
    """
    Validate that narration segments match the video timeline.

    Returns a list of validation errors (empty if valid).
    """
    errors = []
    segments = narration.get("segments", [])

    if not segments:
        errors.append("No narration segments found")
        return errors

    # Check first segment starts at 0
    first_start = segments[0]["start"]
    if first_start > tolerance:
        errors.append(f"First segment starts at {first_start}s, expected ~0.0s")

    # Check last segment ends at video duration
    last_end = segments[-1]["end"]
    if abs(last_end - video_duration) > tolerance:
        errors.append(
            f"Last segment ends at {last_end}s, "
            f"expected ~{video_duration}s (diff: {abs(last_end - video_duration):.2f}s)"
        )

    # Check for gaps or overlaps
    for i in range(len(segments) - 1):
        current_end = segments[i]["end"]
        next_start = segments[i + 1]["start"]
        gap = next_start - current_end

        if abs(gap) > tolerance:
            if gap > 0:
                errors.append(
                    f"Gap of {gap:.2f}s between segment {i+1} and {i+2}"
                )
            else:
                errors.append(
                    f"Overlap of {abs(gap):.2f}s between segment {i+1} and {i+2}"
                )

    # Check each segment has text
    for seg in segments:
        if not seg.get("text", "").strip():
            errors.append(f"Segment {seg['id']} has no narration text")

    return errors
