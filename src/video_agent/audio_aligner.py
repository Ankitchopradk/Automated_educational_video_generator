"""
Audio alignment and duration matching system.

Ensures that generated audio segments match the video timeline exactly,
with proper duration matching and gap filling.
"""

import asyncio
import subprocess
import wave
from pathlib import Path
from typing import Any


async def generate_segment_audio(
    segment: dict,
    voice: str,
    language: str,
    kokoro_tts_func,
    output_dir: Path,
    max_speed: float = 1.2,
    min_speed: float = 0.85,
) -> tuple[str, float]:
    """
    Generate audio for a single narration segment.

    Args:
        segment: Narration segment dict with text and timing
        voice: Kokoro voice name
        language: Language code
        kokoro_tts_func: Async function to call Kokoro TTS
        output_dir: Directory to save segment audio
        max_speed: Maximum allowed speech speed
        min_speed: Minimum allowed speech speed

    Returns:
        (audio_path, actual_duration)
    """
    segment_id = segment["id"]
    text = segment["text"].strip()
    target_duration = segment["end"] - segment["start"]

    if not text:
        # Silent segment - create empty placeholder
        return "", 0.0

    # Generate at normal speed first
    print(f"[AudioAlign] Segment {segment_id}: generating audio "
          f"(target: {target_duration:.2f}s)...")

    audio_path = await kokoro_tts_func(
        narration_text=text,
        voice=voice,
        speed=1.0,
        language=language,
    )

    # Measure actual duration
    actual_duration = get_audio_duration(audio_path)

    print(f"[AudioAlign] Segment {segment_id}: generated {actual_duration:.2f}s "
          f"(target: {target_duration:.2f}s)")

    # Check if adjustment is needed
    if actual_duration > target_duration:
        # Audio is too long - try speeding up
        ratio = actual_duration / target_duration
        new_speed = min(ratio, max_speed)

        if new_speed <= max_speed:
            print(f"[AudioAlign] Segment {segment_id}: regenerating at speed {new_speed:.2f}...")
            audio_path = await kokoro_tts_func(
                narration_text=text,
                voice=voice,
                speed=new_speed,
                language=language,
            )
            actual_duration = get_audio_duration(audio_path)
            print(f"[AudioAlign] Segment {segment_id}: adjusted to {actual_duration:.2f}s")
        else:
            # Still too long even at max speed - needs text shortening
            print(f"[AudioAlign] WARNING: Segment {segment_id} audio ({actual_duration:.2f}s) "
                  f"exceeds target ({target_duration:.2f}s) even at max speed")
            # TODO: Could implement text shortening here

    elif actual_duration < target_duration * 0.7:
        # Audio is significantly shorter - could slow down slightly
        ratio = actual_duration / target_duration
        new_speed = max(ratio, min_speed)

        if new_speed >= min_speed:
            print(f"[AudioAlign] Segment {segment_id}: regenerating at speed {new_speed:.2f}...")
            audio_path = await kokoro_tts_func(
                narration_text=text,
                voice=voice,
                speed=new_speed,
                language=language,
            )
            actual_duration = get_audio_duration(audio_path)
            print(f"[AudioAlign] Segment {segment_id}: adjusted to {actual_duration:.2f}s")

    # Save to numbered segment file
    segment_output = output_dir / f"segment_{segment_id:03d}.wav"
    if audio_path and Path(audio_path).exists():
        Path(audio_path).rename(segment_output)
        audio_path = str(segment_output)

    return audio_path, actual_duration


def get_audio_duration(audio_path: str) -> float:
    """
    Get exact duration of a WAV file.
    """
    try:
        with wave.open(audio_path, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            duration = frames / rate
            return duration
    except Exception as e:
        raise RuntimeError(f"Failed to read audio duration from {audio_path}: {e}")


async def align_audio_to_timeline(
    narration: dict,
    video_duration: float,
    voice: str,
    language: str,
    kokoro_tts_func,
    output_dir: Path,
) -> tuple[str, dict]:
    """
    Generate and align all audio segments to match the video timeline exactly.

    This is the CORE audio alignment function that ensures:
    1. Each segment is generated separately
    2. Each segment fits its timeline slot
    3. Gaps are filled with silence
    4. Final audio matches video duration exactly

    Returns:
        (final_audio_path, alignment_report)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = narration["segments"]
    alignment_report = {
        "segments": [],
        "total_segments": len(segments),
        "video_duration": video_duration,
        "audio_duration": 0.0,
        "duration_match": False,
    }

    # Generate each segment
    segment_files = []
    for segment in segments:
        audio_path, actual_duration = await generate_segment_audio(
            segment=segment,
            voice=voice,
            language=language,
            kokoro_tts_func=kokoro_tts_func,
            output_dir=output_dir,
        )

        segment_report = {
            "id": segment["id"],
            "start": segment["start"],
            "end": segment["end"],
            "target_duration": segment["end"] - segment["start"],
            "actual_duration": actual_duration,
            "audio_path": audio_path,
            "text": segment["text"][:60] + "..." if len(segment["text"]) > 60 else segment["text"],
        }

        alignment_report["segments"].append(segment_report)
        segment_files.append((segment, audio_path, actual_duration))

    # Build timeline-aligned audio
    print(f"[AudioAlign] Assembling final audio timeline...")
    final_audio_path = await assemble_timeline_audio(
        segments=segment_files,
        video_duration=video_duration,
        output_path=output_dir / "final_audio.wav",
    )

    # Measure final audio duration
    final_duration = get_audio_duration(final_audio_path)
    alignment_report["audio_duration"] = final_duration

    duration_diff = abs(final_duration - video_duration)
    alignment_report["duration_difference"] = duration_diff
    alignment_report["duration_match"] = duration_diff <= 0.05

    print(f"[AudioAlign] Final audio: {final_duration:.2f}s (video: {video_duration:.2f}s, diff: {duration_diff:.2f}s)")

    if not alignment_report["duration_match"]:
        print(f"[AudioAlign] WARNING: Duration mismatch exceeds tolerance!")

    return final_audio_path, alignment_report


async def assemble_timeline_audio(
    segments: list[tuple[dict, str, float]],
    video_duration: float,
    output_path: Path,
) -> str:
    """
    Assemble individual audio segments into a single timeline-aligned WAV file.

    Places each audio segment at its exact start timestamp, filling gaps with silence.
    """
    if not segments:
        raise ValueError("No audio segments to assemble")

    # We'll build the audio by placing each segment at its exact timestamp
    # using FFmpeg's complex filter

    # First, we need to get audio parameters from the first valid segment
    sample_segment = None
    for seg_info, audio_path, duration in segments:
        if audio_path and Path(audio_path).exists():
            sample_segment = audio_path
            break

    if not sample_segment:
        raise RuntimeError("No valid audio segments found")

    with wave.open(sample_segment, "rb") as w:
        sample_rate = w.getframerate()
        channels = w.getnchannels()
        sample_width = w.getsampwidth()

    # Build the final audio by placing segments at exact timestamps
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create silence for the full video duration
    bytes_per_sample = sample_width * channels
    bytes_per_second = sample_rate * bytes_per_sample
    total_bytes = int(video_duration * bytes_per_second)

    # Align to frame boundary
    total_bytes -= total_bytes % bytes_per_sample

    # Create base silence
    audio_buffer = bytearray(total_bytes)

    # Overlay each segment at its timestamp
    for seg_info, audio_path, duration in segments:
        if not audio_path or not Path(audio_path).exists():
            continue

        start_time = seg_info["start"]
        start_byte = int(start_time * bytes_per_second)
        start_byte -= start_byte % bytes_per_sample

        # Read segment audio
        with wave.open(audio_path, "rb") as w:
            segment_data = w.readframes(w.getnframes())

        # Overlay into buffer
        end_byte = min(start_byte + len(segment_data), total_bytes)
        overlay_len = end_byte - start_byte

        audio_buffer[start_byte:end_byte] = segment_data[:overlay_len]

    # Write final audio
    with wave.open(str(output_path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(sample_width)
        out.setframerate(sample_rate)
        out.writeframes(bytes(audio_buffer))

    print(f"[AudioAlign] Assembled final audio: {output_path}")

    return str(output_path)


def validate_audio_duration(
    audio_path: str,
    video_duration: float,
    tolerance: float = 0.05,
) -> dict:
    """
    Validate that audio duration matches video duration.

    Returns validation report.
    """
    audio_duration = get_audio_duration(audio_path)
    diff = abs(audio_duration - video_duration)

    return {
        "audio_path": audio_path,
        "audio_duration": audio_duration,
        "video_duration": video_duration,
        "difference": diff,
        "tolerance": tolerance,
        "passed": diff <= tolerance,
    }
