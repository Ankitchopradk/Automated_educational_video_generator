"""
Comprehensive QA validation system.

Validates the complete video generation pipeline including:
- Timeline integrity
- Narration synchronization
- Audio-video duration matching
- File existence and integrity
"""

import json
from pathlib import Path
from typing import Any


def validate_complete_pipeline(
    video_path: str,
    audio_path: str,
    final_video_path: str,
    timeline_path: str,
    narration_path: str,
    audio_alignment_report: dict,
    video_duration: float,
) -> dict:
    """
    Run comprehensive QA validation on all pipeline outputs.

    This is the master validation function that checks everything.

    Returns:
        Complete QA report with pass/fail status
    """
    report = {
        "timestamp": "",
        "video_duration": video_duration,
        "audio_duration": 0.0,
        "duration_difference": 0.0,
        "duration_pass": False,
        "segment_count": 0,
        "segments_passed": 0,
        "segments_failed": 0,
        "sync_pass": False,
        "files_exist": {},
        "timeline_valid": False,
        "narration_valid": False,
        "errors": [],
        "warnings": [],
    }

    # Check file existence
    files = {
        "video": video_path,
        "audio": audio_path,
        "final_video": final_video_path,
        "timeline": timeline_path,
        "narration": narration_path,
    }

    for name, path in files.items():
        exists = Path(path).exists() if path else False
        report["files_exist"][name] = exists
        if not exists:
            report["errors"].append(f"Missing file: {name} ({path})")

    # Validate timeline
    if report["files_exist"]["timeline"]:
        timeline_result = validate_timeline_file(timeline_path, video_duration)
        report["timeline_valid"] = timeline_result["valid"]
        if not timeline_result["valid"]:
            report["errors"].extend(timeline_result["errors"])
        report["timeline_info"] = timeline_result
    else:
        report["errors"].append("Timeline file missing - cannot validate synchronization")

    # Validate narration
    if report["files_exist"]["narration"]:
        narration_result = validate_narration_file(narration_path, video_duration)
        report["narration_valid"] = narration_result["valid"]
        if not narration_result["valid"]:
            report["errors"].extend(narration_result["errors"])
        report["narration_info"] = narration_result
        report["segment_count"] = narration_result.get("segment_count", 0)
    else:
        report["errors"].append("Narration file missing - cannot validate synchronization")

    # Validate audio alignment
    if audio_alignment_report:
        report["audio_duration"] = audio_alignment_report.get("audio_duration", 0.0)
        report["duration_difference"] = audio_alignment_report.get("duration_difference", 999.0)
        report["duration_pass"] = audio_alignment_report.get("duration_match", False)

        # Count segment pass/fail
        segments = audio_alignment_report.get("segments", [])
        report["segment_count"] = len(segments)

        for seg in segments:
            target = seg.get("target_duration", 0)
            actual = seg.get("actual_duration", 0)
            if target > 0:
                ratio = actual / target
                if 0.7 <= ratio <= 1.3:  # Within 30% is acceptable
                    report["segments_passed"] += 1
                else:
                    report["segments_failed"] += 1
                    report["warnings"].append(
                        f"Segment {seg['id']} duration mismatch: "
                        f"target {target:.2f}s, actual {actual:.2f}s"
                    )

        report["audio_alignment"] = audio_alignment_report
    else:
        report["errors"].append("No audio alignment report available")

    # Overall synchronization check
    report["sync_pass"] = (
        report["timeline_valid"]
        and report["narration_valid"]
        and report["duration_pass"]
    )

    # Overall pass/fail
    report["overall_pass"] = (
        all(report["files_exist"].values())
        and report["sync_pass"]
        and len(report["errors"]) == 0
    )

    return report


def validate_timeline_file(timeline_path: str, video_duration: float) -> dict:
    """
    Validate timeline.json structure and content.
    """
    result = {
        "valid": False,
        "errors": [],
        "warnings": [],
    }

    try:
        timeline = json.loads(Path(timeline_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        result["errors"].append(f"Invalid timeline JSON: {e}")
        return result
    except Exception as e:
        result["errors"].append(f"Failed to read timeline: {e}")
        return result

    # Check required fields
    required = ["video_duration", "total_frames", "frame_rate", "scenes"]
    missing = [f for f in required if f not in timeline]
    if missing:
        result["errors"].append(f"Timeline missing fields: {missing}")
        return result

    # Validate duration match
    timeline_duration = timeline["video_duration"]
    diff = abs(timeline_duration - video_duration)
    if diff > 0.1:
        result["warnings"].append(
            f"Timeline duration ({timeline_duration:.2f}s) differs from "
            f"video duration ({video_duration:.2f}s) by {diff:.2f}s"
        )

    # Validate scenes
    scenes = timeline["scenes"]
    if not scenes:
        result["errors"].append("Timeline has no scenes")
        return result

    scene_count = len([s for s in scenes if s.get("type") == "scene_start"])
    if scene_count == 0:
        result["errors"].append("Timeline has no scene_start events")

    result["scene_count"] = scene_count
    result["frame_rate"] = timeline["frame_rate"]
    result["total_frames"] = timeline["total_frames"]
    result["valid"] = len(result["errors"]) == 0

    return result


def validate_narration_file(narration_path: str, video_duration: float) -> dict:
    """
    Validate narration.json structure and content.
    """
    result = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "segment_count": 0,
    }

    try:
        narration = json.loads(Path(narration_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        result["errors"].append(f"Invalid narration JSON: {e}")
        return result
    except Exception as e:
        result["errors"].append(f"Failed to read narration: {e}")
        return result

    # Check required fields
    if "segments" not in narration:
        result["errors"].append("Narration missing 'segments' field")
        return result

    segments = narration["segments"]
    result["segment_count"] = len(segments)

    if not segments:
        result["errors"].append("Narration has no segments")
        return result

    # Validate each segment
    for i, seg in enumerate(segments):
        required = ["id", "start", "end", "text"]
        missing = [f for f in required if f not in seg]
        if missing:
            result["errors"].append(f"Segment {i+1} missing fields: {missing}")

        # Check timing
        if "start" in seg and "end" in seg:
            if seg["end"] <= seg["start"]:
                result["errors"].append(
                    f"Segment {seg.get('id', i+1)}: end time <= start time"
                )

    # Check first segment starts at 0
    if segments[0].get("start", 0) > 0.1:
        result["warnings"].append(
            f"First segment starts at {segments[0]['start']}s, expected ~0.0s"
        )

    # Check last segment ends at video duration
    last_end = segments[-1].get("end", 0)
    diff = abs(last_end - video_duration)
    if diff > 0.1:
        result["warnings"].append(
            f"Last segment ends at {last_end:.2f}s, "
            f"expected ~{video_duration:.2f}s (diff: {diff:.2f}s)"
        )

    # Check for gaps or overlaps
    for i in range(len(segments) - 1):
        current_end = segments[i].get("end", 0)
        next_start = segments[i + 1].get("start", 0)
        gap = next_start - current_end

        if abs(gap) > 0.1:
            if gap > 0:
                result["warnings"].append(
                    f"Gap of {gap:.2f}s between segment {i+1} and {i+2}"
                )
            else:
                result["warnings"].append(
                    f"Overlap of {abs(gap):.2f}s between segment {i+1} and {i+2}"
                )

    result["valid"] = len(result["errors"]) == 0

    return result


def save_qa_report(report: dict, output_path: str) -> None:
    """
    Save QA report to JSON file.
    """
    from datetime import datetime

    report["timestamp"] = datetime.utcnow().isoformat() + "Z"

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"[QA] Report saved to {output_path}")


def print_qa_summary(report: dict) -> None:
    """
    Print a human-readable QA summary.
    """
    print("\n" + "=" * 60)
    print("QA VALIDATION REPORT")
    print("=" * 60)

    status = "✓ PASSED" if report["overall_pass"] else "✗ FAILED"
    print(f"\nOverall Status: {status}")

    print(f"\nDuration Check:")
    print(f"  Video:      {report['video_duration']:.2f}s")
    print(f"  Audio:      {report['audio_duration']:.2f}s")
    print(f"  Difference: {report['duration_difference']:.3f}s")
    print(f"  Status:     {'✓ PASS' if report['duration_pass'] else '✗ FAIL'}")

    print(f"\nSegments:")
    print(f"  Total:   {report['segment_count']}")
    print(f"  Passed:  {report['segments_passed']}")
    print(f"  Failed:  {report['segments_failed']}")

    print(f"\nSynchronization:")
    print(f"  Timeline: {'✓ Valid' if report['timeline_valid'] else '✗ Invalid'}")
    print(f"  Narration: {'✓ Valid' if report['narration_valid'] else '✗ Invalid'}")
    print(f"  Sync Status: {'✓ PASS' if report['sync_pass'] else '✗ FAIL'}")

    if report["errors"]:
        print(f"\nErrors ({len(report['errors'])}):")
        for err in report["errors"]:
            print(f"  - {err}")

    if report["warnings"]:
        print(f"\nWarnings ({len(report['warnings'])}):")
        for warn in report["warnings"][:10]:  # Limit to first 10
            print(f"  - {warn}")
        if len(report["warnings"]) > 10:
            print(f"  ... and {len(report['warnings']) - 10} more warnings")

    print("\n" + "=" * 60)
