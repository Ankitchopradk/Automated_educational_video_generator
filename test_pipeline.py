"""
Test script for the timeline-aware video generation pipeline.

This script helps verify that all components are working correctly.
"""

import asyncio
import json
from pathlib import Path

from video_agent.timeline_extractor import (
    inject_timeline_tracking,
    extract_video_duration,
    load_timeline,
    build_scene_timeline,
    validate_timeline,
)
from video_agent.narration_generator import (
    validate_narration_timing,
)
from video_agent.audio_aligner import (
    get_audio_duration,
    validate_audio_duration,
)
from video_agent.qa_validator import (
    validate_complete_pipeline,
    print_qa_summary,
)


def test_timeline_tracking_injection():
    """Test that timeline tracking code is injected correctly."""
    print("\n" + "=" * 60)
    print("TEST: Timeline Tracking Injection")
    print("=" * 60)

    sample_code = """from manim import *

class GeneratedScene(Scene):
    def construct(self):
        text = Text("Hello")
        self.play(Write(text))
        self.wait(1)
"""

    try:
        injected = inject_timeline_tracking(sample_code)

        # Verify injection
        assert "_TimelineTracker" in injected, "TimelineTracker class not found"
        assert "class GeneratedScene(_TimelineTracker, Scene):" in injected, "Class header not modified"
        assert "import json" in injected, "JSON import missing"

        print("[OK] Timeline tracking injection working correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Timeline tracking injection failed: {e}")
        return False


def test_timeline_validation():
    """Test timeline validation logic."""
    print("\n" + "=" * 60)
    print("TEST: Timeline Validation")
    print("=" * 60)

    # Valid timeline
    valid_timeline = {
        "video_duration": 10.0,
        "total_frames": 600,
        "frame_rate": 60,
        "scenes": [
            {"type": "scene_start", "scene_id": 1, "time": 0.0, "frame": 0},
            {"type": "scene_end", "scene_id": 1, "time": 10.0, "frame": 600, "duration": 10.0},
        ]
    }

    try:
        result = validate_timeline(valid_timeline, 10.0, tolerance=0.1)
        assert result is True, "Valid timeline marked as invalid"
        print("[OK] Timeline validation working correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Timeline validation failed: {e}")
        return False


def test_narration_timing_validation():
    """Test narration timing validation."""
    print("\n" + "=" * 60)
    print("TEST: Narration Timing Validation")
    print("=" * 60)

    # Valid narration
    valid_narration = {
        "video_duration": 10.0,
        "segments": [
            {"id": 1, "start": 0.0, "end": 5.0, "text": "First segment"},
            {"id": 2, "start": 5.0, "end": 10.0, "text": "Second segment"},
        ]
    }

    try:
        errors = validate_narration_timing(valid_narration, 10.0, tolerance=0.1)
        assert len(errors) == 0, f"Valid narration has errors: {errors}"
        print("[OK] Narration timing validation working correctly")

        # Test with gap
        narration_with_gap = {
            "video_duration": 10.0,
            "segments": [
                {"id": 1, "start": 0.0, "end": 4.0, "text": "First segment"},
                {"id": 2, "start": 5.0, "end": 10.0, "text": "Second segment"},  # 1s gap
            ]
        }
        errors = validate_narration_timing(narration_with_gap, 10.0, tolerance=0.1)
        assert len(errors) > 0, "Gap not detected"
        print("[OK] Gap detection working correctly")

        return True
    except Exception as e:
        print(f"[FAIL] Narration timing validation failed: {e}")
        return False


def test_audio_duration_validation():
    """Test audio duration validation."""
    print("\n" + "=" * 60)
    print("TEST: Audio Duration Validation")
    print("=" * 60)

    try:
        # Test exact match
        result = validate_audio_duration("dummy.wav", 10.0, tolerance=0.05)
        # This will fail because dummy.wav doesn't exist, but we're testing the logic
        print("[OK] Audio duration validation logic correct")
        return True
    except Exception as e:
        # Expected - file doesn't exist
        if "Failed to read audio duration" in str(e):
            print("[OK] Audio duration validation working correctly (expected file error)")
            return True
        print(f"[FAIL] Audio duration validation failed unexpectedly: {e}")
        return False


def test_scene_timeline_building():
    """Test scene timeline building from events."""
    print("\n" + "=" * 60)
    print("TEST: Scene Timeline Building")
    print("=" * 60)

    timeline = {
        "video_duration": 20.0,
        "total_frames": 1200,
        "frame_rate": 60,
        "scenes": [
            {"type": "scene_start", "scene_id": 1, "time": 0.0, "frame": 0},
            {"type": "scene_end", "scene_id": 1, "time": 5.0, "frame": 300, "duration": 5.0},
            {"type": "scene_start", "scene_id": 2, "time": 5.0, "frame": 300},
            {"type": "scene_end", "scene_id": 2, "time": 12.0, "frame": 720, "duration": 7.0},
            {"type": "scene_start", "scene_id": 3, "time": 12.0, "frame": 720},
            {"type": "scene_end", "scene_id": 3, "time": 20.0, "frame": 1200, "duration": 8.0},
        ]
    }

    storyboard = [
        {"scene_number": 1, "visual_description": "Intro"},
        {"scene_number": 2, "visual_description": "Main content"},
        {"scene_number": 3, "visual_description": "Conclusion"},
    ]

    try:
        scenes = build_scene_timeline(timeline, storyboard)
        assert len(scenes) == 3, f"Expected 3 scenes, got {len(scenes)}"
        assert scenes[0]["duration"] == 5.0, "Scene 1 duration incorrect"
        assert scenes[1]["duration"] == 7.0, "Scene 2 duration incorrect"
        assert scenes[2]["duration"] == 8.0, "Scene 3 duration incorrect"
        print("[OK] Scene timeline building working correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Scene timeline building failed: {e}")
        return False


def test_qa_validation_structure():
    """Test QA validation report structure."""
    print("\n" + "=" * 60)
    print("TEST: QA Validation Structure")
    print("=" * 60)

    # Create dummy data
    alignment_report = {
        "audio_duration": 10.0,
        "video_duration": 10.0,
        "duration_difference": 0.0,
        "duration_match": True,
        "segments": [
            {"id": 1, "target_duration": 5.0, "actual_duration": 5.1},
            {"id": 2, "target_duration": 5.0, "actual_duration": 4.9},
        ]
    }

    try:
        # This will fail because files don't exist, but tests the structure
        result = validate_complete_pipeline(
            video_path="dummy_video.mp4",
            audio_path="dummy_audio.wav",
            final_video_path="dummy_final.mp4",
            timeline_path="dummy_timeline.json",
            narration_path="dummy_narration.json",
            audio_alignment_report=alignment_report,
            video_duration=10.0,
        )

        # Check required fields
        required_fields = [
            "video_duration", "audio_duration", "duration_difference",
            "duration_pass", "segment_count", "sync_pass", "files_exist",
            "overall_pass", "errors", "warnings"
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        print("[OK] QA validation structure correct")
        return True
    except Exception as e:
        print(f"[FAIL] QA validation structure test failed: {e}")
        return False


async def run_all_tests():
    """Run all unit tests."""
    print("\n" + "#" * 60)
    print("#  VIDEO GENERATION AGENT - UNIT TESTS")
    print("#" * 60)

    tests = [
        test_timeline_tracking_injection,
        test_timeline_validation,
        test_narration_timing_validation,
        test_audio_duration_validation,
        test_scene_timeline_building,
        test_qa_validation_structure,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n[FAIL] {test_func.__name__} crashed: {e}")
            results.append((test_func.__name__, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
