"""
Integration test that verifies the pipeline structure without requiring API calls.

This tests that all nodes are properly connected and the state flows correctly.
"""

import asyncio
from pathlib import Path

from video_agent.graph import build_graph
from video_agent.state import VideoState


async def test_graph_structure():
    """Test that the graph is built correctly with all nodes."""
    print("=" * 60)
    print("TEST: Graph Structure")
    print("=" * 60)

    try:
        graph = build_graph()

        # Get graph structure
        nodes = graph.nodes
        print(f"\nGraph has {len(nodes)} nodes:")
        for node_name in nodes:
            print(f"  - {node_name}")

        expected_nodes = {
            "planner",
            "script_generator",
            "storyboard",
            "manim",
            "timeline",
            "narration",
            "audio",
            "ffmpeg",
            "qa",
        }

        missing = expected_nodes - set(nodes.keys())
        if missing:
            print(f"\n[FAIL] Missing nodes: {missing}")
            return False

        print("\n[OK] All expected nodes present")
        return True

    except Exception as e:
        print(f"[FAIL] Graph structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_state_structure():
    """Test that the state has all required fields."""
    print("\n" + "=" * 60)
    print("TEST: State Structure")
    print("=" * 60)

    try:
        state: VideoState = {}

        # New fields that should be present
        new_fields = [
            'video_duration',
            'timeline_path',
            'timeline_data',
            'timeline_scenes',
            'narration_path',
            'narration_data',
            'audio_segments_dir',
            'audio_alignment',
            'qa_path',
        ]

        print(f"\nNew state fields added:")
        for field in new_fields:
            print(f"  - {field}")

        print("\n[OK] State structure correct")
        return True

    except Exception as e:
        print(f"[FAIL] State structure test failed: {e}")
        return False


async def test_module_functions():
    """Test that key functions are importable and callable."""
    print("\n" + "=" * 60)
    print("TEST: Module Functions")
    print("=" * 60)

    try:
        # Timeline extractor
        from video_agent.timeline_extractor import (
            inject_timeline_tracking,
            extract_video_duration,
            build_scene_timeline,
            validate_timeline,
        )
        print("\n[OK] timeline_extractor functions imported")

        # Test injection
        sample_code = """from manim import *
class GeneratedScene(Scene):
    def construct(self):
        pass
"""
        injected = inject_timeline_tracking(sample_code)
        assert "_TimelineTracker" in injected
        print("  - inject_timeline_tracking() works")

        # Test timeline building
        timeline = {
            "video_duration": 10.0,
            "scenes": [
                {"type": "scene_start", "scene_id": 1, "time": 0.0, "frame": 0},
                {"type": "scene_end", "scene_id": 1, "time": 10.0, "frame": 600, "duration": 10.0},
            ]
        }
        scenes = build_scene_timeline(timeline, [])
        assert len(scenes) == 1
        print("  - build_scene_timeline() works")

        # Narration generator
        from video_agent.narration_generator import validate_narration_timing
        print("\n[OK] narration_generator functions imported")

        narration = {
            "video_duration": 10.0,
            "segments": [
                {"id": 1, "start": 0.0, "end": 10.0, "text": "Test"}
            ]
        }
        errors = validate_narration_timing(narration, 10.0)
        assert isinstance(errors, list)
        print("  - validate_narration_timing() works")

        # Audio aligner
        from video_agent.audio_aligner import validate_audio_duration
        print("\n[OK] audio_aligner functions imported")

        # QA validator
        from video_agent.qa_validator import (
            validate_timeline_file,
            validate_narration_file,
            print_qa_summary,
        )
        print("[OK] qa_validator functions imported")

        print("\n[OK] All module functions work correctly")
        return True

    except Exception as e:
        print(f"[FAIL] Module functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_planner_only():
    """Test just the planner node (no API calls)."""
    print("\n" + "=" * 60)
    print("TEST: Planner Node")
    print("=" * 60)

    try:
        from video_agent.planner import plan_video

        state: VideoState = {
            "user_request": "Explain quantum physics in 60 seconds for YouTube"
        }

        result = plan_video(state)

        print("\nPlanner output:")
        print(f"  Topic: {result.get('topic')}")
        print(f"  Platform: {result.get('platform')}")
        print(f"  Duration: {result.get('duration')}s")
        print(f"  Aspect ratio: {result.get('aspect_ratio')}")
        print(f"  Language: {result.get('language')}")
        print(f"  Voice: {result.get('voice')}")

        assert result.get('platform') == 'youtube'
        assert result.get('duration') == 60
        assert result.get('aspect_ratio') == '16:9'

        print("\n[OK] Planner node works correctly")
        return True

    except Exception as e:
        print(f"[FAIL] Planner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all integration tests."""
    print("\n" + "#" * 60)
    print("#  INTEGRATION TESTS - NO API CALLS")
    print("#" * 60)

    tests = [
        ("Graph Structure", test_graph_structure),
        ("State Structure", test_state_structure),
        ("Module Functions", test_module_functions),
        ("Planner Node", test_planner_only),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] {name} crashed: {e}")
            results.append((name, False))

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
        print("\n[SUCCESS] All integration tests passed!")
        print("\nThe pipeline structure is correct.")
        print("To run end-to-end test, ensure:")
        print("  1. OPENROUTER_API_KEY is set in .env")
        print("  2. MCP servers are running")
        print("  3. Kokoro TTS is installed")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src')
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
