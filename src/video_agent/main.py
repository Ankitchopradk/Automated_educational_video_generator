import asyncio
import sys
import time

from .graph import build_graph


async def main():
    graph = build_graph()

    print("\n" + "=" * 60)
    print("VIDEO GENERATION AGENT")
    print("=" * 60)
    print("\nExamples:")
    print('  "Explain quantum physics in 45 seconds"')
    print('  "Create a 2 minute video about photosynthesis"')
    print('  "Make a 90 second tutorial on binary search"')
    print('  "Explain the water cycle in 1:30"')
    print("=" * 60)

    user_request = input("\nVideo request: ").strip()
    if not user_request:
        print("[Agent] No request provided. Exiting.")
        return

    print(f"\n{'#' * 60}")
    print(f"#  VIDEO GENERATION AGENT")
    print(f"#  Request: {user_request[:50]}{'…' if len(user_request) > 50 else ''}")
    print(f"{'#' * 60}")

    start_time = time.time()

    try:
        result = await graph.ainvoke({"user_request": user_request})
    except Exception as e:
        print(f"\n[Agent] Pipeline failed: {e}")
        return

    elapsed = time.time() - start_time

    # ── Final Summary ─────────────────────────────────────
    print(f"\n{'#' * 60}")
    print(f"#  PIPELINE COMPLETED  ({elapsed:.1f}s)")
    print(f"{'#' * 60}")

    print(f"\n[Result] topic    = {result.get('topic', 'N/A')}")
    print(f"[Result] platform = {result.get('platform', 'N/A')}")
    print(f"[Result] duration = {result.get('duration', 'N/A')}s")
    print(f"[Result] aspect   = {result.get('aspect_ratio', 'N/A')}")
    print(f"[Result] language = {result.get('language', 'N/A')}")
    print(f"[Result] voice    = {result.get('voice', 'N/A')}")

    # Script preview
    script = result.get("selected_script", "")
    if script:
        preview = script[:200] + ("…" if len(script) > 200 else "")
        print(f"\n[Result] Script preview:\n  {preview}")

    # Storyboard summary
    storyboard = result.get("storyboard", [])
    if storyboard:
        print(f"\n[Result] Storyboard: {len(storyboard)} scene(s)")

    # Timeline info
    video_duration = result.get("video_duration", 0)
    timeline_scenes = result.get("timeline_scenes", [])
    if timeline_scenes:
        print(f"\n[Result] Timeline: {len(timeline_scenes)} segments, {video_duration:.2f}s total")

    # Narration info
    narration_data = result.get("narration_data", {})
    if narration_data:
        segments = narration_data.get("segments", [])
        print(f"[Result] Narration: {len(segments)} segments generated from timeline")

    # Audio alignment
    audio_alignment = result.get("audio_alignment", {})
    if audio_alignment:
        audio_dur = audio_alignment.get("audio_duration", 0)
        video_dur = audio_alignment.get("video_duration", 0)
        diff = audio_alignment.get("duration_difference", 0)
        match = audio_alignment.get("duration_match", False)
        print(f"[Result] Audio alignment: {audio_dur:.2f}s (video: {video_dur:.2f}s, diff: {diff:.3f}s) {'✓' if match else '✗'}")

    # Output paths
    print(f"\n[Output] Video        : {result.get('video_path', 'N/A')}")
    print(f"[Output] Timeline     : {result.get('timeline_path', 'N/A')}")
    print(f"[Output] Narration    : {result.get('narration_path', 'N/A')}")
    print(f"[Output] Audio dir    : {result.get('audio_segments_dir', 'N/A')}")
    print(f"[Output] Final audio  : {result.get('audio_path', 'N/A')}")
    print(f"[Output] Final video  : {result.get('final_video_path', 'N/A')}")
    print(f"[Output] QA report    : {result.get('qa_path', 'N/A')}")

    # QA
    qa = result.get("qa_result", {})
    if qa:
        status = "✓ PASSED" if qa.get("overall_pass") else "✗ FAILED"
        print(f"\n[QA] Overall: {status}")
        if qa.get("duration_pass") is not None:
            print(f"[QA] Duration: {'✓ PASS' if qa['duration_pass'] else '✗ FAIL'}")
        if qa.get("sync_pass") is not None:
            print(f"[QA] Sync: {'✓ PASS' if qa['sync_pass'] else '✗ FAIL'}")
        if qa.get("segment_count"):
            print(f"[QA] Segments: {qa['segments_passed']}/{qa['segment_count']} passed")

    final = result.get("final_video_path", "")
    if final:
        print(f"\n{'=' * 60}")
        print(f"  FINAL VIDEO: {final}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())