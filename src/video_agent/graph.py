import asyncio
import json
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from .state import VideoState
from .planner import plan_video
from .tools import (
    generate_script,
    generate_storyboard,
    generate_manim_code,
    render_manim_video,
    generate_voiceover,
    merge_audio_video,
    qa_validate,
    fix_manim_code,
    _get_nemotron,
    OUTPUT_DIR,
)
from .timeline_extractor import (
    extract_video_duration,
    load_timeline,
    build_scene_timeline,
    validate_timeline,
)
from .narration_generator import (
    generate_timeline_narration,
    validate_narration_timing,
)
from .audio_aligner import (
    align_audio_to_timeline,
)
from .qa_validator import (
    validate_complete_pipeline,
    save_qa_report,
    print_qa_summary,
)


# ────────────────────────────────────────────────────────────
# NODE: Planner
# ────────────────────────────────────────────────────────────

async def planner_node(state: VideoState) -> VideoState:
    print("\n" + "=" * 60)
    print("[1/7 · Planner] Analyzing request…")
    print("=" * 60)
    return plan_video(state)


# ────────────────────────────────────────────────────────────
# NODE: Script Generation
# ────────────────────────────────────────────────────────────

async def script_generator_node(state: VideoState) -> VideoState:
    print("\n" + "=" * 60)
    print("[2/7 · Nemotron] Generating narration script…")
    print("=" * 60)

    script = await generate_script(
        user_request=state["user_request"],
        duration=state.get("duration", 60),
        platform=state.get("platform", "youtube"),
        aspect_ratio=state.get("aspect_ratio", "16:9"),
        language_name=state.get("language_name", "English"),
    )

    if not script.strip():
        raise RuntimeError("Generated script is empty.")

    state["selected_script"] = script
    return state


# ────────────────────────────────────────────────────────────
# NODE: Storyboard
# ────────────────────────────────────────────────────────────

async def storyboard_node(state: VideoState) -> VideoState:
    print("\n" + "=" * 60)
    print("[3/7 · Storyboard] Generating storyboard…")
    print("=" * 60)

    scenes = await generate_storyboard(
        selected_script=state["selected_script"],
        user_request=state["user_request"],
        duration=state.get("duration", 60),
        aspect_ratio=state.get("aspect_ratio", "16:9"),
    )

    if not scenes:
        raise RuntimeError("Storyboard generation returned no scenes.")

    state["storyboard"] = scenes
    return state


# ────────────────────────────────────────────────────────────
# NODE: Manim Code Generation + Rendering
# ────────────────────────────────────────────────────────────

async def manim_node(state: VideoState) -> VideoState:
    print("\n" + "=" * 60)
    print("[4/8 · Manim] Generating code & rendering video…")
    print("=" * 60)

    MAX_RENDER_ATTEMPTS = 3

    # Step 1: Generate initial Manim code
    code = await generate_manim_code(
        storyboard=state["storyboard"],
        user_request=state["user_request"],
        aspect_ratio=state.get("aspect_ratio", "16:9"),
    )

    # Step 2: Try rendering, feed errors back to LLM if it fails
    for attempt in range(1, MAX_RENDER_ATTEMPTS + 1):
        try:
            video_path, timeline_path = await render_manim_video(code)
            state["manim_code"] = code
            state["video_path"] = video_path
            state["timeline_path"] = timeline_path
            return state
        except RuntimeError as e:
            error_msg = str(e)
            print(f"[Manim] Render attempt {attempt}/{MAX_RENDER_ATTEMPTS} failed.")

            if attempt < MAX_RENDER_ATTEMPTS:
                print("[Manim] Asking Nemotron to fix the code…")
                try:
                    fixed_code = await fix_manim_code(code, error_msg)
                except Exception as fix_err:
                    # Fixer failed (timeout, bad output, etc.).
                    # Do NOT kill the run — fall through to the next attempt
                    # with the same code so the remaining retries are used.
                    print(
                        f"[Manim] Fixer failed ({fix_err}); "
                        f"will retry render with original code."
                    )
                    await asyncio.sleep(2)
                    continue

                if fixed_code.strip() == code.strip():
                    print(
                        "[Manim] Fixer returned identical code; "
                        "will retry render with original code."
                    )
                    await asyncio.sleep(2)
                    continue

                code = fixed_code
                await asyncio.sleep(1)
            else:
                raise RuntimeError(
                    f"Manim rendering failed after {MAX_RENDER_ATTEMPTS} attempts. "
                    f"Last error (tail):\n{error_msg[-2500:]}\n\n"
                    f"Full code saved to output/debug/last_scene.py"
                )

    # Unreachable but keeps type checkers happy
    raise RuntimeError("Manim rendering failed.")


# ────────────────────────────────────────────────────────────
# NODE: Timeline Extraction (NEW)
# ────────────────────────────────────────────────────────────

async def timeline_node(state: VideoState) -> VideoState:
    print("\n" + "=" * 60)
    print("[5/8 · Timeline] Extracting video timeline…")
    print("=" * 60)

    video_path = state["video_path"]
    timeline_path = state["timeline_path"]

    # Extract exact video duration
    video_duration = extract_video_duration(video_path)
    print(f"[Timeline] Video duration: {video_duration:.2f}s")
    state["video_duration"] = video_duration

    # Load timeline data
    timeline_data = load_timeline(timeline_path)
    state["timeline_data"] = timeline_data

    # Validate timeline matches video
    if not validate_timeline(timeline_data, video_duration):
        print("[Timeline] WARNING: Timeline validation failed")

    # Build scene-level timeline
    timeline_scenes = build_scene_timeline(timeline_data, state.get("storyboard", []))
    state["timeline_scenes"] = timeline_scenes

    # Save timeline to organized location
    timeline_dir = OUTPUT_DIR / "timeline"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    organized_timeline_path = timeline_dir / "timeline.json"
    organized_timeline_path.write_text(
        json.dumps(timeline_data, indent=2),
        encoding="utf-8"
    )
    state["timeline_path"] = str(organized_timeline_path)

    print(f"[Timeline] Extracted {len(timeline_scenes)} scene segments")
    for scene in timeline_scenes:
        print(f"  Scene {scene['segment_id']}: "
              f"{scene['start_time']:.2f}s - {scene['end_time']:.2f}s "
              f"({scene['duration']:.2f}s)")
    print(f"[Timeline] Saved to {organized_timeline_path}")

    return state


# ────────────────────────────────────────────────────────────
# NODE: Narration Generation (NEW - FROM timeline)
# ────────────────────────────────────────────────────────────

async def narration_node(state: VideoState) -> VideoState:
    print("\n" + "=" * 60)
    print("[6/8 · Narration] Generating timeline-aware narration…")
    print("=" * 60)

    timeline_scenes = state["timeline_scenes"]
    user_request = state["user_request"]

    # Generate narration from the actual video timeline
    llm = _get_nemotron(temperature=0.7)
    narration_data = await generate_timeline_narration(
        timeline_scenes=timeline_scenes,
        user_request=user_request,
        llm=llm,
    )

    # Validate narration timing
    video_duration = state["video_duration"]
    errors = validate_narration_timing(narration_data, video_duration)
    if errors:
        print("[Narration] WARNING: Timing validation issues:")
        for err in errors:
            print(f"  - {err}")

    # Save narration to file
    narration_dir = OUTPUT_DIR / "timeline"
    narration_dir.mkdir(parents=True, exist_ok=True)
    narration_path = narration_dir / "narration.json"
    narration_path.write_text(json.dumps(narration_data, indent=2), encoding="utf-8")

    state["narration_data"] = narration_data
    state["narration_path"] = str(narration_path)

    print(f"[Narration] Generated {len(narration_data['segments'])} segments")
    print(f"[Narration] Saved to {narration_path}")

    return state


# ────────────────────────────────────────────────────────────
# NODE: Audio Generation & Alignment (NEW)
# ────────────────────────────────────────────────────────────

async def audio_node(state: VideoState) -> VideoState:
    print("\n" + "=" * 60)
    print("[7/8 · Audio] Generating & aligning audio segments…")
    print("=" * 60)

    narration_data = state["narration_data"]
    video_duration = state["video_duration"]
    voice = state.get("voice", "am_michael")
    language = state.get("language", "en-us")

    # Create audio segments directory
    audio_dir = OUTPUT_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    state["audio_segments_dir"] = str(audio_dir)

    # Generate and align audio to timeline
    final_audio_path, alignment_report = await align_audio_to_timeline(
        narration=narration_data,
        video_duration=video_duration,
        voice=voice,
        language=language,
        kokoro_tts_func=generate_voiceover,
        output_dir=audio_dir,
    )

    state["audio_path"] = final_audio_path
    state["audio_alignment"] = alignment_report

    print(f"[Audio] Final audio: {final_audio_path}")
    print(f"[Audio] Duration match: {'✓ PASS' if alignment_report['duration_match'] else '✗ FAIL'}")

    return state


# ────────────────────────────────────────────────────────────
# NODE: FFmpeg Merge
# ────────────────────────────────────────────────────────────

async def ffmpeg_node(state: VideoState) -> VideoState:
    print("\n" + "=" * 60)
    print("[8/8 · FFmpeg] Merging video + audio…")
    print("=" * 60)

    final_path = await merge_audio_video(
        video_path=state["video_path"],
        audio_path=state["audio_path"],
    )

    state["final_video_path"] = final_path
    return state


# ────────────────────────────────────────────────────────────
# NODE: QA (NEW - Comprehensive validation)
# ────────────────────────────────────────────────────────────

async def qa_node(state: VideoState) -> VideoState:
    print("\n" + "=" * 60)
    print("[QA] Running comprehensive validation…")
    print("=" * 60)

    # Create QA directory
    qa_dir = OUTPUT_DIR / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    qa_path = qa_dir / "qa_report.json"

    # Run comprehensive validation
    result = validate_complete_pipeline(
        video_path=state["video_path"],
        audio_path=state["audio_path"],
        final_video_path=state["final_video_path"],
        timeline_path=state["timeline_path"],
        narration_path=state["narration_path"],
        audio_alignment_report=state["audio_alignment"],
        video_duration=state["video_duration"],
    )

    # Save QA report
    save_qa_report(result, str(qa_path))

    state["qa_result"] = result
    state["qa_path"] = str(qa_path)

    # Print summary
    print_qa_summary(result)

    # If QA failed critically, raise an error
    if not result["overall_pass"]:
        critical_errors = [e for e in result["errors"] if "Missing file" in e or "duration" in e.lower()]
        if critical_errors:
            print("\n[QA] CRITICAL FAILURES DETECTED:")
            for err in critical_errors:
                print(f"  - {err}")

    return state


# ────────────────────────────────────────────────────────────
# BUILD GRAPH
# ────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(VideoState)

    graph.add_node("planner", planner_node)
    graph.add_node("script_generator", script_generator_node)
    graph.add_node("storyboard", storyboard_node)
    graph.add_node("manim", manim_node)
    graph.add_node("timeline", timeline_node)      # NEW: Extract timeline from video
    graph.add_node("narration", narration_node)    # NEW: Generate narration FROM timeline
    graph.add_node("audio", audio_node)            # NEW: Generate & align audio segments
    graph.add_node("ffmpeg", ffmpeg_node)
    graph.add_node("qa", qa_node)

    # NEW PIPELINE ORDER:
    # 1. Plan
    # 2. Generate script
    # 3. Generate storyboard
    # 4. Render Manim video (with timeline tracking)
    # 5. Extract timeline from rendered video
    # 6. Generate narration based on actual timeline
    # 7. Generate and align audio segments to timeline
    # 8. Merge audio + video
    # 9. Comprehensive QA validation

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "script_generator")
    graph.add_edge("script_generator", "storyboard")
    graph.add_edge("storyboard", "manim")          # Video FIRST (source of truth)
    graph.add_edge("manim", "timeline")            # Extract timeline
    graph.add_edge("timeline", "narration")        # Narration FROM video
    graph.add_edge("narration", "audio")           # Audio FROM narration
    graph.add_edge("audio", "ffmpeg")
    graph.add_edge("ffmpeg", "qa")
    graph.add_edge("qa", END)

    return graph.compile()
