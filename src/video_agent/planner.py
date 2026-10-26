import re

from .state import VideoState


def plan_video(state: VideoState) -> VideoState:
    """Extract topic, platform, duration, language, voice, and aspect ratio
    from the user request using simple heuristics — no LLM call needed."""

    request = state["user_request"]
    lower = request.lower()

    # ── Topic ──────────────────────────────────────────────
    state["topic"] = request.strip()

    # ── Platform ───────────────────────────────────────────
    if "instagram" in lower or "reel" in lower:
        state["platform"] = "instagram"
    elif "tiktok" in lower:
        state["platform"] = "tiktok"
    elif "shorts" in lower:
        state["platform"] = "youtube_shorts"
    else:
        state["platform"] = "youtube"

    # ── Duration ───────────────────────────────────────────
    # Try to extract duration from various formats

    # First, try "X:YY" format (e.g., "1:30" = 1 minute 30 seconds)
    time_match = re.search(r"(\d+):(\d{2})", request)
    if time_match:
        minutes = int(time_match.group(1))
        seconds = int(time_match.group(2))
        state["duration"] = minutes * 60 + seconds
    else:
        # Try minute formats first (to catch "2 minute" before falling through)
        minute_match = re.search(r"(\d+)\s*(?:minute|minutes|min|mins)\b", lower)
        if minute_match:
            state["duration"] = int(minute_match.group(1)) * 60
        else:
            # Try second formats
            second_match = re.search(r"(\d+)\s*(?:second|seconds|sec|secs|s)\b", lower)
            if second_match:
                state["duration"] = int(second_match.group(1))
            else:
                # No duration specified - use context-based default
                # For shorts/reels: 30s, for regular videos: 60s
                if state.get("platform") in ("instagram", "tiktok", "youtube_shorts"):
                    state["duration"] = 30
                else:
                    state["duration"] = 60

    # ── Aspect ratio (derived from platform) ───────────────
    if state["platform"] in ("instagram", "tiktok", "youtube_shorts"):
        state["aspect_ratio"] = "9:16"
    else:
        state["aspect_ratio"] = "16:9"

    # ── Language ───────────────────────────────────────────
    wants_female = "female" in lower or "woman" in lower
    if "hindi" in lower or "हिन्दी" in request:
        state["language"] = "hi"            # Kokoro language code
        state["language_name"] = "Hindi"
        _voice = "hf_alpha" if wants_female else "hm_omega"   # hm_* = Hindi male
    else:
        state["language"] = "en-us"
        state["language_name"] = "English"
        _voice = "af_heart" if wants_female else "am_michael"

    # ── Voice ──────────────────────────────────────────────
    state["voice"] = _voice

    print(f"[Planner] topic    = {state['topic']}")
    print(f"[Planner] platform = {state['platform']}")
    print(f"[Planner] duration = {state['duration']}s")
    print(f"[Planner] aspect   = {state['aspect_ratio']}")
    print(f"[Planner] language = {state['language']}")
    print(f"[Planner] voice    = {state['voice']}")

    return state