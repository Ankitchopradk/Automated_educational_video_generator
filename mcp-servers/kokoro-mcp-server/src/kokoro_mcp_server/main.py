import asyncio
import os
import uuid
from pathlib import Path

import soundfile as sf
from fastmcp import FastMCP
from kokoro_onnx import Kokoro

# ---------- Configuration ----------
MODEL_PATH = Path(os.getenv("KOKORO_MODEL", r"C:\Kokoro\kokoro-v1.0.onnx"))
VOICES_PATH = Path(os.getenv("KOKORO_VOICES", r"C:\Kokoro\voices-v1.0.bin"))
OUTPUT_DIR = Path(
    os.getenv("KOKORO_OUTPUT", str(Path(__file__).resolve().parents[2] / "output"))
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for label, path in (("model", MODEL_PATH), ("voices file", VOICES_PATH)):
    if not path.exists():
        raise FileNotFoundError(f"Kokoro {label} not found: {path}")

kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
mcp = FastMCP("kokoro-mcp-server")

# Voice prefix -> (language, gender)
VOICE_MAPPING = {
    "af": ("American English", "Female"), "am": ("American English", "Male"),
    "bf": ("British English", "Female"),  "bm": ("British English", "Male"),
    "ef": ("Spanish", "Female"),          "em": ("Spanish", "Male"),
    "ff": ("French", "Female"),
    "hf": ("Hindi", "Female"),            "hm": ("Hindi", "Male"),
    "if": ("Italian", "Female"),          "im": ("Italian", "Male"),
    "jf": ("Japanese", "Female"),         "jm": ("Japanese", "Male"),
    "pf": ("Portuguese", "Female"),       "pm": ("Portuguese", "Male"),
    "zf": ("Chinese", "Female"),          "zm": ("Chinese", "Male"),
}


# ---------- Helpers (plain functions, callable from any tool) ----------
def _catalog() -> list[dict]:
    return [
        {
            "voice_id": v,
            "language": VOICE_MAPPING.get(v[:2], ("Unknown", "Unknown"))[0],
            "gender": VOICE_MAPPING.get(v[:2], ("Unknown", "Unknown"))[1],
            "style": v[3:],
        }
        for v in kokoro.get_voices()
    ]


def _languages() -> list[str]:
    return sorted({v["language"] for v in _catalog() if v["language"] != "Unknown"})


# ---------- Tools ----------
@mcp.tool
async def generate_speech(
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    language: str = "en-us",
    filename: str | None = None,
) -> dict:
    """Generate speech locally using Kokoro TTS and save it as a WAV file."""
    if not text.strip():
        raise ValueError("Text cannot be empty.")
    if speed <= 0:
        raise ValueError("Speed must be greater than 0.")
    if voice not in kokoro.get_voices():
        raise ValueError(f"Unknown voice: {voice}")

    if filename:
        filename = Path(filename).name
        if not filename.lower().endswith(".wav"):
            filename += ".wav"
    else:
        filename = f"speech_{uuid.uuid4().hex[:8]}.wav"

    output_path = OUTPUT_DIR / filename

    # Run blocking work in threads so the MCP event loop stays responsive
    samples, sample_rate = await asyncio.to_thread(
        kokoro.create, text, voice, speed, language
    )
    await asyncio.to_thread(sf.write, str(output_path), samples, sample_rate)

    return {
        "success": True,
        "audio_path": str(output_path),
        "duration_seconds": round(len(samples) / sample_rate, 2),
        "sample_rate": sample_rate,
        "voice": voice,
        "speed": speed,
        "language": language,
    }


@mcp.tool
async def get_voice_catalog() -> list[dict]:
    """Return voice_id, language, gender and style for every installed voice."""
    return _catalog()


@mcp.tool
async def list_voices() -> list[str]:
    """Return all locally available Kokoro voice IDs."""
    return list(kokoro.get_voices())


@mcp.tool
async def list_languages() -> list[str]:
    """Return all languages represented by the installed voices."""
    return _languages()


@mcp.tool
async def find_voices(
    language: str | None = None, gender: str | None = None
) -> list[dict]:
    """Find voices by language and/or gender, e.g. language="Hindi", gender="Female"."""
    language = language.strip().lower() if language else None
    gender = gender.strip().lower() if gender else None
    return [
        v for v in _catalog()
        if (not language or v["language"].lower() == language)
        and (not gender or v["gender"].lower() == gender)
    ]


@mcp.tool
async def get_kokoro_status() -> dict:
    """Return the status and configuration of the local Kokoro engine."""
    return {
        "status": "ready",
        "model": str(MODEL_PATH),
        "voices": str(VOICES_PATH),
        "output_directory": str(OUTPUT_DIR),
        "voice_count": len(kokoro.get_voices()),
        "languages": _languages(),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")