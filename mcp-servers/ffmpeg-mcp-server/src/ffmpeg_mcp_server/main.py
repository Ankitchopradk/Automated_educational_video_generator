import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from fastmcp import FastMCP

# ---------- Configuration ----------
OUTPUT_DIR = Path(
    os.getenv("FFMPEG_OUTPUT", str(Path(__file__).resolve().parents[2] / "output"))
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEOUT_SECONDS = float(os.getenv("FFMPEG_TIMEOUT", "600"))


def _find_binary(name: str, env_var: str) -> str:
    found = os.getenv(env_var) or shutil.which(name)
    if not found or not Path(found).exists():
        raise FileNotFoundError(f"{name} not found. Add it to PATH or set {env_var}.")
    return found


FFMPEG = _find_binary("ffmpeg", "FFMPEG_PATH")
FFPROBE = _find_binary("ffprobe", "FFPROBE_PATH")

mcp = FastMCP("ffmpeg-mcp-server")


# ---------- Helpers ----------
async def _run(binary: str, args: list[str]) -> tuple[str, str]:
    """Run a binary without a shell; return (stdout, stderr) or raise on failure."""
    proc = await asyncio.create_subprocess_exec(
        binary,
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), TIMEOUT_SECONDS)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Timed out after {TIMEOUT_SECONDS:.0f}s.") from None

    out_text = out.decode(errors="replace")
    err_text = err.decode(errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{Path(binary).name} failed (exit {proc.returncode}): {err_text.strip()[-2000:]}"
        )
    return out_text, err_text


async def _ffmpeg(args: list[str]) -> tuple[str, str]:
    return await _run(FFMPEG, ["-hide_banner", "-loglevel", "error", "-nostdin", *args])


def _input(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Input file not found: {p}")
    return p


def _output(
    output: str | None, source: Path, tag: str, suffix: str | None, overwrite: bool
) -> Path:
    """Resolve the output path. Relative names go into OUTPUT_DIR."""
    if output:
        path = Path(output).expanduser()
        if not path.is_absolute():
            path = OUTPUT_DIR / path
    else:
        path = OUTPUT_DIR / f"{source.stem}_{tag}{suffix or source.suffix}"

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.resolve() == source.resolve():
        raise ValueError("Output file must be different from the input file.")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Use overwrite=true to replace it.")
    return path


def _result(path: Path) -> dict:
    return {"success": True, "output_path": str(path), "size_bytes": path.stat().st_size}


# ---------- Tools ----------
@mcp.tool
async def get_ffmpeg_status() -> dict:
    """Return FFmpeg/FFprobe locations, version and the output directory."""
    out, _ = await _run(FFMPEG, ["-version"])
    return {
        "status": "ready",
        "version": out.splitlines()[0] if out else "unknown",
        "ffmpeg": FFMPEG,
        "ffprobe": FFPROBE,
        "output_directory": str(OUTPUT_DIR),
        "timeout_seconds": TIMEOUT_SECONDS,
    }


@mcp.tool
async def get_media_info(input_path: str) -> dict:
    """Inspect a media file: container info plus a summary of every stream."""
    src = _input(input_path)
    out, _ = await _run(
        FFPROBE,
        ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(src)],
    )
    data = json.loads(out)

    fmt_keys = ("format_name", "duration", "size", "bit_rate")
    stream_keys = (
        "index", "codec_type", "codec_name", "width", "height",
        "r_frame_rate", "sample_rate", "channels", "bit_rate", "duration",
    )
    return {
        "file": str(src),
        "format": {k: data.get("format", {}).get(k) for k in fmt_keys},
        "streams": [
            {k: s[k] for k in stream_keys if k in s} for s in data.get("streams", [])
        ],
    }


@mcp.tool
async def convert_media(
    input_path: str,
    output: str,
    video_codec: str | None = None,
    audio_codec: str | None = None,
    overwrite: bool = False,
) -> dict:
    """
    Convert a file to another format. The output extension decides the container
    (e.g. .mp4, .mkv, .webm, .mp3). Codecs are optional, e.g. video_codec="libx264",
    audio_codec="aac"; use "copy" to keep a stream as-is.
    """
    src = _input(input_path)
    out = _output(output, src, "converted", None, overwrite)

    args = ["-y", "-i", str(src)]
    if video_codec:
        args += ["-c:v", video_codec]
    if audio_codec:
        args += ["-c:a", audio_codec]
    await _ffmpeg([*args, str(out)])
    return _result(out)


@mcp.tool
async def extract_audio(
    input_path: str,
    format: str = "mp3",
    bitrate: str | None = None,
    output: str | None = None,
    overwrite: bool = False,
) -> dict:
    """
    Extract the audio track. format: mp3, wav, aac, m4a, flac or ogg.
    bitrate (e.g. "192k") applies to lossy formats only.
    """
    codecs = {
        "mp3": "libmp3lame", "wav": "pcm_s16le", "aac": "aac",
        "m4a": "aac", "flac": "flac", "ogg": "libvorbis",
    }
    fmt = format.lower().lstrip(".")
    if fmt not in codecs:
        raise ValueError(f"Unsupported format '{format}'. Choose from: {', '.join(codecs)}")

    src = _input(input_path)
    out = _output(output, src, "audio", f".{fmt}", overwrite)

    args = ["-y", "-i", str(src), "-vn", "-c:a", codecs[fmt]]
    if bitrate and fmt not in ("wav", "flac"):
        args += ["-b:a", bitrate]
    await _ffmpeg([*args, str(out)])
    return _result(out)


@mcp.tool
async def trim_media(
    input_path: str,
    start: str = "0",
    end: str | None = None,
    duration: str | None = None,
    output: str | None = None,
    copy_streams: bool = True,
    overwrite: bool = False,
) -> dict:
    """
    Cut a section from a video/audio file. Times can be seconds ("12.5") or
    "HH:MM:SS". Give either end or duration. copy_streams=true is instant but
    cuts on keyframes; set false for frame-accurate (re-encoded) cuts.
    """
    if end and duration:
        raise ValueError("Provide either 'end' or 'duration', not both.")
    if not (end or duration):
        raise ValueError("Provide 'end' or 'duration'.")

    src = _input(input_path)
    out = _output(output, src, "trim", None, overwrite)

    args = ["-y", "-i", str(src), "-ss", start]
    args += ["-to", end] if end else ["-t", duration]
    if copy_streams:
        args += ["-c", "copy"]
    await _ffmpeg([*args, str(out)])
    return _result(out)


@mcp.tool
async def resize_video(
    input_path: str,
    width: int | None = None,
    height: int | None = None,
    output: str | None = None,
    overwrite: bool = False,
) -> dict:
    """
    Resize a video. Give width and/or height; if only one is given the other is
    scaled automatically to keep the aspect ratio.
    """
    if not (width or height):
        raise ValueError("Provide width and/or height.")

    src = _input(input_path)
    out = _output(output, src, "resized", None, overwrite)

    scale = f"scale={width or -2}:{height or -2}"
    await _ffmpeg(
        ["-y", "-i", str(src), "-vf", scale, "-c:v", "libx264", "-c:a", "copy", str(out)]
    )
    return _result(out)


@mcp.tool
async def compress_video(
    input_path: str,
    crf: int = 28,
    preset: str = "medium",
    audio_bitrate: str = "128k",
    output: str | None = None,
    overwrite: bool = False,
) -> dict:
    """
    Compress a video with H.264. Lower crf = better quality/larger file
    (18 near-lossless, 23 default-quality, 28 smaller). preset: ultrafast..veryslow.
    """
    if not 0 <= crf <= 51:
        raise ValueError("crf must be between 0 and 51.")

    src = _input(input_path)
    out = _output(output, src, "compressed", ".mp4", overwrite)

    await _ffmpeg([
        "-y", "-i", str(src),
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-c:a", "aac", "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        str(out),
    ])
    return _result(out)


@mcp.tool
async def extract_frame(
    input_path: str,
    timestamp: str = "0",
    format: str = "png",
    output: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Save a single frame at the given time ("12.5" or "HH:MM:SS") as an image."""
    src = _input(input_path)
    out = _output(output, src, "frame", f".{format.lower().lstrip('.')}", overwrite)

    await _ffmpeg(["-y", "-ss", timestamp, "-i", str(src), "-frames:v", "1", str(out)])
    return _result(out)


@mcp.tool
async def concat_media(
    input_paths: list[str],
    output: str | None = None,
    reencode: bool = False,
    overwrite: bool = False,
) -> dict:
    """
    Join files end-to-end in the given order. Fast stream-copy works when all
    files share the same codecs/settings; set reencode=true if they differ.
    """
    if len(input_paths) < 2:
        raise ValueError("Provide at least two input files.")

    sources = [_input(p) for p in input_paths]
    out = _output(output, sources[0], "concat", None, overwrite)

    lines = [
        "file '{}'".format(p.resolve().as_posix().replace("'", "'\\''")) for p in sources
    ]
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(lines))
        list_file = Path(f.name)

    try:
        args = ["-y", "-f", "concat", "-safe", "0", "-i", str(list_file)]
        if not reencode:
            args += ["-c", "copy"]
        await _ffmpeg([*args, str(out)])
    finally:
        list_file.unlink(missing_ok=True)
    return _result(out)


@mcp.tool
async def merge_audio_video(
    video_path: str,
    audio_path: str,
    mix_with_original: bool = False,
    shortest: bool = True,
    output: str | None = None,
    overwrite: bool = False,
) -> dict:
    """
    Combine a video and a separate audio file into ONE file (audio plays together
    with the video). By default the audio replaces the video's own sound; set
    mix_with_original=true to layer it over the original audio instead (the video
    must have an audio track). shortest=true ends the output when the shorter
    input ends. The video stream is copied, so this is fast.
    """
    video = _input(video_path)
    audio = _input(audio_path)
    out = _output(output, video, "merged", ".mp4", overwrite)

    args = ["-y", "-i", str(video), "-i", str(audio)]
    if mix_with_original:
        args += [
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[a]",
            "-map", "0:v:0", "-map", "[a]",
        ]
    else:
        args += ["-map", "0:v:0", "-map", "1:a:0"]
    args += ["-c:v", "copy", "-c:a", "aac"]
    if shortest:
        args.append("-shortest")

    await _ffmpeg([*args, str(out)])
    return _result(out)


@mcp.tool
async def run_ffmpeg(args: list[str]) -> dict:
    """
    Advanced: run FFmpeg with your own arguments (no shell is used), e.g.
    ["-y", "-i", "C:/in.mp4", "-vf", "hue=s=0", "C:/out.mp4"].
    Do not include the 'ffmpeg' executable itself.
    """
    if not args:
        raise ValueError("args cannot be empty.")
    out, err = await _ffmpeg(args)
    return {"success": True, "stdout": out.strip(), "stderr": err.strip()}


# ---------- Start MCP Server ----------
def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
