# EduMotion AI

**Turn an educational concept into an animated video with AI-generated narration.**

EduMotion AI is an asynchronous Python agent that uses LangGraph to coordinate educational script generation, storyboarding, Manim animation rendering, timeline extraction, text-to-speech, audio/video merging, and quality checks.

## Features

- Converts a natural-language topic and target duration into a video-generation workflow.
- Uses OpenRouter-compatible chat models for planning, scripts, storyboards, and Manim code.
- Renders mathematical and educational animations through a separate **Manim MCP server**.
- Generates local text-to-speech narration through **Kokoro TTS MCP**.
- Uses the **FFmpeg MCP server** to merge audio and video and inspect media.
- Extracts the rendered video timeline before generating narration, helping align speech with the visuals.
- Produces timeline, narration, audio, final-video, and QA artifacts.
- Includes two sample videos in `samples/`.

## Architecture

```text
User request
    |
    v
LangGraph workflow
    |
    +--> Plan topic and generate script
    |
    +--> Create storyboard
    |
    +--> Manim MCP server ----------> Render animation + timeline
    |
    +--> Timeline extraction
    |
    +--> Kokoro TTS MCP server -----> Narration audio
    |
    +--> Audio alignment
    |
    +--> FFmpeg MCP server ---------> Merge audio + video
    |
    +--> QA validation
    |
    v
Final educational video + reports
```

## Repository layout

```text
edumotion-ai/
├── src/video_agent/                 # Main LangGraph application
├── mcp-servers/
│   ├── ffmpeg-mcp-server/           # FFmpeg MCP server source
│   └── kokoro-mcp-server/           # Kokoro TTS MCP server source
├── samples/                         # Example generated videos
├── test_pipeline.py
├── test_integration.py
├── pyproject.toml
├── uv.lock
├── .env.example                     # Create this; never commit real secrets
├── .gitignore
└── README.md
```

The Manim MCP server is maintained in a separate GitHub repository and is not duplicated here. Replace the placeholder URL below with the actual repository URL:

- **Manim MCP server:** `https://github.com/Ankitchopradk/manim-mcp-server`

## Requirements

This setup is written for **Windows PowerShell** and Python **3.12 or later**.

Install the following before running the project:

1. **Python 3.12+** — [python.org](https://www.python.org/downloads/)
2. **uv** — [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
3. **FFmpeg and FFprobe** — [FFmpeg downloads](https://ffmpeg.org/download.html). Ensure both commands are available on `PATH`, or set their executable paths in the MCP server configuration.
4. **Manim Community Edition and its system dependencies** — install these according to the [Manim installation guide](https://docs.manim.community/en/stable/installation.html). The separate Manim MCP server may have its own setup instructions.
5. **Kokoro ONNX model and voices file** — required by the Kokoro MCP server. Configure the file paths as described below.
6. An **OpenRouter API key** — [OpenRouter](https://openrouter.ai/).

> Manim and Kokoro have their own runtime requirements. Installing the main project's Python dependencies does not automatically install their external system tools, model files, or separate virtual environments.

## 1. Clone the repository

```powershell
git clone https://github.com/Ankitchopradk/Automated_educational_video_generator
cd edumotion-ai
```

Replace the URL with the actual GitHub repository URL.

## 2. Install the main project's dependencies

From the repository root:

```powershell
uv sync
```

This creates or updates the main project's virtual environment from `pyproject.toml` and `uv.lock`.

If you use the three separate requirements files instead, install each into its **own** environment; do not mix server dependencies into one environment unless you deliberately choose to do so.

## 3. Configure environment variables

Create a `.env` file in the repository root. Do not commit this file.

Example:

```dotenv
OPENROUTER_API_KEY=replace_with_your_openrouter_api_key

# Optional model selection
LLM_MODEL=
CODE_LLM_MODEL=
LLM_MODELS=

# Optional generation tuning
TTS_SPEED=0.9
SCENE_PAD_SECONDS=0.8
SCENE_MAX_TOKENS=8000
SCENE_TIMEOUT=150
```

The application loads environment variables with `python-dotenv`. `OPENROUTER_API_KEY` is required. Model IDs can change or become unavailable; choose current IDs from [OpenRouter's model catalogue](https://openrouter.ai/models).

Never paste a real key into README files, source code, screenshots, issues, or commits. If a key has already been published, revoke it and create a replacement.

## 4. Set up the MCP servers

The main application connects to the MCP servers through **stdio transports**. The current `src/video_agent/tools.py` configuration is designed for this Windows Desktop layout:

```text
Desktop/
├── edumotion-ai/                 # Main project repository
├── manim-mcp-server/             # Separate Manim MCP repository
├── kokoro-mcp-server/            # Included MCP server folder
└── ffmpeg-mcp-server/            # Included MCP server folder
```

**Warning:** Keep the three MCP server folders and the EduMotion AI project in the locations expected by your current `tools.py`. If you clone the repositories into another directory, use different folder names, or place the included MCP servers under `edumotion-ai/mcp-servers/`, the configured paths may not match. Follow the path-update instructions below before running the agent.

> In this repository's suggested layout, Kokoro and FFmpeg are under `edumotion-ai/mcp-servers/`. If your current `tools.py` instead points to separate Desktop folders, either move the server folders to match the code or update `KOKORO_ROOT` and `FFMPEG_ROOT` to match your actual layout. The filesystem layout and Python paths must agree.

### A. Manim MCP server — separate repository

Clone and install the Manim MCP server using the instructions in its own repository:

```powershell
git clone https://github.com/Ankitchopradk/Automated_educational_video_generator C:\AI-Tools\manim-mcp-server
```

Use the repository's README to create its virtual environment and install its dependencies. Also install Manim and its required system dependencies.

In `src/video_agent/tools.py`, update the Manim transport configuration to point to:

- The Python executable in the Manim MCP server's environment.
- The Manim MCP server's actual entry-point script.
- The Manim MCP server's working directory.

The current code also defines `MANIM_PYTHON` for the linting subprocess. Update that path to the Python executable in the environment where Manim is installed. The MCP server's Python and the Manim linting Python should be compatible with your setup.

#### If you store a server somewhere else

Open `src/video_agent/tools.py` and change the root paths to the real directories on your computer. For example, if all repositories are directly on your Desktop, the configuration can use:

```python
from pathlib import Path

DESKTOP = Path.home() / "Desktop"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIM_ROOT = DESKTOP / "manim-mcp-server"
KOKORO_ROOT = DESKTOP / "kokoro-mcp-server"
FFMPEG_ROOT = DESKTOP / "ffmpeg-mcp-server"
```

If Kokoro and FFmpeg are instead inside the main repository, use:

```python
KOKORO_ROOT = PROJECT_ROOT / "mcp-servers" / "kokoro-mcp-server"
FFMPEG_ROOT = PROJECT_ROOT / "mcp-servers" / "ffmpeg-mcp-server"
```

Then ensure each transport uses the corresponding root for all three values:

- `command`: the Python executable in that server's virtual environment, for example `ROOT / ".venv" / "Scripts" / "python.exe"`.
- `args`: the correct server entry-point path, for example `ROOT / "src" / "kokoro_mcp_server" / "main.py"`.
- `cwd`: the server's root directory.

For the Manim server, `MANIM_PYTHON` must point to the Python executable that has Manim installed, for example:

```python
MANIM_PYTHON = str(MANIM_ROOT / "manim-venv" / "Scripts" / "python.exe")
```

These entry-point names are examples based on the current project layout; check that the files exist in your cloned server repositories. If your Desktop folder is redirected (for example, to OneDrive), update `DESKTOP` to the actual path. You can check each configured path in PowerShell with `Test-Path "C:\path\to\file"`.

### B. FFmpeg MCP server — included in this repository

Source code: `mcp-servers/ffmpeg-mcp-server/`

Install it in its own environment:

```powershell
cd mcp-servers/ffmpeg-mcp-server
uv sync
cd ../..
```

Install FFmpeg and FFprobe separately. Check that the commands work:

```powershell
ffmpeg -version
ffprobe -version
```

The server supports these optional environment variables:

- `FFMPEG_PATH` — full path to the FFmpeg executable.
- `FFPROBE_PATH` — full path to the FFprobe executable.
- `FFMPEG_OUTPUT` — output directory for generated media.
- `FFMPEG_TIMEOUT` — command timeout in seconds.

In `src/video_agent/tools.py`, update `FFMPEG_TRANSPORT` so its `command`, `args`, and `cwd` point to the FFmpeg MCP server's Python executable, `src/ffmpeg_mcp_server/main.py`, and server directory.

### C. Kokoro TTS MCP server — included in this repository

Source code: `mcp-servers/kokoro-mcp-server/`

Install it in its own environment:

```powershell
cd mcp-servers/kokoro-mcp-server
uv sync
cd ../..
```

The Kokoro server needs an ONNX model file and a voices file. Obtain these assets from the official/model source you choose and keep them outside Git unless their licence and size make redistribution appropriate.

Configure these variables for the Kokoro server process:

- `KOKORO_MODEL` — absolute path to the Kokoro ONNX model.
- `KOKORO_VOICES` — absolute path to the voices `.bin` file.
- `KOKORO_OUTPUT` — output directory for generated audio.

The source currently defaults to `C:\Kokoro\kokoro-v1.0.onnx` and `C:\Kokoro\voices-v1.0.bin`. Set the variables to your actual paths if they differ.

In `src/video_agent/tools.py`, update `KOKORO_TRANSPORT` so its `command`, `args`, and `cwd` point to the Kokoro MCP server's Python executable, `src/kokoro_mcp_server/main.py`, and server directory.

### Important: how MCP connections start

The main application creates `StdioTransport` clients for Manim, Kokoro, and FFmpeg. When the workflow connects to these clients, the server processes are started using the configured commands. You generally do not need to start those same stdio servers in separate terminals.

The current code uses absolute local paths. Cloning the repositories alone will **not** make the connections work until those paths are updated. Keep local machine paths and secrets out of the public repository.

## 5. Run EduMotion AI

From the repository root, use PowerShell:

```powershell
$env:PYTHONPATH = "src"
uv run python -m video_agent.main
```

Enter a request when prompted, for example:

```text
Explain quantum physics in 45 seconds
```

Other examples:

```text
Create a 2 minute video about photosynthesis
Make a 90 second tutorial on binary search
Explain the water cycle in 1:30
```

The pipeline can take time because it makes model requests, renders animations, generates narration, and validates the output.

## 6. Sample videos and generated outputs

The `samples/` directory contains example videos committed for demonstration. Newly generated artifacts are written to the output paths reported by the application. Depending on the pipeline run, these include:

- Rendered animation video.
- Timeline JSON.
- Narration JSON.
- Individual and final audio segments.
- Final merged video.
- QA report.
- Debug logs and generated scene code.

Debug files and temporary outputs should generally remain local and should not be committed.


## 7. Tests

The repository includes:

- `test_pipeline.py` — pipeline-oriented tests.
- `test_integration.py` — integration-oriented tests.

Run them using the test framework configured by the project. If these files use `pytest`, install it in the main environment and run:

```powershell
uv run pytest
```

Tests that invoke LLM APIs, MCP servers, Manim, FFmpeg, or Kokoro may require the relevant environment variables, local assets, and services. A test file existing in the repository does not by itself guarantee that every test passes on a clean machine.

## 8. Troubleshooting

### MCP server fails to start

- Confirm the server's virtual environment exists.
- Verify the `command`, `args`, and `cwd` values in `src/video_agent/tools.py`.
- Run the server entry-point manually to inspect missing dependencies or configuration errors.
- Check that the server process can access the files it needs.

### OpenRouter authentication or model error

- Confirm `OPENROUTER_API_KEY` is present in the root `.env`.
- Check that the selected model is currently available and that your account has access.
- Review rate limits and provider errors.

### Manim rendering fails

- Follow the separate Manim MCP repository's installation instructions.
- Confirm Manim and its system dependencies are installed in the Python environment used by the server and linting process.
- Inspect the local debug output and server logs.

### Kokoro cannot load its model

- Confirm the ONNX model and voices file exist.
- Set `KOKORO_MODEL` and `KOKORO_VOICES` for the MCP server process.
- Ensure the Kokoro server environment has all dependencies installed.

### FFmpeg cannot be found

- Run `ffmpeg -version` and `ffprobe -version`.
- Add the executables to `PATH` or configure `FFMPEG_PATH` and `FFPROBE_PATH`.

## Security and repository hygiene

Before pushing:

- Exclude `.env`, `.venv/`, `.git/` nested inside component folders, `__pycache__/`, `*.pyc`, generated debug output, and temporary build artifacts.
- Keep `.env.example` limited to variable names and safe placeholders.
- Do not commit API keys, access tokens, private model files, or local absolute paths specific to your computer.
- Review sample videos for private information and confirm you are allowed to publish them.
- If a secret was committed in the past, remove it from Git history where appropriate and revoke/rotate the credential. `.gitignore` does not erase old commits.

## Technologies

- Python
- LangGraph and LangChain
- OpenRouter-compatible chat models
- Model Context Protocol (MCP) with FastMCP
- Manim Community Edition
- Kokoro ONNX text-to-speech
- FFmpeg / FFprobe
- `uv` for Python dependency management

MIT License

Copyright (c) 2026 Ankit Chopra

**Maintainer:** EduMotion AI project
