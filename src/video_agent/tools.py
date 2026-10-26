import asyncio
import ast
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from langchain_openai import ChatOpenAI

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

# Default first choice for general use. Per-step chains are in ROLE_MODELS below.
NEMOTRON_MODEL = os.getenv("LLM_MODEL") or "nvidia/nemotron-3.5-super-vl:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OPENROUTER CLIENT
# ============================================================

# Free OpenRouter models rotate and get retired, so no single ID is trusted.
# Every pipeline step ("role") has its OWN ordered chain, picked for what that step
# needs. The agent probes a chain in order, uses the first model that answers, and
# falls back to the rest if a call fails mid-run. Free-tier limits are PER MODEL
# (about 20 requests/min, 200/day), so falling through also gives you more quota.
#
# Notes on each model (OpenRouter free list, Oct 22 2026; "quality" = Theozard score):
#   nvidia/nemotron-3.5-super-vl   quality 46 (best here), 1M ctx, no Reasoning flag
#                                  -> first choice for code and strict JSON
#   poolside/laguna-s-2.1          described by OpenRouter as Poolside's coding-agent model
#                                  (unscored) -> 2nd choice for code. Poolside models were
#                                  reported to expire 31 Oct 2026; the chain handles that.
#   xingchen-agi/xing4.0-29b-a4b   unscored, ~29B MoE with ~4B active -> middle fallback
#   nvidia/nemotron-3.5-lightning  quality 22, 1M ctx, light -> fast fallback
#   dots-studio/dots-3-note-preview unscored, 512K ctx -> general/writing fallback
#   liquid/lfm-2.5-2.6b            quality 15, only 2.6B params: it invented Manim classes
#                                  in the first run -> NEVER used for code or JSON,
#                                  only as the very last resort for the script
#   recursive/flint-v1.0           OpenRouter answered HTTP 400 "not a valid model ID"
#                                  -> removed (add it back via LLM_MODELS if it returns)
_SUPER_VL = "nvidia/nemotron-3.5-super-vl:free"
_LAGUNA = "poolside/laguna-s-2.1:free"
_XING = "xingchen-agi/xing4.0-29b-a4b:free"
_LIGHTNING = "nvidia/nemotron-3.5-lightning:free"
_DOTS = "dots-studio/dots-3-note-preview:free"
_LIQUID = "liquid/lfm-2.5-2.6b:free"

ROLE_MODELS = {
    # Manim scene code + the code fixer: needs accurate API knowledge
    "code": [_SUPER_VL, _LAGUNA, _XING, _LIGHTNING, _DOTS],
    # Storyboard: strict JSON output
    "storyboard": [_SUPER_VL, _XING, _LIGHTNING, _DOTS, _LAGUNA],
    # Narration script (Hindi/Devanagari writing): best general model first
    "script": [_SUPER_VL, _DOTS, _XING, _LIGHTNING, _LIQUID],
}

# Env overrides (all optional) -- how they are applied:
#   CODE_LLM_MODEL = first choice for the CODE steps only (scene code + fixer)
#   LLM_MODEL      = first choice for every step (CODE_LLM_MODEL wins for code)
#   LLM_MODELS     = comma-separated extra models tried before the built-in chain
_ACTIVE_MODEL: str | None = None  # last model chosen (for log messages)
_ACTIVE_BY_ROLE: dict[str, str] = {}
_PROBE_CACHE: dict[str, bool] = {}


def _model_chain(role: str = "script") -> list[str]:
    """Ordered, de-duplicated list of models to try for a pipeline step."""
    wanted: list[str | None] = []
    if role == "code":
        wanted.append(os.getenv("CODE_LLM_MODEL"))
    wanted.append(os.getenv("LLM_MODEL"))
    wanted += [m.strip() for m in os.getenv("LLM_MODELS", "").split(",")]
    wanted += ROLE_MODELS.get(role, ROLE_MODELS["script"])

    chain: list[str] = []
    for m in wanted:
        m = (m or "").strip()
        if m and m not in chain:
            chain.append(m)
    return chain


def _reasoning_kwargs() -> dict:
    """Extra request params shared by the probe and real calls.

    Reasoning models spend hidden 'thinking' tokens out of max_tokens, which cut
    off long code requests. Ask for low effort; models without reasoning ignore it.
    Set LLM_REASONING_EFFORT= (empty) in .env to stop sending this parameter.
    """
    effort = os.getenv("LLM_REASONING_EFFORT", "low").strip()
    return {"extra_body": {"reasoning": {"effort": effort}}} if effort else {}


def _probe_model(model: str) -> bool:
    """One tiny request to check the model ID is currently being served.

    Cached per model, so a model shared by several steps is probed only once.
    """
    if model in _PROBE_CACHE:
        return _PROBE_CACHE[model]
    try:
        ChatOpenAI(
            model=model,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
            max_tokens=32,
            timeout=45,
            max_retries=0,
            **_reasoning_kwargs(),
        ).invoke("Reply with the single word: ok")
        ok = True
    except Exception as e:
        print(f"[LLM] {model} unavailable: {str(e)[:140]}")
        ok = False
    _PROBE_CACHE[model] = ok
    return ok


def _active_model(role: str = "script") -> str:
    """First model in this step's chain that actually answers (once per run)."""
    global _ACTIVE_MODEL
    if role in _ACTIVE_BY_ROLE:
        return _ACTIVE_BY_ROLE[role]
    chain = _model_chain(role)
    for m in chain:
        if m not in _PROBE_CACHE:
            print(f"[LLM:{role}] Checking {m} …")
        if _probe_model(m):
            print(f"[LLM:{role}] Using model: {m}")
            _ACTIVE_BY_ROLE[role] = m
            _ACTIVE_MODEL = m
            return m
    raise RuntimeError(
        f"No working LLM model found for '{role}'. Tried: " + ", ".join(chain) + ".\n"
        "Free model IDs change often: pick current ones from "
        "https://openrouter.ai/models?q=free and set LLM_MODEL / CODE_LLM_MODEL / "
        "LLM_MODELS in .env. (Also check OPENROUTER_API_KEY and the free-tier limit of "
        "about 200 requests/day per model.)"
    )


def _get_nemotron(
    temperature: float = 0.7,
    timeout: float = 180,
    max_retries: int = 1,
    max_tokens: int = 12000,
    model: str | None = None,
    role: str = "script",
) -> ChatOpenAI:
    """Return an OpenRouter chat client for a pipeline step.

    role = "script" | "storyboard" | "code" selects that step's model chain
    (see ROLE_MODELS). `model` forces a specific first choice.
    """
    kwargs = _reasoning_kwargs()

    def build(m: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=m,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
            max_tokens=max_tokens,
            **kwargs,
        )

    primary = model or _active_model(role)
    llm = build(primary)
    # If a call fails mid-run (model retired, 404, 429 rate limit), try the next
    # models in THIS step's chain.
    others = [m for m in _model_chain(role) if m != primary]
    if others:
        llm = llm.with_fallbacks([build(m) for m in others])
    return llm


def _ensure_complete(response) -> None:
    """Raise if the model stopped because it ran out of output tokens."""
    meta = getattr(response, "response_metadata", None) or {}
    if meta.get("finish_reason") == "length":
        usage = meta.get("token_usage") or {}
        details = usage.get("completion_tokens_details") or {}
        note = (
            f" (completion_tokens={usage.get('completion_tokens')}, "
            f"reasoning_tokens={details.get('reasoning_tokens')}, "
            f"visible_chars={len(str(getattr(response, 'content', '')))})"
        )
        raise ValueError("LLM output was truncated (hit max_tokens)" + note)


def _extract_code(text) -> str:
    """Pull clean Python out of an LLM reply.

    Handles: ```python fences anywhere, text before/after the block, <think>
    blocks, unclosed fences, and stray ``` lines left at the end.
    """
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        text = max(blocks, key=len)

    # Drop any chatter before the code starts
    idx = text.find("from manim import")
    if idx > 0:
        text = text[idx:]

    # Remove any leftover fence lines. Only strip blank lines at the ends: a
    # plain .strip() would eat the first line's indentation and break dedent.
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
    return "\n".join(lines).strip("\n").rstrip()


def _validate_python_syntax(code: str, stage: str) -> None:
    """Reject incomplete model output before it reaches Manim."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise ValueError(
            f"{stage} returned invalid Python at line {e.lineno}: {e.msg}"
        ) from e


async def _ainvoke_with_progress(llm, prompt: str, *, label: str, timeout: float):
    """Bound a model request and report progress while a provider is slow."""
    task = asyncio.create_task(llm.ainvoke(prompt))
    loop = asyncio.get_running_loop()
    started = loop.time()

    try:
        while True:
            remaining = timeout - (loop.time() - started)
            if remaining <= 0:
                raise TimeoutError(f"{label} timed out after {timeout:.0f} seconds.")

            done, _ = await asyncio.wait({task}, timeout=min(20, remaining))
            if done:
                return task.result()

            elapsed = int(loop.time() - started)
            print(f"[{label}] Still waiting for Nemotron… ({elapsed}s/{timeout:.0f}s)")
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass


# ============================================================
# MCP SERVER TRANSPORTS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIM_ROOT = Path.home() / "Desktop" / "manim-mcp-server"
KOKORO_ROOT = PROJECT_ROOT / "mcp-servers" / "kokoro-mcp-server"
FFMPEG_ROOT = PROJECT_ROOT / "mcp-servers" / "ffmpeg-mcp-server"

MANIM_TRANSPORT = StdioTransport(
    command=str(MANIM_ROOT / "manim-venv" / "Scripts" / "python.exe"),
    args=[str(MANIM_ROOT / "src" / "manim_server.py")],
    cwd=str(MANIM_ROOT),
)

KOKORO_TRANSPORT = StdioTransport(
    command=str(KOKORO_ROOT / ".venv" / "Scripts" / "python.exe"),
    args=[str(KOKORO_ROOT / "src" / "kokoro_mcp_server" / "main.py")],
    cwd=str(KOKORO_ROOT),
)

FFMPEG_TRANSPORT = StdioTransport(
    command=str(FFMPEG_ROOT / ".venv" / "Scripts" / "python.exe"),
    args=[str(FFMPEG_ROOT / "src" / "ffmpeg_mcp_server" / "main.py")],
    cwd=str(FFMPEG_ROOT),
)

MCP_SERVERS = {
    "manim": Client(MANIM_TRANSPORT),
    "kokoro": Client(KOKORO_TRANSPORT),
    "ffmpeg": Client(FFMPEG_TRANSPORT),
}


# ============================================================
# MCP TOOL INVOCATION
# ============================================================

async def call_mcp_tool(
    server: str,
    tool_name: str,
    arguments: dict | None = None,
    timeout: float | None = 300,
) -> dict | str | list:
    """Call a tool on one of the three MCP servers."""
    if server not in MCP_SERVERS:
        raise ValueError(
            f"Unknown MCP server '{server}'. "
            f"Choose from: {', '.join(MCP_SERVERS)}"
        )

    client = MCP_SERVERS[server]

    async with client:
        result = await client.call_tool(
            name=tool_name,
            arguments=arguments or {},
            timeout=timeout,
        )

    # Extract text content from the CallToolResult
    if hasattr(result, "content"):
        texts = []
        for block in result.content:
            if hasattr(block, "text"):
                texts.append(block.text)
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except (json.JSONDecodeError, TypeError):
                return texts[0]
        return texts if texts else result

    return result


# ============================================================
# MCP TOOL DISCOVERY
# ============================================================

async def discover_mcp_tools() -> dict[str, list[str]]:
    """Connect to all three MCP servers and return their available tool names."""
    discovered: dict[str, list[str]] = {}
    for name, client in MCP_SERVERS.items():
        try:
            async with client:
                tools = await client.list_tools()
                discovered[name] = [t.name for t in tools]
                print(f"[MCP] {name}: discovered {len(tools)} tool(s)")
                for t in tools:
                    print(f"       - {t.name}")
        except Exception as e:
            print(f"[MCP] {name}: connection failed — {e}")
            discovered[name] = []
    return discovered


# ============================================================
# GENERATE SCRIPT  (Nemotron)
# ============================================================

async def generate_script(
    user_request: str,
    duration: int = 60,
    platform: str = "youtube",
    aspect_ratio: str = "16:9",
    retries: int = 3,
    language_name: str = "English",
) -> str:
    """Generate a single educational narration script using Nemotron."""
    from .prompts import SCRIPT_PROMPT

    llm = _get_nemotron(temperature=0.7, role="script")

    prompt = f"""\
{SCRIPT_PROMPT}

PLATFORM: {platform}
ASPECT RATIO: {aspect_ratio}
TARGET DURATION: {duration} seconds
WORD LIMIT: {int(duration * 1.7)} words maximum
NARRATION LANGUAGE: {language_name}

USER REQUEST:
{user_request}
"""

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            print(f"[Nemotron] Generating script… attempt {attempt}/{retries}")
            response = await llm.ainvoke(prompt)
            script = response.content.strip()
            if not script:
                raise ValueError("Nemotron returned an empty script.")
            print("[Nemotron] Script generated successfully.")
            return script
        except Exception as e:
            last_error = str(e)
            print(f"[Nemotron] Attempt {attempt} failed: {last_error}")
            if attempt < retries:
                await asyncio.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"Nemotron script generation failed after {retries} attempts: {last_error}"
    )


# ============================================================
# GENERATE STORYBOARD  (Nemotron)
# ============================================================

async def generate_storyboard(
    selected_script: str,
    user_request: str,
    duration: int = 60,
    aspect_ratio: str = "16:9",
    retries: int = 3,
) -> list:
    """Generate a structured storyboard (list of scene dicts) from the script."""
    from .prompts import STORYBOARD_PROMPT

    llm = _get_nemotron(temperature=0.2, role="storyboard")

    prompt = f"""\
{STORYBOARD_PROMPT}

TARGET DURATION: {duration} seconds
ASPECT RATIO: {aspect_ratio}

USER REQUEST:
{user_request}

SELECTED SCRIPT:
{selected_script}
"""

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            print(f"[Storyboard] Generating storyboard… attempt {attempt}/{retries}")
            response = await llm.ainvoke(prompt)
            content = response.content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.replace("```json", "")
                content = content.replace("```", "")
                content = content.strip()

            data = json.loads(content)

            if isinstance(data, dict) and "scenes" in data:
                scenes = data["scenes"]
            elif isinstance(data, list):
                scenes = data
            else:
                raise ValueError("Storyboard JSON must contain a 'scenes' key or be a list.")

            if not scenes:
                raise ValueError("Storyboard contains zero scenes.")

            print(f"[Storyboard] Generated {len(scenes)} scene(s) successfully.")
            return scenes
        except Exception as e:
            last_error = str(e)
            print(f"[Storyboard] Attempt {attempt} failed: {last_error}")
            if attempt < retries:
                await asyncio.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"Storyboard generation failed after {retries} attempts: {last_error}"
    )


# ============================================================
# GENERATE MANIM CODE  (Nemotron)
# ============================================================

SCENE_CODE_PROMPT = '''\
You write ONE scene of a Manim Community Edition 0.21 animation.

Return ONLY the Python statements for this scene, as the body of a method:
no imports, no class, no def, no markdown fences, no explanations.
Do NOT call self.begin_scene(), self.end_scene() or self.clear(); the caller adds them.
The screen is empty when your code starts.

RULES (each one caused real render failures):
- Keep it SHORT: 10-30 lines, a few Mobjects, 2-5 self.play() calls.
- Only use classes that exist in Manim CE 0.21: Axes, NumberLine, Text, Line, Arrow, Dot,
  Circle, Rectangle, Square, Polygon, VGroup, Write, Create, FadeIn, FadeOut, Transform,
  ReplacementTransform, Indicate, GrowFromEdge, SurroundingRectangle, etc.
  NEVER invent classes (SineFunction, Highlight, ShowResult, ...).
- NEVER use MathTex, Tex, BulletedList or anything needing LaTeX. Use Text("...") for ALL text.
  numpy is available as np; PI is Manim's constant (never pi).
- self.play() and self.wait() return None: never chain them. self.play() takes only
  Animation objects, e.g. self.play(Write(title)). Create every object BEFORE animating it.
- Axes(x_range=[0, 3.5, 1], y_range=[-1.5, 1.5, 0.5], x_length=8, y_length=4)   (NOT xrange/yrange)
  Do NOT pass include_numbers=True.
  curve = axes.plot(lambda x: np.sin(x), x_range=[0, PI], color=BLUE)
  area = axes.get_area(curve, x_range=(0, PI), color=BLUE, opacity=0.4)
  axes.plot() has no 'opacity' kwarg (use stroke_opacity).
- The background is black: never use color=BLACK. Use WHITE, YELLOW, BLUE, GREEN, ORANGE...
- Any Hindi/Devanagari text MUST be Text("...", font="Nirmala UI"). Keep math such as
  "sin(x)" or "= 2" in Latin characters. Keep each Text under ~40 characters, font_size 30-44,
  and make sure everything fits the screen (use .to_edge(), .next_to(), .shift()).
- Mobjects have no set_height()/set_width(); use stretch_to_fit_height()/stretch_to_fit_width().
- ArrowTip is abstract; use Arrow(...).
- Reveal things step by step so the screen follows the narration.
'''


# Per-scene code generation knobs (override in .env)
#   SCENE_MAX_TOKENS: output budget for attempt 1 (attempt 2 gets x2, attempt 3 x3).
#     Thinking models burn most of it on hidden reasoning, so keep it generous.
#   SCENE_TIMEOUT:    seconds to wait for attempt 1 (scaled up on later attempts).
#   CODE_LLM_MODEL:   first choice for the code steps (scene code AND the fixer);
#     the 'code' chain in ROLE_MODELS is used after it.
SCENE_MAX_TOKENS = int(os.getenv("SCENE_MAX_TOKENS", "8000"))
SCENE_TIMEOUT = float(os.getenv("SCENE_TIMEOUT", "150"))


def _indent_block(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join((pad + ln) if ln.strip() else "" for ln in text.splitlines())


def _clean_scene_body(raw) -> str:
    """Turn a model reply into plain statements for one scene."""
    import textwrap

    text = _extract_code(raw)
    kept = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith(("from manim", "import ", "from numpy")):
            continue
        if s.startswith(("self.begin_scene", "self.end_scene")) or s == "self.clear()":
            continue
        kept.append(ln)
    return textwrap.dedent("\n".join(kept)).strip("\n")


def _validate_scene_body(body: str) -> None:
    if not body.strip():
        raise ValueError("empty scene body")
    if re.search(r"(?m)^\s*(class|def)\s", body):
        raise ValueError("return only statements, not a class or def")
    if "self.play(" not in body and "self.add(" not in body:
        raise ValueError("scene has no self.play()/self.add() call")
    for bad in ("MathTex", "Tex(", "BulletedList", "SingleStringMathTex"):
        if bad in body:
            raise ValueError(f"forbidden construct: {bad}")
    # Quality: warn about FadeOut before introduction
    if "FadeOut(" in body and "FadeIn(" not in body and "Create(" not in body and "Write(" not in body and "self.add(" not in body:
        raise ValueError(
            "FadeOut used but no FadeIn/Create/Write/self.add — objects must "
            "be introduced before they are removed"
        )
    try:
        ast.parse("def _f(self):\n" + _indent_block(body, 4))
    except SyntaxError as e:
        raise ValueError(f"invalid Python at line {e.lineno}: {e.msg}") from e


def _scene_fallback_body(scene: dict) -> str:
    """Plain caption scene used when the model cannot produce a valid one."""
    label = str(
        scene.get("on_screen_text") or scene.get("title") or ""
    ).replace("\n", " ").strip()[:60] or "..."
    return (
        f"caption = Text({label!r}, font=\"Nirmala UI\", font_size=40)\n"
        "self.play(FadeIn(caption))"
    )


async def _generate_scene_body(
    llm, scene: dict, idx: int, total: int, user_request: str,
    aspect_ratio: str, retries: int,
) -> str:
    brief = {
        k: scene[k]
        for k in ("title", "visual_description", "animation", "on_screen_text", "duration_seconds")
        if k in scene
    }
    base_prompt = (
        f"{SCENE_CODE_PROMPT}\n"
        f"VIDEO TOPIC: {user_request}\n"
        f"ASPECT RATIO: {aspect_ratio}\n"
        f"SCENE {idx} of {total}:\n{json.dumps(brief, ensure_ascii=False, indent=2)}\n"
    )

    last_error = ""
    for attempt in range(1, retries + 1):
        prompt = base_prompt
        if last_error:
            prompt += (
                f"\nYour previous answer was rejected: {last_error}\n"
                "Return a shorter, simpler version.\n"
            )
        try:
            # Bigger budget + longer wait on each retry (reasoning models truncate
            # when they think too long, so the same budget would fail again).
            response = await _ainvoke_with_progress(
                llm.bind(max_tokens=SCENE_MAX_TOKENS * attempt),
                prompt,
                label=f"ManimCode scene {idx}",
                timeout=SCENE_TIMEOUT * attempt,
            )
            _ensure_complete(response)
            body = _clean_scene_body(response.content)
            _validate_scene_body(body)
            print(f"[ManimCode] Scene {idx}/{total} OK ({len(body.splitlines())} lines)")
            return body
        except Exception as e:
            last_error = str(e)
            # A wrong model ID / bad API key will fail identically for every scene:
            # stop now instead of silently building a video from caption fallbacks.
            if "not a valid model" in last_error or "Error code: 401" in last_error:
                raise RuntimeError(
                    f"LLM configuration error: {last_error}\n"
                    "Check LLM_MODEL / CODE_LLM_MODEL / OPENROUTER_API_KEY in your .env."
                ) from e
            print(f"[ManimCode] Scene {idx} attempt {attempt}/{retries} failed: {last_error}")
            if attempt < retries:
                await asyncio.sleep(2 ** (attempt - 1))

    print(f"[ManimCode] Scene {idx}: using plain caption fallback.")
    return _scene_fallback_body(scene)


async def generate_manim_code(
    storyboard: list,
    user_request: str,
    aspect_ratio: str = "16:9",
    retries: int = 3,
) -> str:
    """Generate the Manim file ONE SCENE PER LLM CALL, then assemble it.

    Small requests cannot be truncated, a bad scene is retried (or replaced by
    a caption) without losing the others, and begin_scene()/end_scene(duration)
    are added here so timing always matches the voiceover.
    """
    llm = _get_nemotron(
        temperature=0.2,
        timeout=SCENE_TIMEOUT * 3 + 30,
        max_tokens=SCENE_MAX_TOKENS,
        role="code",  # honors CODE_LLM_MODEL, then the 'code' chain
    )
    total = len(storyboard)
    sem = asyncio.Semaphore(4)  # free-tier allows ~20 req/min/model

    print(
        f"[ManimCode] Model for scene code: {_ACTIVE_BY_ROLE.get('code', '?')} "
        f"(next in chain: {', '.join(_model_chain('code')[1:3])} …)"
    )
    print(f"[ManimCode] Generating {total} scene(s), one request each…")

    async def one(i: int, scene: dict) -> str:
        async with sem:
            return await _generate_scene_body(
                llm, scene, i, total, user_request, aspect_ratio, retries
            )

    bodies = await asyncio.gather(*(one(i, s) for i, s in enumerate(storyboard, 1)))

    blocks = []
    for i, (scene, body) in enumerate(zip(storyboard, bodies), 1):
        duration = float(scene.get("duration_seconds", 6))
        blocks.append(
            f"        # --- Scene {i} ---\n"
            "        self.begin_scene()\n"
            f"{_indent_block(body, 8)}\n"
            f"        self.end_scene({duration})\n"
        )

    code = (
        "from manim import *\n"
        "import numpy as np\n\n\n"
        "class GeneratedScene(Scene):\n"
        "    def construct(self):\n"
        + "\n".join(blocks)
    )
    _validate_python_syntax(code, "Manim scene assembler")
    print(f"[ManimCode] Assembled {len(code.splitlines())} lines of Manim code.")
    return code


async def _generate_manim_code_single_shot(
    storyboard: list,
    user_request: str,
    aspect_ratio: str = "16:9",
    retries: int = 3,
) -> str:
    """(Unused backup) Generate a complete Manim Python file in ONE request."""
    from .prompts import MANIM_CODE_PROMPT

    llm = _get_nemotron(temperature=0.2, role="code")

    storyboard_json = json.dumps(storyboard, indent=2)

    prompt = f"""\
{MANIM_CODE_PROMPT}

ASPECT RATIO: {aspect_ratio}

STORYBOARD:
{storyboard_json}

USER REQUEST:
{user_request}
"""

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            print(f"[ManimCode] Generating Manim code… attempt {attempt}/{retries}")
            attempt_prompt = prompt
            if last_error:
                attempt_prompt += (
                    "\n\nThe previous code response was rejected: "
                    f"{last_error}\nReturn a complete corrected Python file."
                )
            response = await llm.ainvoke(attempt_prompt)
            _ensure_complete(response)
            code = _extract_code(response.content)

            # Basic validation
            if "class GeneratedScene" not in code:
                raise ValueError("Generated code is missing 'class GeneratedScene'.")
            if "def construct" not in code:
                raise ValueError("Generated code is missing 'def construct'.")

            _validate_python_syntax(code, "Manim code generator")

            # Safety checks: reject forbidden constructs
            forbidden = ["MathTex", "Tex(", "BulletedList", "SingleStringMathTex"]
            for f in forbidden:
                if f in code:
                    raise ValueError(f"Generated code contains forbidden construct: {f}")

            print("[ManimCode] Manim code generated successfully.")
            return code
        except Exception as e:
            last_error = str(e)
            print(f"[ManimCode] Attempt {attempt} failed: {last_error}")
            if attempt < retries:
                await asyncio.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"Manim code generation failed after {retries} attempts: {last_error}"
    )


# ============================================================
# RENDER MANIM VIDEO  (Manim MCP)
# ============================================================

MANIM_ROOT = Path.home() / "Desktop" / "manim-mcp-server"

MANIM_PYTHON = str(MANIM_ROOT / "manim-venv" / "Scripts" / "python.exe")
LINT_SCRIPT = Path(__file__).with_name("manim_lint.py")

DEBUG_DIR = OUTPUT_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def _tail(text: str, limit: int = 1800) -> str:
    """Keep the END of a traceback - the real exception is at the bottom."""
    return text if len(text) <= limit else "...[truncated]...\n" + text[-limit:]


def repair_known_manim_api_issues(code: str) -> tuple[str, list[str]]:
    """Correct common generated setter names that Manim Mobjects do not expose."""
    fixes = []
    repaired_lines = []

    for line in code.splitlines():
        for dimension in ("height", "width"):
            invalid = f".set_{dimension}"
            if invalid not in line:
                continue

            valid = f".stretch_to_fit_{dimension}"
            if re.search(rf"{re.escape(invalid)}\(\s*0(?:\.0+)?\s*,", line) and re.search(
                r"\bstretch\s*=\s*True\b", line
            ):
                # Keep a tiny nonzero size so the following stretch animation
                # has a valid starting dimension; Manim cannot rescale from 0.
                line = re.sub(
                    rf"{re.escape(invalid)}\(\s*0(?:\.0+)?\s*,",
                    f"{valid}(0.001,",
                    line,
                )
            line = re.sub(rf"\.set_{dimension}\b", valid, line)
            if re.search(r"\bstretch\s*=\s*True\b", line):
                line = re.sub(
                    r"\bstretch\s*=\s*True\b", "about_edge=DOWN", line
                )
            else:
                line = re.sub(r",\s*stretch\s*=\s*False\b", "", line)
            fixes.append(f"set_{dimension} -> stretch_to_fit_{dimension}")

        arrow_tip = re.search(
            r"ArrowTip\((?P<args>[^()\n]*)\)\.move_to\((?P<line>\w+)\.get_end\(\)\)",
            line,
        )
        if arrow_tip:
            args = re.sub(r"\bheight\s*=", "length=", arrow_tip.group("args"))
            line_name = arrow_tip.group("line")
            replacement = (
                f"ArrowTriangleFilledTip({args})"
                f".rotate({line_name}.get_angle() - PI)"
                f".move_to({line_name}.get_end())"
            )
            line = line[:arrow_tip.start()] + replacement + line[arrow_tip.end():]
            fixes.append("ArrowTip -> ArrowTriangleFilledTip")

        repaired_lines.append(line)

    return "\n".join(repaired_lines), fixes


NAMES_CACHE = Path(__file__).with_name("manim_names.json")


def _extract_model_code(full_code: str) -> str:
    """Return only the model-authored portion of the assembled Manim file.

    Infrastructure code (_TimedSceneMixin, _TimelineTracker, etc.) is injected
    by the pipeline and should never be linted — it's known-good Python.
    We keep `from manim import *` so the linter can resolve names, and only
    lint from the GeneratedScene class onward.
    """
    marker = "class GeneratedScene"
    idx = full_code.find(marker)
    if idx < 0:
        return full_code  # fall through to normal lint

    # Keep the import line so the linter's name resolution still works.
    # The scene header has also been augmented with these injected helpers;
    # remove those bases from the lint-only copy because their definitions are
    # intentionally excluded above.
    model_code = full_code[idx:]

    def _without_injected_bases(match: re.Match) -> str:
        bases = [base.strip() for base in match.group(2).split(",") if base.strip()]
        bases = [
            base for base in bases
            if base not in {"_TimelineTracker", "_TimedSceneMixin"}
        ]
        return match.group(1) + ", ".join(bases) + match.group(3)

    model_code = re.sub(
        r"(?m)^(class\s+GeneratedScene\s*\()([^)]*)(\)\s*:)",
        _without_injected_bases,
        model_code,
        count=1,
    )
    return "from manim import *\nimport numpy as np\n\n" + model_code


async def lint_manim_code(code: str) -> list[str]:
    """Fast pre-flight check (runs in the Manim venv). Returns a list of issues.

    Infrastructure problems (timeout, crash) never block rendering: the lint is
    skipped and Manim itself reports any real error.
    Only the model-authored portion (from `class GeneratedScene` onward) is
    linted, so injected helpers can never trigger false positives.
    """
    import subprocess

    # Only lint the model-written part — not injected infrastructure.
    lint_input = _extract_model_code(code)

    def _run(extra: list, stdin_text: str, timeout: int):
        return subprocess.run(
            [MANIM_PYTHON, str(LINT_SCRIPT), *extra],
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )

    try:
        if not NAMES_CACHE.exists():
            print("[Manim] Building Manim name cache (one-time, may take a minute)…")
            await asyncio.to_thread(_run, ["--build-cache"], "", 300)
        proc = await asyncio.to_thread(_run, [], lint_input, 120)
    except subprocess.TimeoutExpired:
        print("[Manim] Lint timed out; skipping pre-flight lint for this attempt.")
        return []
    except Exception as e:
        print(f"[Manim] Lint unavailable ({e}); skipping.")
        return []

    try:
        return json.loads(proc.stdout.strip() or "[]")
    except json.JSONDecodeError:
        print(f"[Manim] Lint script error, skipping: {_tail(proc.stderr, 300)}")
        return []


TIMED_SCENE_HELPER = '''
class _TimedSceneMixin:
    """Timing helpers mixed into GeneratedScene without adding a Manim Scene."""

    def begin_scene(self):
        self._scene_t0 = self.renderer.time

    def end_scene(self, duration):
        elapsed = self.renderer.time - getattr(self, "_scene_t0", self.renderer.time)
        remaining = float(duration) - elapsed
        if remaining > 0.05:
            self.wait(remaining)
        self.clear()
'''


def inject_timed_scene(code: str) -> str:
    """Add scene timing methods without exposing a second renderable Scene."""
    has_mixin = "class _TimedSceneMixin" in code
    if not has_mixin:
        # Migrate older model output that defined TimedScene(Scene).
        timed_class = re.search(
            r"(?m)^class\s+TimedScene\s*(?:\([^)]*\))?\s*:", code
        )
        if timed_class:
            code = (
                code[:timed_class.start()]
                + "class _TimedSceneMixin:"
                + code[timed_class.end():]
            )
            has_mixin = True

    scene_class = re.search(
        r"(?m)^class\s+GeneratedScene\s*\(([^)]*)\)\s*:", code
    )
    if scene_class:
        bases = [base.strip() for base in scene_class.group(1).split(",") if base.strip()]
        bases = [base for base in bases if base not in {"TimedScene", "_TimedSceneMixin"}]
        if "Scene" not in bases:
            bases.insert(0, "Scene")
        bases.append("_TimedSceneMixin")
        header = f"class GeneratedScene({', '.join(bases)}):"
        code = code[:scene_class.start()] + header + code[scene_class.end():]

    if has_mixin:
        return code

    marker = "from manim import *"
    if marker in code:
        return code.replace(marker, marker + "\n" + TIMED_SCENE_HELPER, 1)
    return "from manim import *\n" + TIMED_SCENE_HELPER + "\n" + code


async def render_manim_video(manim_code: str) -> tuple[str, str]:
    """Send Manim code to the Manim MCP server and return the video file path + timeline path.

    Returns:
        (video_path, timeline_path)
    """
    from .timeline_extractor import inject_timeline_tracking
    from .manim_validator import validate_manim_code, fix_common_errors

    manim_code, api_fixes = repair_known_manim_api_issues(manim_code)
    if api_fixes:
        unique_fixes = ", ".join(dict.fromkeys(api_fixes))
        print(f"[Manim] Applied Manim API correction: {unique_fixes}.")

    manim_code = inject_timed_scene(manim_code)

    # CRITICAL: Inject timeline tracking to capture animation timeline
    print("[Manim] Injecting timeline tracking...")
    manim_code = inject_timeline_tracking(manim_code)

    # NEW: Apply automatic fixes for common errors
    fixed_code, auto_fixes = fix_common_errors(manim_code)
    if auto_fixes:
        print(f"[Manim] Auto-fixed: {', '.join(auto_fixes)}")
        manim_code = fixed_code

    # NEW: Validate code before rendering
    print("[Manim] Validating code...")
    is_valid, validation_errors = validate_manim_code(manim_code)
    if not is_valid:
        print(f"[Manim] Validation found {len(validation_errors)} issue(s):")
        for i, error in enumerate(validation_errors, 1):
            print(f"  {i}. {error}")
        # Don't fail immediately - let the lint catch it or let it fail with better error
        print("[Manim] Proceeding with caution...")

    # Always save the code + full result so failures can be inspected.
    (DEBUG_DIR / "last_scene.py").write_text(manim_code, encoding="utf-8")

    issues = await lint_manim_code(manim_code)
    if issues:
        msg = "Pre-flight lint found problems:\n- " + "\n- ".join(issues)
        print(f"[Manim] {msg}")
        raise RuntimeError(msg)

    print("[Manim MCP] Rendering video…")

    result = await call_mcp_tool(
        "manim",
        "execute_manim_code",
        {"manim_code": manim_code},
        timeout=300,
    )

    result_text = result if isinstance(result, str) else str(result)
    (DEBUG_DIR / "last_manim_result.txt").write_text(result_text, encoding="utf-8")

    if "Execution successful" not in result_text:
        raise RuntimeError(f"Manim rendering failed:\n{_tail(result_text, 2500)}")

    # Parse out the video path: "Video: <path>"
    match = re.search(r"Video:\s*(.+\.mp4)", result_text)
    if not match:
        raise RuntimeError(
            f"Manim succeeded but no video path found in response:\n{result_text[:500]}"
        )

    video_path = match.group(1).strip()

    if not Path(video_path).exists():
        raise RuntimeError(f"Manim reported video at {video_path} but file does not exist.")

    # timeline.json is written to Manim's working directory, which the server sets
    # to the per-request folder ("Output directory: ..."), NOT next to the video
    # (that is several levels deeper: media/videos/scene/480p15/). Search all of them.
    video_dir = Path(video_path).parent
    candidates = []
    out_dir_match = re.search(r"Output directory:\s*(.+)", result_text)
    if out_dir_match:
        candidates.append(Path(out_dir_match.group(1).strip()) / "timeline.json")
    candidates += [parent / "timeline.json" for parent in [video_dir, *video_dir.parents][:6]]
    candidates.append(Path("timeline.json"))

    timeline_path = next((p for p in candidates if p.exists()), None)

    if timeline_path is None:
        raise RuntimeError(
            "Timeline file not found. Looked in:\n  "
            + "\n  ".join(str(p) for p in candidates)
            + "\nTimeline tracking may have failed during rendering."
        )

    print(f"[Manim MCP] Video rendered: {video_path}")
    print(f"[Manim MCP] Timeline saved: {timeline_path}")
    return str(video_path), str(timeline_path)


# ============================================================
# FIX MANIM CODE  (feed error back to Nemotron)
# ============================================================

def _find_failing_scene(code: str, error_message: str) -> tuple[int | None, str, int, int]:
    """Locate which scene caused the error and return its body + line range.

    Returns (scene_number, scene_body, start_line_idx, end_line_idx).
    If the scene cannot be identified, returns (None, "", 0, 0).
    """
    # Try to extract the failing line number from the error traceback
    line_match = re.search(r'line (\d+)', error_message)
    if not line_match:
        return None, "", 0, 0

    error_line = int(line_match.group(1))
    lines = code.splitlines()

    # Find all scene markers: "# --- Scene N ---"
    scene_markers: list[tuple[int, int]] = []  # (line_idx, scene_number)
    for i, ln in enumerate(lines):
        m = re.match(r'\s*# --- Scene (\d+) ---', ln)
        if m:
            scene_markers.append((i, int(m.group(1))))

    if not scene_markers:
        return None, "", 0, 0

    # Find which scene contains the error line (1-indexed → 0-indexed)
    error_idx = error_line - 1
    for pos, (start_idx, scene_num) in enumerate(scene_markers):
        # End of this scene is the start of the next, or end of file
        if pos + 1 < len(scene_markers):
            end_idx = scene_markers[pos + 1][0]
        else:
            end_idx = len(lines)

        if start_idx <= error_idx < end_idx:
            body = "\n".join(lines[start_idx:end_idx])
            return scene_num, body, start_idx, end_idx

    return None, "", 0, 0


async def fix_manim_code(broken_code: str, error_message: str) -> str:
    """Ask Nemotron to fix Manim code that failed to render.

    Strategy: if the error can be traced to a specific scene (via the
    ``# --- Scene N ---`` markers the assembler writes), only that scene's
    body (~15 lines) is sent for repair. This is far faster than asking
    a slow free-tier model to rewrite 150+ lines.
    """
    scene_num, scene_body, start_idx, end_idx = _find_failing_scene(
        broken_code, error_message
    )

    # A stalled free endpoint must not keep this pipeline waiting forever,
    # but 90s was too short for 150+ line files.  Give the fixer enough time
    # for a realistic multi-scene file (~240s on slow free endpoints).
    fix_timeout = float(os.getenv("FIXER_TIMEOUT", "240"))
    llm = _get_nemotron(
        temperature=0.1, timeout=fix_timeout, max_retries=1, role="code"
    )

    # Keep the END of the error (the real exception), not the start.
    error_trimmed = _tail(error_message, 1800)

    if scene_num and scene_body:
        # ── Per-scene fix (fast, ~15 lines) ──────────────────
        return await _fix_single_scene(
            llm, broken_code, scene_num, scene_body,
            start_idx, end_idx, error_trimmed, fix_timeout,
        )
    else:
        # ── Whole-file fallback ──────────────────────────────
        return await _fix_whole_file(
            llm, broken_code, error_trimmed, fix_timeout,
        )


async def _fix_single_scene(
    llm, full_code: str, scene_num: int, scene_body: str,
    start_idx: int, end_idx: int, error_trimmed: str, timeout: float,
) -> str:
    """Fix a single scene body and splice it back into the full file."""
    prompt = f"""\
You are a Manim Community Edition 0.21.x debugging expert.

Scene {scene_num} of a larger file FAILED to render. Fix ONLY this scene.

RULES (each one caused real failures):
- NEVER use MathTex, Tex, SingleStringMathTex, or BulletedList.
- Use Text("...") for ALL text including math.
- ax.plot() NOT ax.get_graph() (deprecated).
- Do NOT use include_numbers=True.
- Create every variable BEFORE animating it in self.play().
- Every object must be introduced (FadeIn, Create, Write, self.add) before
  being animated or faded out.
- self.play() and self.wait() return None — never chain them.
- Keep Text under 40 chars, font_size 30-44, and on screen (.to_edge, .shift).
- The scene MUST start with self.begin_scene() and end with self.end_scene(<duration>).

BROKEN SCENE CODE:
```python
{scene_body}
```

ERROR:
{error_trimmed}

Return ONLY the fixed scene code (the block from "# --- Scene {scene_num} ---" through
"self.end_scene(...)"). No explanations, no markdown fences, no full file.
"""
    print(f"[ManimFix] Fixing Scene {scene_num} (~{len(scene_body.splitlines())} lines)…")
    response = await _ainvoke_with_progress(
        llm, prompt, label=f"ManimFix scene {scene_num}", timeout=timeout,
    )
    _ensure_complete(response)
    fixed_body = _extract_code(response.content)

    # Basic sanity
    if "self.begin_scene()" not in fixed_body or "self.end_scene(" not in fixed_body:
        raise ValueError("Fixed scene is missing begin_scene/end_scene.")
    for bad in ("MathTex", "Tex(", "BulletedList", "SingleStringMathTex"):
        if bad in fixed_body:
            raise ValueError(f"Fixed scene still contains forbidden: {bad}")

    # Splice back into the full file
    lines = full_code.splitlines()
    fixed_lines = fixed_body.splitlines()
    # Ensure indentation matches (scene blocks are 8-space indented in construct)
    if lines[start_idx].startswith("        "):
        # Re-indent fixed lines to 8 spaces
        import textwrap
        dedented = textwrap.dedent(fixed_body)
        fixed_lines = [
            ("        " + ln) if ln.strip() else ""
            for ln in dedented.splitlines()
        ]
    result = "\n".join(lines[:start_idx] + fixed_lines + lines[end_idx:])

    if "class GeneratedScene" not in result:
        raise ValueError("Spliced code lost GeneratedScene class.")

    _validate_python_syntax(result, "Manim single-scene fixer")
    print(f"[ManimFix] Scene {scene_num} fixed successfully.")
    return result


async def _fix_whole_file(
    llm, broken_code: str, error_trimmed: str, timeout: float,
) -> str:
    """Fallback: fix the entire file when the failing scene can't be identified."""

    # Identify specific error type for better guidance
    error_type = "unknown"
    if "TypeError: Animation only works on Mobjects" in error_trimmed:
        error_type = "undefined_variable"
    elif "NameError" in error_trimmed or "not defined" in error_trimmed:
        error_type = "undefined_variable"
    elif "AttributeError" in error_trimmed:
        error_type = "wrong_method"
    elif "SyntaxError" in error_trimmed:
        error_type = "syntax_error"

    # Build error-specific guidance
    specific_guidance = ""
    if error_type == "undefined_variable":
        specific_guidance = """
CRITICAL ERROR DETECTED: Variable used before definition!

The error shows you're trying to animate a variable that doesn't exist yet.
Common causes:
1. Variable defined AFTER the self.play() call
2. Variable defined inside a loop but used outside
3. Typo in variable name

FIX PATTERN:
WRONG:
    self.play(FadeIn(point_on_x), FadeIn(vert_new))
    point_on_x = axes.c2p(x_new, 0)
    vert_new = Line(point_on_x, point_pred, color=YELLOW)

RIGHT:
    point_on_x = axes.c2p(x_new, 0)
    vert_new = Line(point_on_x, point_pred, color=YELLOW)
    self.play(FadeIn(point_on_x), FadeIn(vert_new))

CHECK EVERY self.play() call - all variables must be created BEFORE the call.
"""

    prompt = f"""\
You are a Manim Community Edition 0.21.x debugging expert.

The following Manim code FAILED to render. Fix it.

ERROR TYPE: {error_type}
{specific_guidance}

CRITICAL RULES:
- Define exactly one renderable class: class GeneratedScene(Scene).
- Do NOT define TimedScene or any other Scene subclass. The runner adds timing helpers.
- Every scene block must start with self.begin_scene() and end with self.end_scene(<duration_seconds>) (no self.clear() needed).
- NEVER use MathTex, Tex, SingleStringMathTex, or BulletedList.
- NEVER use any feature requiring a LaTeX compiler.
- Use Text("...") for ALL displayed text including math.
- For axes/graphs use ax.plot() NOT ax.get_graph() (deprecated).
- Do NOT use include_numbers=True on NumberLine or Axes.
- Only import from manim: "from manim import *"
- Make sure all variables are defined before use.
- Make sure all Mobjects are created correctly.
- ONLY use classes that really exist in Manim Community 0.21 (Axes, Text, Line,
  Dot, Arrow, Rectangle, VGroup, Write, Create, FadeIn, FadeOut, Transform,
  ReplacementTransform, Indicate, SurroundingRectangle ...). Do NOT invent
  classes such as SineFunction, Highlight, ShowResult, ShadeAreaBelowCurve.
- self.play(...) and self.wait(...) return None: NEVER chain them
  (write self.play(...) then self.wait(1) on separate lines).
- self.play() accepts only Animation objects, e.g. self.play(Write(Text("hi"))).
- Axes uses x_range=[0, 3.14, 1] and y_range=[-1, 1, 0.5] (NOT xrange/yrange).
- Mobjects have no set_height()/set_width(); use stretch_to_fit_height()/
  stretch_to_fit_width(). For bars anchored at the bottom, pass about_edge=DOWN.
- ArrowTip is abstract; use Arrow(...) or ArrowTriangleFilledTip(...).
  Start at a small nonzero size (for example 0.001), not exactly zero, before
  animating a stretch; Manim cannot rescale an object whose current size is zero.
- Use PI (not pi). Shaded area: ax.get_area(graph, x_range=(0, PI), color=BLUE, opacity=0.4).
- Any Hindi/Devanagari text MUST be created as Text("...", font="Nirmala UI").
- Unicode characters (✓, ✗, →) may cause issues. Use ASCII or Text objects.

BROKEN CODE:
```python
{broken_code}
```

ERROR:
{error_trimmed}

DEBUGGING CHECKLIST:
1. Find the line number in the error
2. Look at what variables are used in that line
3. Check if those variables are defined BEFORE that line
4. Move variable definitions BEFORE they're used
5. Ensure ALL self.play() arguments are valid Mobjects

Return ONLY the fixed Python code. No explanations, no markdown fences.
"""

    print("[ManimFix] Asking Nemotron to fix whole file…")
    response = await _ainvoke_with_progress(
        llm, prompt, label="ManimFix", timeout=timeout
    )
    _ensure_complete(response)
    code = _extract_code(response.content)

    # Validate basics
    if "class GeneratedScene" not in code:
        raise ValueError("Fixed code is missing 'class GeneratedScene'.")
    _validate_python_syntax(code, "Manim fixer")

    # Safety checks
    forbidden = ["MathTex", "Tex(", "BulletedList", "SingleStringMathTex"]
    for f in forbidden:
        if f in code:
            raise ValueError(f"Fixed code still contains forbidden: {f}")

    print("[ManimFix] Code fixed successfully.")
    return code


# ============================================================
# GENERATE VOICEOVER  (Kokoro MCP)
# ============================================================

async def generate_voiceover(
    narration_text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    language: str = "en-us",
) -> str:
    """Send narration text to Kokoro MCP and return the audio file path."""
    print("[Kokoro MCP] Generating voiceover…")

    result = await call_mcp_tool(
        "kokoro",
        "generate_speech",
        {
            "text": narration_text,
            "voice": voice,
            "speed": speed,
            "language": language,
        },
        timeout=120,
    )

    # result is a dict like {"success": True, "audio_path": "...", ...}
    if isinstance(result, dict):
        audio_path = result.get("audio_path", "")
        if not audio_path:
            raise RuntimeError(f"Kokoro returned no audio_path: {result}")
    else:
        raise RuntimeError(f"Unexpected Kokoro response: {result}")

    if not Path(audio_path).exists():
        raise RuntimeError(f"Kokoro reported audio at {audio_path} but file does not exist.")

    duration = result.get("duration_seconds", "?")
    print(f"[Kokoro MCP] Voiceover generated: {audio_path} ({duration}s)")
    return audio_path


# Speech speed (Kokoro range 0.5-2.0). <1.0 = slower. Override with TTS_SPEED in .env
TTS_SPEED = float(os.getenv("TTS_SPEED", "0.9"))
# Silence added after each scene's narration so the viewer can breathe
SCENE_PAD_SECONDS = float(os.getenv("SCENE_PAD_SECONDS", "0.8"))


async def generate_scene_voiceover(
    storyboard: list,
    voice: str,
    language: str,
    speed: float | None = None,
) -> tuple[str, list]:
    """Synthesize narration PER SCENE, pad each to a fixed length, and join them.

    Returns (combined_wav_path, storyboard) where every scene's
    `duration_seconds` is overwritten with the real audio length, so the
    Manim video can be built to match the voice exactly.
    """
    import math
    import wave

    speed = speed or TTS_SPEED
    parts: list[tuple[bytes, int]] = []  # (frames, scene_target_seconds*2)
    params = None

    for i, scene in enumerate(storyboard, 1):
        text = (scene.get("narration") or "").strip()
        if not text:  # silent scene: keep its planned length
            parts.append((b"", float(scene.get("duration_seconds", 3))))
            continue
        print(f"[Kokoro] Scene {i}/{len(storyboard)}…")
        path = await generate_voiceover(text, voice=voice, speed=speed, language=language)
        with wave.open(path, "rb") as w:
            if params is None:
                params = w.getparams()
            rate = w.getframerate()
            frames = w.readframes(w.getnframes())
            dur = w.getnframes() / rate

        target = math.ceil((dur + SCENE_PAD_SECONDS) * 2) / 2  # round up to 0.5s
        scene["duration_seconds"] = target
        parts.append((frames, target))
        print(f"[Kokoro] Scene {i}: speech {dur:.1f}s -> scene length {target:.1f}s")

    if not parts or params is None:
        raise RuntimeError("No narration text found in storyboard.")

    bytes_per_sec = params.framerate * params.sampwidth * params.nchannels
    out_path = OUTPUT_DIR / "voiceover.wav"
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(params.nchannels)
        out.setsampwidth(params.sampwidth)
        out.setframerate(params.framerate)
        for frames, target in parts:
            silence = int(target * bytes_per_sec) - len(frames)
            silence -= silence % (params.sampwidth * params.nchannels)
            out.writeframes(frames + b"\x00" * max(silence, 0))

    total = sum(t for _, t in parts)
    print(f"[Kokoro] Combined voiceover: {out_path} (~{total:.0f}s)")
    return str(out_path), storyboard


# ============================================================
# MERGE AUDIO + VIDEO  (FFmpeg MCP)
# ============================================================

async def merge_audio_video(
    video_path: str,
    audio_path: str,
    output_name: str = "final_video.mp4",
) -> str:
    """Merge visual video + voiceover audio into final video via FFmpeg MCP.

    IMPORTANT: Audio and video MUST have matching durations before calling this.
    The audio alignment system ensures this.
    """
    print("[FFmpeg MCP] Merging audio + video…")

    output_path = str(OUTPUT_DIR / output_name)

    # Do NOT use shortest=True - both inputs should already match exactly
    result = await call_mcp_tool(
        "ffmpeg",
        "merge_audio_video",
        {
            "video_path": video_path,
            "audio_path": audio_path,
            "output": output_path,
            "shortest": False,  # Changed: durations already match
            "overwrite": True,
        },
        timeout=120,
    )

    # result is dict like {"success": True, "output_path": "...", "size_bytes": ...}
    if isinstance(result, dict):
        final_path = result.get("output_path", output_path)
    else:
        final_path = output_path

    if not Path(final_path).exists():
        raise RuntimeError(f"FFmpeg merge reported output at {final_path} but file does not exist.")

    size_bytes = Path(final_path).stat().st_size
    print(f"[FFmpeg MCP] Final video: {final_path} ({size_bytes:,} bytes)")
    return final_path


# ============================================================
# QA VALIDATION (LEGACY - keeping for compatibility)
# ============================================================

async def qa_validate(
    video_path: str,
    audio_path: str,
    final_video_path: str,
) -> dict:
    """Run basic QA checks on the pipeline outputs.

    NOTE: This is the legacy QA function. The new comprehensive QA system
    is in qa_validator.py and should be used instead.
    """
    print("[QA] Running basic validation…")

    checks = {
        "video_exists": Path(video_path).exists(),
        "audio_exists": Path(audio_path).exists(),
        "final_video_exists": Path(final_video_path).exists(),
        "video_size_bytes": 0,
        "audio_size_bytes": 0,
        "final_video_size_bytes": 0,
    }

    if checks["video_exists"]:
        checks["video_size_bytes"] = Path(video_path).stat().st_size
    if checks["audio_exists"]:
        checks["audio_size_bytes"] = Path(audio_path).stat().st_size
    if checks["final_video_exists"]:
        checks["final_video_size_bytes"] = Path(final_video_path).stat().st_size

    # Try to get media info from FFmpeg for the final video
    try:
        info = await call_mcp_tool(
            "ffmpeg",
            "get_media_info",
            {"input_path": final_video_path},
            timeout=30,
        )
        if isinstance(info, dict):
            checks["media_info"] = info
            fmt = info.get("format", {})
            checks["final_duration_seconds"] = fmt.get("duration", "unknown")
            checks["final_format"] = fmt.get("format_name", "unknown")
            streams = info.get("streams", [])
            checks["has_video_stream"] = any(
                s.get("codec_type") == "video" for s in streams
            )
            checks["has_audio_stream"] = any(
                s.get("codec_type") == "audio" for s in streams
            )
    except Exception as e:
        checks["media_info_error"] = str(e)

    # Overall pass/fail
    checks["passed"] = (
        checks["final_video_exists"]
        and checks["final_video_size_bytes"] > 0
        and checks.get("has_video_stream", False)
        and checks.get("has_audio_stream", False)
    )

    status = "PASSED" if checks["passed"] else "FAILED"
    print(f"[QA] Basic validation {status}")
    if checks.get("final_duration_seconds"):
        print(f"[QA] Final duration: {checks['final_duration_seconds']}s")
    print(f"[QA] Final size: {checks['final_video_size_bytes']:,} bytes")

    return checks
