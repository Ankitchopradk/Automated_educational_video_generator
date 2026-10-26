"""Pre-flight linter for LLM-generated Manim code.

Run with the *Manim venv* Python (so it can see the real manim namespace):

    manim-venv\\Scripts\\python.exe manim_lint.py < scene.py

Reads code from stdin, prints a JSON list of human-readable issues to stdout.
An empty list means the code passed.
"""
import ast
import builtins
import json
import sys


def _load_known_names() -> set:
    """Names exported by `from manim import *`, cached in manim_names.json.

    Importing manim is slow (10-60 s on Windows), so do it once and cache.
    """
    import pathlib

    cache = pathlib.Path(__file__).with_name("manim_names.json")
    if cache.exists():
        try:
            return set(json.loads(cache.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    ns: dict = {}
    exec("from manim import *", ns)
    names = sorted(n for n in ns if n != "__builtins__")
    cache.write_text(json.dumps(names), encoding="utf-8")
    return set(names)


def main() -> None:
    if "--build-cache" in sys.argv:
        print(len(_load_known_names()))
        return

    code = sys.stdin.read()
    issues: list[str] = []

    # 1. Syntax --------------------------------------------------------
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(json.dumps([f"SyntaxError line {e.lineno}: {e.msg}"]))
        return

    # 2. Real names exported by `from manim import *` (cached) --------
    known = _load_known_names() | set(dir(builtins))

    # 3. Names defined inside the code itself --------------------------
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
            if isinstance(node, ast.FunctionDef):
                for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
                    defined.add(a.arg)
                if node.args.vararg:           # *args
                    defined.add(node.args.vararg.arg)
                if node.args.kwarg:            # **kwargs
                    defined.add(node.args.kwarg.arg)
        elif isinstance(node, ast.Lambda):
            for a in node.args.args:
                defined.add(a.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            defined.add(node.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                defined.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                defined.add(a.asname or a.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.comprehension):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    defined.add(t.id)

    seen: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ArrowTip"
        ):
            issues.append(
                f"line {node.lineno}: ArrowTip is abstract; use a concrete tip "
                "such as ArrowTriangleFilledTip or create an Arrow."
            )

        if isinstance(node, ast.Attribute) and node.attr in {"set_height", "set_width"}:
            dimension = node.attr.removeprefix("set_")
            issues.append(
                f"line {node.lineno}: Manim Mobjects do not provide '{node.attr}'; "
                f"use stretch_to_fit_{dimension}(...)."
            )

        # 3a. Undefined / hallucinated names (SineFunction, ShowResult, x, pi ...)
        if isinstance(node, ast.Name) and node.id == "BLACK":
            issues.append(
                f"line {node.lineno}: BLACK is invisible on the black background; use WHITE/GRAY/YELLOW."
            )
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in known and node.id not in defined and node.id not in seen:
                seen.add(node.id)
                hint = ""
                if node.id == "pi":
                    hint = " (use PI)"
                issues.append(
                    f"line {node.lineno}: '{node.id}' does not exist in Manim "
                    f"and is not defined in the file{hint}."
                )

        # 3b. self.play(...).wait(...)  -> play() returns None
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr in ("play", "wait", "add", "remove")
            and node.attr in ("wait", "play")
        ):
            issues.append(
                f"line {node.lineno}: chained call '.{node.value.func.attr}(...).{node.attr}(...)'. "
                f"self.play()/self.wait() return None - put them on separate lines."
            )

        if isinstance(node, ast.Call):
            # 3c. Wrong Axes kwargs
            for kw in node.keywords:
                if kw.arg == "opacity" and isinstance(node.func, ast.Attribute) and node.func.attr in ("plot", "get_graph"):
                    issues.append(
                        f"line {node.lineno}: ax.plot() has no 'opacity' kwarg; use stroke_opacity=0.5."
                    )
                if kw.arg in ("xrange", "yrange"):
                    issues.append(
                        f"line {node.lineno}: invalid kwarg '{kw.arg}'; "
                        f"use x_range=[min, max, step] / y_range=[min, max, step]."
                    )
                if kw.arg == "include_numbers" and isinstance(kw.value, ast.Constant) and kw.value.value:
                    issues.append(f"line {node.lineno}: include_numbers=True needs LaTeX; remove it.")

            # 3d. Text(...) / self.play(<str>) misuse: play() args must be Animations
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "play"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        issues.append(
                            f"line {node.lineno}: self.play() got a raw string; "
                            f"wrap it: self.play(Write(Text(\"...\")))."
                        )
                for kw in node.keywords:
                    if kw.arg not in ("run_time", "rate_func", "lag_ratio", "subcaption",
                                      "subcaption_duration", "subcaption_offset"):
                        issues.append(
                            f"line {node.lineno}: self.play() has unexpected kwarg '{kw.arg}'."
                        )

            # 3e. Devanagari text without a font
            if isinstance(node.func, ast.Name) and node.func.id == "Text":
                has_font = any(kw.arg == "font" for kw in node.keywords)
                for arg in node.args[:1]:
                    if (
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and any("\u0900" <= ch <= "\u097f" for ch in arg.value)
                        and not has_font
                    ):
                        issues.append(
                            f"line {node.lineno}: Hindi text needs font=\"Nirmala UI\" "
                            f"(Text(\"...\", font=\"Nirmala UI\"))."
                        )

    # 4. Structure -----------------------------------------------------
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if not any(c.name == "GeneratedScene" for c in classes):
        issues.append("missing class GeneratedScene(Scene).")

    renderable_classes = [
        c for c in classes
        if any(isinstance(base, ast.Name) and base.id == "Scene" for base in c.bases)
    ]
    if len(renderable_classes) != 1 or renderable_classes[0].name != "GeneratedScene":
        issues.append(
            "define exactly one top-level Scene subclass named GeneratedScene; "
            "helper classes must not inherit from Scene."
        )

    if "end_scene(" not in code.split("class GeneratedScene")[-1]:
        issues.append(
            "scenes must call self.begin_scene() at the start and "
            "self.end_scene(<duration_seconds>) at the end (keeps video in sync with voice)."
        )

    # 5. Object introduction: FadeOut/animate used before FadeIn/Create/Write/add
    issues.extend(_check_object_introduction(tree))

    # 6. Keep-inside-screen: stacked text that drifts off the bottom
    issues.extend(_check_stacked_text_bounds(tree))

    print(json.dumps(issues[:15]))


def _check_object_introduction(tree: ast.Module) -> list:
    """Warn when a Mobject is FadedOut or animated before being introduced.

    Tracks variables that were added to the scene via FadeIn, Create, Write,
    self.add, or DrawBorderThenFill. Warns if FadeOut or .animate references
    a variable that was never introduced.

    This is a *best-effort* heuristic — it walks construct() linearly and
    tracks simple names only (no attribute chains or subscripts).
    """
    issues = []
    introduce_funcs = {"FadeIn", "Create", "Write", "DrawBorderThenFill", "GrowFromCenter", "ShowCreation"}

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "construct"):
            continue

        introduced: set = set()
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Call):
                continue
            func = stmt.func

            # self.add(x, y, ...) — introduces objects
            if (isinstance(func, ast.Attribute) and func.attr == "add"
                    and isinstance(func.value, ast.Name) and func.value.id == "self"):
                for arg in stmt.args:
                    if isinstance(arg, ast.Name):
                        introduced.add(arg.id)
                continue

            # self.play(FadeIn(x), Create(y), FadeOut(z), ...)
            if (isinstance(func, ast.Attribute) and func.attr == "play"
                    and isinstance(func.value, ast.Name) and func.value.id == "self"):
                for arg in stmt.args:
                    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                        anim_name = arg.func.id
                        # Introduction animations
                        if anim_name in introduce_funcs:
                            for sub in arg.args:
                                if isinstance(sub, ast.Name):
                                    introduced.add(sub.id)
                        # Removal / mutation — check introduced
                        elif anim_name == "FadeOut":
                            for sub in arg.args:
                                if isinstance(sub, ast.Name) and sub.id not in introduced:
                                    issues.append(
                                        f"line {stmt.lineno}: FadeOut({sub.id}) but "
                                        f"'{sub.id}' was never added to the scene "
                                        "(use FadeIn/Create/Write/self.add first)."
                                    )

    return issues


def _check_stacked_text_bounds(tree: ast.Module) -> list:
    """Warn when stacked text is likely to overflow the bottom of the frame.

    Manim's visible frame runs from y = -4 to y = +4. If code stacks Text
    objects using .shift(DOWN * N) or .next_to(..., DOWN) in a loop or with
    increasing offsets, warn when the cumulative offset exceeds ~3 units
    below the origin.
    """
    issues = []
    # Heuristic: look for patterns like  .shift(DOWN * N)  where N >= 3
    import re as _re
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "shift":
                for arg in node.args:
                    # DOWN * 3.5  or  3.5 * DOWN
                    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mult):
                        left, right = arg.left, arg.right
                        name_node = const_node = None
                        if isinstance(left, ast.Name) and isinstance(right, ast.Constant):
                            name_node, const_node = left, right
                        elif isinstance(right, ast.Name) and isinstance(left, ast.Constant):
                            name_node, const_node = right, left
                        if (name_node and const_node
                                and name_node.id == "DOWN"
                                and isinstance(const_node.value, (int, float))
                                and const_node.value >= 3):
                            issues.append(
                                f"line {node.lineno}: .shift(DOWN * {const_node.value}) "
                                f"may push objects below the visible frame (y < -4). "
                                f"Consider smaller offsets or .to_edge(DOWN)."
                            )
    return issues


if __name__ == "__main__":
    main()
