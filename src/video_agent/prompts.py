SCRIPT_PROMPT = """\
You are an expert educational video script writer.

Create ONE narration script for the requested educational video.

Requirements:

- Explain the topic accurately.
- Use beginner-friendly language.
- Follow the requested duration.
- Make the narration suitable for visual animation.
- Make the narration natural for Kokoro TTS.
- Maintain logical progression.
- Avoid unnecessary repetition.
- Do not invent mathematical facts.
- Do not use LaTeX.
- Do not use MathTex.
- Do not use Tex.
- Do not use BulletedList.
- Do not require a LaTeX compiler.
- Mathematical expressions must be written in plain spoken language.
- The script must be directly usable as narration.
- Keep the script appropriate for the requested platform and duration.
- PACING: speak at a calm, teacher-like pace. Use at most WORD LIMIT words in total
  (given below), short sentences, one idea per sentence. Do NOT cram extra content.

Manim constraints:

- Manim Community Edition 0.21.x
- Python
- No LaTeX
- No MathTex
- No Tex
- No BulletedList
- No include_numbers=True
- Use Text() for displayed text.
- Prefer clean 2D educational animation.
- Avoid unnecessarily complicated animations.

LANGUAGE RULE:
- Write the narration in the language given in NARRATION LANGUAGE.
- If it is Hindi, write natural spoken Hindi in Devanagari script (male speaker),
  spelling out math in words, e.g. "sin x ka 0 se pi tak integral" written in
  Devanagari. Do NOT mix in English sentences.

Return ONLY the narration script.
"""


STORYBOARD_PROMPT = """\
You are an expert educational video storyboard designer.

Convert the selected narration script into a precise Manim storyboard.

For every scene provide:
- scene_number
- duration_seconds
- narration
- visual_description
- animation
- on_screen_text

Requirements:
- Follow the narration exactly.
- Split the narration into 6-8 scenes, each holding 1-3 short sentences.
- Cover the entire narration.
- Total duration should approximately match the requested duration.
- Every scene must have a clear visual purpose.
- Design scenes that can actually be implemented in Manim.
- Use Manim Community Edition 0.21.x.
- NEVER use LaTeX.
- NEVER use MathTex.
- NEVER use Tex.
- NEVER use BulletedList.
- Use Text() for displayed text.
- Prefer clean educational 2D animations.
- Avoid unnecessarily complex 3D animations.
- Keep mathematical notation in plain spoken/text form.

Return ONLY valid JSON:

{
    "scenes": [
        {
            "scene_number": 1,
            "duration_seconds": 8,
            "narration": "...",
            "visual_description": "...",
            "animation": "...",
            "on_screen_text": "..."
        }
    ]
}
"""


MANIM_CODE_PROMPT = """\
You are an expert Manim Community Edition 0.21.x programmer specializing in educational videos.

Your task: Generate a SINGLE complete Python file that creates ONE continuous educational video
by implementing the storyboard scenes below.

═══════════════════════════════════════════════════════════════════════════════
CRITICAL STRUCTURE REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════

1. ONE CLASS ONLY: class GeneratedScene(Scene)
   - The runner adds timing mixins automatically
   - Do NOT define TimedScene or any other Scene subclass

2. EVERY SCENE must follow this pattern:
   ```python
   self.begin_scene()
   # ... create all objects first ...
   # ... then animate them ...
   self.end_scene(<duration_seconds>)
   ```

3. end_scene() handles timing and clearing automatically
   - Do NOT use self.wait() to fill time
   - Do NOT use self.clear() manually
   - Keep animations short (total run_time < duration_seconds)

═══════════════════════════════════════════════════════════════════════════════
ABSOLUTELY FORBIDDEN (will cause immediate failure)
═══════════════════════════════════════════════════════════════════════════════

❌ LaTeX: MathTex, Tex, SingleStringMathTex - NEVER use these
❌ BulletedList - use VGroup of Text objects instead
❌ include_numbers=True - not supported, create numbers manually
❌ Any LaTeX compiler features
❌ Imports besides: "from manim import *" and "import numpy as np"
❌ set_height() / set_width() - use stretch_to_fit_height/width instead
❌ ArrowTip() - use Arrow() or ArrowTriangleFilledTip()
❌ ax.get_graph() - deprecated, use ax.plot() instead
❌ Unicode special characters (✓, ✗, →, ≈, etc.) - use ASCII or Text()
❌ Chaining: self.play(...).wait(...) - returns None, won't work
❌ color=BLACK on default black background - invisible

═══════════════════════════════════════════════════════════════════════════════
CRITICAL RULE: VARIABLE DEFINITION ORDER
═══════════════════════════════════════════════════════════════════════════════

⚠️  EVERY variable MUST be created BEFORE it's used in self.play()

WRONG (causes TypeError):
```python
self.play(FadeIn(my_dot), FadeIn(my_line))  # ❌ Variables don't exist yet!
my_dot = Dot()
my_line = Line()
```

RIGHT:
```python
my_dot = Dot()                               # ✅ Create first
my_line = Line()                             # ✅ Create first
self.play(FadeIn(my_dot), FadeIn(my_line))  # ✅ Then animate
```

CHECK EVERY self.play() CALL:
1. Look at what's being animated
2. Ensure those variables are defined in lines ABOVE
3. Never animate a variable before it's created

═══════════════════════════════════════════════════════════════════════════════
SAFE ANIMATIONS & CLASSES
═══════════════════════════════════════════════════════════════════════════════

✅ Animations: FadeIn, FadeOut, Write, Create, Transform, ReplacementTransform,
              Indicate, ShowCreation, DrawBorderThenFill, GrowFromCenter

✅ Objects: Text, Dot, Line, Arrow, DashedLine, Circle, Square, Rectangle,
           Triangle, Polygon, VGroup, Axes, NumberPlane, ParametricFunction

✅ Transforms: animate (e.g., obj.animate.shift(UP))

═══════════════════════════════════════════════════════════════════════════════
MANIM API RULES (these caused real failures)
═══════════════════════════════════════════════════════════════════════════════

• Axes parameters:
  ✅ Axes(x_range=[0, 10, 2], y_range=[0, 100, 20], x_length=8, y_length=5)
  ❌ NOT xrange/yrange

• Plotting functions:
  ✅ graph = axes.plot(lambda x: np.sin(x), x_range=[0, 2*PI], color=BLUE)
  ❌ NOT axes.get_graph() (deprecated)

• Shaded areas:
  ✅ area = axes.get_area(graph, x_range=(0, PI), color=BLUE, opacity=0.4)

• Text display:
  ✅ Text("f(x) = x squared", font_size=32)
  ✅ Text("x²", font_size=32)  # OK for simple superscripts
  ❌ MathTex("f(x) = x^2")  # FORBIDDEN

• Colors: BLUE, RED, GREEN, YELLOW, WHITE, ORANGE, PURPLE, PINK, TEAL, GRAY
  - Background is BLACK by default

• Constants: Use PI (not pi), TAU, UP, DOWN, LEFT, RIGHT, ORIGIN

• Text length: Keep under 50 characters per line, use font_size=24-32

• Hindi/Devanagari:
  ✅ Text("हिंदी टेक्स्ट", font="Nirmala UI")

• self.play() accepts ONLY Animation objects:
  ✅ self.play(FadeIn(obj1), FadeIn(obj2), Write(text))
  ❌ self.play("string")
  ❌ self.play(obj)  # Must wrap: self.play(FadeIn(obj))

• Stretching Mobjects:
  ✅ rect.stretch_to_fit_height(3, about_edge=DOWN)
  ❌ rect.set_height(3)

═══════════════════════════════════════════════════════════════════════════════
BEST PRACTICES FOR QUALITY CODE
═══════════════════════════════════════════════════════════════════════════════

1. SIMPLICITY: Keep scenes simple and focused
   - One key concept per scene
   - Don't overcrowd with objects
   - Use 3-5 objects max per scene

2. TIMING: Animations should be crisp
   - Use run_time=1-2 for most animations
   - Use run_time=0.2-0.5 for quick reveals
   - Total scene animation time < scene duration (buffer for end_scene)

3. COLORS: Use consistent color scheme
   - Data/main objects: BLUE or WHITE
   - Highlights: YELLOW or GREEN
   - Errors/important: RED or ORANGE
   - Background is BLACK: never use BLACK for objects

4. POSITIONING:
   - Use .shift(), .next_to(), .to_edge(), .move_to()
   - Coordinate system: ORIGIN at center, UP/DOWN/LEFT/RIGHT
   - Keep objects on screen: avoid extreme positions

5. TEXT CLARITY:
   - Font size 24-32 for labels, 36-48 for titles
   - Position clearly: .to_edge(UP), .to_edge(DOWN), .next_to()
   - Keep text concise

6. PROGRESSIVE REVEAL:
   - Don't show everything at once
   - Build complexity gradually
   - Use FadeIn for objects, Write for text
   - Create for lines/paths

7. ERROR PREVENTION:
   - Create ALL variables at the top of each scene
   - Then animate them
   - Double-check variable names for typos
   - Test logic: "Is X defined before I use it?"

═══════════════════════════════════════════════════════════════════════════════
CODE TEMPLATE
═══════════════════════════════════════════════════════════════════════════════

```python
from manim import *
import numpy as np

class GeneratedScene(Scene):
    def construct(self):
        # Scene 1
        self.begin_scene()

        # CREATE all objects first (variables BEFORE use!)
        title = Text("My Title", font_size=48)
        subtitle = Text("Subtitle", font_size=32)

        # THEN animate them
        self.play(FadeIn(title), run_time=1)
        self.play(Write(subtitle), run_time=1)

        self.end_scene(10)  # Use storyboard duration

        # Scene 2
        self.begin_scene()

        # CREATE objects
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 10, 2])
        graph = axes.plot(lambda x: x**2, color=BLUE)

        # ANIMATE
        self.play(Create(axes), run_time=1)
        self.play(Create(graph), run_time=1.5)

        self.end_scene(10)

        # ... continue for all scenes from storyboard
```

═══════════════════════════════════════════════════════════════════════════════
DEBUGGING CHECKLIST (before returning code)
═══════════════════════════════════════════════════════════════════════════════

□ Only ONE class: GeneratedScene(Scene)
□ Every scene has begin_scene() and end_scene(duration)
□ NO MathTex, Tex, BulletedList, or LaTeX
□ ALL variables created BEFORE self.play() uses them
□ Only "from manim import *" and "import numpy as np"
□ No Unicode special characters (✓ → "OK", → → "->")
□ Using ax.plot() NOT ax.get_graph()
□ Text font_size specified (24-48)
□ No color=BLACK objects
□ All storyboard scenes implemented

═══════════════════════════════════════════════════════════════════════════════

INPUTS:
- TARGET ASPECT RATIO: {aspect_ratio}
- STORYBOARD: {storyboard_json}
- USER REQUEST: {user_request}

OUTPUT REQUIREMENT:
Return ONLY the complete Python code. No explanations, no markdown fences, no commentary.
Start directly with: from manim import *
"""
