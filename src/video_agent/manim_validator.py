"""
Enhanced Manim code validation to catch common errors before rendering.

This module performs static analysis on generated Manim code to catch
errors that would cause rendering failures.
"""

import ast
import re
from typing import List, Tuple


def validate_variable_usage(code: str) -> List[str]:
    """
    Check that all variables used in self.play() calls are defined before use.

    This catches the common error where the LLM tries to animate a variable
    before it's been created.
    """
    errors = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    # Find the construct method
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GeneratedScene":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "construct":
                    # Analyze the construct method
                    errors.extend(_check_construct_method(item))

    return errors


def _check_construct_method(construct_node: ast.FunctionDef) -> List[str]:
    """Check the construct method for variable definition/usage errors."""
    errors = []
    defined_vars = set()

    for stmt in construct_node.body:
        # Track variable assignments
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    defined_vars.add(target.id)

        # Check self.play() calls
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if (isinstance(call.func, ast.Attribute) and
                isinstance(call.func.value, ast.Name) and
                call.func.value.id == "self" and
                call.func.attr == "play"):

                # Check all arguments to self.play()
                for arg in call.args:
                    undefined_vars = _find_undefined_variables(arg, defined_vars)
                    for var in undefined_vars:
                        errors.append(
                            f"Variable '{var}' used in self.play() before it's defined"
                        )

    return errors


def _find_undefined_variables(node: ast.AST, defined_vars: set) -> List[str]:
    """Find variables used in an AST node that aren't defined yet.

    Ignores Manim classes/constants (CamelCase / UPPER_CASE such as Create,
    FadeIn, PI, BLUE), builtins, np, and lambda/comprehension parameters, which
    are not local variables.
    """
    ignore = {
        "np", "len", "range", "min", "max", "abs", "sum", "int", "float",
        "str", "list", "tuple", "enumerate", "zip", "round", "True", "False", "None",
    }
    local_params: set = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Lambda):
            local_params.update(a.arg for a in child.args.args)
        elif isinstance(child, ast.comprehension):
            for t in ast.walk(child.target):
                if isinstance(t, ast.Name):
                    local_params.add(t.id)

    undefined = []

    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            name = child.id
            if (
                name not in defined_vars
                and name not in ignore
                and name not in local_params
                and name != "self"
                and not name[:1].isupper()
            ):
                undefined.append(name)

    return undefined


def check_unicode_issues(code: str) -> List[str]:
    """Check for Unicode characters that might cause encoding issues."""
    errors = []

    # Common problematic Unicode characters
    problematic_chars = {
        '✓': 'checkmark',
        '✗': 'cross mark',
        '→': 'arrow',
        '←': 'left arrow',
        '↑': 'up arrow',
        '↓': 'down arrow',
        '≈': 'approximately',
        '≠': 'not equal',
        '≤': 'less than or equal',
        '≥': 'greater than or equal',
    }

    for char, name in problematic_chars.items():
        if char in code:
            errors.append(
                f"Unicode character '{char}' ({name}) found. "
                f"This may cause encoding issues on Windows. "
                f"Use Text() with proper font or ASCII alternatives."
            )

    return errors


def check_forbidden_constructs(code: str) -> List[str]:
    """Check for forbidden Manim constructs."""
    errors = []

    forbidden = {
        'MathTex': 'Use Text() instead',
        'Tex(': 'Use Text() instead',
        'SingleStringMathTex': 'Use Text() instead',
        'BulletedList': 'Use VGroup of Text objects instead',
        'include_numbers=True': 'Not supported, create numbers manually',
        '.set_height(': 'Use .stretch_to_fit_height() instead',
        '.set_width(': 'Use .stretch_to_fit_width() instead',
    }

    for construct, suggestion in forbidden.items():
        if construct in code:
            errors.append(f"Forbidden: '{construct}'. {suggestion}")

    return errors


def check_play_calls(code: str) -> List[str]:
    """Check for common self.play() usage errors."""
    errors = []

    # Check for chained play().wait()
    if re.search(r'self\.play\([^)]+\)\.wait\(', code):
        errors.append(
            "Chained self.play().wait() detected. "
            "self.play() returns None. Put self.wait() on a separate line."
        )

    # Check for self.play with non-Animation objects (strings, etc.)
    # This is harder to detect statically, but we can look for obvious cases
    if re.search(r'self\.play\(\s*["\']', code):
        errors.append(
            "self.play() called with a string. "
            "Wrap strings in Text() and an animation like Write() or FadeIn()."
        )

    return errors


def validate_manim_code(code: str) -> Tuple[bool, List[str]]:
    """
    Run all validation checks on Manim code.

    Returns:
        (is_valid, list_of_errors)
    """
    all_errors = []

    # Run all checks
    all_errors.extend(validate_variable_usage(code))
    all_errors.extend(check_unicode_issues(code))
    all_errors.extend(check_forbidden_constructs(code))
    all_errors.extend(check_play_calls(code))

    is_valid = len(all_errors) == 0

    return is_valid, all_errors


def fix_common_errors(code: str) -> Tuple[str, List[str]]:
    """
    Attempt to automatically fix common errors in Manim code.

    Returns:
        (fixed_code, list_of_fixes_applied)
    """
    fixes_applied = []
    fixed_code = code

    # Fix: Replace Unicode checkmarks with text
    if '✓' in fixed_code:
        fixed_code = fixed_code.replace('✓', 'OK')
        fixes_applied.append("Replaced '✓' with 'OK'")

    if '✗' in fixed_code:
        fixed_code = fixed_code.replace('✗', 'X')
        fixes_applied.append("Replaced '✗' with 'X'")

    # Fix: Remove include_numbers=True
    if 'include_numbers=True' in fixed_code:
        fixed_code = re.sub(r',?\s*include_numbers=True', '', fixed_code)
        fixes_applied.append("Removed 'include_numbers=True'")

    return fixed_code, fixes_applied
