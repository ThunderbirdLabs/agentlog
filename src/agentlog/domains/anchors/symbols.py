"""Symbol anchors: the functions and classes a change actually touched.

Given changed line ranges, find the enclosing definition. Python is parsed with
`ast`, so the qualified name (`InspectionExportService.render`) is exact.
TypeScript and JavaScript use an indentation-aware scan — good enough to name a
symbol, and cheap enough to run offline on every segment.
"""

from __future__ import annotations

import ast
import re

from agentlog.core.logging import get_logger

log = get_logger("anchors.symbols")

_TS_DECL = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?"
    r"(?:"
    r"(?P<kind_class>class)\s+(?P<class_name>[A-Za-z_$][\w$]*)"
    r"|(?:async\s+)?(?P<kind_fn>function)\s*\*?\s*(?P<fn_name>[A-Za-z_$][\w$]*)"
    r"|(?:const|let|var)\s+(?P<const_name>[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*"
    r"(?:async\s*)?(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|[A-Za-z_$][\w$]*\s*=>)"
    r"|(?P<method_name>[A-Za-z_$][\w$]*)\s*\([^)]*\)\s*(?::[^{]+)?\{"
    r")"
)


def _python_symbols(source: str, ranges: list[tuple[int, int]]) -> list[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        log.debug("could not parse python source: %s", exc)
        return []

    spans: list[tuple[int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}.{child.name}" if prefix else child.name
                end = getattr(child, "end_lineno", None) or child.lineno
                spans.append((child.lineno, end, name))
                walk(child, name)
            else:
                walk(child, prefix)

    walk(tree, "")

    found: dict[str, None] = {}
    for start, stop in ranges:
        # The innermost span containing the change is the one worth naming, so
        # prefer the shortest match.
        matches = [s for s in spans if s[0] <= stop and s[1] >= start]
        if not matches:
            continue
        matches.sort(key=lambda s: s[1] - s[0])
        found[matches[0][2]] = None
    return list(found)


def _ts_symbols(source: str, ranges: list[tuple[int, int]]) -> list[str]:
    lines = source.splitlines()
    # (line number, indent width, name) for every declaration we recognise.
    decls: list[tuple[int, int, str]] = []
    for number, line in enumerate(lines, start=1):
        match = _TS_DECL.match(line)
        if not match:
            continue
        name = (
            match.group("class_name")
            or match.group("fn_name")
            or match.group("const_name")
            or match.group("method_name")
        )
        if not name or name in ("if", "for", "while", "switch", "catch", "return"):
            continue
        indent = len(match.group("indent").expandtabs(4))
        decls.append((number, indent, name))

    def enclosing(line_number: int) -> str | None:
        # Walk backwards to the nearest preceding declaration, then keep
        # walking to collect any less-indented declarations enclosing it.
        stack: list[str] = []
        current_indent: int | None = None
        for number, indent, name in reversed(decls):
            if number > line_number:
                continue
            if current_indent is None or indent < current_indent:
                stack.append(name)
                current_indent = indent
            if current_indent == 0:
                break
        if not stack:
            return None
        return ".".join(reversed(stack))

    found: dict[str, None] = {}
    for start, stop in ranges:
        for line_number in (start, stop):
            name = enclosing(line_number)
            if name:
                found[name] = None
    return list(found)


def extract(file_path: str, source: str, ranges: list[tuple[int, int]]) -> list[str]:
    if not ranges:
        return []
    if file_path.endswith(".py"):
        return _python_symbols(source, ranges)
    if file_path.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs")):
        return _ts_symbols(source, ranges)
    return []
