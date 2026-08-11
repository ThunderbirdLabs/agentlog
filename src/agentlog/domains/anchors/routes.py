"""Route anchors: FastAPI decorators and Next.js app-router file conventions.

A route is often the most durable name a piece of work has. Files get split and
renamed; `POST /api/v1/inspections/{id}/export` survives both.
"""

from __future__ import annotations

import ast
import re

from agentlog.core.logging import get_logger

log = get_logger("anchors.routes")

_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head", "options"})

_NEXT_ROUTE_METHODS = re.compile(
    r"export\s+(?:async\s+)?function\s+(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b"
)
_NEXT_ROUTE_CONST = re.compile(r"export\s+const\s+(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b")


def _normalize(path: str) -> str:
    path = path.strip()
    if not path:
        return "/"
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return path or "/"


def _fastapi_prefix(tree: ast.Module) -> str:
    """The `prefix=` of an `APIRouter(...)` constructed at module level."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "APIRouter":
            continue
        for keyword in node.keywords:
            if keyword.arg == "prefix" and isinstance(keyword.value, ast.Constant):
                value = keyword.value.value
                if isinstance(value, str):
                    prefix = value.strip()
                    if prefix and not prefix.startswith("/"):
                        prefix = "/" + prefix
                    return prefix.rstrip("/")
    return ""


def from_python(source: str) -> list[str]:
    """FastAPI-style routes declared in a Python module.

    Parsed with `ast`, not regex, so a route string that merely *appears* in
    the file — inside a test assertion, a docstring, a comment — cannot become
    an anchor. Only a real decorator on a real function counts.

    A file that does not parse yields nothing rather than falling back to a
    looser scan. Half-written files are common mid-session, and a wrong anchor
    is worse than a missing one: it sends a future agent's retrieval to the
    wrong code.

    The router prefix is taken from an `APIRouter(prefix=...)` in the same
    file. Prefixes composed at `include_router` time in another module are not
    followed — a partially-correct route is a worse anchor than a consistently
    shaped one.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        log.debug("could not parse python source for routes: %s", exc)
        return []

    prefix = _fastapi_prefix(tree)
    routes: dict[str, None] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not isinstance(func, ast.Attribute) or func.attr not in _METHODS:
                continue
            if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
                continue
            path = decorator.args[0].value
            if not isinstance(path, str):
                continue
            full = _normalize(prefix + _normalize(path))
            routes[f"{func.attr.upper()} {full}"] = None
    return list(routes)


def _next_url_path(file_path: str) -> str | None:
    """Turn an app-router file path into a URL path.

    `app/api/v1/inspections/[id]/export/route.ts`
      -> `/api/v1/inspections/{id}/export`

    Route groups (`(marketing)`) and private folders (`_lib`) contribute no URL
    segment, matching Next.js routing.
    """
    parts = file_path.split("/")
    if not parts:
        return None
    filename = parts[-1]
    stem = filename.rsplit(".", 1)[0]
    if stem not in ("route", "page"):
        return None

    try:
        app_index = max(i for i, part in enumerate(parts[:-1]) if part == "app")
    except ValueError:
        return None

    segments: list[str] = []
    for part in parts[app_index + 1 : -1]:
        if part.startswith("(") and part.endswith(")"):
            continue
        if part.startswith("_"):
            continue
        if part.startswith("[") and part.endswith("]"):
            inner = part[1:-1]
            inner = inner.removeprefix("...").removeprefix("[").removesuffix("]")
            segments.append("{" + inner + "}")
            continue
        segments.append(part)
    return "/" + "/".join(segments) if segments else "/"


def from_next(file_path: str, source: str) -> list[str]:
    url = _next_url_path(file_path)
    if url is None:
        return []
    stem = file_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if stem == "page":
        return [f"PAGE {url}"]

    methods: dict[str, None] = {}
    for pattern in (_NEXT_ROUTE_METHODS, _NEXT_ROUTE_CONST):
        for match in pattern.finditer(source):
            methods[match.group(1)] = None
    if not methods:
        return []
    return [f"{method} {url}" for method in methods]


def extract(file_path: str, source: str) -> list[str]:
    if file_path.endswith(".py"):
        return from_python(source)
    if file_path.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs")):
        return from_next(file_path, source)
    return []
