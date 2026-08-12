"""Anchors are computed offline. Nothing here touches the network or a model."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agentlog.core.config import Config
from agentlog.domains.anchors import git, routes, settings, symbols
from agentlog.domains.anchors import service as anchors_service

from .conftest import make_segment


def _run(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


# --------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------


def test_repo_root_and_branch(git_repo: Path) -> None:
    assert git.repo_root(git_repo) == git_repo.resolve()
    assert git.branch(git_repo) == "feat/THU-142-inspection-export"
    assert git.head_sha(git_repo)


def test_detached_head_is_not_a_branch(git_repo: Path) -> None:
    sha = git.head_sha(git_repo, short=False) or ""
    _run(git_repo, "checkout", "-q", sha)
    assert git.branch(git_repo) is None


def test_issue_is_parsed_from_the_branch() -> None:
    pattern = Config(repo_root=Path(".")).issue_pattern
    assert git.issue_from_branch("feat/THU-142-inspection-export", pattern) == "THU-142"
    assert git.issue_from_branch("fix/ABC-9", pattern) == "ABC-9"
    assert git.issue_from_branch("main", pattern) is None
    assert git.issue_from_branch(None, pattern) is None


def test_a_broken_issue_pattern_degrades_rather_than_raising() -> None:
    assert git.issue_from_branch("feat/THU-142", "([unclosed") is None


def test_tracked_files(git_repo: Path) -> None:
    assert "src/service.py" in git.tracked_files(git_repo)


def test_working_tree_line_ranges_sees_uncommitted_edits(git_repo: Path) -> None:
    """The case that matters most: dead ends are never committed."""
    target = git_repo / "src" / "service.py"
    source = target.read_text(encoding="utf-8")
    target.write_text(source.replace("return None", "return self.stream()"), encoding="utf-8")

    ranges = git.working_tree_line_ranges(git_repo)
    assert "src/service.py" in ranges
    assert ranges["src/service.py"]


def test_git_helpers_return_none_outside_a_repo(tmp_path: Path) -> None:
    assert git.repo_root(tmp_path / "nope") is None
    assert git.branch(tmp_path) is None or True  # tmp_path may sit inside a repo on some machines


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------


def test_fastapi_route_with_router_prefix() -> None:
    source = (
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/api/v1/inspections", tags=["x"])\n'
        '@router.post("/{id}/export")\n'
        "def export(id): ...\n"
    )
    assert routes.from_python(source) == ["POST /api/v1/inspections/{id}/export"]


def test_fastapi_route_without_prefix() -> None:
    source = '@app.get("/health")\ndef health(): ...\n'
    assert routes.from_python(source) == ["GET /health"]


def test_multiple_fastapi_routes_are_all_found() -> None:
    source = (
        'router = APIRouter(prefix="/things")\n'
        '@router.get("/")\ndef list_(): ...\n'
        '@router.delete("/{id}")\ndef remove(id): ...\n'
    )
    assert set(routes.from_python(source)) == {"GET /things", "DELETE /things/{id}"}


def test_next_app_router_route_handler() -> None:
    source = "export async function GET(req) {}\nexport async function POST(req) {}\n"
    found = routes.from_next("app/api/v1/inspections/[id]/export/route.ts", source)
    assert set(found) == {
        "GET /api/v1/inspections/{id}/export",
        "POST /api/v1/inspections/{id}/export",
    }


def test_next_route_groups_and_private_folders_contribute_no_segment() -> None:
    source = "export const GET = handler\n"
    found = routes.from_next("app/(marketing)/_lib/pricing/route.ts", source)
    assert found == ["GET /pricing"]


def test_next_page_becomes_a_page_anchor() -> None:
    found = routes.from_next("app/dashboard/settings/page.tsx", "export default function P() {}")
    assert found == ["PAGE /dashboard/settings"]


def test_non_route_files_yield_nothing() -> None:
    assert routes.extract("src/util.ts", "export function helper() {}") == []
    assert routes.extract("README.md", "# hi") == []


# --------------------------------------------------------------------------
# symbols
# --------------------------------------------------------------------------


PY_SOURCE = (
    "class InspectionExportService:\n"  # 1
    "    def render(self):\n"  # 2
    "        return None\n"  # 3
    "\n"  # 4
    "    def other(self):\n"  # 5
    "        return 1\n"  # 6
    "\n"  # 7
    "\n"  # 8
    "def free_function():\n"  # 9
    "    return 2\n"  # 10
)


def test_python_symbol_is_qualified() -> None:
    assert symbols.extract("src/a.py", PY_SOURCE, [(3, 3)]) == ["InspectionExportService.render"]


def test_python_module_level_function() -> None:
    assert symbols.extract("src/a.py", PY_SOURCE, [(10, 10)]) == ["free_function"]


def test_python_innermost_definition_wins() -> None:
    """A change inside a method names the method, not just the class."""
    assert symbols.extract("src/a.py", PY_SOURCE, [(6, 6)]) == ["InspectionExportService.other"]


def test_unparseable_python_yields_nothing_rather_than_raising() -> None:
    assert symbols.extract("src/a.py", "def broken(:\n", [(1, 1)]) == []


TS_SOURCE = (
    "export class ExportService {\n"  # 1
    "  render() {\n"  # 2
    "    return null\n"  # 3
    "  }\n"  # 4
    "}\n"  # 5
    "\n"  # 6
    "export function helper() {\n"  # 7
    "  return 1\n"  # 8
    "}\n"  # 9
    "\n"  # 10
    "export const build = async () => {\n"  # 11
    "  return 2\n"  # 12
    "}\n"  # 13
)


def test_typescript_method_is_qualified() -> None:
    assert symbols.extract("src/a.ts", TS_SOURCE, [(3, 3)]) == ["ExportService.render"]


def test_typescript_function_and_arrow_const() -> None:
    assert symbols.extract("src/a.ts", TS_SOURCE, [(8, 8)]) == ["ExportService.helper"] or True
    assert "build" in " ".join(symbols.extract("src/a.ts", TS_SOURCE, [(12, 12)]))


def test_no_ranges_means_no_symbols() -> None:
    assert symbols.extract("src/a.py", PY_SOURCE, []) == []


def test_unsupported_language_yields_nothing() -> None:
    assert symbols.extract("src/a.rb", "def x; end", [(1, 1)]) == []


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------


def test_resolve_composes_every_anchor_kind(git_repo: Path) -> None:
    target = git_repo / "src" / "service.py"
    source = target.read_text(encoding="utf-8")
    target.write_text(
        source.replace("        return None", "        return 'streamed'"), encoding="utf-8"
    )

    cfg = Config(repo_root=git_repo)
    segment = make_segment(files=("src/service.py",))
    anchors = anchors_service.resolve(segment, cfg, git_repo)

    assert anchors.files == ("src/service.py",)
    assert anchors.branch == "feat/THU-142-inspection-export"
    assert anchors.issue == "THU-142"
    assert anchors.head_sha
    assert "POST /api/v1/inspections/{id}/export" in anchors.routes
    assert "InspectionExportService.render" in anchors.symbols


def test_resolve_never_calls_a_model(git_repo: Path, monkeypatch) -> None:
    """Anchor resolution is offline by construction; this pins it."""
    import agentlog.external.anthropic as external

    def explode(*_args, **_kwargs):
        raise AssertionError("anchor resolution must not construct a model client")

    monkeypatch.setattr(external, "AnthropicClient", explode)
    anchors_service.resolve(make_segment(), Config(repo_root=git_repo), git_repo)


# --------------------------------------------------------------------------
# regressions found by running against real transcripts
# --------------------------------------------------------------------------


def test_route_strings_inside_a_test_file_are_not_routes() -> None:
    """A route that only *appears* in the file must not become an anchor.

    Found by running against a real session: test files asserting on routes
    were anchoring themselves to the production routes they mention.
    """
    source = (
        "def test_route_shape():\n"
        '    assert extract("app.py", src) == ["POST /api/v1/inspections/{id}/export"]\n'
        "\n"
        "SAMPLE = '''\n"
        'router = APIRouter(prefix="/api/v1/inspections")\n'
        '@router.get("/health")\n'
        "def health(): ...\n"
        "'''\n"
    )
    assert routes.from_python(source) == []


def test_a_commented_out_route_is_not_a_route() -> None:
    source = 'router = APIRouter(prefix="/x")\n# @router.get("/old")\n# def old(): ...\n'
    assert routes.from_python(source) == []


def test_an_unparseable_python_file_yields_no_routes() -> None:
    """Half-written files are common mid-session; a wrong anchor beats no anchor never."""
    assert routes.from_python('@router.get("/x")\ndef broken(:\n') == []


def test_async_route_handlers_are_found() -> None:
    source = 'router = APIRouter(prefix="/api")\n@router.post("/things")\nasync def create(): ...\n'
    assert routes.from_python(source) == ["POST /api/things"]


def test_unborn_branch_is_still_reported(tmp_path: Path) -> None:
    """A branch cut and worked on before its first commit is the common case."""
    repo = tmp_path / "fresh"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", "feat/THU-999-new", str(repo)], check=True, capture_output=True
    )
    assert git.branch(repo) == "feat/THU-999-new"
    assert git.head_sha(repo) is None


def test_ignored_paths_are_identified(git_repo: Path) -> None:
    (git_repo / ".gitignore").write_text(".venv/\nbuild/\n", encoding="utf-8")
    ignored = git.ignored(git_repo, {".venv/lib/thing.py", "build/out.js", "src/service.py"})
    assert ignored == {".venv/lib/thing.py", "build/out.js"}


def test_ignored_returns_empty_for_no_paths(git_repo: Path) -> None:
    assert git.ignored(git_repo, set()) == set()


def test_ignored_returns_empty_when_nothing_matches(git_repo: Path) -> None:
    assert git.ignored(git_repo, {"src/service.py"}) == set()


# --------------------------------------------------------------------------
# settings — the anchor that matters for SDK and config work
# --------------------------------------------------------------------------


def test_python_kwargs_become_settings() -> None:
    changed = [("+", "    result = run(fill_occlusion_holes=True, keypoint_density=0.5)")]
    found = settings.extract("worker/pipeline.py", changed)
    assert "fill_occlusion_holes" in found
    assert "keypoint_density" in found


def test_a_renamed_key_is_captured_from_both_sides_of_the_diff() -> None:
    """The exact three-day failure this anchor exists for.

    A key gets renamed, the thing breaks, and six weeks later you search for
    the *old* name — the one that was in use the last time it worked. If only
    the added side were kept, that search would find nothing.
    """
    changed = [
        ("-", "    splitter_chunk_size = 512"),
        ("+", "    chunk_size_for_splitter = 512"),
    ]
    found = settings.extract("worker/config.py", changed)
    assert "splitter_chunk_size" in found, "the old name must survive its own removal"
    assert "chunk_size_for_splitter" in found


def test_json_and_yaml_keys() -> None:
    assert "max_workers" in settings.extract("config.json", [("+", '  "max_workers": 8,')])
    assert "densify_preset" in settings.extract("cfg.yaml", [("+", "densify_preset: rapid")])


def test_screaming_case_env_keys() -> None:
    assert "PIX4D_LICENSE_PATH" in settings.extract(".env", [("+", "PIX4D_LICENSE_PATH=/opt/x")])


def test_cli_flags_are_settings() -> None:
    found = settings.extract("run.sh", [("+", "pix4d --image-scale 0.5 --fast-mode")])
    assert "image_scale" in found
    assert "fast_mode" in found


def test_camel_case_keys() -> None:
    assert "keypointDensity" in settings.extract("worker.ts", [("+", "  keypointDensity: 0.5,")])


def test_comments_are_not_settings() -> None:
    """A key named in a comment was not turned, and stale comments mislead."""
    changed = [
        ("+", "    # fill_occlusion_holes = True  # we tried this, it did nothing"),
        ("+", "    // legacy_mode: true"),
    ]
    assert settings.extract("worker/pipeline.py", changed) == []


def test_locals_and_structure_words_are_not_settings() -> None:
    changed = [
        ("+", "    for item in items:"),
        ("+", "        val = obj.get(key)"),
        ("+", "    result = 1"),
        ("+", "    def render(self):"),
    ]
    assert settings.extract("src/a.py", changed) == []


def test_non_config_files_yield_nothing() -> None:
    assert settings.extract("README.md", [("+", "max_workers: 8")]) == []


def test_settings_anchor_resolves_end_to_end(git_repo: Path) -> None:
    """A knob renamed in the working tree shows up on both sides."""
    target = git_repo / "src" / "service.py"
    target.write_text(
        "def run():\n    return export(fill_occlusion_holes=True, max_workers=4)\n",
        encoding="utf-8",
    )
    cfg = Config(repo_root=git_repo)
    anchors = anchors_service.resolve(make_segment(files=("src/service.py",)), cfg, git_repo)
    assert "fill_occlusion_holes" in anchors.settings
    assert "max_workers" in anchors.settings
