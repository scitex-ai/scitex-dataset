"""Exact bundled-page lookup shared by the Dataset CLI and MCP adapters."""

from pathlib import Path


def _skills_dir() -> Path:
    return Path(__file__).resolve().parent / "_skills" / "scitex-dataset"


def _skill_pages() -> dict[str, Path]:
    root = _skills_dir().resolve()
    pages = {}
    for candidate in root.glob("*.md"):
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError, RuntimeError):
            continue
        if resolved.is_file():
            pages[candidate.stem] = resolved
    return pages


def _skill_names() -> list[str]:
    return sorted(name for name in _skill_pages() if name != "SKILL")


def _read_skill_page(name: str) -> str | None:
    # Look up an exact bundled identifier; never interpret it as a path.
    page = _skill_pages().get(name)
    return page.read_text(encoding="utf-8") if page is not None else None
