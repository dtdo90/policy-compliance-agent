"""Project-relative path helpers."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "demo.yaml"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
