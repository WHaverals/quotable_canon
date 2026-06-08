"""Repository path constants and ``src/`` import bootstrap.

Notebooks at the repo root should add ``src/`` to ``sys.path`` once, then import
from this module::

    import sys
    from pathlib import Path

    for _p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (_p / "src" / "paths.py").is_file():
            sys.path.insert(0, str(_p / "src"))
            break
    else:
        raise FileNotFoundError(
            "Could not find src/paths.py — run notebooks with cwd = repository root"
        )

    from paths import REPO_ROOT, DATA_DIR, EXPORTS_DIR, passim_data_paths
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

CHADWYCK_TEXT_DIRNAME = "chadwyckhealey"
PASSIM_DATASET_DIRNAME = "ppa_found_poems"


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from *start* (or ``cwd``) until ``src/poem_corpus.py`` is found."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "src" / "poem_corpus.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find repo root (expected src/poem_corpus.py). "
        "Run notebooks with cwd = repository root."
    )


def bootstrap_src_path(start: Path | None = None) -> Path:
    """Ensure ``src/`` is on ``sys.path`` and return the repo root."""
    root = find_repo_root(start)
    src = root / "src"
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return root


REPO_ROOT = find_repo_root()
DATA_DIR = REPO_ROOT / "data"
EXPORTS_DIR = REPO_ROOT / "exports"
FIGURES_DIR = EXPORTS_DIR / "figures"
MODEL_DIR = EXPORTS_DIR / "model"
CACHE_DIR = REPO_ROOT / ".cache"

CHADWYCK_TEXT_DIR = DATA_DIR / CHADWYCK_TEXT_DIRNAME
PASSIM_DATA_DIR = DATA_DIR / PASSIM_DATASET_DIRNAME
POETRY_METADATA_PATH = DATA_DIR / "poetry_metadata.csv"
INTERNET_POEMS_METADATA_PATH = (
    DATA_DIR / "internet_poems" / "internet_poems_metadata_enriched.csv"
)


def passim_data_paths() -> dict[str, Path]:
    """Paths to the compiled Passim dataset inputs under ``PASSIM_DATA_DIR``."""
    return {
        "excerpts": PASSIM_DATA_DIR / "excerpts.csv.gz",
        "ppa_work_metadata": PASSIM_DATA_DIR / "ppa_work_metadata.csv",
        "poem_meta": PASSIM_DATA_DIR / "poem_meta.csv",
    }


def assert_passim_data_paths(paths: Mapping[str, Path] | None = None) -> dict[str, Path]:
    """Return Passim input paths, raising if any file is missing."""
    resolved = dict(paths or passim_data_paths())
    for key, path in resolved.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {key}: {path}")
    return resolved
