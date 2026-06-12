"""Utilitaires partagés pour la configuration Jupyter du projet."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def pandoc_dir() -> str:
    import pypandoc

    path = pypandoc.get_pandoc_path()
    if os.path.sep in path or (os.path.altsep and os.path.altsep in path):
        return str(Path(path).resolve().parent)

    found = shutil.which(path) or shutil.which("pandoc")
    if found:
        return str(Path(found).resolve().parent)

    bundled = Path(pypandoc.__file__).resolve().parent / "files" / "pandoc"
    if bundled.is_file():
        return str(bundled.parent)

    raise RuntimeError("pandoc introuvable : lancez `uv run python scripts/setup_pandoc.py`")


def prepend_pandoc_to_path() -> None:
    pandoc = pandoc_dir()
    os.environ["PATH"] = pandoc + os.pathsep + os.environ.get("PATH", "")
