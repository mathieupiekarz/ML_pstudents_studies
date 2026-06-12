"""Prépare l'environnement notebook : pandoc + Chromium (webpdf)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _venv_bin() -> Path:
    if virtual_env := os.environ.get("VIRTUAL_ENV"):
        return Path(virtual_env) / "bin"

    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".venv" / "bin"
        if candidate.is_dir():
            return candidate

    raise RuntimeError("Répertoire .venv/bin introuvable. Lancez d'abord : uv sync")


def _pandoc_binary() -> Path:
    import pypandoc

    path = pypandoc.get_pandoc_path()
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()

    found = shutil.which(path) or shutil.which("pandoc")
    if found:
        return Path(found).resolve()

    bundled = Path(pypandoc.__file__).resolve().parent / "files" / "pandoc"
    if bundled.is_file():
        return bundled.resolve()

    raise RuntimeError("pandoc introuvable dans pypandoc-binary")


def _ensure_pandoc_link() -> None:
    pandoc_src = _pandoc_binary()
    pandoc_link = _venv_bin() / "pandoc"

    if pandoc_link.is_symlink() and pandoc_link.resolve() == pandoc_src:
        print(f"pandoc déjà configuré : {pandoc_link} -> {pandoc_src}")
        return

    if pandoc_link.exists() or pandoc_link.is_symlink():
        pandoc_link.unlink()

    pandoc_link.symlink_to(pandoc_src)
    print(f"pandoc configuré : {pandoc_link} -> {pandoc_src}")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_jupyter_config() -> None:
    """Installe la config Jupyter dans le venv (sans JUPYTER_CONFIG_DIR)."""
    src = _project_root() / "jupyter_config"
    dst = _venv_bin().parent / "etc" / "jupyter"
    dst.mkdir(parents=True, exist_ok=True)

    for name in (
        "jupyter_server_config.py",
        "jupyter_nbconvert_config.py",
        "_env.py",
    ):
        link = dst / name
        target = (src / name).resolve()
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)

    print(f"config Jupyter installée : {dst}")


def _ensure_chromium() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright non installé (nbconvert[webpdf] manquant)")
        return

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
            browser.close()
            print("chromium déjà disponible pour l'export webpdf")
            return
        except Exception:
            pass

    print("installation de chromium pour l'export PDF (webpdf)…")
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        check=True,
    )
    print("chromium installé")


def main() -> int:
    _ensure_pandoc_link()
    _ensure_jupyter_config()
    _ensure_chromium()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
