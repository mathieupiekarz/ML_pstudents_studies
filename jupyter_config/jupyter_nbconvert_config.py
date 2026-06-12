"""Configuration nbconvert locale du projet SY09."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import prepend_pandoc_to_path  # noqa: E402

prepend_pandoc_to_path()

c.PDFExporter.enabled = False  # noqa: F821
c.WebPDFExporter.disable_sandbox = True  # noqa: F821
