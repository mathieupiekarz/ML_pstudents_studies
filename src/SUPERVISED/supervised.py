#!/usr/bin/env python3
"""Point d'entrée CLI de l'analyse supervisée post-clustering (Phases A/B/C)."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
for p in (str(SCRIPT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS_DIR = SCRIPT_DIR / "results"

import config  # noqa: E402
import report  # noqa: E402
from _utils import build_typology, load_data  # noqa: E402
from loaders import build_joined, target_frame  # noqa: E402
from phase_a import run_phase_a  # noqa: E402
from phase_a_plots import (  # noqa: E402
    plot_factorial_by_cluster,
    plot_factorial_by_target,
    plot_target_distribution,
    plot_target_heatmap,
)
from phase_b import run_phase_b  # noqa: E402
from phase_c import run_phase_c  # noqa: E402

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

_HELP_TEXT = """\
Usage
-----
uv run python src/SUPERVISED/supervised.py <source> <source_run> <cluster_run> <analysis_run> [--method cah|kmeans|nuees]

Arguments
---------
source       : acp | acm | famd | aftd
source_run   : nom du run de réduction (ex. no_G1_G2_G3)
cluster_run  : nom du run de clustering (dossier src/CLUSTERING/results/)
analysis_run : nom libre ; sorties dans src/SUPERVISED/results/<analysis_run>/
--method     : cah (défaut) | kmeans | nuees
"""


class _HelpAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        print(_HELP_TEXT)
        parser.exit()


def _valid_name(name: str) -> str:
    if not _NAME_RE.match(name):
        raise argparse.ArgumentTypeError(
            f"Nom invalide {name!r} — lettres, chiffres, '.', '_', '-' uniquement."
        )
    return name


def run_pipeline(source, source_run, cluster_run, analysis_run, method) -> int:
    joined, cl = build_joined(source, source_run, cluster_run, method)
    typology = build_typology(config.apply_preprocessing(load_data()))

    print(f"\n=== Analyse supervisée — {source.upper()}/{source_run} "
          f"× clustering {cluster_run} ({method}, k={cl.k}) ===")
    print(f"Individus : {len(joined.frame)} | axes : {len(joined.dim_cols)} | "
          f"explicatives : {len(joined.feature_cols)}")
    if joined.excluded_vars:
        print(f"Exclues (INCLUDE_VARS=False) : {', '.join(joined.excluded_vars)}")
    if joined.cluster_in_reduction:
        print("ATTENTION : cible(s) présente(s) dans la réduction (fuite possible) : "
              f"{joined.cluster_in_reduction}")

    for spec in joined.target_specs:
        target, kind = spec.name, spec.kind
        na = joined.n_dropped_na.get(target, 0)
        print(f"\n--- Cible : {target} ({kind})"
              + (f" | {na} NA retirés" if na else "") + " ---")

        out_dir = RESULTS_DIR / analysis_run / target.replace(".", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        sub = target_frame(joined, spec)

        # ── Phase A ─────────────────────────────────────────────────────
        print("  Phase A : tests + visualisations …")
        a = run_phase_a(sub, target, kind, joined.feature_cols)
        plot_target_distribution(sub, target, kind, out_dir / "A_target_distribution.png")
        plot_factorial_by_cluster(sub, joined.dim_cols, out_dir / "A_factorial_by_cluster.png")
        plot_factorial_by_target(sub, joined.dim_cols, target, kind,
                                 out_dir / "A_factorial_by_target.png")
        plot_target_heatmap(sub, target, kind, out_dir / "A_heatmap_target_by_cluster.png")
        report.write_phase_a(a, out_dir)

        # ── Phase B ─────────────────────────────────────────────────────
        print("  Phase B : panel de modèles + évaluation …")
        b = run_phase_b(sub, typology, target, kind, joined.feature_cols, out_dir)
        report.write_phase_b(b, out_dir)

        # ── Phase C ─────────────────────────────────────────────────────
        print("  Phase C : triangulation + conclusion …")
        c = run_phase_c(a, b)
        report.write_phase_c(c, out_dir)
        report.write_synthese(target, a, b, c, out_dir)

        print(f"  → {out_dir.resolve()}")
        print(f"    Phase A : {'significatif' if a.significant else 'non significatif'} "
              f"({a.effect_name}={a.effect_value:.3f})")
        print(f"    Phase B : meilleur modèle = {b.best_model}")
        print(f"    Phase C : {c.scenario}")

    out_base = RESULTS_DIR / analysis_run
    n_files = sum(1 for _ in out_base.rglob("*") if _.is_file())
    print(f"\n=== Terminé — {n_files} fichiers écrits ===")
    print(f"Dossier : {out_base.resolve()}/")
    print("(gitignoré : masqué dans l'explorateur si les fichiers ignorés sont cachés)")

    return 0


def main() -> int:
    if len(sys.argv) == 1:
        print(_HELP_TEXT)
        return 0

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action=_HelpAction, nargs=0)
    parser.add_argument("source", choices=["acp", "acm", "famd", "aftd"])
    parser.add_argument("source_run", type=_valid_name)
    parser.add_argument("cluster_run", type=_valid_name)
    parser.add_argument("analysis_run", type=_valid_name)
    parser.add_argument("--method", choices=["cah", "kmeans", "nuees"], default="cah")
    args = parser.parse_args()

    return run_pipeline(
        args.source, args.source_run, args.cluster_run, args.analysis_run, args.method
    )


if __name__ == "__main__":
    raise SystemExit(main())
