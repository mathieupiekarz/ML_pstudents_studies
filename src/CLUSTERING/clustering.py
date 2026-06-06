#!/usr/bin/env python3
"""Point d'entrée CLI du module clustering."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

from cah import compare_linkages, cophenetic_quality, plot_dendrogram, run_cah
from compare import compare_partitions
from kmeans import (
    plot_elbow_silhouette,
    run_kmeans,
    run_nuees_dynamiques,
    select_k_silhouette,
)
from loaders import load_coordinates
from plots import plot_cluster_scatter


# ── Validation du nom ─────────────────────────────────────────────────────────

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _valid_name(name: str) -> str:
    if not _NAME_RE.match(name):
        raise argparse.ArgumentTypeError(
            f"Nom invalide {name!r} — uniquement lettres, chiffres, '.', '_', '-'."
        )
    return name


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    source: str,
    source_run: str,
    cluster_run: str,
    k_forced: int | None,
    n_axes: int | None,
    variance_threshold: float = 0.80,
) -> int:
    # ── Dossier de sortie ─────────────────────────────────────────────────
    out_dir = RESULTS_DIR / cluster_run
    out_dir.mkdir(parents=True, exist_ok=True)

    source_label = f"{source.upper()} — {source_run}"
    print(f"\n=== Clustering {source_label} → {cluster_run} ===\n")

    # ── Chargement ────────────────────────────────────────────────────────
    X, n_axes_used = load_coordinates(
        source, source_run, n_axes=n_axes, variance_threshold=variance_threshold
    )
    print(f"Coordonnées chargées : {X.shape[0]} individus × {n_axes_used} axes")

    # ── Sélection de k ────────────────────────────────────────────────────
    k_range = range(2, 9)
    if k_forced is not None:
        k = k_forced
        print(f"k imposé : {k}")
    else:
        k = select_k_silhouette(X, k_range)
        print(f"k optimal (silhouette) : {k}")

    # ── 03 : Elbow + silhouette ───────────────────────────────────────────
    print("  03 elbow + silhouette …")
    plot_elbow_silhouette(X, k_range, out_dir / "03_elbow_silhouette.png",
                          source_label=source_label)

    # ── CAH ───────────────────────────────────────────────────────────────
    print("  CAH Ward …")
    cah_labels, Z = run_cah(X, k, linkage_method="ward")
    coph = cophenetic_quality(X, Z)
    print(f"  Coefficient cophénétique (Ward) : {coph:.4f}")

    # ── 01 : Dendrogramme Ward ────────────────────────────────────────────
    print("  01 dendrogramme …")
    plot_dendrogram(
        Z, k,
        out_dir / "01_dendrogram_ward.png",
        title=f"CAH — Dendrogramme Ward (k={k})",
        source_label=source_label,
    )

    # ── 02 : Comparaison des critères ─────────────────────────────────────
    print("  02 comparaison critères …")
    coeff_dict = compare_linkages(X, out_dir / "02_linkage_comparison.png",
                                  source_label=source_label)

    # ── 04 : Scatter CAH ─────────────────────────────────────────────────
    print("  04 scatter CAH …")
    plot_cluster_scatter(
        X, cah_labels,
        out_dir / f"04_cah_k{k}_scatter.png",
        title=f"CAH Ward — k={k}",
        method_name=f"coeff. coph. = {coph:.3f}",
        source_label=source_label,
    )

    # ── K-means ──────────────────────────────────────────────────────────
    print("  k-means …")
    km_labels = run_kmeans(X, k)

    # ── 05 : Scatter k-means ──────────────────────────────────────────────
    print("  05 scatter k-means …")
    plot_cluster_scatter(
        X, km_labels,
        out_dir / f"05_kmeans_k{k}_scatter.png",
        title=f"K-means — k={k}",
        method_name="init k-means++",
        source_label=source_label,
    )

    # ── Nuées dynamiques ─────────────────────────────────────────────────
    print("  nuées dynamiques …")
    nuees_labels = run_nuees_dynamiques(X, k, Z)

    # ── 06 : Scatter nuées dynamiques ────────────────────────────────────
    print("  06 scatter nuées dynamiques …")
    plot_cluster_scatter(
        X, nuees_labels,
        out_dir / f"06_nuees_k{k}_scatter.png",
        title=f"Nuées dynamiques — k={k}",
        method_name="init centroïdes CAH",
        source_label=source_label,
    )

    # ── 07 : Comparaison de partitions ────────────────────────────────────
    print("  07 comparaison partitions …")
    labels_dict = {
        f"CAH Ward k={k}": cah_labels,
        f"K-means k={k}": km_labels,
        f"Nuées dyn. k={k}": nuees_labels,
    }
    summary_df = compare_partitions(
        labels_dict, X,
        out_dir / "07_partition_comparison.png",
        source_label=source_label,
    )

    # ── CSV labels ────────────────────────────────────────────────────────
    import numpy as np
    import pandas as pd

    for fname, lbl in [
        (f"cah_labels_k{k}.csv", cah_labels),
        (f"kmeans_labels_k{k}.csv", km_labels),
        (f"nuees_labels_k{k}.csv", nuees_labels),
    ]:
        pd.DataFrame({"individu": range(len(lbl)), "cluster": lbl}).to_csv(
            out_dir / fname, index=False
        )

    # ── Résumé texte ──────────────────────────────────────────────────────
    lines = [
        f"=== Clustering {source.upper()} — {source_run} ===",
        f"Run clustering : {cluster_run}",
        f"Individus : {X.shape[0]} | Axes utilisés : {n_axes_used}",
        f"k retenu  : {k}",
        "",
        "Coefficient cophénétique (ultramétrique CAH)",
        *[f"  {m:10s}: {c:.4f}" for m, c in coeff_dict.items()],
        "",
        "Qualité des partitions",
        summary_df.to_string(index=False),
        "",
        "Fichiers générés :",
        *[f"  {p.name}" for p in sorted(out_dir.iterdir())],
    ]
    summary_path = out_dir / "clustering_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nRésultats : {out_dir.resolve()}")
    print(summary_df.to_string(index=False))
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

_HELP_TEXT = """\
    uv run python src/CLUSTERING/clustering.py <source> <source_run> <cluster_run> [--k K] [--axes N]

Arguments
---------
source       : acp | acm | famd | aftd
source_run   : nom du run de réduction (ex. no_G1_G2_G3_avg)
cluster_run  : nom libre pour cette exécution ; les résultats vont dans
               src/CLUSTERING/results/<cluster_run>/ (ex. aftd_k4)
"""


class _HelpAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        print(_HELP_TEXT)
        parser.exit()


def main() -> int:
    # Ajout du dossier src/CLUSTERING au path pour les imports locaux
    for p in (str(SCRIPT_DIR),):
        if p not in sys.path:
            sys.path.insert(0, p)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action=_HelpAction, nargs=0)
    parser.add_argument(
        "source",
        choices=["acp", "acm", "famd", "aftd"],
    )
    parser.add_argument("source_run", type=_valid_name)
    parser.add_argument("cluster_run", type=_valid_name)
    parser.add_argument("--k", type=int, default=None, metavar="K")
    parser.add_argument("--axes", type=int, default=None, metavar="N")
    args = parser.parse_args()

    return run_pipeline(
        source=args.source,
        source_run=args.source_run,
        cluster_run=args.cluster_run,
        k_forced=args.k,
        n_axes=args.axes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
