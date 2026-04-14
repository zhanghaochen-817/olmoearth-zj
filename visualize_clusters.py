#!/usr/bin/env python3
"""Visualize kNN / clustering results from OlmoEarth embeddings.

This script is designed to work with the outputs from `knn_cluster.py`:
- `features_flat.npy`
- `clusters.csv` and/or `cluster_labels.npy`
- `metadata.csv`

It produces a simple 2D scatter plot using PCA, colored by cluster label.
This is useful when you want a quick PNG/JPG figure for inspection.

Examples:

```bash
python3 scripts/visualize_clusters.py \
  --features ./knn_results/features_flat.npy \
  --clusters ./knn_results/cluster_labels.npy \
  --output ./knn_results/cluster_plot.png
```

Or with CSV clusters:

```bash
python3 scripts/visualize_clusters.py \
  --features ./knn_results/features_flat.npy \
  --clusters-csv ./knn_results/clusters.csv \
  --output ./knn_results/cluster_plot.jpg
```
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize clustering results")
    parser.add_argument("--features", type=Path, required=True, help="Path to features_flat.npy")
    parser.add_argument("--clusters", type=Path, default=None, help="Path to cluster_labels.npy")
    parser.add_argument("--clusters-csv", type=Path, default=None, help="Path to clusters.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output image path (.png/.jpg)")
    parser.add_argument("--title", type=str, default="OlmoEarth Cluster Visualization", help="Plot title")
    return parser.parse_args()


def load_features(path: Path) -> np.ndarray:
    features = np.load(path)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features array, got shape {features.shape}")
    return features.astype(np.float32)


def load_labels(args: argparse.Namespace, n: int) -> np.ndarray:
    if args.clusters is not None:
        labels = np.load(args.clusters)
        if labels.shape[0] != n:
            raise ValueError("Cluster label count does not match number of features")
        return labels.astype(int)

    if args.clusters_csv is not None:
        labels = []
        with args.clusters_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(int(row["cluster"]))
        labels_arr = np.asarray(labels, dtype=int)
        if labels_arr.shape[0] != n:
            raise ValueError("Cluster CSV row count does not match number of features")
        return labels_arr

    raise ValueError("Provide either --clusters or --clusters-csv")


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    features = load_features(args.features)
    labels = load_labels(args, features.shape[0])

    # Reduce to 2D for plotting
    if features.shape[1] > 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(features)
    else:
        coords = features[:, :2]

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20")

    plt.figure(figsize=(10, 8), dpi=200)
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.85,
            color=cmap(i % 20),
            label=f"Cluster {lab}",
            edgecolors="none",
        )

    plt.title(args.title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc="best", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()

    print(f"[OK] Saved cluster visualization to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
