#!/usr/bin/env python3
"""Run kNN search and optional clustering on exported OlmoEarth embeddings.

This script expects the output from `extract_olmoearth_embeddings.py`:
- `embeddings.npy` or `embeddings.npz`
- `metadata.csv`

It can:
1) compute k-nearest neighbors for each sample
2) optionally run KMeans clustering
3) save all results to CSV files for inspection

Examples:

kNN only:

```bash
python3 scripts/knn_cluster.py \
  --embeddings ./exported_embeddings/embeddings.npy \
  --metadata ./exported_embeddings/metadata.csv \
  --output-dir ./knn_results \
  --k 10
```

kNN + clustering:

```bash
python3 scripts/knn_cluster.py \
  --embeddings ./exported_embeddings/embeddings.npy \
  --metadata ./exported_embeddings/metadata.csv \
  --output-dir ./knn_results \
  --k 10 \
  --clusters 8
```
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run kNN / clustering on embeddings")
    parser.add_argument("--embeddings", type=Path, required=True, help="Path to embeddings.npy or embeddings.npz")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save results")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    parser.add_argument("--clusters", type=int, default=0, help="If >0, run KMeans with this many clusters")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"], help="Distance metric for kNN")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for clustering")
    return parser.parse_args()


def load_embeddings(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
        if arr.ndim < 2:
            raise ValueError(f"Expected at least 2D array in {path}, got shape {arr.shape}")
        return arr

    if path.suffix == ".npz":
        npz = np.load(path)
        arrays = [npz[key] for key in npz.files]
        if not arrays:
            raise ValueError(f"No arrays found in {path}")
        try:
            return np.stack(arrays, axis=0)
        except ValueError as e:
            raise ValueError(
                "Embeddings in the NPZ must have matching shapes to stack. "
                "If you exported variable-sized maps with --pooling none, "
                "please first convert them to fixed-size vectors."
            ) from e

    raise ValueError(f"Unsupported embeddings file type: {path}")


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def flatten_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim == 2:
        return embeddings.astype(np.float32)
    return embeddings.reshape(embeddings.shape[0], -1).astype(np.float32)


def save_neighbors(
    output_path: Path,
    metadata: list[dict[str, str]],
    indices: np.ndarray,
    distances: np.ndarray,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["index", "file", "stem", "neighbor_rank", "neighbor_index", "neighbor_file", "neighbor_stem", "distance"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(metadata):
            for rank, (nbr_idx, dist) in enumerate(zip(indices[i], distances[i])):
                nbr_row = metadata[int(nbr_idx)]
                writer.writerow(
                    {
                        "index": i,
                        "file": row.get("file", ""),
                        "stem": row.get("stem", ""),
                        "neighbor_rank": rank,
                        "neighbor_index": int(nbr_idx),
                        "neighbor_file": nbr_row.get("file", ""),
                        "neighbor_stem": nbr_row.get("stem", ""),
                        "distance": float(dist),
                    }
                )


def save_clusters(output_path: Path, metadata: list[dict[str, str]], labels: np.ndarray) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["index", "file", "stem", "cluster"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(metadata):
            writer.writerow(
                {
                    "index": i,
                    "file": row.get("file", ""),
                    "stem": row.get("stem", ""),
                    "cluster": int(labels[i]),
                }
            )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = load_embeddings(args.embeddings)
    metadata = load_metadata(args.metadata)

    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"Metadata rows ({len(metadata)}) do not match embeddings count ({embeddings.shape[0]})"
        )

    features = flatten_embeddings(embeddings)

    # kNN search
    k = min(args.k + 1, len(features))  # include self, then drop it
    knn = NearestNeighbors(n_neighbors=k, metric=args.metric)
    knn.fit(features)
    distances, indices = knn.kneighbors(features)

    # Remove self-match in column 0
    if k > 1:
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    else:
        distances = distances[:, :0]
        indices = indices[:, :0]

    save_neighbors(args.output_dir / "knn_neighbors.csv", metadata, indices, distances)

    np.save(args.output_dir / "features_flat.npy", features)

    print(f"[OK] Saved kNN results to {args.output_dir / 'knn_neighbors.csv'}")
    print(f"[OK] Saved flattened features to {args.output_dir / 'features_flat.npy'}")

    # Optional KMeans clustering
    if args.clusters and args.clusters > 0:
        if args.clusters > len(features):
            raise ValueError("Number of clusters cannot exceed number of samples")

        km = KMeans(n_clusters=args.clusters, random_state=args.random_state, n_init="auto")
        labels = km.fit_predict(features)
        save_clusters(args.output_dir / "clusters.csv", metadata, labels)
        np.save(args.output_dir / "cluster_labels.npy", labels)
        np.save(args.output_dir / "cluster_centers.npy", km.cluster_centers_)
        print(f"[OK] Saved clustering results to {args.output_dir / 'clusters.csv'}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
