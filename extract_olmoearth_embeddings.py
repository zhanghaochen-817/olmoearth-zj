#!/usr/bin/env python3
"""Export embedding rasters produced by OlmoEarth into KNN-friendly arrays.

This script is meant to be used after `compute_olmoearth_embeddings.py` or any
other rslearn/OlmoEarth run that writes an `embeddings` raster layer.

What it does:
1) finds embedding GeoTIFFs under the provided run directory
2) loads each embedding raster
3) optionally pools the spatial map into one vector per window
4) saves a `.npy` array plus CSV metadata for downstream KNN / clustering

Typical usage:

```bash
python3 scripts/extract_olmoearth_embeddings.py \
  --input-dir ./olmoearth_embedding_runs/bbox_xxx/dataset/windows \
  --output-dir ./exported_embeddings \
  --glob "**/layers/embeddings/**/geotiff.tif" \
  --pooling mean
```

If your goal is pixel-level clustering, use `--pooling none` to keep the full
`C x H x W` embeddings per file, then post-process them yourself.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export OlmoEarth embedding rasters")
    p.add_argument("--input-dir", type=Path, required=True, help="Run directory or windows directory containing embedding GeoTIFFs")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to save exported arrays")
    p.add_argument("--glob", type=str, default="**/layers/embeddings/**/geotiff.tif", help="Glob for embedding GeoTIFFs")
    p.add_argument("--pooling", type=str, choices=["mean", "max", "none"], default="mean", help="Pool over HxW or keep full map")
    return p.parse_args()


def read_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    return arr


def pool_embedding(arr: np.ndarray, pooling: str) -> np.ndarray:
    if pooling == "none":
        return arr
    if arr.ndim == 3:
        if pooling == "mean":
            return arr.mean(axis=(1, 2))
        return arr.max(axis=(1, 2))
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported raster shape: {arr.shape}")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tif_paths = sorted(args.input_dir.glob(args.glob))
    if not tif_paths:
        raise ValueError(f"No embedding files matched {args.glob} under {args.input_dir}")

    all_embeddings: list[np.ndarray] = []
    rows: list[dict[str, str]] = []

    for p in tif_paths:
        arr = read_raster(p)
        pooled = pool_embedding(arr, args.pooling)
        all_embeddings.append(pooled.astype(np.float32))
        rows.append({"file": str(p), "stem": p.stem, "shape": str(tuple(arr.shape))})
        print(f"[OK] {p.name} -> raw shape {arr.shape}, exported shape {pooled.shape}")

    embeddings_path = args.output_dir / "embeddings.npy"
    metadata_path = args.output_dir / "metadata.csv"

    if args.pooling == "none":
        np.savez_compressed(embeddings_path.with_suffix(".npz"), *all_embeddings)
        saved_path = embeddings_path.with_suffix(".npz")
    else:
        embs = np.stack(all_embeddings, axis=0)
        np.save(embeddings_path, embs)
        saved_path = embeddings_path

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "stem", "shape"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print(f"Saved embeddings: {saved_path}")
    print(f"Saved metadata:   {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
