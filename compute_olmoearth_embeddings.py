#!/usr/bin/env python3
"""Compute OlmoEarth embeddings for a user-specified geographic box.

This script follows the `OlmoEarthEmbeddings.md` workflow, but hides the dataset
setup behind a single command:
1. create a fresh working directory under `./olmoearth_embedding_runs/`
2. write the rslearn dataset/model configs
3. add the requested bbox window(s)
4. download/materialize Sentinel imagery from the internet
5. run OlmoEarth and save embeddings for later kNN / clustering

Linux example:
    python3 scripts/compute_olmoearth_embeddings.py \
        --box -122.4 47.6 -122.3 47.7 \
        --start 2024-01-01T00:00:00+00:00 \
        --end 2025-01-01T00:00:00+00:00

If your region is large, add `--grid-size 1024` so rslearn splits it into
multiple windows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime


DATASET_CONFIG = {
    "layers": {
        "landsat": {
            "band_sets": [
                {
                    "bands": [
                        "B1",
                        "B2",
                        "B3",
                        "B4",
                        "B5",
                        "B6",
                        "B7",
                        "B8",
                        "B9",
                        "B10",
                        "B11",
                    ],
                    "dtype": "uint16",
                }
            ],
            "data_source": {
                "class_path": "rslearn.data_sources.aws_landsat.LandsatOliTirs",
                "init_args": {
                    "metadata_cache_dir": "cache/landsat",
                    "sort_by": "cloud_cover",
                },
                "ingest": False,
                "query_config": {
                    "max_matches": 12,
                    "period_duration": "30d",
                    "space_mode": "PER_PERIOD_MOSAIC",
                },
            },
            "type": "raster",
        },
        "sentinel1": {
            "band_sets": [
                {
                    "bands": ["vv", "vh"],
                    "dtype": "float32",
                    "nodata_vals": [-32768, -32768],
                }
            ],
            "data_source": {
                "class_path": "rslearn.data_sources.planetary_computer.Sentinel1",
                "init_args": {
                    "cache_dir": "cache/planetary_computer",
                    "query": {
                        "sar:instrument_mode": {"eq": "IW"},
                        "sar:polarizations": {"eq": ["VV", "VH"]},
                    },
                },
                "ingest": False,
                "query_config": {
                    "max_matches": 12,
                    "period_duration": "30d",
                    "space_mode": "PER_PERIOD_MOSAIC",
                },
            },
            "type": "raster",
        },
        "sentinel2_l2a": {
            "band_sets": [
                {
                    "bands": [
                        "B01",
                        "B02",
                        "B03",
                        "B04",
                        "B05",
                        "B06",
                        "B07",
                        "B08",
                        "B8A",
                        "B09",
                        "B11",
                        "B12",
                    ],
                    "dtype": "uint16",
                }
            ],
            "data_source": {
                "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
                "init_args": {
                    "cache_dir": "cache/planetary_computer",
                    "harmonize": True,
                    "sort_by": "eo:cloud_cover",
                },
                "ingest": False,
                "query_config": {
                    "max_matches": 12,
                    "period_duration": "30d",
                    "space_mode": "PER_PERIOD_MOSAIC",
                },
            },
            "type": "raster",
        },
        "embeddings": {
            "band_sets": [{"dtype": "float32", "num_bands": 768}],
            "type": "raster",
        },
    }
}

MODEL_YAML_TEMPLATE = """model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
            init_args:
              model_id: OLMOEARTH_V1_BASE
              patch_size: 4
        decoder:
          - class_path: rslearn.train.tasks.embedding.EmbeddingHead
    optimizer:
      class_path: rslearn.train.optimizer.AdamW
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: {dataset_path}
    inputs:
      sentinel2_l2a:
        data_type: raster
        layers: [sentinel2_l2a]
        bands: [B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09]
        passthrough: true
        dtype: FLOAT32
        load_all_layers: true
      sentinel1:
        data_type: raster
        layers: [sentinel1]
        bands: [vv, vh]
        passthrough: true
        dtype: FLOAT32
        load_all_layers: true
    task:
      class_path: rslearn.train.tasks.embedding.EmbeddingTask
    batch_size: 8
    num_workers: 8
    predict_config:
      transforms:
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: [B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09]
              sentinel1: [vv, vh]
      load_all_crops: true
      crop_size: 64
      overlap_pixels: 32
trainer:
  callbacks:
    - class_path: rslearn.train.prediction_writer.RslearnWriter
      init_args:
        output_layer: embeddings
        merger:
          class_path: rslearn.train.prediction_writer.RasterMerger
          init_args:
            overlap_pixels: 8
            downsample_factor: 4
"""


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_run_dir(base_dir: Path, box: list[float]) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_box = "_".join(f"{v:g}" for v in box)
    return (base_dir / f"bbox_{safe_box}_{stamp}").resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute OlmoEarth embeddings for a geographic box.")
    parser.add_argument("--box", nargs=4, type=float, metavar=("LON1", "LAT1", "LON2", "LAT2"), required=True, help="Bounding box in EPSG:4326")
    parser.add_argument("--start", required=True, help="ISO-8601 start time")
    parser.add_argument("--end", required=True, help="ISO-8601 end time")
    parser.add_argument("--grid-size", type=int, default=None, help="Optional grid size for large regions")
    parser.add_argument("--workers", type=int, default=32, help="Worker count for prepare/materialize")
    parser.add_argument("--base-dir", default="./olmoearth_embedding_runs", help="Where to create the working directory")
    parser.add_argument("--skip-materialize", action="store_true", help="Only write configs and add windows")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    run_dir = make_run_dir(base_dir, args.box)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = run_dir / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset_config_path = dataset_root / "config.json"
    model_config_path = run_dir / "model.yaml"

    write_text(dataset_config_path, json.dumps(DATASET_CONFIG, indent=2) + "\n")
    write_text(model_config_path, MODEL_YAML_TEMPLATE.format(dataset_path=str(dataset_root)))
    box_str = ",".join(str(v) for v in args.box)
    add_window_cmd = [
        "rslearn",
        "dataset",
        "add_windows",
        "--root",
        str(dataset_root),
        "--group",
        "default",
        "--name",
        "default",
        "--utm",
        "--resolution",
        "10",
        "--src_crs",
        "EPSG:4326",
        "--box",
        box_str,
        "--start",
        args.start,
        "--end",
        args.end,
    ]
    if args.grid_size is not None:
        add_window_cmd += ["--grid_size", str(args.grid_size)]
    run(add_window_cmd)

    if args.skip_materialize:
        print(f"Run directory: {run_dir}")
        print(f"Dataset config: {dataset_config_path}")
        print(f"Model config: {model_config_path}")
        return 0

    run([
        "rslearn",
        "dataset",
        "prepare",
        "--root",
        str(dataset_root),
        "--workers",
        str(args.workers),
        "--disabled-layers",
        "landsat",
        "--retry-max-attempts",
        "5",
        "--retry-backoff-seconds",
        "5",
    ])
    run([
        "rslearn",
        "dataset",
        "materialize",
        "--root",
        str(dataset_root),
        "--workers",
        str(args.workers),
        "--no-use-initial-job",
        "--disabled-layers",
        "landsat",
        "--retry-max-attempts",
        "5",
        "--retry-backoff-seconds",
        "5",
    ])
    run(["rslearn", "model", "predict", "--config", str(model_config_path)], cwd=run_dir)

    print("\nDone.")
    print(f"Run directory: {run_dir}")
    print(f"Embeddings written under: {dataset_root / 'windows'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
