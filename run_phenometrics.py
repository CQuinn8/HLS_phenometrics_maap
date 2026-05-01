#!/usr/bin/env python3
"""
run_phenometrics.py
-------------------
Full pipeline orchestrator: HLS download -> EVI calculation -> phenometrics.

Each upstream step is optional (skip flags) and the script paths are
soft-coded via arguments, so they can be swapped without touching this file.

Usage
-----
# Full pipeline
python run_phenometrics.py \
    --data_dir /projects/my-public-bucket/hls/testing/ \
    --output_path /projects/my-public-bucket/hls/testing/ \
    --tile 18SUJ \
    --target_year 2022 \
    --download_script /projects/scripts/hls_download.py \
    --evi_script /projects/scripts/hls_evi_calc.py

# Skip download, reuse existing scenes, only run phenometrics
python run_phenometrics.py \
    --data_dir /projects/my-public-bucket/hls/testing/ \
    --output_path /projects/my-public-bucket/hls/testing/ \
    --tile 18SUJ \
    --target_year 2022 \
    --skip_download \
    --skip_evi
"""

import argparse
import sys
from pathlib import Path
from functools import partial
import geopandas as gpd

from phenometric_algorithm import *


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full pipeline: HLS download -> EVI calculation -> phenometrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required I/O ---
    p.add_argument("--data_dir",         required=True,  type=Path,
                   help="Root directory containing (or to receive) EVI2 files.")
    p.add_argument("--output_path",      required=True,  type=Path,
                   help="Root output directory for phenometrics results.")
    p.add_argument("--tile",             required=True,  type=str,
                   help="MGRS tile ID (e.g. 18SUJ).")
    p.add_argument("--target_year",      required=True,  type=int,
                   help="Year to process.")
    
    # -------------------------------------------------------------------------
    # OPTIONAL ARGS SET WITH DEFAULTS
    # -------------------------------------------------------------------------
    # --- Context window ---
    p.add_argument("--context_months",  default=12, type=int,
                   help="Months of context passed to enter_processing_stage.")

    # --- Optional ROI ---
    p.add_argument("--roi_file",         default=None,   type=Path)
    p.add_argument("--tile_epsg",        default=None,  type=int)

    # --- Runtime / memory ---
    p.add_argument("--chunk_size",       default=600,    type=int)
    p.add_argument("--chunks_in_memory", default=10,     type=int)
    p.add_argument("--run_label",        default=None,   type=str)
    p.add_argument("--n_workers",        default=1,   type=int)

    return p.parse_args()


# =============================================================================
# Core runner
# =============================================================================

def run_phenometrics(
    data_dir: Path,
    output_path: Path,
    tile: str,
    target_year: int,
    context_months: int = 12,
    roi_file: Path | None = None,
    tile_epsg: int = None,
    chunk_size: int = 600,
    chunks_in_memory: int = 10,
    run_label: str | None = None,    
    n_workers: int = 1,
) -> dict:
    """
    Full pipeline: phenometrics
    """

    # Output directory
    out_subdir = f"{tile}"
    if run_label:
        out_subdir = f"{out_subdir}-{run_label}"
    if roi_file is not None:
        out_subdir = f"{out_subdir}-{roi_file.parent.name}"

    output_dir = output_path / out_subdir / str(target_year)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Tile         : {tile}")
    print(f"  Target year  : {target_year}")
    print(f"  Input dir    : {data_dir}")
    print(f"  Output dir   : {output_dir}")
    print("=" * 70)

    data_config = ProcessingConfig(
        base_path=Path(data_dir),
        tile_id=tile
    )
    scenes = build_scene_index(data_config)
    available_years = sorted({s.year for s in scenes})
    if target_year not in available_years:
        raise ValueError(
            f"target_year={target_year} has no EVI scenes in local directory {data_dir}. "
            f"Available years: {available_years}"
        )

    roi_reproj = None
    if roi_file is not None:
        roi = gpd.read_file(roi_file)
        roi_reproj = roi.to_crs(f"EPSG:{tile_epsg}")
        print(f"  ROI loaded & reprojected to EPSG:{tile_epsg}")

    reader = ChunkedTimeSeriesReaderStreaming(
        scenes,
        chunk_size=(chunk_size, chunk_size),
        roi=roi_reproj,
        duplicate_handling="mean",
        output_dir=output_dir,
        context_months=context_months,
        target_year=target_year,
        default_crs=tile_epsg,
    )

    configured_pipeline = partial(
        full_pipeline_chunk,
        apply_threshold=True
    )

    reader.enter_processing_stage(
        process_fn=configured_pipeline,
        chunks_in_memory=chunks_in_memory,
        context_months=context_months,    
        n_workers = n_workers,
    )

    print(f"\n  Done – {target_year} outputs written to {output_dir}")


if __name__ == "__main__":
    
    args = parse_args()
    print(args)
    run_phenometrics(
        data_dir          = args.data_dir,
        output_path       = args.output_path,
        tile              = args.tile,
        target_year       = args.target_year,
        context_months    = args.context_months,
        roi_file          = args.roi_file,
        tile_epsg         = args.tile_epsg,
        chunk_size        = args.chunk_size,
        chunks_in_memory  = args.chunks_in_memory,
        run_label         = args.run_label,
        n_workers = args.n_workers,
    )
    # python run_phenometrics.py --data_dir=/projects/my-public-bucket/hls/testing/daily-subset-SERC/ --output_path=temp_out --tile=18SUJ --target_year=2020 --chunk_size=10 --n_workers=2
    
    ## python run_phenometrics.py --data_dir=/projects/my-public-bucket/hls/testing/daily/ --output_path=temp_out --tile=18SUJ --target_year=2020 --chunk_size=500 --n_workers=6 --roi_file="/projects/my-public-bucket/hls/testing/ROIs/MD_transect/POLYGON.shp"
