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

# from phenometric_algorithm import *


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
    # HLS Download and EVI calculation args 
    # -------------------------------------------------------------------------
    p.add_argument("--download_script",  default=None,   type=Path,
                   help="Path to the HLS download script (e.g. hls_download.py). "
                        "Omit or use --skip_download to bypass this step.")
    p.add_argument("--evi_script",       default=None,   type=Path,
                   help="Path to the EVI calculation script (e.g. hls_evi_calc.py). "
                        "Omit or use --skip_evi to bypass this step.")

    p.add_argument("--download_args",    default="",     type=str,
                   help="Extra arguments forwarded verbatim to the download script, "
                        "as a single quoted string (e.g. '--max_cloud 20 --sensor S30').")
    p.add_argument("--evi_args",         default="",     type=str,
                   help="Extra arguments forwarded verbatim to the EVI script.")

    p.add_argument("--skip_download",    action="store_true",
                   help="Skip the HLS download step (data already on disk).")
    p.add_argument("--skip_evi",         action="store_true",
                   help="Skip the EVI calculation step (EVI2 tifs already exist).")
    p.add_argument("--skip_phenometrics", action="store_true",
                   help="Skip phenometrics (useful to test only the upstream steps).")
    
    # -------------------------------------------------------------------------
    # OPTIONAL ARGS SET WITH DEFAULTS
    # -------------------------------------------------------------------------
    # --- Context window ---
    p.add_argument("--context_months",  default=12, type=int,
                   help="Months of context passed to enter_processing_stage.")

    # --- Optional ROI ---
    p.add_argument("--roi_file",         default=None,   type=Path)
    p.add_argument("--tile_epsg",        default=32618,  type=int)

    # --- Runtime / memory ---
    p.add_argument("--chunk_size",       default=600,    type=int)
    p.add_argument("--chunks_in_memory", default=10,     type=int)
    p.add_argument("--rebuild",          action="store_true",
                   help="Force rebuild of the scene index JSON.")
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
    cadence: str = "10day",
    composite_method: str = "median",
    context_months: int = 12,
    roi_file: Path | None = None,
    tile_epsg: int = 32618,
    chunk_size: int = 600,
    chunks_in_memory: int = 10,
    rebuild: bool = False,
    run_label: str | None = None,
    
    # --- upstream steps ---
    download_script: Path | None = None,
    evi_script: Path | None = None,
    download_args: str = "",
    evi_args: str = "",
    skip_download: bool = False,
    skip_evi: bool = False,
    
    skip_phenometrics: bool = False,
    n_workers: int = 1,
) -> dict:
    """
    Full pipeline: HLS download → EVI calculation → phenometrics.

    Upstream steps are only executed when their script path is provided AND
    the corresponding skip flag is not set. This makes the function safe to
    call with any combination of steps active.
    """

    use_doy_files = (cadence == "10day")
    is_monthly    = (cadence == "monthly")

    # Shared kwargs forwarded to upstream scripts
    upstream_kwargs = dict(
        tile=tile,
        target_year=target_year,
        data_dir=data_dir,
        cadence=cadence,
    )

    # Output directory
    out_subdir = f"{tile}_{cadence}-{composite_method}"
    if run_label:
        out_subdir = f"{out_subdir}-{run_label}"
    if roi_file is not None:
        out_subdir = f"{out_subdir}-{roi_file.parent.name}"

    output_dir = output_path / out_subdir / str(target_year)

    if not skip_download or not skip_evi or not skip_phenometrics:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("No processing requested")

    # ------------------------------------------------------------------
    # Step 1 – HLS Download
    # ------------------------------------------------------------------
    if not skip_download:
        if download_script is not None:
            run_upstream_script(
                download_script,
                "HLS download",
                extra_args=download_args,
                **upstream_kwargs,
            )
        else:
            print("  [HLS download] No --download_script provided, skipping.")
    else:
        print("  [HLS download] Skipped.")

    # ------------------------------------------------------------------
    # Step 2 – EVI Calculation
    # ------------------------------------------------------------------
    if not skip_evi:
        if evi_script is not None:
            run_upstream_script(
                evi_script,
                "EVI calculation",
                extra_args=evi_args,
                **upstream_kwargs,
            )
        else:
            print("  [EVI calculation] No --evi_script provided, skipping.")
    else:
        print("  [EVI calculation] Skipped.")

    # ------------------------------------------------------------------
    # Step 3 – Phenometrics
    # ------------------------------------------------------------------
    if skip_phenometrics:
        print("  [Phenometrics] Skipped via --skip_phenometrics.")
        return {}

    print("=" * 70)
    print(f"  Tile         : {tile}")
    print(f"  Target year  : {target_year}")
    print(f"  Input dir    : {data_dir}")
    print(f"  Output dir   : {output_dir}")
    print("=" * 70)

    # data_config = ProcessingConfig(
    #     base_path=Path(data_dir),
    #     tile_id=tile,
    #     cadence=cadence,
    # )
    # scenes = get_or_build_index(data_config, rebuild=rebuild)
    # available_years = sorted({s.year for s in scenes})
    # if target_year not in available_years:
    #     raise ValueError(
    #         f"target_year={target_year} has no EVI scenes in local directory {data_dir}. "
    #         f"Available years: {available_years}"
    #     )

    # roi_reproj = None
    # if roi_file is not None:
    #     roi = gpd.read_file(roi_file)
    #     roi_reproj = roi.to_crs(f"EPSG:{tile_epsg}")
    #     print(f"  ROI loaded & reprojected to EPSG:{tile_epsg}")

    # reader = ChunkedTimeSeriesReaderStreaming(
    #     scenes,
    #     chunk_size=(chunk_size, chunk_size),
    #     roi=roi_reproj,
    #     duplicate_handling="mean",
    #     use_doy_files=use_doy_files,
    #     output_dir=output_dir,
    #     context_months=context_months,
    #     target_year=target_year,
    #     default_crs=tile_epsg,
    # )

    # configured_pipeline = partial(
    #     full_pipeline_chunk,
    #     apply_threshold=True,
    #     is_monthly=is_monthly,
    # )

    # reader.enter_processing_stage(
    #     process_fn=configured_pipeline,
    #     chunks_in_memory=chunks_in_memory,
    #     context_months=context_months,
    #     skip_timeseries=True,
    #     skip_pixel_processing=False,        
    #     n_workers = n_workers,
    # )

    print(f"\n  Done – {target_year} outputs written to {output_dir}")


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    args = parse_args()
    print(args)
    run_phenometrics(
        data_dir          = args.data_dir,
        output_path       = args.output_path,
        tile              = args.tile,
        target_year       = args.target_year,
        # cadence           = args.cadence,
        # composite_method  = args.composite_method,
        context_months    = args.context_months,
        roi_file          = args.roi_file,
        tile_epsg         = args.tile_epsg,
        chunk_size        = args.chunk_size,
        chunks_in_memory  = args.chunks_in_memory,
        rebuild           = args.rebuild,
        run_label         = args.run_label,
        download_script   = args.download_script,
        evi_script        = args.evi_script,
        download_args     = args.download_args,
        evi_args          = args.evi_args,
        skip_download     = args.skip_download,
        skip_evi          = args.skip_evi,
        skip_phenometrics = args.skip_phenometrics,
        n_workers = args.n_workers,
    )


if __name__ == "__main__":
    main()

    #  python run_phenometrics.py --data_dir="~/Library/CloudStorage/OneDrive-NASA/NASA/HLS/data/" --output_path="~/Library/CloudStorage/OneDrive-NASA/NASA/HLS/results/" --tile=18SUJ --target_year=2020 --cadence=10day --skip_download --skip_evi --context_months=12

    #  python run_phenometrics.py --data_dir="/projects/my-public-bucket/hls/testing/10day/" --output_path="/projects/my-public-bucket/hls/testing/operational_phenometrics/10day_median/" --tile=18SUJ --target_year=2020 --cadence=10day --skip_download --skip_evi --context_months=12 --chunk_size=3600

    # python run_phenometrics.py --data_dir="/projects/my-public-bucket/hls/testing/10day-subset-SERC/" --output_path="/projects/my-public-bucket/hls/testing/operational_phenometrics/10day_subset_SERC/" --tile=18SUJ --target_year=2020 --skip_download --skip_evi --context_months=12 --chunk_size=100
