# from pathlib import Path
# from datetime import datetime, timedelta
# import re
# import xarray as xr
# import rioxarray as rxr
# import numpy as np
# import pandas as pd
# import gc
# from dataclasses import dataclass, field
# from typing import Generator
# import json
import time
from scipy.integrate import trapezoid
import os
import tempfile
# import time as timer
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore', message='invalid value encountered in cast')

from phenometrics_utils import *

from scipy.interpolate import LSQUnivariateSpline


def _precompute_knots(
    x:       np.ndarray,
    n_knots: int,
    k:       int = 5,
) -> np.ndarray | None:
    interior = np.unique(np.percentile(x, np.linspace(10, 90, n_knots)))
    interior = interior[(interior > x[0]) & (interior < x[-1])]
    return interior if len(interior) >= 3 else None


def _make_worker_slices(ny: int, n_workers: int) -> list[tuple[int, int]]:
    """ Takes n_rows of data and n_workers and calculates slice coords for each worker """
    base, extra = divmod(ny, n_workers)
    slices, start = [], 0
    for i in range(n_workers):
        end = start + base + (1 if i < extra else 0)
        if start < end:
            slices.append((start, end))
        start = end
    return slices


def _process_worker_slice(
    evi_mmap_path:      str,
    evi_shape:          tuple,          # (n_times, ny, nx)
    # row range owned by this worker
    row_start:          int,
    row_end:            int,
    
    t_nominal:          np.ndarray,     # (n_times,) shared time axis
    weights_template:   np.ndarray,     # (n_times,) Gaussian-decay weights
    t_daily:            np.ndarray,     # (n_output,) evaluation points
    precomputed_knots:  np.ndarray | None,
    min_valid_points:   int,
    value_min:          float,
    value_max:          float,
    fill_low_data:      str,
    k:                  int,
    n_output:           int,
) -> tuple[int, int, np.ndarray]:

    evi_data  = np.memmap(evi_mmap_path,  dtype=np.float32,
                          mode="r", shape=evi_shape)

    n_rows = row_end - row_start
    nx     = evi_shape[2]
    result = np.full((n_output, n_rows, nx), np.nan, dtype=np.float32)

    # for each local row 
    for local_yi, yi in enumerate(range(row_start, row_end)):
        # process all x pixels in row yi
        for xi in range(nx):
            
            # 1. Extract single x,y pixel time series
            ts      = evi_data[:, yi, xi]
            t_pixel = t_nominal
            valid   = np.isfinite(ts) 
            n_valid = valid.sum()

            # 2. Low-data handling (ToDo: add in fill logic) 
            if n_valid < min_valid_points:
                if fill_low_data == "mean" and n_valid > 0:
                    result[:, local_yi, xi] = np.nanmean(ts[valid])
                continue   # leave as NaN for default

            x_valid = t_pixel[valid]
            y_valid = ts[valid].astype(np.float64)
            w_valid = weights_template[valid].copy()

            # 3. Monotonicity check + deduplication
            if not np.all(np.diff(x_valid) > 1e-6):
                idx     = np.argsort(x_valid, kind="stable")
                x_valid = x_valid[idx]
                y_valid = y_valid[idx]
                w_valid = w_valid[idx]
                # Remove duplicate x positions (LSQ requires strictly increasing)
                keep    = np.concatenate([[True], np.diff(x_valid) > 1e-6])
                x_valid = x_valid[keep]
                y_valid = y_valid[keep]
                w_valid = w_valid[keep]

            if len(x_valid) < min_valid_points:
                continue

            # 4. Up-weight EVI extremes to allow spline to capture peaks/troughs
            y_range = y_valid.max() - y_valid.min()
            if y_range > 0.1:
                lo = y_valid.min() + 0.20 * y_range
                hi = y_valid.min() + 0.80 * y_range
                w_valid[y_valid < lo] *= 2.0
                w_valid[y_valid > hi] *= 2.0

            # 5. Knots: trim the precomputed set to this pixel's range
            if precomputed_knots is not None:
                # test precompute: filter dates to the precomputed range
                interior = precomputed_knots[
                    (precomputed_knots > x_valid[0]) &
                    (precomputed_knots < x_valid[-1])
                ]
            
                # Check knot density vs valid observations
                # If fewer than 3 valid obs per knot interval, recompute per-pixel
                obs_per_knot = len(x_valid) / max(len(interior), 1)
                if obs_per_knot < 3 or len(interior) < 3:
                    # Fall back to per-pixel knots based on actual valid observations
                    n_dates     = len(x_valid)
                    n_knots      = min(max(n_dates // 5, 6), n_dates - k - 1)
                    pct      = np.linspace(10, 90, n_knots)
                    interior = np.unique(np.percentile(x_valid, pct))
                    interior = interior[(interior > x_valid[0]) &
                                        (interior < x_valid[-1])]
            else:
                n_dates     = len(x_valid)
                n_knots      = min(max(n_dates // 5, 6), n_dates - k - 1)
                pct      = np.linspace(10, 90, n_knots)
                interior = np.unique(np.percentile(x_valid, pct))
                interior = interior[(interior > x_valid[0]) &
                                    (interior < x_valid[-1])]

            if len(interior) < 3:
                continue

            # 6. Fit spline to full context observation dates and then evaluate on daily ts
            try:
                spl = LSQUnivariateSpline(x_valid, y_valid, interior, w=w_valid, k=k)
                # evaluate and make sure predictions are clamped to EVI range for all dates at this y,x
                result[:, local_yi, xi] = np.clip(spl(t_daily), value_min, value_max).astype(np.float32)
            except Exception:
                if fill_low_data == "mean":
                    result[:, local_yi, xi] = float(np.nanmean(y_valid))

    return row_start, row_end, result

def smooth_evi_chunk_for_year(
    chunk:                xr.DataArray,
    target_year:          int,
    # --- algorithm config ---
    context_months:       int   = 12,
    min_valid_points:     int   = 30,
    min_valid_frac:       float = 0.30,
    fill_low_data:        str   = "nan",  # currently no gap filling, Bolton uses the context years to "grab" similar values but that's a weaker method 
    value_min:            float = -1.0,
    value_max:            float = 1.0,
    daily_output:         bool  = True,
    k:                    int   = 5,      # spline degree, 4 = cubic
    testing_mode:         bool  = False,
    _pool:                Parallel | None = None,  # warm pool from caller
    n_jobs=-1,
) -> xr.DataArray:
    """
    Fit a pixel-wise LSQ smoothing spline over a ±context_months window
    around target_year, returning daily smoothed EVI for target_year only.

    Parameters
    ----------
    chunk                : (time, y, x) EVI DataArray covering at least
                           target_year ± context_months of data.
    target_year          : Year to produce output for.
    context_months       : Months of data on each side used for fitting.
    min_valid_points     : Pixels with fewer finite observations are skipped.
    fill_low_data        : "nan" — leave skipped pixels as NaN.
                           "mean" — fill with the pixel's temporal mean.
    value_min/max        : Output clipping bounds.
    k                    : Spline degree (5 recommended for EVI phenology).
    doy_data             : Optional (time, y, x) actual-DOY DataArray for
                           10-day composites. When provided with
                           composite_start_doys, each pixel gets its own
                           time axis derived from actual observation DOYs.
    composite_start_doys : 1-D array of composite-period start DOYs aligned
                           to chunk.time. Required when doy_data is provided.
    testing_mode         : If True, output spans the full fitting window
                           instead of target_year only (used for QC plots).
    _pool                : Pre-warmed joblib.Parallel instance. Pass this in
                           from process_all_chunks_yearly so the loky pool
                           startup cost is paid once per run, not per chunk.

    Returns
    -------
    xr.DataArray : (time, y, x) daily smoothed EVI.
                   time = 365 days of target_year (or full context window if testing_mode).
    """
    t_start   = time.time()
    if _pool is not None and hasattr(_pool, 'n_jobs'):
        pool_workers = _pool.n_jobs
        n_workers    = os.cpu_count() if pool_workers == -1 else max(1, pool_workers)
        print(f"  Workers   : {n_workers} (from warm pool)")
    else:
        n_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        print(f"  Workers   : {n_workers} (local pool)")

    # ----------------------------------------------------------------
    # 1. Temporal subset — restrict to context window (should be this window incoming)
    # ----------------------------------------------------------------
    fit_start = (pd.Timestamp(f"{target_year}-01-01")
                 - pd.DateOffset(months=context_months))
    fit_end   = (pd.Timestamp(f"{target_year}-12-31")
                 + pd.DateOffset(months=context_months))

    fit_chunk = chunk.sel(time=slice(fit_start, fit_end))
    if fit_chunk.sizes["time"] == 0:
        raise ValueError(
            f"No data in fitting window {fit_start.date()} – {fit_end.date()}. "
            f"Check that target_year={target_year} is within the loaded data range."
        )

    # ----------------------------------------------------------------
    # 2. Drop entirely-NaN timesteps
    # ----------------------------------------------------------------
    valid_ts  = np.any(np.isfinite(fit_chunk.values), axis=(1, 2))
    n_dropped = int((~valid_ts).sum())
    if n_dropped > 0:
        fit_chunk = fit_chunk.isel(time=valid_ts)

    # dim size of data (n timesteps, y pixel cnt, x pixel cnt)
    n_times, ny, nx = fit_chunk.shape
    n_pixels        = ny * nx

    # ----------------------------------------------------------------
    # 3. Calc effective min_valid_points
    #     Hard floor  : k+1 (minimum for a degree-k spline)
    #     Hard ceiling: n_times (can't require more points than time steps exist)
    # ----------------------------------------------------------------        
    K_FLOOR     = k + 1                                    # e.g. 6 for k=5
    frac_floor  = int(np.ceil(n_times * min_valid_frac))   # e.g. 11 from 35×0.3

    effective_min_valid = min(
        max(K_FLOOR, frac_floor),    # adaptive floor when large amount of data
        min_valid_points,            # args ceiling — lowers threshold if < floor
        n_times,                     # hard ceiling - don't overfit
    )

    print(f"  min_valid : {effective_min_valid} "
          f"(k+1={K_FLOOR}, "
          f"{min_valid_frac*100:.0f}%×{n_times}={frac_floor}, "
          f"user={min_valid_points}) "
          f"  effective={effective_min_valid}")

    print(f"  Chunk     : {ny}×{nx} = {n_pixels:,} pixels | "
          f"{n_times} timesteps ({n_dropped} all-NaN dropped) | "
          f"min_valid={effective_min_valid}")
        
    # ----------------------------------------------------------------
    # 4. Nominal time axis
    #    Days since (target_year-1)-01-01 — keeps values in a sensible
    #    range for spline fit regardless of year in context window
    # ----------------------------------------------------------------
    ref_date  = np.datetime64(f"{target_year - 1}-01-01")
    ref_year  = target_year - 1

    # drop any all-NaN timesteps (esp case for tiles near coasts)
    valid_mask = ~np.isnan(chunk.values).all(axis=(1, 2))
    chunk_valid = chunk.isel(time=valid_mask)
    
    t_nominal = ((fit_chunk.time.values - ref_date)
                 / np.timedelta64(1, "D")).astype(np.float64)

    if len(t_nominal) == 0:
        print(f"   WARNING: 0 valid timesteps for {target_year} current chunk;"
              f"     chunk is entirely masked. Returning NaN output.")
        daily_times = pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31", freq="D")        
        nan_data = np.full(
            (len(daily_times), chunk.shape[1], chunk.shape[2]),
            np.nan,
            dtype=np.float32,
        )
        return xr.DataArray(
            nan_data,
            dims=["time", "y", "x"],
            coords={"time": daily_times, "y": chunk.y, "x": chunk.x,},
        )
    
    # ----------------------------------------------------------------
    # 5. Output time axis - infill days so EVI2 is continuous across DOY of target-year
    # ----------------------------------------------------------------
    daily_dates = (pd.date_range(fit_start, fit_end)
                   if testing_mode
                   else pd.date_range(f"{target_year}-01-01",
                                      f"{target_year}-12-31"))
    t_daily  = ((daily_dates.values - ref_date)
                / np.timedelta64(1, "D")).astype(np.float64)
    n_output = len(daily_dates)
    t_daily = np.clip(t_daily, t_nominal[0], t_nominal[-1])
    # print(f"t_nominal: {t_nominal[0]:.1f} - {t_nominal[-1]:.1f}", flush=True)
    # print(f"t_daily:   {t_daily[0]:.1f}  - {t_daily[-1]:.1f}", flush=True)
    # print(f"overlap:   {t_daily[0] >= t_nominal[0]} to {t_daily[-1] <= t_nominal[-1]}", flush=True)
    
    # ----------------------------------------------------------------
    # 6. Gaussian-decay weight template
    #    Observations near the centre of target_year get full weight;
    #    context observations are downweighted by distance
    # ----------------------------------------------------------------
    target_center    = float(
        (np.datetime64(f"{target_year}-07-01") - ref_date)
        / np.timedelta64(1, "D")
    )
    days_from_center = np.abs(t_nominal - target_center)
    weights_template = np.exp(-0.25 * days_from_center / 365) * 0.85 + 0.15

    # ----------------------------------------------------------------
    # 7. Knot precomputation
    #    compute once from the full t_nominal vector,
    #    then trim to each pixel's valid range inside the worker.
    #    This is annoying but allows for slightly faster compute.
    # ----------------------------------------------------------------
    n_dates  = len(t_nominal)
    n_knots   = min(max(n_dates // 3, 12), # exceed n_months
                    n_dates - k - 1)       # don't allow to exceed dof
    precomputed_knots = _precompute_knots(t_nominal, n_knots, k)
    if precomputed_knots is None:
        raise ValueError("Could not compute interior knots from time axis — "
                         "check that n_times is sufficient.")
    print(f"  Time axis : shared | {n_knots} knots precomputed once "
          f" {len(precomputed_knots)} interior after trim")

    print(f"  Workers   : {n_workers} processes | "
          f"Output: {n_output} days")
    
    # ----------------------------------------------------------------
    # 8. Write memmap temp files for distributed processing
    #    Parent writes once - workers read zero-copy via OS page mapping.
    #    TemporaryDirectory cleans up automatically on exit.
    # ----------------------------------------------------------------
    evi_shape    = (n_times, ny, nx)
    smoothed_out = np.full((n_output, ny, nx), np.nan, dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="smooth_evi_") as tmpdir:

        # EVI memmap
        evi_path = str(Path(tmpdir) / "evi.mmap")
        mm       = np.memmap(evi_path, dtype=np.float32, mode="w+", shape=evi_shape)
        mm[:]    = fit_chunk.values.astype(np.float32)
        mm.flush(); del mm

        # ----------------------------------------------------------------
        # 10. Dispatch
        #     One task per worker, each covering ~ny/n_workers rows of data.
        #     Use a warm pool if provided, otherwise create a local one.
        # ----------------------------------------------------------------
        worker_slices = _make_worker_slices(ny, n_workers)
        assert len(worker_slices) == n_workers, (
            f"Slice count {len(worker_slices)} != worker count {n_workers} — "
            f"check n_jobs/pool alignment"
        )
        
        print(f"  Slices    : {len(worker_slices)} × ~{ny // n_workers} rows each")
        t_dispatch = time.time()

        # Shared kwargs — same for every worker
        worker_kwargs = dict(
            evi_mmap_path    = evi_path,
            evi_shape        = evi_shape,
            t_nominal        = t_nominal,
            weights_template = weights_template,
            t_daily          = t_daily,
            precomputed_knots= precomputed_knots,
            min_valid_points = effective_min_valid,
            value_min        = value_min,
            value_max        = value_max,
            fill_low_data    = fill_low_data,
            k                = k,
            n_output         = n_output,
        )

        executor = _pool or Parallel(
            n_jobs=n_workers, prefer="processes", batch_size="auto"
        )

        # use precomputed row-wise worker slices to distribute with kwargs to workers
        results = executor(
            delayed(_process_worker_slice)(
                row_start=s, row_end=e, **worker_kwargs
            )
            for s, e in worker_slices
        )

        # ----------------------------------------------------------------
        # 11. Reassemble
        # ----------------------------------------------------------------
        n_fitted = n_skipped = 0
        for row_start, row_end, row_result in results:
            smoothed_out[:, row_start:row_end, :] = row_result
            finite_mask = np.any(np.isfinite(row_result), axis=0)   # (n_rows, nx)
            n_fitted  += int(finite_mask.sum())
            n_skipped += int((~finite_mask).sum())

    # ----------------------------------------------------------------
    # 12. Timing summary
    # ----------------------------------------------------------------
    t_total    = time.time() - t_start
    t_compute  = time.time() - t_dispatch
    rate       = n_pixels / max(t_compute, 1e-6)
    print(f"  Done      : {n_fitted:,} fitted | {n_skipped:,} skipped | "
          f"{t_total:.1f}s total | {rate:,.0f} px/s")

    gc.collect()

    # ----------------------------------------------------------------
    # 13. Return as xr.DataArray
    # ----------------------------------------------------------------
    return xr.DataArray(
        smoothed_out,
        dims=["time", "y", "x"],
        coords={
            "time": daily_dates,
            "y":    fit_chunk.y,
            "x":    fit_chunk.x,
        },
    )


def apply_thresholds_chunk(chunk: xr.DataArray,
                           min_val: float = 0.1,
                           max_val: float = 0.95) -> xr.DataArray:
    """Apply min/max thresholds to chunk."""
    return chunk.where((chunk >= min_val) & (chunk <= max_val))


# per Bolton et al., 2020 eq.3 pg4
def despike_timeseries_chunk(
        chunk: xr.DataArray,
        max_gap_days: int = 45,
        abs_threshold: float = 0.1,
        rel_threshold: float = 2.0,
        handle_edges: bool = True,
) -> xr.DataArray:
    """
    Three-point de-spiking with optional per-pixel DOY awareness.

    Args:
        chunk:                DataArray (time, y, x) of EVI values
        doy_data:             DataArray (time, y, x) of DOY offsets within composite
        composite_start_doys: Array of composite start DOY per timestep
        max_gap_days:         Max gap between pre/post for despiking
        abs_threshold:        Absolute difference threshold
        rel_threshold:        Relative difference threshold
        handle_edges:         Check first/last observations for spikes
    """
    n_times = len(chunk.time)
    chunk_values = chunk.values  # (time, y, x)

    times = pd.to_datetime(chunk.time.values)
    if len(times) == 0:
        return chunk        
    nominal_days = (times - times[0]).days.astype(np.float32)

    # Spike detection
    spike_mask = np.zeros_like(chunk_values, dtype=bool)

    # nominal time gaps (same for all pixels when DOY.tif is NaN)
    time_days_da = xr.DataArray(nominal_days, dims=['time'],
                                coords={'time': chunk.time})

    evi_pre = chunk.shift(time=1)
    evi_post = chunk.shift(time=-1)
    time_pre = time_days_da.shift(time=1)
    time_post = time_days_da.shift(time=-1)

    gap = time_post - time_pre
    weight = (time_days_da - time_pre) / (time_post - time_pre)
    evi_fit = evi_pre + (evi_post - evi_pre) * weight

    amplitude = evi_post - evi_pre
    diff = evi_fit - chunk
    abs_diff = np.abs(diff)
    rel_diff = np.abs(diff / amplitude.where(np.abs(amplitude) > 0.001))

    spike_da = (
            (abs_diff > abs_threshold)
            & (rel_diff > rel_threshold)
            & (gap < max_gap_days)
            & (~evi_pre.isnull())
            & (~evi_post.isnull())
    )
    spike_mask = spike_da.values

    if handle_edges and n_times >= 2:
        t_gap_first = nominal_days[1] - nominal_days[0]
        if t_gap_first < max_gap_days:
            diff_first = np.abs(chunk_values[0] - chunk_values[1])
            spike_mask[0] = (
                    (diff_first > abs_threshold * 1.5)
                    & (~np.isnan(chunk_values[0]))
                    & (~np.isnan(chunk_values[1]))
            )

        t_gap_last = nominal_days[-1] - nominal_days[-2]
        if t_gap_last < max_gap_days:
            diff_last = np.abs(chunk_values[-1] - chunk_values[-2])
            spike_mask[-1] = (
                    (diff_last > abs_threshold * 1.5)
                    & (~np.isnan(chunk_values[-1]))
                    & (~np.isnan(chunk_values[-2]))
            )

    chunk_despiked = chunk.where(~spike_mask)

    n_spikes = int(spike_mask.sum())
    n_total = int((~chunk.isnull()).sum())
    if n_spikes > 0:
        pct = 100 * n_spikes / n_total if n_total > 0 else 0
        print(f"  De-spiking: removed {n_spikes} spikes ({pct:.2f}%) [nominal gaps]")

    return chunk_despiked

def compute_scene_quality_metrics(
    chunk: xr.DataArray,
    target_year: int,
) -> tuple[np.ndarray, np.ndarray]:

    chunk_target_year = chunk.where(chunk.time.dt.year == target_year)
    values = chunk_target_year.values                                   
    doys   = chunk_target_year.time.dt.dayofyear.values.astype(np.float32) 

    valid_mask  = ~np.isnan(values)                        
    valid_count = valid_mask.sum(axis=0).astype(np.float32) 

    # Mask invalid timesteps and compute per-pixel DOY range
    doys_3d    = np.broadcast_to(doys[:, None, None], values.shape)
    doys_valid = np.where(valid_mask, doys_3d, np.nan) 

    doy_max = np.nanmax(doys_valid, axis=0)
    doy_min = np.nanmin(doys_valid, axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_revisit = np.select(
            condlist=[
                valid_count > 1,    # normal case
                valid_count == 1,   # single observation → fill with 1
            ],
            choicelist=[
                (doy_max - doy_min) / (valid_count - 1),
                np.ones_like(valid_count),
            ],
            default=np.nan,         # 0 valid observations
        ).astype(np.float32)

    quality_pixels = np.where(
        valid_count > 0, valid_count, np.nan
    ).astype(np.float32)

    return mean_revisit, quality_pixels
    

def annual_phenometrics_chunk(chunk: xr.DataArray,
                              year: int = None,
                              threshold_greenup_pct: float = 0.15) -> dict[str, np.ndarray]:
    """
    Calculate annual phenometrics for a chunk.

    Args:
        chunk: DataArray (time, y, x) - should span multiple years
        doy_data: Optional DataArray (time, y, x) of actual observation DOY
                  (for composites where DOY varies per pixel)
        year: Specific year to process (None = all years in data)
        threshold_greenup_pct: Percentage of amplitude for greenup/dormancy thresholds (default 15%)
        composite_start_doys: Array of start DOY for each time step (for 10day composites)

    Returns:
        Dict with 3D arrays (year, y, x) for each metric
    """

    ny, nx = chunk.shape[1], chunk.shape[2]
    n_years = 1  # n years

    # Initialize output phenometric arrays
    annual_mean = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    annual_max = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    annual_max_doy = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    annual_min = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    annual_min_doy = np.full((n_years, ny, nx), np.nan, dtype=np.float32)

    greenup_evi = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    greenup_doy = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    greenup_threshold = np.full((n_years, ny, nx), np.nan, dtype=np.float32)

    dormancy_evi = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    dormancy_doy = np.full((n_years, ny, nx), np.nan, dtype=np.float32)

    annual_amplitude = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    growing_season_length = np.full((n_years, ny, nx), np.nan, dtype=np.float32)

    auc_full = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    auc_net = np.full((n_years, ny, nx), np.nan, dtype=np.float32)

    greenup_rate = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    greenup_rate_doy = np.full((n_years, ny, nx), np.nan, dtype=np.float32)

    senescence_rate = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    senescence_rate_doy = np.full((n_years, ny, nx), np.nan, dtype=np.float32)
    
    year_evi = chunk
    if len(year_evi.time) == 0:
        return None

    nominal_doys = year_evi.time.dt.dayofyear.values

    # 1. MEAN
    annual_mean[0] = year_evi.mean(dim='time').values

    # 2. MAX
    year_max = year_evi.max(dim='time').values
    annual_max[0] = year_max

    # 3. MIN
    year_min = year_evi.min(dim='time').values
    annual_min[0] = year_min

    # 4. AMPLITUDE
    amplitude = year_max - year_min
    annual_amplitude[0] = amplitude

    # 5. Max DOY
    fill_value = -9999
    year_filled = year_evi.fillna(fill_value)
    max_indices = year_filled.argmax(dim='time').values  # (y, x)

    # 6. Min DOY
    min_fill_value = 9999
    year_min_filled = year_evi.fillna(min_fill_value)
    min_indices = year_min_filled.argmin(dim='time').values

    # prep annual values and mask
    year_evi_values = year_evi.values  # (time, y, x)
    all_nan_mask = year_evi.isnull().all(dim='time').values  # (y, x)

    for yi in range(ny):
        for xi in range(nx):
            if not all_nan_mask[yi, xi]:
                max_idx = max_indices[yi, xi]
                annual_max_doy[0, yi, xi] = nominal_doys[max_idx]
                min_idx = min_indices[yi, xi]
                annual_min_doy[0, yi, xi] = nominal_doys[min_idx]

    # 7. GREENUP, 8. DORMANCY, 9. AUC full, 10. AUC net, 11. GREENUP Inflection, 12. Senescences Inflection
    threshold = year_min + (amplitude * threshold_greenup_pct)
    greenup_threshold[0] = threshold
    for yi in range(ny):
        for xi in range(nx):
            if all_nan_mask[yi, xi]:
                continue

            pixel_evi = year_evi_values[:, yi, xi]
            pixel_max_doy = annual_max_doy[0, yi, xi]
            pixel_threshold = threshold[yi, xi]

            if np.isnan(pixel_max_doy) or np.isnan(pixel_threshold):
                continue

            pixel_doys = nominal_doys.astype(float)

            valid_pixel = ~np.isnan(pixel_evi) & ~np.isnan(pixel_doys)

            if valid_pixel.sum() < 3:
                continue

            pixel_evi_valid = pixel_evi[valid_pixel]
            pixel_doys_valid = pixel_doys[valid_pixel]

            # Greenup: first DOY exceeding threshold BEFORE peak
            pre_peak_mask = pixel_doys < pixel_max_doy
            if pre_peak_mask.sum() > 0:
                pre_peak_evi = pixel_evi[pre_peak_mask]
                pre_peak_doys = pixel_doys[pre_peak_mask]

                sort_idx = np.argsort(pre_peak_doys)
                pre_peak_evi = pre_peak_evi[sort_idx]
                pre_peak_doys = pre_peak_doys[sort_idx]

                valid = ~np.isnan(pre_peak_evi)
                if valid.sum() > 0:
                    pre_peak_evi = pre_peak_evi[valid]
                    pre_peak_doys = pre_peak_doys[valid]

                    above_thresh = pre_peak_evi >= pixel_threshold
                    if above_thresh.any():
                        first_above_idx = np.argmax(above_thresh)
                        greenup_doy[0, yi, xi] = pre_peak_doys[first_above_idx]
                        greenup_evi[0, yi, xi] = pre_peak_evi[first_above_idx]

            # Dormancy: first DOY below threshold AFTER peak
            post_peak_mask = pixel_doys > pixel_max_doy
            if post_peak_mask.sum() > 0:
                post_peak_evi = pixel_evi[post_peak_mask]
                post_peak_doys = pixel_doys[post_peak_mask]

                sort_idx = np.argsort(post_peak_doys)
                post_peak_evi = post_peak_evi[sort_idx]
                post_peak_doys = post_peak_doys[sort_idx]

                valid = ~np.isnan(post_peak_evi)
                if valid.sum() > 0:
                    post_peak_evi = post_peak_evi[valid]
                    post_peak_doys = post_peak_doys[valid]

                    below_thresh = post_peak_evi <= pixel_threshold
                    if below_thresh.any():
                        first_below_idx = np.argmax(below_thresh)
                        dormancy_doy[0, yi, xi] = post_peak_doys[first_below_idx]
                        dormancy_evi[0, yi, xi] = post_peak_evi[first_below_idx]

            # 11. AUC FULL and 12. AUC NET
            pix_greenup = greenup_doy[0, yi, xi]
            pix_dormancy = dormancy_doy[0, yi, xi]
            pix_min = year_min[yi, xi]

            if not np.isnan(pix_greenup) and not np.isnan(pix_dormancy):
                gs_mask = (pixel_doys_valid >= pix_greenup) & (pixel_doys_valid <= pix_dormancy)

                if gs_mask.sum() >= 3:
                    gs_doy = pixel_doys_valid[gs_mask]
                    gs_evi = pixel_evi_valid[gs_mask]

                    gs_valid = ~np.isnan(gs_evi)
                    if gs_valid.sum() >= 3:
                        gs_doy = gs_doy[gs_valid]
                        gs_evi = gs_evi[gs_valid]

                        # AUC Full: total area under curve from greenup to dormancy
                        auc_full[0, yi, xi] = trapezoid(gs_evi, gs_doy)

                        # AUC Net: area above the minimum baseline
                        gs_evi_above_min = gs_evi - pix_min
                        auc_net[0, yi, xi] = trapezoid(gs_evi_above_min, gs_doy)

            # 13 & 14. INFLECTION POINTS (steepest greenup and senescence)
            if len(pixel_doys_valid) >= 4:
                evi_derivative = np.gradient(pixel_evi_valid, pixel_doys_valid)

                # Steepest greenup: max positive derivative before peak
                pre_peak = pixel_doys_valid < pixel_max_doy
                if pre_peak.sum() >= 2:
                    pre_derivs = evi_derivative[pre_peak]
                    pre_doys = pixel_doys_valid[pre_peak]

                    max_rate_idx = np.argmax(pre_derivs)
                    greenup_rate[0, yi, xi] = pre_derivs[max_rate_idx]
                    greenup_rate_doy[0, yi, xi] = pre_doys[max_rate_idx]

                # Steepest senescence: max negative derivative after peak
                post_peak = pixel_doys_valid > pixel_max_doy
                if post_peak.sum() >= 2:
                    post_derivs = evi_derivative[post_peak]
                    post_doys = pixel_doys_valid[post_peak]

                    min_rate_idx = np.argmin(post_derivs)
                    senescence_rate[0, yi, xi] = post_derivs[min_rate_idx]
                    senescence_rate_doy[0, yi, xi] = post_doys[min_rate_idx]

    # 15. Growing season length
    valid_both = ~np.isnan(greenup_doy[0]) & ~np.isnan(dormancy_doy[0])
    growing_season_length[0, valid_both] = dormancy_doy[0, valid_both] - greenup_doy[0, valid_both]

    return {
        'annual_mean': annual_mean,
        'annual_max': annual_max,
        'annual_min': annual_min,
        'annual_max_doy': annual_max_doy,
        'annual_amplitude': annual_amplitude,
        'greenup_doy': greenup_doy,
        'dormancy_doy': dormancy_doy,
        'growing_season_length': growing_season_length,

        'annual_min_doy': annual_min_doy,
        'greenup_evi': greenup_evi,
        'dormancy_evi': dormancy_evi,
        'greenup_threshold': greenup_threshold,

        'auc_full': auc_full,
        'auc_net': auc_net,
        'greenup_rate': greenup_rate,
        'greenup_rate_doy': greenup_rate_doy,
        'senescence_rate': senescence_rate,
        'senescence_rate_doy': senescence_rate_doy,
    }


def full_pipeline_chunk(chunk: xr.DataArray,
                        doy_data: xr.DataArray = None,
                        apply_threshold: bool = True,
                        min_evi_threshold: float = -1.0,
                        max_evi_threshold: float = 1.0,
                        threshold_greenup_pct: float = 0.15,
                        despike: bool = True,
                        despike_max_gap: int = 45,
                        despike_abs_threshold: float = 0.1,
                        despike_rel_threshold: float = 2.0,
                        target_year: int = None,
                        testing_mode: bool = False,
                        _pool = None,
                        n_jobs:int = -1,
                        **kwargs) -> dict[str, np.ndarray]:
    """
    Full processing pipeline for a chunk.

    Pipeline:
        1. Apply EVI thresholds: ensures any anomalous EVI values are clipped
        2. TBD Positive/bright pixel filtering
        3. De-spike (three-point method): removes
        4. Interpolate gaps
        5. Calculate annual phenometrics

    Args:
        chunk: Raw EVI DataArray (time, y, x)
        doy_data: Optional DOY offset DataArray for 10day composites
        apply_threshold: Whether to threshold EVI values
        min_evi_threshold, max_evi_threshold: EVI thresholds
        interp_method: Interpolation method
        threshold_pct: Percentage of amplitude for greenup/dormancy
        despike: Whether to apply de-spiking
        despike_max_gap: Max gap (days) between pre/post for despiking
        despike_abs_threshold: Absolute difference threshold for spike detection
        despike_rel_threshold: Relative difference threshold for spike detection
        target_year: current year beign processed
        _pool: warm pool cpus

    Returns:
        Dict with 2D arrays for each metric-year combination
    """
    metric_mapping = {
        'annual_mean': 'mean_evi',
        'annual_max': 'max_evi',
        'annual_min': 'min_evi',
        'annual_max_doy': 'max_doy',
        'annual_amplitude': 'amplitude',
        'greenup_doy': 'greenup_doy',
        'dormancy_doy': 'dormancy_doy',
        'growing_season_length': 'growing_season_length',
        'annual_min_doy': 'min_doy',
        'greenup_evi': 'greenup_evi',
        'dormancy_evi': 'dormancy_evi',
        'greenup_threshold': 'greenup_threshold',
        'auc_full': 'auc_full',
        'auc_net': 'auc_net',
        'greenup_rate': 'greenup_rate',
        'greenup_rate_doy': 'greenup_rate_doy',
        'senescence_rate': 'senescence_rate',
        'senescence_rate_doy': 'senescence_rate_doy'
        # 'mean_revisit_time': 'mean_revisit_time',
        # 'quality_pixel_cnt': 'quality_pixel_cnt'
    }
    
    chunk_original = chunk.copy(deep=True) if testing_mode else None

    # Step 1: Threshold
    print("Step1: Thresholding")
    if apply_threshold:
        chunk = apply_thresholds_chunk(
            chunk,
            min_evi_threshold,
            max_evi_threshold
        )

    chunk_post_threshold = chunk.copy(deep=True) if testing_mode else None

    # TODO Step: Positive/Bright pixel filtering (blue and red bands)
    
    # Step 2: Negative pixel filtering using DOY (EVI2 despiking - cloud shadows)
    # - uses target year +/- 1 year, if edge case remove the non-existing year
    if despike:
        print("Step2: Despiking")
        chunk = despike_timeseries_chunk(
            chunk,
            max_gap_days=despike_max_gap,
            abs_threshold=despike_abs_threshold,
            rel_threshold=despike_rel_threshold,
        )
    chunk_post_despike = chunk.copy(deep=True) if testing_mode else None

    # calculate scene revisit and quality pixels before the spline fit, 365 DOY data is generated
    print("Step3: Scene quality metrics")
    scene_mean_revisit, scene_quality_pixels = compute_scene_quality_metrics(chunk, target_year)

    valid_timesteps = (~np.isnan(chunk.values)).any(axis=(1, 2)).sum()
    if valid_timesteps == 0:
        print(f"  WARNING: chunk has 0 valid timesteps for {target_year}. All metrics will be NaN — skipping spline and phenometrics.")
        return {
            f'{name}_{target_year}': np.full((chunk.shape[1], chunk.shape[2]), np.nan, dtype=np.float32)
                for name in metric_mapping.values()
        } | {
            f'mean_revisit_time_{target_year}': scene_mean_revisit,
            f'quality_pixel_cnt_{target_year}': scene_quality_pixels,
        }
        
    # Step 4: apply penalized cubic spline interpolation
    print("Step4: Apply spline")
    smoothed_daily = smooth_evi_chunk_for_year(
        chunk,
        target_year,
        testing_mode=testing_mode,
        _pool=_pool,
        n_jobs=n_jobs
    )
    chunk_post_spline = smoothed_daily.copy(deep=True) if testing_mode else None

    # Step 5: Annual phenometrics
    smoothed_year = smoothed_daily.where(smoothed_daily.time.dt.year == target_year)
    print("  Step5: Calculate phenometrics")
    pheno = annual_phenometrics_chunk(
        smoothed_year,
        threshold_greenup_pct=threshold_greenup_pct,
        year=target_year,
    )
    results = {}
    for internal_name, output_name in metric_mapping.items():
        results[f'{output_name}_{target_year}'] = pheno[internal_name][0]
        
    results[f'mean_revisit_time_{target_year}'] = scene_mean_revisit
    results[f'quality_pixel_cnt_{target_year}'] = scene_quality_pixels

    if testing_mode:
        results['_intermediate'] = {
            'original': chunk_original,
            'post_threshold': chunk_post_threshold,
            'post_despike': chunk_post_despike,
            'post_spline': chunk_post_spline,
        }

    return results