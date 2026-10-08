import time
from scipy.integrate import trapezoid
import os
import tempfile
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore', message='invalid value encountered in cast')

from phenometrics_utils import *
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import find_peaks
from phenometric_gap_fill import *
import bottleneck


##############################################
##### Spline fitting funcitons and helpers ###
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
    min_valid_points:   int,
    value_min:          float,
    value_max:          float,
    k:                  int,
    n_output:           int,
    use_context_months: bool,
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

            # 2. Low-data handling
            if n_valid < min_valid_points:
                continue

            x_valid = t_pixel[valid]
            y_valid = ts[valid].astype(np.float64)
            w_valid = weights_template[valid].copy()

            # 3. Monotonicity check + deduplication
            if not np.all(np.diff(x_valid) > 1e-6):
                idx     = np.argsort(x_valid, kind="stable")
                x_valid = x_valid[idx]
                y_valid = y_valid[idx]
                w_valid = w_valid[idx]
                # Remove duplicate obs dates (LSQ requires strictly increasing)
                keep    = np.concatenate([[True], np.diff(x_valid) > 1e-6])
                x_valid = x_valid[keep]
                y_valid = y_valid[keep]
                w_valid = w_valid[keep]

            if len(x_valid) < min_valid_points:
                continue

            # 4. Up-weight EVI extremes to allow spline to capture peaks/troughs
            y_range = y_valid.max() - y_valid.min()
            if y_range > 0.1 and use_context_months:
                lo = y_valid.min() + 0.20 * y_range
                hi = y_valid.min() + 0.80 * y_range
                w_valid[y_valid < lo] *= 2.0
                w_valid[y_valid > hi] *= 2.0

            # 5. Knots: trim the precomputed time series to this pixel's range
            x_range  = x_valid[-1] - x_valid[0]            
            if use_context_months:
                n_knots  = min(max(len(x_valid) // 3, 12), len(x_valid) - k - 1)
                interior = np.unique(
                    np.percentile(x_valid, np.linspace(10, 90, n_knots))
                )
                interior = interior[
                    (interior > x_valid[0]) & (interior < x_valid[-1])
                ]
                if len(interior) > 1:
                    keep     = np.concatenate([[True], np.diff(interior) >= 10.0])
                    interior = interior[keep]
                if len(interior) < 3:
                    continue

            # 6. Fit spline to full context observation dates and then evaluate on daily ts
            if not use_context_months:
                peak_idx    = np.argmax(y_valid)
                peak_t      = x_valid[peak_idx]
                pre_peak_t  = x_valid[0] + (peak_t - x_valid[0]) * 0.5
                post_peak_t = peak_t     + (x_valid[-1] - peak_t) * 0.5
                interior    = np.array([pre_peak_t, peak_t, post_peak_t])
                interior    = interior[
                    (interior > x_valid[0] + 1) & (interior < x_valid[-1] - 1)
                ]
                if len(interior) < 2:
                    continue
                spl = LSQUnivariateSpline(x_valid, y_valid, interior, w=w_valid, k=3)
            else:
                spl = LSQUnivariateSpline(x_valid, y_valid, interior, w=w_valid, k=k)

            result[:, local_yi, xi] = np.clip(
                spl(t_daily), value_min, value_max
            ).astype(np.float32)

    return row_start, row_end, result


def _process_worker_slice_savgol(
    evi_mmap_path:    str,
    evi_shape:        tuple,
    row_start:        int,
    row_end:          int,
    t_nominal:        np.ndarray,
    t_daily:          np.ndarray,
    min_valid_points: int,
    value_min:        float,
    value_max:        float,
    fill_low_data:    str,
    n_output:         int,
    window_length:    int,
    polyorder:        int,
) -> tuple[int, int, np.ndarray]:
    """
   !!! Currently not default implementation in Pheno_Algo (Sept 2026) !!!

    Savitzky-Golay smoothing worker.
    Linearly interpolates sparse obs to daily then applies SG filter.
    """
    from scipy.signal import savgol_filter

    evi_data = np.memmap(evi_mmap_path, dtype=np.float32, mode="r", shape=evi_shape)
    n_rows   = row_end - row_start
    nx       = evi_shape[2]
    result   = np.full((n_output, n_rows, nx), np.nan, dtype=np.float32)

    for local_yi, yi in enumerate(range(row_start, row_end)):
        for xi in range(nx):
            ts    = evi_data[:, yi, xi]
            valid = np.isfinite(ts)

            if valid.sum() < min_valid_points:
                if fill_low_data == "mean" and valid.sum() > 0:
                    result[:, local_yi, xi] = np.nanmean(ts[valid])
                continue

            x_valid = t_nominal[valid]
            y_valid = ts[valid].astype(np.float64)

            # 1. Linear interpolation to daily grid
            daily_interp = np.interp(t_daily, x_valid, y_valid)

            # 2. SG filter — ensure window_length doesn't exceed series length
            wl = min(window_length, len(daily_interp))
            wl = wl if wl % 2 == 1 else wl - 1   # must be odd
            if wl <= polyorder:
                result[:, local_yi, xi] = daily_interp.astype(np.float32)
                continue

            smoothed = savgol_filter(daily_interp, window_length=wl, polyorder=polyorder)
            result[:, local_yi, xi] = np.clip(
                smoothed, value_min, value_max
            ).astype(np.float32)

    return row_start, row_end, result


def smooth_evi_chunk_for_year(
    chunk:                xr.DataArray,
    target_year:          int,
    smoother:             str = "spline",
    savgol_window:        int = 31,
    savgol_polyorder:     int = 3,
    min_valid_points:     int   = 6,
    min_valid_frac:       float = 0.30,
    value_min:            float = -1.0,
    value_max:            float = 1.0,
    k:                    int   = 5,
    use_context_months:   bool  = True,
    testing_mode:         bool  = False,
    _pool:                Parallel | None = None,
    n_jobs=-1,
) -> xr.DataArray:
    """
    Smooth a spatial EVI chunk for a target year using pixel-wise spline or Savitzky-Golay fitting.

    Fits a smoother over a +/- 12-month context window (36 month total) around `target_year` to reduce
    edge artifacts (if use_context_months), then returns a continuous daily EVI time series for the target
    year only (365 days). Pixels with insufficient valid observations are returned as NaN.

    Parameters
    ----------
    chunk : xr.DataArray
        (time, y, x) EVI DataArray spanning at least `target_year` +/- context months.
    target_year : int
        Calendar year for which to produce output.
    smoother : {'spline', 'savgol'}
        Smoothing method. 'spline' fits a weighted LSQ spline; 'savgol' applies a Savitzky-Golay filter.
    savgol_window : int
        Window length (days, must be odd) for the Savitzky-Golay filter.
    savgol_polyorder : int
        Polynomial order for the Savitzky-Golay filter.
    min_valid_points : int
        Threshold for valid observations required per pixel; the effective floor is also constrained
        by `k+1` knots and `min_valid_frac`.
    min_valid_frac : float
        Minimum fraction of timesteps that must be finite for a pixel to be fit.
    value_min, value_max : float
        Clipping bounds applied to the smoothed output (EVI range).
    k : int
        Spline degree (3 = cubic, 5 = quintic).
    use_context_months : bool
        If True, fits over a 36 month window to anchor edge behaviour.
        If False, fits only over the target year (data-sparse regions).
    testing_mode : bool
        If True, output spans the full fitting window rather than target_year only,
        useful for visual QC of edge and context behaviour.
    _pool : joblib.Parallel or None
        Pre-warmed Parallel instance. Reusing a pool avoids repeated loky worker
        startup costs when processing many chunks in sequence.
    n_jobs : int
        Number of parallel workers. Ignored if `_pool` is provided.

    Returns
    -------
    xr.DataArray
        (time, y, x) daily smoothed EVI with dtype float32.
        ``time`` covers every day of `target_year` (365 values), or the full
        context window if ``testing_mode=True``.
    """

    t_start   = time.time()
    if _pool is not None and hasattr(_pool, 'n_jobs'):
        pool_workers = _pool.n_jobs
        n_workers    = os.cpu_count() if pool_workers == -1 else max(1, pool_workers)
        print(f"  Workers   : {n_workers} (from warm pool)")
    else:
        n_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        print(f"  Workers   : {n_workers} (local pool)")

    if use_context_months:
        context_months = 12
    else:
        context_months = 0
        
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
    t_nominal = ((fit_chunk.time.values - ref_date)
                 / np.timedelta64(1, "D")).astype(np.float64)

    if len(t_nominal) == 0:
        print(f"   WARNING: 0 valid timesteps for {target_year} current chunk;"
              f"     chunk is entirely masked. Returning NaN output.")
        daily_times = pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31", freq="D")        
        nan_data = np.full(
            (len(daily_times), fit_chunk.shape[1], fit_chunk.shape[2]),
            np.nan,
            dtype=np.float32,
        )
        return xr.DataArray(
            nan_data,
            dims=["time", "y", "x"],
            coords={"time": daily_times, "y": fit_chunk.y, "x": fit_chunk.x,},
        )
        
    if len(t_nominal) < min_valid_points:
        print(f"  [smooth_evi] WARNING: only {len(t_nominal)} valid timesteps ")
        daily_times = pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31", freq="D")
        nan_data = np.full(
            (len(daily_times), fit_chunk.shape[1], fit_chunk.shape[2]),
            np.nan,
            dtype=np.float32,
        )
        return xr.DataArray(
            nan_data,
            dims=["time", "y", "x"],
            coords={"time": daily_times,"y": fit_chunk.y,"x": fit_chunk.x,},
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
    print(f"t_nominal: {t_nominal[0]:.1f} - {t_nominal[-1]:.1f}", flush=True)
    print(f"t_daily:   {t_daily[0]:.1f}  - {t_daily[-1]:.1f}", flush=True)
    print(f"overlap:   {t_daily[0] >= t_nominal[0]} to {t_daily[-1] <= t_nominal[-1]}", flush=True)
    
    # ----------------------------------------------------------------
    # 6. Gaussian-decay weight template
    #    Observations near the centre of target_year get full weight;
    #    context observations are downweighted by distance
    # ----------------------------------------------------------------
    target_center = float(
        (np.datetime64(f"{target_year}-07-01") - ref_date)
        / np.timedelta64(1, "D")
    )
    days_from_center = np.abs(t_nominal - target_center)
    weights_template = np.exp(-0.25 * days_from_center / 365) * 0.85 + 0.15
    
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
        print(f"  Knot mode : {'sparse/Arctic' if not use_context_months else 'full context'} | "  f"k={k}")
        print(f"  Workers   : {n_workers} processes | Output: {n_output} days")
        print(f"  Slices    : {len(worker_slices)} × ~{ny // n_workers} rows each")
        t_dispatch = time.time()

        # Shared kwargs — same for every worker
        if smoother == "savgol":
            worker_kwargs = dict(
                evi_mmap_path=evi_path,
                evi_shape=evi_shape,
                t_nominal=t_nominal,
                t_daily=t_daily,
                min_valid_points=effective_min_valid,
                value_min=value_min,
                value_max=value_max,
                n_output=n_output,
                window_length=savgol_window,
                polyorder=savgol_polyorder,
            )
            worker_fn = _process_worker_slice_savgol
        else:
            worker_kwargs = dict(
                evi_mmap_path=evi_path,
                evi_shape=evi_shape,
                t_nominal=t_nominal,
                weights_template=weights_template,
                t_daily=t_daily,
                min_valid_points=effective_min_valid,
                value_min=value_min,
                value_max=value_max,
                k=k,
                n_output=n_output,
                use_context_months=use_context_months,
            )
            worker_fn = _process_worker_slice
        print(f"  Smoother  : {smoother}"
              + (f" (window={savgol_window}, polyorder={savgol_polyorder})"
                 if smoother == "savgol" else f" (k={k})"))

        executor = _pool or Parallel(
            n_jobs=n_workers, prefer="processes", batch_size="auto"
        )

        # use precomputed row-wise worker slices to distribute with kwargs to workers
        results = executor(
            delayed(_process_worker_slice)(row_start=s, row_end=e, **worker_kwargs)
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


#############################################
##### Thresholding Function #################
def apply_thresholds_chunk(chunk: xr.DataArray,
                           min_val: float = 0.1,
                           max_val: float = 0.95) -> xr.DataArray:
    """Apply min/max thresholds to chunk."""
    return chunk.where((chunk >= min_val) & (chunk <= max_val))


#############################################
##### Despiking function ####################
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

    Parameters:
    ----------
        chunk:                DataArray (time, y, x) of EVI values
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
    spike_mask = np.zeros_like(chunk_values, dtype=bool)
    time_days_da = xr.DataArray(nominal_days, dims=['time'],
                                coords={'time': chunk.time})
    
    evi_pre   = chunk.ffill(dim="time").shift(time=1)    
    evi_post  = chunk.bfill(dim="time").shift(time=-1)  
    time_pre  = time_days_da.ffill(dim="time").shift(time=1)
    time_post = time_days_da.bfill(dim="time").shift(time=-1)

    gap = time_post - time_pre
    weight = (time_days_da - time_pre) / (time_post - time_pre)
    evi_fit = evi_pre + (evi_post - evi_pre) * weight

    amplitude = evi_post - evi_pre
    diff = evi_fit - chunk
    abs_diff = np.abs(diff)
    rel_diff = np.abs(diff / amplitude.where(np.abs(amplitude) > 0.001))


    spike_da = (
        # Case 1: large brightness spike
        (abs_diff > abs_threshold)
        & (rel_diff > rel_threshold)
        & (gap < max_gap_days)
        & (~evi_pre.isnull())
        & (~evi_post.isnull())
    ) | (
        # Case 2: large absolute dip regardless of relative, catches cases where
        # neighbour amplitude is large enough to suppress rel below threshold
        (abs_diff > abs_threshold * 1.5)
        & (chunk < evi_pre - abs_threshold)   # must be below pre neighbour
        & (chunk < evi_post - abs_threshold)  # must be below post neighbour
        & (gap < max_gap_days)
        & (~evi_pre.isnull())
        & (~evi_post.isnull())
    ) | (
        # Case 3: near-flat neighbours — rel is noisy when amplitude ~ 0
        (np.abs(amplitude) <= 0.05)
        & (evi_pre - chunk > abs_threshold * 0.6)
        & (evi_post - chunk > abs_threshold * 0.6)
        & (gap < max_gap_days)
        & (~evi_pre.isnull())
        & (~evi_post.isnull())
    )

    spike_mask = spike_da.values

    if handle_edges and n_times >= 3:
        # ── First obs ─────────────────────────────────────────────
        t_gap_first = nominal_days[1] - nominal_days[0]
        if t_gap_first < max_gap_days:
            diff_first = np.abs(chunk_values[0] - chunk_values[1])
            spike_mask[0] = (
                (diff_first > abs_threshold * 1.5)
                & (~np.isnan(chunk_values[0]))
                & (~np.isnan(chunk_values[1]))
            )
    
        # ── Second obs — must be low relative to BOTH neighbours ──
        t_gap_second = nominal_days[2] - nominal_days[0]
        if t_gap_second < max_gap_days:
            spike_mask[1] = (
                (chunk_values[1] < chunk_values[0])   # lower than first
                & (chunk_values[1] < chunk_values[2]) # lower than third
                & ((np.abs(chunk_values[1] - chunk_values[0]) > abs_threshold) | (np.abs(chunk_values[1] - chunk_values[2]) > abs_threshold))
                & (~np.isnan(chunk_values[0]))
                & (~np.isnan(chunk_values[1]))
                & (~np.isnan(chunk_values[2]))
            )
    
        # ── Last obs ──────────────────────────────────────────────
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


######################################################
##### Product: Target year quality pixels function ###
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


#############################################
##### Main phenology metrics function #######
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
    n_cycles = 2  # Primary and Secondary growing seasons

    # Initialize output phenometric arrays (shape: n_cycles, ny, nx)
    core_metric_keys = [
        'annual_mean', 'annual_max', 'annual_max_doy', 'annual_min', 'annual_min_doy',
        'greenup_evi', 'greenup_doy', 'greenup_threshold', 'dormancy_evi', 'dormancy_doy',
        'annual_amplitude', 'growing_season_length', 'auc_full', 'auc_net',
        # 'greenup_rate', 'greenup_rate_doy', 'senescence_rate', 'senescence_rate_doy'
        'mid_greenup_doy', 'mid_greendown_doy'
    ]
    metric_keys = core_metric_keys + ['qa_valid_metrics']
    metrics = {k: np.full((n_cycles, ny, nx), np.nan, dtype=np.float32) for k in metric_keys}
    cycle_count = np.full((ny, nx), np.nan, dtype=np.float32)

    if len(chunk.time) == 0: return metrics

    # nominal_doys = chunk.time.dt.dayofyear.values
    target_jan1 = np.datetime64(f'{year}-01-01')
    nominal_doys = (chunk.time.values - target_jan1) / np.timedelta64(1, 'D') + 1
    years_array = chunk.time.dt.year.values
    chunk_vals = chunk.values
    continuous_days = np.arange(len(nominal_doys))

    for yi in range(ny):
        for xi in range(nx):
            pixel_evi = chunk_vals[:, yi, xi]
            if np.isnan(pixel_evi).all():
                continue

            global_amp = np.nanmax(pixel_evi) - np.nanmin(pixel_evi)

            # 1. Identify candidate peaks (1st derivative == 0, pos to neg slope)
            peaks, _ = find_peaks(pixel_evi)

            # Follow MODIS MCD12Q2 peak finding logic (Gray et al., 2022)
            valid_cycles = []
            if len(peaks) > 0:
                active_peaks = sorted(list(peaks))

                # Iteratively eliminate non-candidate peaks to dynamically expand neighboring search windows
                while len(active_peaks) > 0:
                    current_cycles = []

                    for i, current_peak in enumerate(active_peaks):
                        prior_peak = active_peaks[i - 1] if i > 0 else 0
                        next_peak = active_peaks[i + 1] if i < len(active_peaks) - 1 else len(pixel_evi) - 1

                        # Bound the searches to max 185-30 days prior/post current peak
                        # Threshold 1: Greenup (greendown) >30 days prior (after) to current peak & after (before) prior peak
                        gu_start = max(prior_peak, prior_peak - 185)
                        gu_end = current_peak - 30
                        gd_start = current_peak + 30
                        gd_end = min(next_peak, current_peak + 185)

                        if gu_end <= gu_start or gd_start >= gd_end:
                            current_cycles.append(
                                {'peak_idx': current_peak, 'amp': -1.0, 'peak_evi': pixel_evi[current_peak]})
                            continue

                        gu_segment = pixel_evi[gu_start:gu_end]
                        gd_segment = pixel_evi[gd_start:gd_end]

                        if np.isnan(gu_segment).all() or np.isnan(gd_segment).all():
                            current_cycles.append(
                                {'peak_idx': current_peak, 'amp': -1.0, 'peak_evi': pixel_evi[current_peak]})
                            continue

                        gu_min_idx = gu_start + np.nanargmin(gu_segment)
                        gd_min_idx = gd_start + np.nanargmin(gd_segment)

                        amp = pixel_evi[current_peak] - pixel_evi[gu_min_idx]

                        current_cycles.append({
                            'peak_idx': current_peak,
                            'gu_min_idx': gu_min_idx,
                            'gd_min_idx': gd_min_idx,
                            'amp': amp,
                            'peak_evi': pixel_evi[current_peak],
                            'min_evi': pixel_evi[gu_min_idx]
                        })

                    # Find the smallest cycle (Smallest amplitude, Break ties using peak_evi for tight windows)
                    smallest_cycle = min(current_cycles, key=lambda x: (x['amp'], x['peak_evi']))

                    # Threshold 2: cycle amplitude >= 0.1 and >= 35% global amplitude
                    if smallest_cycle['amp'] < 0.1 or smallest_cycle['amp'] < 0.35 * global_amp:
                        # Failed: Remove it and repeat loop. Neighbor windows will now expand to include this segment as potential growing cycle.
                        active_peaks.remove(smallest_cycle['peak_idx'])
                    else:
                        # The SMALLEST active peak passed: ALL active peaks pass, progress to phenometrics
                        valid_cycles = current_cycles
                        break

            # 2. Filter cycles where the peak falls strictly within target year
            target_cycles = [c for c in valid_cycles if years_array[c['peak_idx']] == year]
            cycle_count[yi, xi] = len(target_cycles)

            # Fallback: If peak logic failed, treat the target year as a single cycle
            if len(target_cycles) == 0:
                year_mask = (years_array == year) & ~np.isnan(pixel_evi)
                if year_mask.any():
                    # Find peak strictly within the target year
                    target_valid_idx = np.where(year_mask)[0]
                    p_idx = target_valid_idx[np.argmax(pixel_evi[target_valid_idx])]

                    # But allow the MINIMUM search to cross into context years (up to 185 days)
                    all_valid_idx = np.where(~np.isnan(pixel_evi))[0]

                    pre_mask = all_valid_idx[(all_valid_idx <= p_idx) & (all_valid_idx >= p_idx - 185)]
                    gu_min_idx = pre_mask[np.argmin(pixel_evi[pre_mask])] if len(pre_mask) > 0 else p_idx

                    post_mask = all_valid_idx[(all_valid_idx >= p_idx) & (all_valid_idx <= p_idx + 185)]
                    gd_min_idx = post_mask[np.argmin(pixel_evi[post_mask])] if len(post_mask) > 0 else p_idx

                    target_cycles = [{
                        'peak_idx': p_idx, 'gu_min_idx': gu_min_idx, 'gd_min_idx': gd_min_idx,
                        'amp': pixel_evi[p_idx] - pixel_evi[gu_min_idx],
                        'peak_evi': pixel_evi[p_idx], 'min_evi': pixel_evi[gu_min_idx]
                    }]

            target_cycles = sorted(target_cycles, key=lambda x: x['peak_evi'], reverse=True)[:n_cycles]

            # 3. Calculate phenometrics accepted cycles
            for c_idx, cycle in enumerate(target_cycles):
                metrics['annual_max'][c_idx, yi, xi] = cycle['peak_evi']
                metrics['annual_max_doy'][c_idx, yi, xi] = nominal_doys[cycle['peak_idx']]
                metrics['annual_min'][c_idx, yi, xi] = cycle['min_evi']
                metrics['annual_min_doy'][c_idx, yi, xi] = nominal_doys[cycle['gu_min_idx']]
                metrics['annual_amplitude'][c_idx, yi, xi] = cycle['amp']

                # Slice times for this specific cycle context
                c_start, c_end = cycle['gu_min_idx'], cycle['gd_min_idx']
                metrics['annual_mean'][c_idx, yi, xi] = np.nanmean(pixel_evi[c_start:c_end])

                thresh_val = cycle['min_evi'] + (cycle['amp'] * threshold_greenup_pct)
                mid_thresh_val = cycle['min_evi'] + (cycle['amp'] * 0.50)
                # thresh_val_10pct = cycle['min_evi'] + (cycle['amp'] * 0.10)
                # thresh_val_25pct = cycle['min_evi'] + (cycle['amp'] * 0.10)
                metrics['greenup_threshold'][c_idx, yi, xi] = thresh_val

                # 1. GREENUP - tracking backwards from peak
                pre_mask = (continuous_days >= cycle['gu_min_idx']) & (continuous_days < cycle['peak_idx'])
                pre_evi = pixel_evi[pre_mask]
                pre_doys = nominal_doys[pre_mask]
                gu_abs_idx = np.nan  # Continuous index tracker for rates/lengths

                if len(pre_evi) > 2:
                    # pre_deriv = np.gradient(pre_evi, continuous_days[pre_mask]) # rate var
                    # Find the most immediate previous time before the peak that EVI dropped below threshold
                    below_thresh = np.where(pre_evi <= thresh_val)[0]
                    if len(below_thresh) > 0:
                        i = below_thresh[-1]
                        if i < len(pre_evi) - 1:
                            # Interpolate DOY to closest exact threshold crossing
                            y0, y1 = pre_evi[i], pre_evi[i + 1]
                            closest_idx = i if abs(y0 - thresh_val) < abs(y1 - thresh_val) else i + 1
                            metrics['greenup_doy'][c_idx, yi, xi] = pre_doys[closest_idx]
                            gu_abs_idx = continuous_days[pre_mask][closest_idx]
                        else:
                            metrics['greenup_doy'][c_idx, yi, xi] = pre_doys[i]
                            gu_abs_idx = continuous_days[pre_mask][i]
                        metrics['greenup_evi'][c_idx, yi, xi] = thresh_val

                    # # Steepest greenup
                    # if not np.isnan(gu_abs_idx):
                    #     min_rise = cycle['min_evi'] + (cycle['amp'] * 0.10)
                    #     inflect_mask = (pre_evi >= min_rise) & (continuous_days[pre_mask] >= gu_abs_idx)
                    #     if inflect_mask.sum() >= 2:
                    #         inf_idx = np.argmax(pre_deriv[inflect_mask])
                    #         metrics['greenup_rate'][c_idx, yi, xi] = pre_deriv[inflect_mask][inf_idx]
                    #         metrics['greenup_rate_doy'][c_idx, yi, xi] = pre_doys[inflect_mask][inf_idx]

                    # Median Greenup Threshold (50% amplitude)
                    below_mid = np.where(pre_evi <= mid_thresh_val)[0]
                    if len(below_mid) > 0:
                        i = below_mid[-1]
                        if i < len(pre_evi) - 1:
                            y0, y1 = pre_evi[i], pre_evi[i + 1]
                            closest_idx = i if abs(y0 - mid_thresh_val) < abs(y1 - mid_thresh_val) else i + 1
                            metrics['mid_greenup_doy'][c_idx, yi, xi] = pre_doys[closest_idx]
                        else:
                            metrics['mid_greenup_doy'][c_idx, yi, xi] = pre_doys[i]

                # 2. DORMANCY
                post_mask = (continuous_days > cycle['peak_idx']) & (continuous_days <= cycle['gd_min_idx'])
                post_evi = pixel_evi[post_mask]
                post_doys = nominal_doys[post_mask]
                dorm_abs_idx = np.nan

                if len(post_evi) > 2:
                    # post_deriv = np.gradient(post_evi, continuous_days[post_mask]) # used in rate
                    # Find the FIRST time after the peak that EVI drops below greenup threshold
                    below_thresh = np.where(post_evi <= thresh_val)[0]
                    if len(below_thresh) > 0:
                        i = below_thresh[0]
                        if i > 0:
                            # Interpolate DOY to closest exact threshold crossing
                            y0, y1 = post_evi[i - 1], post_evi[i]
                            closest_idx = i - 1 if abs(y0 - thresh_val) < abs(y1 - thresh_val) else i
                            metrics['dormancy_doy'][c_idx, yi, xi] = post_doys[closest_idx]
                            dorm_abs_idx = continuous_days[post_mask][closest_idx]
                        else:
                            metrics['dormancy_doy'][c_idx, yi, xi] = post_doys[i]
                            dorm_abs_idx = continuous_days[post_mask][i]
                        metrics['dormancy_evi'][c_idx, yi, xi] = thresh_val

                    # # Steepest senescence
                    # if not np.isnan(dorm_abs_idx):
                    #     max_fall = cycle['peak_evi'] - (cycle['amp'] * 0.10)
                    #     inflect_mask = (post_evi <= max_fall) & (continuous_days[post_mask] <= dorm_abs_idx)
                    #     if inflect_mask.sum() >= 2:
                    #         inf_idx = np.argmin(post_deriv[inflect_mask])
                    #         metrics['senescence_rate'][c_idx, yi, xi] = post_deriv[inflect_mask][inf_idx]
                    #         metrics['senescence_rate_doy'][c_idx, yi, xi] = post_doys[inflect_mask][inf_idx]
                    # 50% Greendown Threshold
                    below_mid = np.where(post_evi <= mid_thresh_val)[0]
                    if len(below_mid) > 0:
                        i = below_mid[0]
                        if i > 0:
                            y0, y1 = post_evi[i - 1], post_evi[i]
                            closest_idx = i - 1 if abs(y0 - mid_thresh_val) < abs(y1 - mid_thresh_val) else i
                            metrics['mid_greendown_doy'][c_idx, yi, xi] = post_doys[closest_idx]
                        else:
                            metrics['mid_greendown_doy'][c_idx, yi, xi] = post_doys[i]

                # 3. AUC and 4. Growing Season Length calculations (Requires both ends)
                if not np.isnan(gu_abs_idx) and not np.isnan(dorm_abs_idx):
                    metrics['growing_season_length'][c_idx, yi, xi] = dorm_abs_idx - gu_abs_idx
                    gs_mask = (continuous_days >= gu_abs_idx) & (continuous_days <= dorm_abs_idx)
                    gs_evi = pixel_evi[gs_mask]
                    gs_valid = ~np.isnan(gs_evi)

                    if gs_valid.sum() >= 3:
                        gs_evi = gs_evi[gs_valid]
                        # Using dx=1 since indices are continuous days to prevent DOY wrap-around bugs
                        metrics['auc_full'][c_idx, yi, xi] = trapezoid(gs_evi, dx=1)
                        metrics['auc_net'][c_idx, yi, xi] = trapezoid(gs_evi - cycle['min_evi'], dx=1)

        # Count the number of valid phenometrics created: a quality proxy
    metrics['qa_valid_metrics'] = np.zeros((n_cycles, ny, nx), dtype=np.float32)
    for k in core_metric_keys:
        metrics['qa_valid_metrics'] += ~np.isnan(metrics[k])

    # count the number of detected cycles
    metrics['cycle_count'] = cycle_count

    return metrics


###############################################################################
##### Fucntions to help gap filling and non growing season ID #################

def get_context_months_from_gaps(
        chunk: xr.DataArray,
        target_year: int,
        gap_threshold_days: int = 70,  # 10week isoline in Bormann et al., 2018
        min_spatial_coverage: float = 0.25,  # Require N% of pixels to have data
) -> bool:
    """
    Check the actual observation record for temporal gaps at the pixel level.
    If the median pixel has a gap at the start or end of the target year
    exceeding gap_threshold_days (a conservative winter estimate),
    context years will cause edge spikes so set use_context_months=False.
    """
    target_obs = chunk.sel(time=str(target_year))
    is_valid = target_obs.notnull()
    has_any = is_valid.any(dim="time")

    if not has_any.any():
        print(f"  [Context Diagnostics] {target_year}: Chunk is completely empty.")
        return False

    # Find the index of the first and last valid observation for every pixel
    first_idx = is_valid.argmax(dim="time")
    n_times = is_valid.sizes["time"]
    last_idx = n_times - 1 - is_valid.isel(time=slice(None, None, -1)).argmax(dim="time")

    # Get DOY and mask NaN
    first_doys = target_obs.time.dt.dayofyear.isel(time=first_idx).where(has_any)
    last_doys = target_obs.time.dt.dayofyear.isel(time=last_idx).where(has_any)

    # Calculate the median gaps
    median_first_doy = float(first_doys.median().values)
    median_last_doy = float(last_doys.median().values)
    gap_start = int(median_first_doy - 1)  # DOY 1 = 0 gap
    gap_end = int(365 - median_last_doy)

    # --- DIAGNOSTICS ---
    # What % of pixels got an observation in the first/last 45 days
    early_pixels = (first_doys <= gap_threshold_days).sum().values
    late_pixels = (last_doys >= (365 - gap_threshold_days)).sum().values
    total_valid = has_any.sum().values

    pct_early = (early_pixels / total_valid) * 100 if total_valid > 0 else 0
    pct_late = (late_pixels / total_valid) * 100 if total_valid > 0 else 0

    print(f"  [Context months diagnostics] {target_year} Median Gaps:")
    print(f"    -> Start Gap: {gap_start} days (Median first obs: DOY {median_first_doy:.0f})")
    print(f"    -> End Gap:   {gap_end} days (Median last obs: DOY {median_last_doy:.0f})")
    print(f"    -> Pixels with data in first {gap_threshold_days} days: {pct_early:.1f}%")
    print(f"    -> Pixels with data in last {gap_threshold_days} days:  {pct_late:.1f}%")

    if gap_start >= gap_threshold_days or gap_end >= gap_threshold_days:
        print("    => RESULT: False (Winter gap detected, turning OFF context months)")
        return False

    print("    => RESULT: True (Sufficient winter data, using 3-year context)")
    return True


def mask_snow_ndfsi_chunk(
    chunk:           xr.DataArray,
    ndfsi:           xr.DataArray,
    ndfsi_threshold: float = 0.4,
    background_pct:  float = 0.05,
) -> tuple[xr.DataArray, np.ndarray]:
    """
    Returns:
        chunk_masked  : EVI2 with snow pixels replaced by background
        ndfsi_hit_cnt : (y, x) int16 array — number of timesteps masked
                        per pixel, 0 = never masked
    """
    # Reindex NDFSI onto EVI2 time axis — some EVI2 acquisition dates may
    # have no companion NDFSI file; those timesteps get NaN which evaluates
    # False in snow_mask, leaving the EVI2 value
    ndfsi = ndfsi.reindex(time=chunk.time, method=None)
    background = (
        chunk
        .quantile(background_pct, dim="time", skipna=True)
        .drop_vars("quantile", errors="ignore")
        .clip(min=0.0)
    )
    snow_mask     = ndfsi > ndfsi_threshold
    ndfsi_hit_cnt = snow_mask.sum(dim="time").values.astype(np.int16)   # (y, x)

    print(f"  NDFSI masked : {int((ndfsi_hit_cnt > 0).sum()):,} px affected | "
          f"{int(ndfsi_hit_cnt.sum()):,} total obs replaced "
          f"(max {int(ndfsi_hit_cnt.max())} per px)")

    return chunk.where(~snow_mask, other=background), ndfsi_hit_cnt


def calc_obs_snow_background(
        chunk: xr.DataArray,
        low_pct: float = 0.10,
        snow_doy_start: int = 300,
        snow_doy_end: int = 100,
        min_snow_obs: int = 3,
        debug_y: int =  None,
        debug_x: int = None,
) -> xr.DataArray:
    doy = chunk.time.dt.dayofyear
    snow_mask = (doy >= snow_doy_start) | (doy <= snow_doy_end)
    snow_obs = chunk.isel(time=snow_mask)
    n_valid = snow_obs.notnull().sum(dim="time")

    # Path 1: winter obs available
    snow_background = (
        snow_obs
        .quantile(low_pct, dim="time", skipna=True)
        .drop_vars("quantile", errors="ignore")
        .clip(min=0.0)
    )

    # Path 2: no winter obs (Arctic / Sparse winter data)
    all_low = (
        chunk
        .quantile(low_pct, dim="time", skipna=True)
        .drop_vars("quantile", errors="ignore")
        .clip(min=0.0)
    )
    precentile_background = all_low.clip(min=0.0)

    print(f"  snow_background    : min={float(snow_background.min(skipna=True)):.4f} "
          f"mean={float(snow_background.mean(skipna=True)):.4f} "
          f"max={float(snow_background.max(skipna=True)):.4f}")
    print(f"  amplitude_background: min={float(precentile_background.min(skipna=True)):.4f} "
          f"mean={float(precentile_background.mean(skipna=True)):.4f} "
          f"max={float(precentile_background.max(skipna=True)):.4f}")

    background = xr.where(n_valid >= min_snow_obs, snow_background, precentile_background)
    background = background.drop_vars("quantile", errors="ignore")

    # --- Pixel-Specific Debug Block ---
    if debug_y is not None and debug_x is not None:
        try:
            pix_valid = int(n_valid.isel(y=debug_y, x=debug_x).values)
            pix_p1 = float(snow_background.isel(y=debug_y, x=debug_x).values)
            pix_p2 = float(precentile_background.isel(y=debug_y, x=debug_x).values)
            pix_final = float(background.isel(y=debug_y, x=debug_x).values)

            print(f"\n  [BG Debug] Pixel (y={debug_y}, x={debug_x}):")
            print(f"    -> Valid Winter Obs (DOY <{snow_doy_end} | >{snow_doy_start}): {pix_valid}")
            print(f"    -> Path 1 (Winter 10%): {pix_p1:.4f}")
            print(f"    -> Path 2 (All-Year 10%): {pix_p2:.4f}")
            print(f"    -> Final Chosen BG: {pix_final:.4f} (Used Path {1 if pix_valid >= min_snow_obs else 2})")
        except Exception as e:
            print(f"  [BG Debug] Could not extract pixel ({debug_y}, {debug_x}): {e}")

    return background


##################################################
##### Main phenology orchestrator function #######
def full_pipeline_chunk(chunk: xr.DataArray,
                        ndfsi: xr.DataArray = None,
                        doy_data: xr.DataArray = None,
                        apply_threshold: bool = True,
                        min_evi_threshold: float = -1.0,
                        max_evi_threshold: float = 1.0,
                        threshold_greenup_pct: float = 0.15,
                        despike: bool = True,
                        despike_max_gap: int = 45,
                        despike_abs_threshold: float = 0.1,
                        despike_rel_threshold: float = 2.0,
                        use_infill: bool = True,
                        smoother: str = "spline",
                        target_year: int = None,
                        testing_mode: bool = False,
                        _pool = None,
                        n_jobs:int = -1,
                        **kwargs) -> dict[str, np.ndarray]:
    """
    Full processing pipeline for a chunk.

    Pipeline:
        1. Apply EVI thresholds: ensures any anomalous EVI values are clipped
        2. Mask snow pixels using NDFSI
        3. De-spike (three-point method): removes
        4. Infill gaps: Use the 36 month time series to fill in gaps in target year
        5. Calculate scene quality pixel counts
        6. Fit spline and generate synthetic daily time series 
        7. Fill snow and non-growing season gaps (useful for snowy climates)
        8. Calculate annual phenometrics

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
        'mid_greenup_doy': 'mid_greenup_doy',
        'mid_greendown_doy': 'mid_greendown_doy',
        # 'greenup_rate': 'greenup_rate',
        # 'greenup_rate_doy': 'greenup_rate_doy',
        # 'senescence_rate': 'senescence_rate',
        # 'senescence_rate_doy': 'senescence_rate_doy',
        'qa_valid_metrics': 'qa_valid_metrics'
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

    # Step 2: Snow masking
    ndfsi_hit_cnt = None
    if ndfsi is not None:
        print("Step 1b: NDFSI snow masking")
        chunk, ndfsi_hit_cnt = mask_snow_ndfsi_chunk(chunk, ndfsi)

    chunk_post_threshold = chunk.copy(deep=True) if testing_mode else None

    # Step 3: Negative pixel filtering using DOY (EVI2 despiking - cloud shadows)
    # - uses target year +/- 1 year, if edge case remove the non-existing year
    if despike:
        print("Step3 : Despiking")
        chunk = despike_timeseries_chunk(
            chunk,
            max_gap_days=despike_max_gap,
            abs_threshold=despike_abs_threshold,
            rel_threshold=despike_rel_threshold,
            # debug_pixel=(1,3),
        )
        target_obs_despiked = chunk.sel(time=str(target_year))
        target_obs_raw = chunk_post_threshold.sel(time=str(target_year)) if testing_mode else None
        if testing_mode:
            removed = target_obs_raw.notnull() & target_obs_despiked.isnull()

    chunk_post_despike = chunk.copy(deep=True) if testing_mode else None

    # -----------------------------------------------------------------------------------------
    # Step 3b: Context-year gap infill
    # Uses despiked observations from context years to fill gaps in the
    # target year before the spline sees the data.
    context_infill_diagnostics = None
    target_da_for_spline = chunk.sel(time=str(target_year))  # default: no infill

    use_context_months = get_context_months_from_gaps(chunk=chunk, target_year=target_year)
    context_years_present = sorted({
        int(y) for y in chunk.time.dt.year.values
        if int(y) != target_year
    })

    if len(context_years_present) >= 1 and use_infill == True:
        print("Step 3b: Context-year observation infill")
        target_da_for_spline, context_infill_diagnostics = build_context_infilled_observations(
            chunk_despiked=chunk,  # full 3-yr despiked DataArray
            target_year=target_year,
            n_harmonics=3,
            min_similarity=0.60,  # tune: lower = more permissive infill
            scale_to_target=True,
            testing_mode=testing_mode,
        )
        # Rebuild a chunk that contains the infilled target year so the spline
        # fitter receives the augmented observations
        other_years = chunk.sel(
            time=~chunk.time.dt.year.isin([target_year])
        )
        chunk_for_spline = xr.concat(
            [other_years, target_da_for_spline],
            dim="time"
        ).sortby("time")
    else:
        print("Step 3b: Context infill skipped "
              f"(use_context_months={use_context_months}, "
              f"context_years={context_years_present})")
        chunk_for_spline = chunk

    chunk_post_context_infill = chunk_for_spline.copy(deep=True) if testing_mode else None

    chunk_for_spline = despike_timeseries_chunk(
        chunk_for_spline,
        max_gap_days=despike_max_gap,
        abs_threshold=despike_abs_threshold,
        rel_threshold=despike_rel_threshold,
    )

    # Step 4: calculate scene revisit and quality pixels before the spline fit, 365 DOY data is generated
    print("Step 4: Scene quality metrics")
    scene_mean_revisit, scene_quality_pixels = compute_scene_quality_metrics(chunk, target_year)

    valid_timesteps = (~np.isnan(chunk.values)).any(axis=(1, 2)).sum()
    if valid_timesteps == 0:
        print(
            f"  WARNING: chunk has 0 valid timesteps for {target_year}. All metrics will be NaN — skipping spline and phenometrics.")
        return {
            f'{name}_{target_year}': np.full((chunk.shape[1], chunk.shape[2]), np.nan, dtype=np.float32)
            for name in metric_mapping.values()
        } | {
            f'mean_revisit_time_{target_year}': scene_mean_revisit,
            f'quality_pixel_cnt_{target_year}': scene_quality_pixels,
        }

    # Step 5: apply penalized cubic spline interpolation
    background_threshold = calc_obs_snow_background(chunk_for_spline)
    print("Step 5: Apply spline")
    fill_snow_gaps = not use_context_months

    print(f"  use_context_months : {use_context_months}")
    print(f"  fill_snow_gaps     : {fill_snow_gaps}")

    smoothed_daily = smooth_evi_chunk_for_year(
        chunk_for_spline,
        target_year,
        smoother=smoother,
        testing_mode=testing_mode,
        use_context_months=use_context_months,
        _pool=_pool,
        n_jobs=n_jobs
    )

    # Step 5b: Spline Floor clamp
    # Prevent the spline from sinusoidally dipping below the lowest actual observation. This fixes inflated amplitudes esp. in multiple growing seasons.
    # obs_min = chunk_for_spline.min(dim="time", skipna=True)
    # smoothed_daily = xr.where(smoothed_daily < obs_min, obs_min, smoothed_daily)
    # chunk_post_spline = smoothed_daily.copy(deep=True) if testing_mode else None

    smoothed_daily = xr.where(smoothed_daily < background_threshold, background_threshold, smoothed_daily)
    chunk_post_spline = smoothed_daily.copy(deep=True) if testing_mode else None

    if fill_snow_gaps:
        # Step 6: Fill snow gaps using naive min EVI2 value
        print("Step 6: Snow gap fill", flush=True)
        target_obs = chunk_for_spline.sel(time=str(target_year))
        is_valid = target_obs.notnull()
        has_any = is_valid.any(dim="time")

        # ID first/last observations
        first_idx = is_valid.argmax(dim="time")
        last_idx = (target_obs.sizes["time"] - 1 - is_valid.isel(time=slice(None, None, -1)).argmax(dim="time"))
        first_obs_doy = target_obs.time.dt.dayofyear.isel(time=first_idx).where(has_any)
        last_obs_doy = target_obs.time.dt.dayofyear.isel(time=last_idx).where(has_any)

        smoothed_year = smoothed_daily.sel(time=str(target_year))
        daily_doy = smoothed_year.time.dt.dayofyear
        bg = background_threshold.drop_vars("quantile", errors="ignore")

        # subset target year data to valid observations
        before_first = daily_doy < first_obs_doy
        after_last = daily_doy > last_obs_doy
        outside_obs = (before_first | after_last | ~has_any)

        # Clamp runaway tails outside observation window
        # Extract the spline's value at the exact DOY of the first and last valid observations.
        val_at_first = smoothed_year.where(daily_doy == first_obs_doy).max(dim="time")
        val_at_last = smoothed_year.where(daily_doy == last_obs_doy).max(dim="time")

        # Identify runaway positive tails: if the extrapolated spline has a positive end behavior
        # than the boundary observation, do not conisder it a missed pheno cycle. Clamp to min/background.
        runaway_pre = before_first & (smoothed_year > val_at_first)
        runaway_post = after_last & (smoothed_year > val_at_last)
        is_runaway = runaway_pre | runaway_post

        # Spline is kept where > floor but is clamped to bg if it runs away upward OR if it drops below bg.
        smoothed_year = xr.where(
            is_runaway,
            bg,  # Replace runaway positive tails directly to bg
            xr.where(  # else, where runaways
                outside_obs,  # if beyond observation window
                smoothed_year.clip(min=bg),  # replace with BG where downward extrapolations occur at tails
                smoothed_year  # Inside obs window: untouched
            )
        )

        #  Transition DOYs
        spline_above_bg = smoothed_year > bg
        rising_idx = spline_above_bg.argmax(dim="time")
        falling_idx = (spline_above_bg.sizes["time"] - 1
                       - spline_above_bg.isel(time=slice(None, None, -1))
                       .argmax(dim="time"))
        any_above = spline_above_bg.any(dim="time")
        rising_doy = daily_doy.isel(time=rising_idx).where(any_above)
        falling_doy = daily_doy.isel(time=falling_idx).where(any_above)

        # Prevent post-season spline rebound
        vals = smoothed_year.values.copy()  # (T, ny, nx)
        doys_1d = daily_doy.values  # (T,)
        bg_vals = bg.values  # (ny, nx)
        ny, nx = vals.shape[1], vals.shape[2]

        rise_doy = rising_doy.values  # (ny, nx)
        fall_doy = falling_doy.values  # (ny, nx)

        # Spline value at the exact transition DOYs = ceiling for outside region
        rise_idx_1d = np.argmin(np.abs(doys_1d[:, None, None] - rise_doy[None]), axis=0)
        fall_idx_1d = np.argmin(np.abs(doys_1d[:, None, None] - fall_doy[None]), axis=0)

        # (ny, nx) — the spline value at each pixel's transition DOY
        spline_at_rise = vals[rise_idx_1d, np.arange(ny)[:, None], np.arange(nx)[None, :]]
        spline_at_fall = vals[fall_idx_1d, np.arange(ny)[:, None], np.arange(nx)[None, :]]

        for t_idx in range(len(doys_1d)):
            d = doys_1d[t_idx]
            before_rise = d < rise_doy  # (ny, nx)
            after_fall = d > fall_doy

            # Pre-season: clamp to [bg, spline_at_rise]
            vals[t_idx] = np.where(
                before_rise,
                np.clip(vals[t_idx], bg_vals, spline_at_rise),
                vals[t_idx]
            )
            # Post-season: clamp to [bg, spline_at_fall]
            vals[t_idx] = np.where(
                after_fall,
                np.clip(vals[t_idx], bg_vals, spline_at_fall),
                vals[t_idx]
            )

        smoothed_year = smoothed_year.copy(data=vals)

        smoothed_year_pheno = smoothed_year.where(
            (daily_doy >= rising_doy) & (daily_doy <= falling_doy)
        )

    else:
        smoothed_year = smoothed_daily
        smoothed_year_pheno = smoothed_daily.sel(time=str(target_year))

    chunk_post_snow_fill = smoothed_year.copy(deep=True) if testing_mode else None

    # Step 7: Annual phenometrics
    print("Step 7: Calculate phenometrics")

    if use_context_months:
        chunk_for_pheno = smoothed_year_pheno
    else:
        chunk_for_pheno = smoothed_year_pheno.where(smoothed_year_pheno.time.dt.year == target_year)

    pheno = annual_phenometrics_chunk(
        chunk_for_pheno,
        year=target_year,
        threshold_greenup_pct=threshold_greenup_pct,
    )

    results = {}
    for internal_name, output_name in metric_mapping.items():
        # Cycle 0 = Primary, Cycle 1 = Secondary
        results[f'{output_name}_{target_year}_primary'] = pheno[internal_name][0]
        results[f'{output_name}_{target_year}_secondary'] = pheno[internal_name][1]

    results[f'cycle_count_{target_year}'] = pheno['cycle_count']
    results[f'mean_revisit_time_{target_year}'] = scene_mean_revisit
    results[f'quality_pixel_cnt_{target_year}'] = scene_quality_pixels

    if testing_mode:
        results['_intermediate'] = {
            'original': chunk_original,
            'post_threshold': chunk_post_threshold,
            'post_despike': chunk_post_despike,
            'post_context_infill': chunk_post_context_infill,
            'context_infill_diag': context_infill_diagnostics,
            'post_spline': chunk_post_spline,
            'post_snow_fill': chunk_post_snow_fill,
            'ndfsi_hit_cnt': ndfsi_hit_cnt,
            'ndfsi': ndfsi.sel(time=str(target_year)) if ndfsi is not None else None,
        }

    return results