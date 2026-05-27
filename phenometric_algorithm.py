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

import bottleneck

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
    fill_low_data:      str,
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
            if y_range > 0.1 and use_context_months:
                lo = y_valid.min() + 0.20 * y_range
                hi = y_valid.min() + 0.80 * y_range
                w_valid[y_valid < lo] *= 2.0
                w_valid[y_valid > hi] *= 2.0

            # 5. Knots: trim the precomputed set to this pixel's range
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
            try:
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
            
            except Exception:
                if fill_low_data == "mean":
                    result[:, local_yi, xi] = float(np.nanmean(y_valid))

    return row_start, row_end, result

def smooth_evi_chunk_for_year(
    chunk:                xr.DataArray,
    target_year:          int,
    # --- algorithm config ---
    min_valid_points:     int   = 6,
    min_valid_frac:       float = 0.30,
    fill_low_data:        str   = "nan",  # currently no gap filling, Bolton uses the context years to "grab" similar values but that's a weaker method 
    value_min:            float = -1.0,
    value_max:            float = 1.0,
    daily_output:         bool  = True,
    k:                    int   = 5,      # spline degree, 4 = cubic
    use_context_months:   bool  = True,   # computed to avoid pits/peaks from overfitting gaps
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
            coords={
                "time": daily_times,
                "y":    fit_chunk.y,
                "x":    fit_chunk.x,
            },
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
        worker_kwargs = dict(
            evi_mmap_path    = evi_path,
            evi_shape        = evi_shape,
            t_nominal        = t_nominal,
            weights_template = weights_template,
            t_daily          = t_daily,
            min_valid_points = effective_min_valid,
            value_min        = value_min,
            value_max        = value_max,
            fill_low_data    = fill_low_data,
            k                = k,
            n_output         = n_output,
            use_context_months=use_context_months,
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
            (abs_diff > abs_threshold)
            & (rel_diff > rel_threshold)
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

            # Greenup: first DOY exceeding threshold BEFORE peak AND on ascending segment
            pre_peak_mask = pixel_doys < pixel_max_doy
            if pre_peak_mask.sum() > 0:
                pre_peak_evi = pixel_evi[pre_peak_mask]
                pre_peak_doys = pixel_doys[pre_peak_mask]
                valid = ~np.isnan(pre_peak_evi)
                
                if valid.sum() > 2:
                    pre_peak_evi  = pre_peak_evi[valid]
                    pre_peak_doys = pre_peak_doys[valid]

                    # Compute derivative on pre-peak valid obs
                    pre_deriv = np.gradient(pre_peak_evi, pre_peak_doys)
                    above_thresh = pre_peak_evi >= pixel_threshold
                    ascending    = pre_deriv > 0
                    valid_greenup = above_thresh & ascending

                    if valid_greenup.any():
                        first_idx = np.argmax(valid_greenup)
                        greenup_doy[0, yi, xi] = pre_peak_doys[first_idx]
                        greenup_evi[0, yi, xi] = pre_peak_evi[first_idx]

            # Dormancy: first DOY below threshold AFTER peak
            post_peak_mask = pixel_doys > pixel_max_doy            
            if post_peak_mask.sum() > 0:
                evi_post = pixel_evi[post_peak_mask]
                doy_post = pixel_doys[post_peak_mask]         

                # reduce to non NAN range of dates
                valid_post = ~np.isnan(evi_post)
                evi_post   = evi_post[valid_post]
                doy_post   = doy_post[valid_post]
                if len(evi_post) == 0:
                    # No valid post-peak obs: dormancy = NaN
                    continue
                
                crossing_idx = np.where(evi_post <= pixel_threshold)[0]            
                if len(crossing_idx) > 0:
                    i = crossing_idx[0]            
                    if i > 0:
                        x0, x1 = doy_post[i-1], doy_post[i]
                        y0, y1 = evi_post[i-1], evi_post[i]            
                        dormancy_doy[0, yi, xi] = (doy_post[i-1] 
                                                   if abs(y0 - pixel_threshold) < abs(y1 - pixel_threshold) 
                                                   else doy_post[i])
                        dormancy_evi[0, yi, xi] = pixel_threshold
                    else:
                        dormancy_doy[0, yi, xi] = doy_post[i]
                        dormancy_evi[0, yi, xi] = evi_post[i]
            
                else:
                    # fallback: last valid observation after peak
                    dormancy_doy[0, yi, xi] = doy_post[-1]
                    dormancy_evi[0, yi, xi] = evi_post[-1]
            
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
                min_rise_scalar = 0.10
                # Steepest greenup: max positive derivative before peak
                pre_peak = pixel_doys_valid < pixel_max_doy
                if pre_peak.sum() >= 2:
                    pre_derivs = evi_derivative[pre_peak]
                    pre_doys   = pixel_doys_valid[pre_peak]
                    pre_evi    = pixel_evi_valid[pre_peak]
                
                    # ── Exclude peak shoulder ─────────────────────────────────
                    # Greenup inflection must be after EVI has risen meaningfully
                    # from baseline — at least 20% of amplitude above min
                    pixel_min_evi  = annual_min[0, yi, xi]
                    amplitude_px   = annual_amplitude[0, yi, xi]
                    min_rise       = pixel_min_evi + (amplitude_px * min_rise_scalar)
                
                    on_ascending_limb = pre_evi >= min_rise
                    if on_ascending_limb.sum() >= 2:
                        pre_derivs = pre_derivs[on_ascending_limb]
                        pre_doys   = pre_doys[on_ascending_limb]
                
                        max_rate_idx = np.argmax(pre_derivs)
                        greenup_rate[0, yi, xi]     = pre_derivs[max_rate_idx]
                        greenup_rate_doy[0, yi, xi] = pre_doys[max_rate_idx]

                # Steepest senescence: max negative derivative after peak
                post_peak = pixel_doys_valid > pixel_max_doy
                if post_peak.sum() >= 2:
                    post_derivs = evi_derivative[post_peak]
                    post_doys = pixel_doys_valid[post_peak]
                    post_evi    = pixel_evi_valid[post_peak]
                    
                    pixel_peak_evi = annual_max[0, yi, xi]
                    amplitude_px   = annual_amplitude[0, yi, xi]
                    min_drop       = pixel_peak_evi - (amplitude_px * min_rise_scalar)

                    on_descending_limb = post_evi <= min_drop
                    if on_descending_limb.sum() >= 2:
                        post_derivs = post_derivs[on_descending_limb]
                        post_doys   = post_doys[on_descending_limb]
                
                        min_rate_idx = np.argmin(post_derivs)
                        senescence_rate[0, yi, xi]     = post_derivs[min_rate_idx]
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


def get_context_months_from_gaps(
    chunk: xr.DataArray,
    target_year: int,
    gap_threshold_days: int = 45,
) -> bool:
    """
    Check the actual observation record.
    If the first observation gap at start or end of target year
    exceeds gap_threshold_days, context years will cause edge spikes.
    Return 0 context months in that case.
    """
    target_obs = chunk.sel(time=str(target_year))
    has_obs = target_obs.notnull().any(dim=["y", "x"])
    if not has_obs.any():
        return False   # all-NaN chunk — handled elsewhere

    obs_times = target_obs.time.values
    valid_times = obs_times[has_obs.values]

    # Gap from Jan 1 to first observation
    jan1 = np.datetime64(f"{target_year}-01-01")
    dec31 = np.datetime64(f"{target_year}-12-31")
    gap_start = int((valid_times[0]  - jan1)  / np.timedelta64(1, 'D'))
    gap_end   = int((dec31 - valid_times[-1]) / np.timedelta64(1, 'D'))

    if gap_start >= gap_threshold_days or gap_end >= gap_threshold_days:
        return False

    return True

def calc_obs_snow_background(
    chunk:                    xr.DataArray,
    threshold_background_pct: float = 0.15,  # fraction of amplitude above min
    low_pct:                  float = 0.10,  # percentile of all valid obs
    snow_doy_start:           int   = 300,
    snow_doy_end:             int   = 100,
    min_snow_obs:             int   = 3,
) -> xr.DataArray:

    doy       = chunk.time.dt.dayofyear
    snow_mask = (doy >= snow_doy_start) | (doy <= snow_doy_end)
    snow_obs  = chunk.isel(time=snow_mask)
    n_valid   = snow_obs.notnull().sum(dim="time")

    # Path 1: winter obs available: low percentile of winter window
    snow_background = (
        snow_obs
        .quantile(low_pct, dim="time", skipna=True)
        .drop_vars("quantile", errors="ignore")
        .clip(min=0.0)
    )

    # Path 2: no winter obs (Arctic)
    #   low percentile of ALL valid obs + amplitude fraction
    #   this sits at the dormant floor, not dragged by noise/edge obs
    all_low = (
        chunk
        .quantile(low_pct, dim="time", skipna=True)
        .drop_vars("quantile", errors="ignore")
        .clip(min=0.0)
    )
    chunk_min = chunk.min(dim="time", skipna=True)
    chunk_max = chunk.max(dim="time", skipna=True)
    amplitude = chunk_max - chunk_min

    # floor = low percentile + small amplitude fraction
    # prevents background from sitting below real dormant signal
    amplitude_background = (all_low + amplitude * threshold_background_pct).clip(min=0.0)

    print(f"  snow_background    : min={float(snow_background.min()):.4f} "
          f"mean={float(snow_background.mean()):.4f} "
          f"max={float(snow_background.max()):.4f}")
    print(f"  amplitude_background: min={float(amplitude_background.min()):.4f} "
          f"mean={float(amplitude_background.mean()):.4f} "
          f"max={float(amplitude_background.max()):.4f}")

    background = xr.where(n_valid >= min_snow_obs, snow_background, amplitude_background)
    background = background.drop_vars("quantile", errors="ignore")

    print(f"  final background   : min={float(background.min()):.4f} "
          f"mean={float(background.mean()):.4f} "
          f"max={float(background.max()):.4f}")

    return background
    

def full_pipeline_chunk(chunk: xr.DataArray,
                        doy_data: xr.DataArray = None,
                        apply_threshold: bool = True,
                        min_evi_threshold: float = -1.0,
                        max_evi_threshold: float = 1.0,
                        threshold_greenup_pct: float = 0.15,
                        fill_snow_gaps: bool = False,
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
        target_obs_despiked = chunk.sel(time=str(target_year))
        target_obs_raw  = chunk_post_threshold.sel(time=str(target_year)) if testing_mode else None
        if testing_mode:
            removed = target_obs_raw.notnull() & target_obs_despiked.isnull()
            removed_times = target_obs_raw.time[removed.any(dim=["y", "x"])].values
            
            print(f"  Despiked dates in {target_year}:")
            for t in removed_times:
                print(f"    {pd.Timestamp(t).date()}")
            
    chunk_post_despike = chunk.copy(deep=True) if testing_mode else None

    # Step 3: calculate scene revisit and quality pixels before the spline fit, 365 DOY data is generated
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
    print("Step 4: Apply spline")
    use_context_months = get_context_months_from_gaps(chunk=chunk,target_year=target_year)    
    fill_snow_gaps     = not use_context_months 
    print(f"  use_context_months : {use_context_months}")
    print(f"  fill_snow_gaps     : {fill_snow_gaps}")  
    
    smoothed_daily = smooth_evi_chunk_for_year(
        chunk,
        target_year,
        testing_mode=testing_mode,
        use_context_months=use_context_months,
        _pool=_pool,
        n_jobs=n_jobs
    )
    chunk_post_spline = smoothed_daily.copy(deep=True) if testing_mode else None
    
    if fill_snow_gaps:
        # Step 5: Fill snow gaps using naive min EVI2 value
        print("Step 5: Snow gap fill", flush=True)
        background_threshold = calc_obs_snow_background(chunk) 
        target_obs  = chunk.sel(time=str(target_year))
        is_valid = target_obs.notnull()
        has_any   = is_valid.any(dim="time")              
        first_idx = is_valid.argmax(dim="time")            
        last_idx  = (target_obs.sizes["time"] - 1 
                     - is_valid.isel(time=slice(None, None, -1)).argmax(dim="time"))                                              
        first_obs_doy = target_obs.time.dt.dayofyear.isel(time=first_idx).where(has_any)
        last_obs_doy  = target_obs.time.dt.dayofyear.isel(time=last_idx).where(has_any)
        smoothed_year = smoothed_daily.sel(time=str(target_year))
        daily_doy     = smoothed_year.time.dt.dayofyear
        bg            = background_threshold.drop_vars("quantile", errors="ignore")
    
        before_first = daily_doy < first_obs_doy
        after_last   = daily_doy > last_obs_doy
        no_data      = ~has_any
    
        # Spline value at the exact boundary day [y, x]
        first_doy_idx  = (daily_doy == first_obs_doy)
        last_doy_idx   = (daily_doy == last_obs_doy)
    
        spline_at_first = smoothed_year.where(first_doy_idx).max(dim="time")
        spline_at_last  = smoothed_year.where(last_doy_idx).max(dim="time")
    
        # Fill = min(background, spline at boundary) — never step up OR down
        lead_fill  = xr.where(spline_at_first < bg, spline_at_first, bg)
        trail_fill = xr.where(spline_at_last  < bg, spline_at_last,  bg)
    
        smoothed_year = smoothed_year.where(~(before_first | no_data), other=lead_fill)
        smoothed_year = smoothed_year.where(~(after_last   | no_data), other=trail_fill)    
        smoothed_year_pheno = smoothed_daily.sel(time=str(target_year)).where(
            (daily_doy >= first_obs_doy) & (daily_doy <= last_obs_doy)
        )
    else:
        smoothed_year_pheno = smoothed_daily.sel(time=str(target_year))
        
    chunk_post_snow_fill = smoothed_year.copy(deep=True) if testing_mode else None

    # Step 6: Annual phenometrics
    # smoothed_year = smoothed_daily.where(smoothed_daily.time.dt.year == target_year)        
    print("  Step 7: Calculate phenometrics")
    pheno = annual_phenometrics_chunk(
        smoothed_year_pheno,
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
            'post_snow_fill': chunk_post_snow_fill,
        }

    return results