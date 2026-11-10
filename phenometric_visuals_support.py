import numpy as np
import xarray as xr
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  context_infill.py   (new module, or append to your phenometrics utils)
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import xarray as xr
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DOY harmonic curve fitter  (vectorised over all pixels at once)
# ─────────────────────────────────────────────────────────────────────────────

def _build_harmonic_matrices(
    doys_fit: np.ndarray,
    doys_pred: np.ndarray,
    n_harmonics: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return design matrices (X_fit, X_pred) for a Fourier harmonic model."""
    def _dm(doys):
        t = 2.0 * np.pi * doys / 365.0
        cols = [np.ones(len(doys))]
        for k in range(1, n_harmonics + 1):
            cols += [np.cos(k * t), np.sin(k * t)]
        return np.column_stack(cols)          # (n, 1+2*H)

    return _dm(doys_fit), _dm(doys_pred)


def fit_annual_harmonic_curves(
    annual_da: xr.DataArray,
    n_harmonics: int = 3,
    doy_out: Optional[np.ndarray] = None,
    min_valid_obs: int = 6,
    mask_outside_obs_range: bool = True,   # ← NEW
    obs_range_buffer_days: int = 16,       # ← NEW
) -> np.ndarray:
    if doy_out is None:
        doy_out = np.arange(1, 366, dtype=np.float32)

    doys_fit  = annual_da.time.dt.dayofyear.values.astype(float)
    values    = annual_da.values
    if values.ndim == 2:                   # (time, y) — missing x dim
        values = values[:, :, np.newaxis]
    if values.ndim == 1:                   # (time,) — single pixel
        values = values[:, np.newaxis, np.newaxis]
    
    ny, nx    = values.shape[1], values.shape[2]
    n_pix     = ny * nx
    vals_flat = values.reshape(len(doys_fit), n_pix)

    X_fit, X_pred = _build_harmonic_matrices(doys_fit, doy_out, n_harmonics)
    curves_flat   = np.full((len(doy_out), n_pix), np.nan, dtype=np.float32)

    valid_per_pix = (~np.isnan(vals_flat)).sum(axis=0)
    fit_pixels    = np.where(valid_per_pix >= min_valid_obs)[0]

    for pix in fit_pixels:
        mask = ~np.isnan(vals_flat[:, pix])
        if mask.sum() < X_fit.shape[1]:
            continue
        try:
            coeffs, *_ = np.linalg.lstsq(X_fit[mask], vals_flat[mask, pix], rcond=None)
            curve = X_pred @ coeffs

            if mask_outside_obs_range:
                # NaN the curve outside the observed DOY range + buffer
                obs_doys  = doys_fit[mask]
                doy_min   = obs_doys.min() - obs_range_buffer_days
                doy_max   = obs_doys.max() + obs_range_buffer_days
                in_range  = (doy_out >= doy_min) & (doy_out <= doy_max)
                curve     = np.where(in_range, curve, np.nan)

            curves_flat[:, pix] = curve
        except np.linalg.LinAlgError:
            pass

    return curves_flat.reshape(len(doy_out), ny, nx)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Per-pixel similarity scoring  (context year vs target year)
# ─────────────────────────────────────────────────────────────────────────────

def score_curve_similarity(
    target_curve: np.ndarray,   # (365, ny, nx)
    context_curve: np.ndarray,  # (365, ny, nx)
) -> np.ndarray:
    """
    Compute per-pixel Pearson correlation between two annual harmonic curves.

    Returns
    -------
    similarity : np.ndarray, shape (ny, nx), values in [-1, 1].
                 NaN where either curve has no valid data.
    """
    ny, nx = target_curve.shape[1], target_curve.shape[2]

    t_flat = target_curve.reshape(365, -1)    # (365, P)
    c_flat = context_curve.reshape(365, -1)

    # Vectorised Pearson r along axis-0
    t_mu  = np.nanmean(t_flat, axis=0)
    c_mu  = np.nanmean(c_flat, axis=0)
    dt    = t_flat - t_mu
    dc    = c_flat - c_mu

    num   = np.nansum(dt * dc,        axis=0)
    denom = np.sqrt(
        np.nansum(dt ** 2, axis=0) *
        np.nansum(dc ** 2, axis=0)
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(denom > 0, num / denom, np.nan)

    return r.reshape(ny, nx).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Gap-filling target year with scaled context observations
# ─────────────────────────────────────────────────────────────────────────────

def _scale_context_obs_to_target(
    context_val: np.ndarray,
    target_curve_at_doy: np.ndarray,
    context_curve_at_doy: np.ndarray,
    min_denominator: float = 0.02,
    min_curve_value: float = 0.02,     # ← NEW: both curves must exceed this
    max_scale: float = 2.5,            # ← tighter than previous 4.0
    min_scale: float = 0.40,           # ← tighter than previous 0.25
) -> np.ndarray:
    """
    Returns NaN (blocking infill) where either curve is near-zero / unreliable.
    """
    either_unreliable = (
        np.isnan(target_curve_at_doy)  |
        np.isnan(context_curve_at_doy) |
        (np.abs(context_curve_at_doy) < min_denominator) |
        (np.abs(target_curve_at_doy)  < min_curve_value) |   # target near zero = bad
        (np.abs(context_curve_at_doy) < min_curve_value)     # context near zero = bad
    )

    with np.errstate(invalid='ignore', divide='ignore'):
        scale = np.where(
            np.abs(context_curve_at_doy) > min_denominator,
            target_curve_at_doy / context_curve_at_doy,
            1.0
        )
    scale  = np.clip(scale, min_scale, max_scale)
    result = context_val * scale

    # Block the infill entirely where curves are unreliable
    return np.where(either_unreliable, np.nan, result)


def infill_gaps_from_context_years(
    chunk_despiked: xr.DataArray,
    target_year: int,
    target_curve: np.ndarray,
    context_curves: dict,
    similarity_scores: dict,
    min_similarity: float = 0.50,
    testing_mode: bool = False,
) -> Tuple[xr.DataArray, Optional[dict]]:

    DOY_WINDOW = 10

    target_da   = chunk_despiked.sel(time=str(target_year))
    target_doys = target_da.time.dt.dayofyear.values.astype(int)

    context_years = sorted([y for y in context_curves if y != target_year])
    context_das   = {y: chunk_despiked.sel(time=str(y)) for y in context_years}

    if not context_years:
        return target_da, None

    # ── Per-pixel best context year ───────────────────────────────────────
    sim_stack       = np.stack([similarity_scores[y] for y in context_years], axis=0)
    sim_stack_gated = np.where(sim_stack >= min_similarity, sim_stack, np.nan)
    any_valid_sim   = np.any(~np.isnan(sim_stack_gated), axis=0)
    best_ctx_idx    = np.where(
        any_valid_sim,
        np.nanargmax(
            np.where(np.isnan(sim_stack_gated), -np.inf, sim_stack_gated), axis=0
        ),
        0
    )

    # ── Step 1: identify context timesteps with no nearby target timestep ─
    # These are genuinely missing windows — need new timestamps injected
    # ── Step 1 ───────────────────────────────────────────────────────────────
    extra_das = []
    target_valid = ~np.isnan(target_da.values)   # (T, ny, nx)  ← ADD THIS

    for ctx_year in context_years:
        ctx_da   = context_das[ctx_year]
        ctx_doys = ctx_da.time.dt.dayofyear.values.astype(int)
        ctx_idx  = context_years.index(ctx_year)
        is_best  = (best_ctx_idx == ctx_idx) & any_valid_sim

        for c_idx, ctx_doy in enumerate(ctx_doys):
            nearby_timestamps = np.abs(target_doys - ctx_doy) <= DOY_WINDOW
            has_nearby_valid  = target_valid[nearby_timestamps].any(axis=0)  # (ny, nx)
            inject_here       = is_best & ~has_nearby_valid

            if not inject_here.any():
                continue

            ctx_obs     = ctx_da.values[c_idx]
            inject_vals = np.where(
                inject_here & ~np.isnan(ctx_obs),
                ctx_obs,
                np.nan
            ).astype(np.float32)

            if np.isnan(inject_vals).all():
                continue

            ts     = pd.Timestamp(ctx_da.time.values[c_idx])
            new_ts = ts.replace(year=target_year)
            new_ts_np = np.datetime64(new_ts)

            # If timestamp already exists, fill its NaN pixels directly
            # rather than injecting a duplicate
            if new_ts_np in target_da.time.values:
                t_match = np.where(target_da.time.values == new_ts_np)[0]
                if len(t_match) > 0:
                    existing = target_da.values[t_match[0]]   # (ny, nx)
                    # Only fill pixels that are NaN in the existing timestamp
                    merged = np.where(
                        np.isnan(existing) & ~np.isnan(inject_vals),
                        inject_vals,
                        existing
                    )
                    target_da.values[t_match[0]] = merged
                continue

            new_da = xr.DataArray(
                inject_vals[np.newaxis, :, :],
                dims=target_da.dims,
                coords={
                    'time': [new_ts_np],
                    'y':    target_da.y,
                    'x':    target_da.x,
                }
            )
            extra_das.append(new_da)

    # ── Step 2: merge injected timesteps into target, sort by time ────────
    if extra_das:
        target_augmented = xr.concat(
            [target_da] + extra_das, dim='time'
        ).sortby('time')
        print(f"  [infill] Injected {len(extra_das)} new timesteps "
              f"from context years")
    else:
        target_augmented = target_da

    target_vals = target_augmented.values.copy()
    target_doys = target_augmented.time.dt.dayofyear.values.astype(int)
    T, ny, nx   = target_vals.shape

    # ── Step 3: fill NaN gaps at existing timestamps (original logic) ─────
    # Compute first/last obs DOY per pixel once before loop
    valid_mask    = ~np.isnan(target_vals)
    any_valid     = valid_mask.any(axis=0)
    first_valid_t = np.argmax(valid_mask, axis=0)
    last_valid_t  = T - 1 - np.argmax(valid_mask[::-1], axis=0)
    first_obs_doy = np.where(any_valid, target_doys[first_valid_t], 999)
    last_obs_doy  = np.where(any_valid, target_doys[last_valid_t],   -1)

    infill_count = np.zeros((ny, nx), dtype=np.int16)

    for t_idx in range(T):
        gap_mask = np.isnan(target_vals[t_idx])
        if not gap_mask.any():
            continue

        doy = target_doys[t_idx]

        # Simpler: just check for any nearby valid obs in either direction
        has_nearby_target = np.zeros((ny, nx), dtype=bool)
        for nb_idx in range(T):
            if nb_idx == t_idx:
                continue
            if abs(int(target_doys[nb_idx]) - doy) > DOY_WINDOW:
                continue
            has_nearby_target |= ~np.isnan(target_vals[nb_idx])

        isolated_gap = gap_mask & ~has_nearby_target & any_valid_sim
         
        if not isolated_gap.any():
            continue

        fill_vals = np.full((ny, nx), np.nan, dtype=np.float32)

        for ctx_year in context_years:
            ctx_da   = context_das[ctx_year]
            ctx_doys = ctx_da.time.dt.dayofyear.values.astype(int)
            ctx_idx  = context_years.index(ctx_year)
            is_best  = best_ctx_idx == ctx_idx

            pixels_to_fill = isolated_gap & is_best
            if not pixels_to_fill.any():
                continue

            doy_diff   = np.abs(ctx_doys - doy)
            nearby_idx = np.where(doy_diff <= DOY_WINDOW)[0]
            if len(nearby_idx) == 0:
                continue

            closest = nearby_idx[np.argmin(doy_diff[nearby_idx])]
            ctx_obs = ctx_da.values[closest]

            fill_vals = np.where(
                pixels_to_fill & ~np.isnan(ctx_obs),
                ctx_obs,
                fill_vals
            )

        filled_here = isolated_gap & ~np.isnan(fill_vals)
        target_vals[t_idx] = np.where(filled_here, fill_vals, target_vals[t_idx])
        infill_count += filled_here.astype(np.int16)

    augmented = target_augmented.copy(data=target_vals)
    diagnostics = {'infill_count': infill_count} if testing_mode else None
    return augmented, diagnostics

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Top-level orchestrator  (called from full_pipeline_chunk)
# ─────────────────────────────────────────────────────────────────────────────

def build_context_infilled_observations(
    chunk_despiked: xr.DataArray,
    target_year: int,
    n_harmonics: int = 3,
    min_similarity: float = 0.50,
    scale_to_target: bool = True,
    min_valid_obs: int = 6,
    testing_mode: bool = False,
) -> Tuple[xr.DataArray, Optional[dict]]:
    """
    Full context-year infill orchestrator.

    1. Identifies available context years in `chunk_despiked`.
    2. Fits per-pixel harmonic annual curves for each year.
    3. Scores similarity of each context year to the target year.
    4. Infills gaps in target year observations using scaled context obs.

    Parameters
    ----------
    chunk_despiked : Despiked DataArray spanning up to 3 years (time, y, x).
    target_year    : The year phenometrics will be extracted for.
    n_harmonics    : Fourier harmonics for annual curve (default 3).
    min_similarity : Per-pixel correlation threshold to allow infilling.
    scale_to_target: Magnitude-scale context obs to match target year curve.
    min_valid_obs  : Min obs per pixel to attempt harmonic fit.
    testing_mode   : Return extended diagnostics if True.

    Returns
    -------
    augmented_target_da : DataArray (time, y, x) for target year,
                          NaN gaps filled where context data was available.
    diagnostics         : dict or None
    """
    all_years = sorted({int(t) for t in chunk_despiked.time.dt.year.values})
    context_years = [y for y in all_years if y != target_year]

    print(f"  [context_infill] target={target_year}, "
          f"context years={context_years}, harmonics={n_harmonics}")

    # ── Fit harmonic curves for each year ────────────────────────────────────
    curves = {}
    for yr in all_years:
        yr_da = chunk_despiked.sel(time=str(yr))
        if len(yr_da.time) == 0:
            print(f"  [context_infill] WARNING: no data for year {yr}, skipping")
            continue
        curves[yr] = fit_annual_harmonic_curves(
            yr_da,
            n_harmonics=n_harmonics,
            min_valid_obs=min_valid_obs,
        )
        print(f"  [context_infill] Harmonic curve fitted for {yr}")

    if target_year not in curves:
        print("  [context_infill] WARNING: No target year curve — returning raw target obs")
        return chunk_despiked.sel(time=str(target_year)), None

    # ── Score context year similarity to target year ──────────────────────────
    similarity_scores = {}
    for yr in context_years:
        if yr not in curves:
            continue
        similarity_scores[yr] = score_curve_similarity(
            curves[target_year], curves[yr]
        )
        mean_sim = np.nanmean(similarity_scores[yr])
        print(f"  [context_infill] Mean similarity {yr}→{target_year}: {mean_sim:.3f}")

    # ── Infill gaps ───────────────────────────────────────────────────────────
    augmented, diagnostics = infill_gaps_from_context_years(
        chunk_despiked      = chunk_despiked,
        target_year         = target_year,
        target_curve        = curves[target_year],
        context_curves      = {yr: curves[yr] for yr in context_years if yr in curves},
        similarity_scores   = similarity_scores,
        min_similarity      = min_similarity,
        testing_mode        = testing_mode,
    )

    if testing_mode and diagnostics is not None:
        diagnostics.update({
            'harmonic_curves': curves,
            'similarity_scores': similarity_scores,
        })

    total_filled = np.nansum(diagnostics['infill_count']) if diagnostics else '?'
    print(f"  [context_infill] Total pixel-timesteps infilled: {total_filled}")

    return augmented, diagnostics


# ─────────────────────────────────────────────────────────────────────────────
#  context_infill_diagnostics.py
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
from typing import Optional, Tuple
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Colour / style constants
# ─────────────────────────────────────────────────────────────────────────────
_YEAR_COLORS   = ['#2c7bb6', '#d7191c', '#1a9641']   # up to 3 context years
_TARGET_COLOR  = '#f4a11d'
_FILL_COLOR    = '#9b59b6'
_CURVE_ALPHA   = 0.75
_OBS_ALPHA     = 0.85


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Spatial maps
# ─────────────────────────────────────────────────────────────────────────────

def plot_infill_spatial_summary(
    diagnostics: dict,
    target_year: int,
    chunk_pre_infill: xr.DataArray,   # despiked, before infill  (time, y, x)
    chunk_post_infill: xr.DataArray,  # after infill              (time, y, x)
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    4-panel spatial summary:
      [A] # valid obs before infill   [B] # valid obs after infill
      [C] # obs added by infill        [D] per-pixel similarity (best context yr)
    """
    pre  = chunk_pre_infill.sel( time=str(target_year))
    post = chunk_post_infill.sel(time=str(target_year))

    valid_pre  = (~np.isnan(pre.values )).sum(axis=0)   # (ny, nx)
    valid_post = (~np.isnan(post.values)).sum(axis=0)
    delta      = valid_post - valid_pre                  # obs added

    # Best-context similarity (max r across context years)
    sim_scores = diagnostics.get('similarity_scores', {})
    if sim_scores:
        best_sim = np.nanmax(
            np.stack(list(sim_scores.values()), axis=0), axis=0
        )
    else:
        best_sim = np.full(delta.shape, np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"Context-year infill spatial summary  |  target year {target_year}",
        fontsize=13, fontweight='bold'
    )

    panels = [
        (valid_pre,  'Valid obs — pre-infill',  'YlGn',  None),
        (valid_post, 'Valid obs — post-infill', 'YlGn',  None),
        (delta,      'Obs added by infill',     'PuRd',  None),
        (best_sim,   'Best context similarity', 'RdYlGn', (-1, 1)),
    ]

    ims = []
    for ax, (data, title, cmap, vrange) in zip(axes, panels):
        vmin, vmax = vrange if vrange else (np.nanmin(data), np.nanmax(data))
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest', aspect='equal')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x pixel"); ax.set_ylabel("y pixel")
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        ims.append(im)

    plt.tight_layout()
    

def plot_similarity_maps(
    diagnostics: dict,
    target_year: int,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    One panel per context year showing per-pixel Pearson r to target year.
    Overlays a hatched mask where r < min_similarity (0.5 by default).
    """
    sim_scores = diagnostics.get('similarity_scores', {})
    if not sim_scores:
        print("  [diag] No similarity scores found in diagnostics.")
        return None

    n = len(sim_scores)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"Per-pixel harmonic curve similarity to {target_year}",
        fontsize=13, fontweight='bold'
    )

    for ax, (yr, sim) in zip(axes, sorted(sim_scores.items())):
        im = ax.imshow(sim, cmap='RdYlGn', vmin=-1, vmax=1,
                       interpolation='nearest', aspect='equal')
        # Hatch pixels below threshold
        below = sim < 0.50
        ax.contourf(
            np.arange(sim.shape[1]), np.arange(sim.shape[0]),
            below.astype(float), levels=[0.5, 1.5],
            hatches=['///'], colors='none', alpha=0.0
        )
        ax.set_title(f"Context {yr} → {target_year}\n"
                     f"mean r = {np.nanmean(sim):.3f}", fontsize=10)
        ax.set_xlabel("x pixel"); ax.set_ylabel("y pixel")
        plt.colorbar(im, ax=ax, shrink=0.75, label='Pearson r')

    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Harmonic curve plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_harmonic_curves_pixel(
    diagnostics: dict,
    chunk_despiked: xr.DataArray,
    target_year: int,
    pixels: list,                   # [(y0, x0), (y1, x1), ...]
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    For each requested pixel, plot:
      • Despiked observations for every year (scatter)
      • Fitted harmonic curve for every year (line)
    Useful for verifying curve quality before infill.
    """
    curves      = diagnostics.get('harmonic_curves', {})
    all_years   = sorted(curves.keys())
    doy_axis    = np.arange(1, 366)
    n_pix       = len(pixels)

    fig, axes = plt.subplots(n_pix, 1, figsize=(14, 4 * n_pix), squeeze=False)
    fig.suptitle(
        f"Harmonic annual curves — target {target_year}",
        fontsize=13, fontweight='bold'
    )

    year_col = {yr: _YEAR_COLORS[i % len(_YEAR_COLORS)]
                for i, yr in enumerate(all_years)}
    year_col[target_year] = _TARGET_COLOR

    for row, (py, px) in enumerate(pixels):
        ax = axes[row, 0]

        for yr in all_years:
            yr_da   = chunk_despiked.sel(time=str(yr))
            obs_doy = yr_da.time.dt.dayofyear.values
            obs_val = yr_da.values[:, py, px]
            valid   = ~np.isnan(obs_val)
            c       = year_col[yr]
            lw      = 2.2 if yr == target_year else 1.4
            ls      = '-'  if yr == target_year else '--'

            ax.scatter(obs_doy[valid], obs_val[valid],
                       color=c, s=30, alpha=_OBS_ALPHA, zorder=4,
                       label=f'{yr} obs' if row == 0 else '_')
            if yr in curves:
                ax.plot(doy_axis, curves[yr][:, py, px],
                        color=c, lw=lw, ls=ls, alpha=_CURVE_ALPHA,
                        label=f'{yr} curve' if row == 0 else '_')

        ax.set_title(f"Pixel ({py}, {px})", fontsize=10)
        ax.set_xlabel("DOY"); ax.set_ylabel("EVI2")
        ax.set_xlim(1, 365); ax.grid(True, alpha=0.3)
        ax.set_ylim(0.1, 0.8) 

    # Shared legend on first panel
    axes[0, 0].legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Before / after infill time series
# ─────────────────────────────────────────────────────────────────────────────

def plot_infill_timeseries_pixel(
    chunk_pre_infill:  xr.DataArray,
    chunk_post_infill: xr.DataArray,
    diagnostics: dict,
    target_year: int,
    pixels: list,
    post_spline: Optional[xr.DataArray] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:

    curves  = diagnostics.get('harmonic_curves',  {})
    sim_all = diagnostics.get('similarity_scores', {})
    doy_axis = np.arange(1, 366)

    pre_da  = chunk_pre_infill.sel( time=str(target_year))
    post_da = chunk_post_infill.sel(time=str(target_year))

    pre_doy  = pre_da.time.dt.dayofyear.values.astype(int)
    post_doy = post_da.time.dt.dayofyear.values.astype(int)

    context_years = [y for y in sorted(curves.keys()) if y != target_year]
    n_pix = len(pixels)

    fig, axes = plt.subplots(n_pix, 1, figsize=(15, 5 * n_pix), squeeze=False)
    fig.suptitle(
        f"Context-year infill — before / after  |  {target_year}",
        fontsize=13, fontweight='bold'
    )

    for row, (py, px) in enumerate(pixels):
        ax = axes[row, 0]

        # ── context harmonic curves ────────────────────────────────────────
        for i, yr in enumerate(context_years):
            if yr in curves:
                ax.plot(doy_axis, curves[yr][:, py, px],
                        color=_YEAR_COLORS[i % len(_YEAR_COLORS)],
                        lw=1.0, ls=':', alpha=0.45,
                        label=f'{yr} harmonic')

        # ── target harmonic curve ──────────────────────────────────────────
        if target_year in curves:
            ax.plot(doy_axis, curves[target_year][:, py, px],
                    color=_TARGET_COLOR, lw=1.5, ls='--', alpha=0.7,
                    label=f'{target_year} harmonic')

        # ── original despiked obs (pre) ────────────────────────────────────
        pre_val   = pre_da.values[:, py, px]
        valid_pre = ~np.isnan(pre_val)
        ax.scatter(pre_doy[valid_pre], pre_val[valid_pre],
                   color='#555555', s=55, zorder=5, alpha=_OBS_ALPHA,
                   marker='o', label='Original despiked obs')

        # ── identify infilled points by DOY alignment ──────────────────────
        # Build DOY→value lookup for pre
        pre_doy_to_val = {d: v for d, v in zip(pre_doy, pre_val)}

        post_val = post_da.values[:, py, px]
        for d, v in zip(post_doy, post_val):
            if np.isnan(v):
                continue
            pre_v = pre_doy_to_val.get(d, np.nan)
            # Infilled = valid in post AND (DOY didn't exist in pre OR was NaN)
            if np.isnan(pre_v):
                ax.scatter(d, v, color=_FILL_COLOR, s=80, zorder=6,
                           alpha=0.95, marker='*',
                           label='Infilled from context' if 'Infilled from context' not in
                           [h.get_label() for h in ax.get_children()] else '_')

        # ── optional post-spline curve ─────────────────────────────────────
        if post_spline is not None:
            sp_yr  = post_spline.sel(time=str(target_year))
            sp_doy = sp_yr.time.dt.dayofyear.values
            sp_val = sp_yr.values[:, py, px]
            ax.plot(sp_doy, sp_val, color='#e74c3c', lw=2.0,
                    alpha=0.85, label='Spline (post infill)')

        # ── annotations ───────────────────────────────────────────────────
        n_added  = sum(
            1 for d, v in zip(post_doy, post_da.values[:, py, px])
            if not np.isnan(v) and np.isnan(pre_doy_to_val.get(d, np.nan))
        )
        sim_txt = "  ".join([
            f"r({yr})={sim_all[yr][py, px]:.2f}"
            for yr in context_years if yr in sim_all
        ])

        ax.set_title(f"Pixel ({py}, {px})  |  obs added: {n_added}  |  {sim_txt}",
                     fontsize=9)
        ax.set_xlabel("DOY"); ax.set_ylabel("EVI2")
        ax.set_xlim(1, 365); ax.set_ylim(0.0, 0.8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, ncol=3)

    plt.tight_layout()

 


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Infill count histogram + seasonal distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_infill_distribution(
    diagnostics: dict,
    chunk_pre_infill:  xr.DataArray,
    chunk_post_infill: xr.DataArray,
    target_year: int,
    save_path: Optional[Path] = None,
) -> plt.Figure:

    pre  = chunk_pre_infill.sel( time=str(target_year))
    post = chunk_post_infill.sel(time=str(target_year))

    pre_doy  = pre.time.dt.dayofyear.values.astype(int)
    post_doy = post.time.dt.dayofyear.values.astype(int)
    pre_vals  = pre.values    # (T_pre,  ny, nx)
    post_vals = post.values   # (T_post, ny, nx)

    ny, nx    = pre_vals.shape[1], pre_vals.shape[2]
    total_px  = ny * nx

    # ── infill_count: per-pixel obs added ─────────────────────────────────
    # Build DOY→valid mask for pre, then compare against post by DOY
    pre_doy_valid = {}                           # {doy: (ny,nx) bool}
    for t_idx, d in enumerate(pre_doy):
        pre_doy_valid[d] = ~np.isnan(pre_vals[t_idx])

    infill_count = diagnostics.get('infill_count', None)
    if infill_count is None:
        infill_count = np.zeros((ny, nx), dtype=np.int16)
        for t_idx, d in enumerate(post_doy):
            post_valid = ~np.isnan(post_vals[t_idx])
            pre_valid  = pre_doy_valid.get(d, np.zeros((ny, nx), dtype=bool))
            infill_count += (post_valid & ~pre_valid).astype(np.int16)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Infill distribution diagnostics  |  {target_year}",
                 fontsize=13, fontweight='bold')

    # ── A: histogram of obs added per pixel ───────────────────────────────
    ax   = axes[0]
    flat = infill_count.flatten()
    flat = flat[flat >= 0]
    max_bin = max(int(flat.max()), 1)
    ax.hist(flat, bins=range(0, max_bin + 2), color=_FILL_COLOR,
            edgecolor='white', alpha=0.85)
    ax.axvline(flat.mean(), color='k', ls='--', lw=1.5,
               label=f'mean = {flat.mean():.1f}')
    ax.set_title("Obs added per pixel", fontsize=10)
    ax.set_xlabel("# observations added"); ax.set_ylabel("# pixels")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── B: seasonal DOY distribution of infilled obs ──────────────────────
    ax = axes[1]
    bins        = np.arange(1, 382, 16)
    doy_centers = 0.5 * (bins[:-1] + bins[1:])

    # Count valid obs per timestep for histogram weighting
    orig_counts = [(~np.isnan(pre_vals[t])).sum()  for t in range(len(pre_doy))]
    fill_counts = []
    for t_idx, d in enumerate(post_doy):
        post_valid = ~np.isnan(post_vals[t_idx])
        pre_valid  = pre_doy_valid.get(d, np.zeros((ny, nx), dtype=bool))
        fill_counts.append(int((post_valid & ~pre_valid).sum()))

    orig_by_doy, _ = np.histogram(
        np.repeat(pre_doy,  orig_counts), bins=bins
    )
    fill_by_doy, _ = np.histogram(
        np.repeat(post_doy, fill_counts), bins=bins
    )

    ax.bar(doy_centers, orig_by_doy, width=14,
           color='#555555', alpha=0.6, label='Original obs')
    ax.bar(doy_centers, fill_by_doy, width=14,
           bottom=orig_by_doy, color=_FILL_COLOR,
           alpha=0.85, label='Infilled obs')
    ax.set_title("Seasonal distribution of obs (16-day bins)", fontsize=10)
    ax.set_xlabel("DOY"); ax.set_ylabel("Total pixel-observations")
    ax.set_xlim(1, 365); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── C: monthly coverage ───────────────────────────────────────────────
    ax = axes[2]
    pre_times  = pd.DatetimeIndex(pre.time.values)
    post_times = pd.DatetimeIndex(post.time.values)

    pre_cov, post_cov = [], []
    for m in range(1, 13):
        pre_m  = pre_vals[ pre_times.month  == m]
        post_m = post_vals[post_times.month == m]
        pre_cov.append(
            float((~np.isnan(pre_m)).any(axis=0).sum())  / total_px * 100
            if len(pre_m)  else 0.0
        )
        post_cov.append(
            float((~np.isnan(post_m)).any(axis=0).sum()) / total_px * 100
            if len(post_m) else 0.0
        )

    x  = np.arange(12)
    bw = 0.35
    ax.bar(x - bw/2, pre_cov,  width=bw, color='#555555', alpha=0.75, label='Pre-infill')
    ax.bar(x + bw/2, post_cov, width=bw, color=_FILL_COLOR, alpha=0.85, label='Post-infill')
    ax.set_xticks(x)
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax.set_title("Monthly pixel coverage (≥1 valid obs)", fontsize=10)
    ax.set_xlabel("Month"); ax.set_ylabel("% pixels covered")
    ax.set_ylim(0, 105); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
