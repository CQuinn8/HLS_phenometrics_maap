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
        return np.column_stack(cols)  # (n, 1+2*H)

    return _dm(doys_fit), _dm(doys_pred)


def fit_annual_harmonic_curves(
        annual_da: xr.DataArray,
        n_harmonics: int = 3,
        doy_out: Optional[np.ndarray] = None,
        min_valid_obs: int = 6,
        mask_outside_obs_range: bool = True,  # don't allow to relate curves outside target year valid obs window
        obs_range_buffer_days: int = 16,
) -> np.ndarray:
    if doy_out is None:
        doy_out = np.arange(1, 366, dtype=np.float32)

    doys_fit = annual_da.time.dt.dayofyear.values.astype(float)
    values = annual_da.values
    if values.ndim == 2:  # (time, y) — missing x dim
        values = values[:, :, np.newaxis]
    if values.ndim == 1:  # (time,) — single pixel
        values = values[:, np.newaxis, np.newaxis]

    ny, nx = values.shape[1], values.shape[2]
    n_pix = ny * nx
    vals_flat = values.reshape(len(doys_fit), n_pix)

    X_fit, X_pred = _build_harmonic_matrices(doys_fit, doy_out, n_harmonics)
    curves_flat = np.full((len(doy_out), n_pix), np.nan, dtype=np.float32)

    valid_per_pix = (~np.isnan(vals_flat)).sum(axis=0)
    fit_pixels = np.where(valid_per_pix >= min_valid_obs)[0]

    for pix in fit_pixels:
        mask = ~np.isnan(vals_flat[:, pix])
        if mask.sum() < X_fit.shape[1]:
            continue
        try:
            coeffs, *_ = np.linalg.lstsq(X_fit[mask], vals_flat[mask, pix], rcond=None)
            curve = X_pred @ coeffs

            if mask_outside_obs_range:
                # NaN the curve outside the observed DOY range + buffer
                obs_doys = doys_fit[mask]
                doy_min = obs_doys.min() - obs_range_buffer_days
                doy_max = obs_doys.max() + obs_range_buffer_days
                in_range = (doy_out >= doy_min) & (doy_out <= doy_max)
                curve = np.where(in_range, curve, np.nan)

            curves_flat[:, pix] = curve
        except np.linalg.LinAlgError:
            pass

    return curves_flat.reshape(len(doy_out), ny, nx)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Per-pixel similarity scoring  (context year vs target year)
# ─────────────────────────────────────────────────────────────────────────────

def score_curve_similarity(
        target_curve: np.ndarray,  # (365, ny, nx)
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

    t_flat = target_curve.reshape(365, -1)
    c_flat = context_curve.reshape(365, -1)

    t_mu = np.nanmean(t_flat, axis=0)
    c_mu = np.nanmean(c_flat, axis=0)
    dt = t_flat - t_mu
    dc = c_flat - c_mu

    num = np.nansum(dt * dc, axis=0)
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
        min_curve_value: float = 0.02,  # both curves must exceed this
        max_scale: float = 2.5,
        min_scale: float = 0.40,
) -> np.ndarray:
    """
    Returns NaN (blocking infill) where either curve is near-zero / unreliable.
    """
    either_unreliable = (
            np.isnan(target_curve_at_doy) |
            np.isnan(context_curve_at_doy) |
            (np.abs(context_curve_at_doy) < min_denominator) |
            (np.abs(target_curve_at_doy) < min_curve_value) |
            (np.abs(context_curve_at_doy) < min_curve_value)
    )

    with np.errstate(invalid='ignore', divide='ignore'):
        scale = np.where(
            np.abs(context_curve_at_doy) > min_denominator,
            target_curve_at_doy / context_curve_at_doy,
            1.0
        )
    scale = np.clip(scale, min_scale, max_scale)
    result = context_val * scale

    # Block the infill entirely where curves are unreliable
    return np.where(either_unreliable, np.nan, result)


def infill_gaps_from_context_years(
        chunk_despiked: xr.DataArray,
        target_year: int,
        context_curves: dict,
        similarity_scores: dict,
        min_similarity: float = 0.50,
        testing_mode: bool = False,
) -> Tuple[xr.DataArray, Optional[dict]]:
    doy_window = 10

    target_da = chunk_despiked.sel(time=str(target_year))
    target_doys = target_da.time.dt.dayofyear.values.astype(int)

    context_years = sorted([y for y in context_curves if y != target_year])
    context_das = {y: chunk_despiked.sel(time=str(y)) for y in context_years}

    if not context_years:
        return target_da, None

    # ── Per-pixel best context year ───────────────────────────────────────
    sim_stack = np.stack([similarity_scores[y] for y in context_years], axis=0)
    sim_stack_gated = np.where(sim_stack >= min_similarity, sim_stack, np.nan)
    any_valid_sim = np.any(~np.isnan(sim_stack_gated), axis=0)
    best_ctx_idx = np.where(
        any_valid_sim,
        np.nanargmax(
            np.where(np.isnan(sim_stack_gated), -np.inf, sim_stack_gated), axis=0
        ),
        0
    )

    # ── Step 1: identify context timesteps with no nearby target timestep ─
    extra_das = []
    target_valid = ~np.isnan(target_da.values)

    for ctx_year in context_years:
        ctx_da = context_das[ctx_year]
        ctx_doys = ctx_da.time.dt.dayofyear.values.astype(int)
        ctx_idx = context_years.index(ctx_year)
        is_best = (best_ctx_idx == ctx_idx) & any_valid_sim

        for c_idx, ctx_doy in enumerate(ctx_doys):
            nearby_timestamps = np.abs(target_doys - ctx_doy) <= doy_window
            has_nearby_valid = target_valid[nearby_timestamps].any(axis=0)  # (ny, nx)
            inject_here = is_best & ~has_nearby_valid

            if not inject_here.any():
                continue

            ctx_obs = ctx_da.values[c_idx]
            inject_vals = np.where(
                inject_here & ~np.isnan(ctx_obs),
                ctx_obs,
                np.nan
            ).astype(np.float32)

            if np.isnan(inject_vals).all():
                continue

            ts = pd.Timestamp(ctx_da.time.values[c_idx])
            new_ts = ts.replace(year=target_year)
            new_ts_np = np.datetime64(new_ts)

            # If timestamp already exists, fill its NaN pixels directly
            # rather than injecting a duplicate
            if new_ts_np in target_da.time.values:
                t_match = np.where(target_da.time.values == new_ts_np)[0]
                if len(t_match) > 0:
                    existing = target_da.values[t_match[0]]  # (ny, nx)
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
                    'y': target_da.y,
                    'x': target_da.x,
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
    T, ny, nx = target_vals.shape

    # ── Step 3: fill NaN gaps at existing timestamps ─────
    # Compute first/last obs DOY per pixel once before loop
    valid_mask = ~np.isnan(target_vals)
    any_valid = valid_mask.any(axis=0)
    infill_count = np.zeros((ny, nx), dtype=np.int16)

    for t_idx in range(T):
        gap_mask = np.isnan(target_vals[t_idx])
        if not gap_mask.any():
            continue

        doy = target_doys[t_idx]

        # check for any nearby valid obs in either direction
        has_nearby_target = np.zeros((ny, nx), dtype=bool)
        for nb_idx in range(T):
            if nb_idx == t_idx:
                continue
            if abs(int(target_doys[nb_idx]) - doy) > doy_window:
                continue
            has_nearby_target |= ~np.isnan(target_vals[nb_idx])

        isolated_gap = gap_mask & ~has_nearby_target & any_valid_sim

        if not isolated_gap.any():
            continue

        fill_vals = np.full((ny, nx), np.nan, dtype=np.float32)

        for ctx_year in context_years:
            ctx_da = context_das[ctx_year]
            ctx_doys = ctx_da.time.dt.dayofyear.values.astype(int)
            ctx_idx = context_years.index(ctx_year)
            is_best = best_ctx_idx == ctx_idx

            pixels_to_fill = isolated_gap & is_best
            if not pixels_to_fill.any():
                continue

            doy_diff = np.abs(ctx_doys - doy)
            nearby_idx = np.where(doy_diff <= doy_window)[0]
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


def build_context_infilled_observations(
    chunk_despiked: xr.DataArray,
    target_year: int,
    n_harmonics: int = 3,
    min_similarity: float = 0.50,
    min_valid_obs: int = 6,
    testing_mode: bool = False,
) -> Tuple[xr.DataArray, Optional[dict]]:
    """
    Full context-year infill orchestrator.

    1. Identifies available context years in chunk_despiked.
    2. Fits per-pixel harmonic annual curves for each year.
    3. Scores similarity of each context year to the target year.
    4. Infills gaps in target year observations using scaled context obs.

    Parameters
    ----------
    chunk_despiked : Despiked DataArray spanning up to 3 years (time, y, x).
    target_year    : The year phenometrics will be extracted for.
    n_harmonics    : Fourier harmonics for annual curve (default 3).
    min_similarity : Per-pixel correlation threshold to allow infilling.
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
        print(f"  [context_infill] Mean similarity {yr} -> {target_year}: {mean_sim:.3f}")

    # ── Infill gaps ───────────────────────────────────────────────────────────
    augmented, diagnostics = infill_gaps_from_context_years(
        chunk_despiked      = chunk_despiked,
        target_year         = target_year,
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
