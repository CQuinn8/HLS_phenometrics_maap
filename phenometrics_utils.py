from pathlib import Path
import xarray as xr
import rioxarray as rxr
import rasterio as rio
from rasterio.windows import from_bounds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
from typing import Optional
import json
import gc
import os
from joblib import Parallel
import sys


# =============================================================================
# Configuration Class
# =============================================================================

@dataclass
class ProcessingConfig:
    """Configuration for EVI processing pipeline."""
    base_path: Path
    tile_id: str
    cadence: str = 'monthly'  # 'monthly', '10day', 'daily'
    evi_pattern: str = "*.EVI2.tif"  # glob pattern for EVI files

    # Date patterns for different cadences (extracts from filename/dirname)
    DATE_PATTERNS: dict = field(default_factory=lambda: {
        'monthly': r'\.(\d{7})\.',  # matches .2018121. (YYYYDOY)
        '10day': r'\.(\d{7})\.(\d{7})\.',  # '10day': r'\.(\d{7})\.',
        'daily': r'\.(\d{7})\.\d+\.\d+\.'
    })

    # For monthly, we need to load DOY from companion file
    HAS_DOY_FILE: dict = field(default_factory=lambda: {
        'daily': False,
        '10day': True,
        'monthly': False
    })

    @property
    def date_pattern(self):
        return self.DATE_PATTERNS.get(self.cadence)

    @property
    def has_doy_file(self):
        return self.HAS_DOY_FILE.get(self.cadence, False)

    @property
    def evi_dir(self) -> Path:
        return self.base_path / self.tile_id

    @property
    def index_file(self) -> Path:
        """JSON index of all EVI files with metadata."""
        return self.evi_dir / f'evi_index_{self.cadence}.json'


# =============================================================================
# Scene Discovery & Indexing
# =============================================================================

@dataclass
class EVIScene:
    """Metadata for a single EVI scene."""
    date: datetime
    doy: int
    year: int
    filepath: Path
    doy_filepath: Path = None

    def to_dict(self):
        return {
            'date': self.date.isoformat(),
            'doy': self.doy,
            'year': self.year,
            'filepath': str(self.filepath),
            'doy_filepath': str(self.doy_filepath) if self.doy_filepath else None
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            date=datetime.fromisoformat(d['date']),
            doy=d['doy'],
            year=d['year'],
            filepath=Path(d['filepath']),
            doy_filepath=Path(d['doy_filepath']) if d.get('doy_filepath') else None
        )


def parse_date_from_filename(filename: str, pattern: str, cadence: str = 'daily') -> datetime | None:
    """Extract datetime from filename using pattern."""
    match = re.search(pattern, filename)
    if not match:
        return None

    try:
        if cadence == 'monthly':
            # For monthly, use midpoint of composite period
            start_str = match.group(1)  # YYYYDOY
            end_str = match.group(2)  # YYYYDOY

            start_year = int(start_str[:4])
            start_doy = int(start_str[4:])
            end_doy = int(end_str[4:])

            mid_doy = (start_doy + end_doy) // 2
            return datetime(start_year, 1, 1) + timedelta(days=mid_doy - 1)
        else:
            # For daily/10day, direct DOY
            date_str = match.group(1)  # YYYYDOY
            year = int(date_str[:4])
            doy = int(date_str[4:])
            return datetime(year, 1, 1) + timedelta(days=doy - 1)
    except (ValueError, OverflowError):
        return None


def discover_evi_scenes(config: ProcessingConfig, recursive: bool = True) -> list[EVIScene]:
    """
    Discover all existing EVI files and build scene index.
    Handles companion DOY files for composite products.
    """
    scenes = []
    print(config.evi_dir)

    glob_method = config.evi_dir.rglob if recursive else config.evi_dir.glob
    evi_files = sorted(glob_method("*.EVI2.tif"))

    print(f"Found {len(evi_files)} EVI files in {config.evi_dir}")

    n_with_doy = 0

    for filepath in evi_files:
        date_obj = parse_date_from_filename(filepath.name, config.date_pattern, config.cadence)

        if date_obj is None:
            print(f"  Warning: Could not parse date from {filepath.name}")
            continue

        # Look for companion DOY file
        doy_filepath = None
        if config.has_doy_file:
            # Try different DOY file naming patterns
            possible_doy_names = [
                filepath.name.replace('.EVI2.tif', '.DOY.tif'),
                filepath.name.replace('.EVI2.tif', '.doy.tif'),
                filepath.name.replace('EVI2', 'DOY'),
            ]

            for doy_name in possible_doy_names:
                potential_doy_path = filepath.parent / doy_name
                if potential_doy_path.exists():
                    doy_filepath = potential_doy_path
                    n_with_doy += 1
                    break

            if doy_filepath is None:
                # Only warn for first few missing
                if n_with_doy == 0 and len(scenes) < 3:
                    print(f"  Warning: No DOY file found for {filepath.name}")
                    print(f"    Tried: {possible_doy_names}")

        scene = EVIScene(
            date=date_obj,
            doy=date_obj.timetuple().tm_yday,
            year=date_obj.year,
            filepath=filepath,
            doy_filepath=doy_filepath
        )
        scenes.append(scene)

    scenes.sort(key=lambda s: s.date)

    if scenes:
        print(f"Indexed {len(scenes)} scenes from {scenes[0].date.date()} to {scenes[-1].date.date()}")
        if config.has_doy_file:
            print(f"  {n_with_doy}/{len(scenes)} scenes have companion DOY files")
    else:
        print("No valid scenes found!")

    return scenes


def save_scene_index(scenes: list[EVIScene], config: ProcessingConfig):
    """Save scene index to JSON for quick re-loading."""
    if not scenes:
        print("No scenes to save")
        return

    index = {
        'cadence': config.cadence,  # monthly, 10day, daily
        'tile_id': config.tile_id,  # MRGS code (18SUJ)
        'n_scenes': len(scenes),  # number of unique observations
        'date_range': [scenes[0].date.isoformat(), scenes[-1].date.isoformat()],  # simple filename based date range
        'scenes': [s.to_dict() for s in scenes]  # list of all scenes using EVIScene class
    }

    with open(config.index_file, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"Saved index to {config.index_file}")


def build_scene_index(config: ProcessingConfig,
                rebuild: bool = False,
                recursive: bool = True) -> list[EVIScene]:
    """Build EVI2 scene index"""
    scenes = discover_evi_scenes(config, recursive=recursive)
    save_scene_index(scenes, config)
    return scenes


# =============================================================================
# Scene Filtering Utilities
# =============================================================================

def filter_scenes_by_year(scenes: list[EVIScene],
                          years: list[int] | int) -> list[EVIScene]:
    """Filter scenes to specific year(s)."""
    if isinstance(years, int):
        years = [years]
    return [s for s in scenes if s.year in years]


def filter_scenes_by_date_range(scenes: list[EVIScene],
                                start_date: datetime,
                                end_date: datetime) -> list[EVIScene]:
    """Filter scenes to date range (inclusive)."""
    return [s for s in scenes if start_date <= s.date <= end_date]


def filter_scenes_by_doy_range(scenes: list[EVIScene],
                               start_doy: int,
                               end_doy: int) -> list[EVIScene]:
    """Filter scenes by day of year range (e.g., growing season DOY 100-300)."""
    if start_doy <= end_doy:
        return [s for s in scenes if start_doy <= s.doy <= end_doy]
    else:
        # Wrap around (e.g., DOY 300-100 for winter)
        return [s for s in scenes if s.doy >= start_doy or s.doy <= end_doy]


class ChunkedTimeSeriesReaderStreaming:
    """Memory-efficient streaming reader with optional ROI clipping and duplicate handling."""

    def __init__(
            self,
            scenes: list,
            chunk_size: tuple = (128, 128),
            roi=None,
            duplicate_handling: str = 'mean',
            use_doy_files: bool = True,
            default_crs: str = None,  
            output_dir: Path = None,
            context_months: int = None,
            target_year: int = None,
    ):
        """
        Args:
            scenes: List of EVIScene objects
            chunk_size: Spatial chunk size
            roi: Optional GeoDataFrame for clipping
            duplicate_handling: How to handle same-date scenes
                               'mean' - average values
                               'first' - keep first scene
                               'last' - keep last scene
                               'max' - take maximum (less cloud impact)
            start_year: Only load scenes from this year onward
            end_year: Bound end date for scenes
            use_doy_files: Whether to use companion DOY files for actual observation dates
            default_crs: CRS to use if file has no/invalid CRS metadata
        """
        self.scenes = scenes
        self.chunk_size = chunk_size
        self.roi = roi
        self.duplicate_handling = duplicate_handling
        self.use_doy_files = use_doy_files
        self.default_crs = default_crs
        self.transform = None
        self._roi_window = None
        self._file_nodata = None
        self.output_dir = output_dir
        self.context_months = context_months
        self.target_year = target_year


        if not self.output_dir.exists():
            print(f"WARNING output dir does not exist, manually build:\n {self.output_dir}")
            return

        # Check if we have DOY files
        self.has_doy_files = any(s.doy_filepath is not None for s in self.scenes)
        if self.has_doy_files and use_doy_files:
            print("DOY files detected - will use actual observation dates for phenometrics")

        self.start_year, self.end_year = self._identify_valid_date_window()
        self._group_scenes_by_date()
        self._extract_composite_start_doys()

        first_valid = next((s for s in self.scenes if s.filepath.exists()), None)
        if first_valid is None:
            raise FileNotFoundError("No valid EVI files found")

        self._initalize_spatial_metadata(scenes[0])

        self._compute_chunk_slices()
        self._estimate_memory()

    def _initalize_spatial_metadata(self, scene):

        with rxr.open_rasterio(scene.filepath) as ds:
            ds = self._ensure_crs(ds)

            if self.roi is not None:
                ds_clipped = ds.rio.clip(self.roi.geometry, self.roi.crs)
                self.ny, self.nx = ds_clipped.shape[1], ds_clipped.shape[2]
                self.x_coords = ds_clipped.x.values
                self.y_coords = ds_clipped.y.values
            else:
                self.ny, self.nx = ds.shape[1], ds.shape[2]
                self.x_coords = ds.x.values
                self.y_coords = ds.y.values

            self.crs = ds.rio.crs

            self._ref_da = xr.DataArray(
                np.zeros((1, len(self.y_coords), len(self.x_coords)),
                         dtype=np.float32),
                dims=["band", "y", "x"],
                coords={"band": [1],
                        "y": self.y_coords,
                        "x": self.x_coords},
            )
            self._ref_da = self._ref_da.rio.write_crs(self.crs)
            self._ref_da = self._ref_da.rio.write_transform()

        # Compute ROI pixel-wise window to bypass clipping
        with rio.open(scene.filepath) as src:
            self._file_nodata = src.nodata
            self._file_height = src.height
            self._file_width = src.width

            if self.roi is not None:
                # Use clipped coordinates to find exact pixel positions
                row_start, col_start = src.index(self.x_coords[0], self.y_coords[0])
                row_end, col_end = src.index(self.x_coords[-1], self.y_coords[-1])

                row_min = min(row_start, row_end)
                row_max = max(row_start, row_end)
                col_min = min(col_start, col_end)
                col_max = max(col_start, col_end)

                self._roi_window = rio.windows.Window(
                    col_off=col_min,
                    row_off=row_min,
                    width=col_max - col_min + 1,
                    height=row_max - row_min + 1,
                )

                # Validate
                win_height = int(self._roi_window.height)
                win_width = int(self._roi_window.width)

                if win_height != self.ny or win_width != self.nx:
                    print(f"  Warning: Window ({win_height}x{win_width}) vs clip ({self.ny}x{self.nx})")
                    print(f"  Adjusting window dimensions")
                    self._roi_window = rio.windows.Window(
                        col_off=col_min,
                        row_off=row_min,
                        width=self.nx,
                        height=self.ny,
                    )

                self.transform = rio.windows.transform(self._roi_window, src.transform)

                read_pct = (self.ny * self.nx) / (src.height * src.width) * 100
                print(f"  ROI window: row={int(self._roi_window.row_off)}, "
                      f"col={int(self._roi_window.col_off)}, "
                      f"{self.ny}x{self.nx} pixels")
                print(f"  Reading {read_pct:.1f}% of full file ({src.height}x{src.width})")
            else:
                self.transform = src.transform
                print(f"  No ROI - reading full file ({src.height}x{src.width})")

        # DOY data
        self._doy_nodata = None
        if self.use_doy_files:
            print(f"  DOY nodata: {self._doy_nodata} (hardcoded)")

    def _ensure_crs(self, ds):
        """Ensure dataset has a valid CRS, assign default if missing/invalid."""
        try:
            crs = ds.rio.crs
            if crs is None:
                return ds.rio.write_crs(self.default_crs)

            epsg = crs.to_epsg()
            if epsg is None:
                return ds.rio.write_crs(self.default_crs)

            return ds

        except Exception as e:
            print(f"  Warning: CRS error ({e}), using default: {self.default_crs}")
            return ds.rio.write_crs(self.default_crs)

    def _group_scenes_by_date(self):
        """Group scenes by date to handle duplicates."""
        from collections import defaultdict

        scenes_by_date = defaultdict(list)
        n_filtered_start = 0
        n_filtered_end = 0

        for scene in self.scenes:
            if self.start_year is not None and scene.year < self.start_year:
                n_filtered_start += 1
                continue

            if self.end_year is not None and scene.year > self.end_year:
                n_filtered_end += 1
                continue

            date_key = scene.date.date()
            scenes_by_date[date_key].append(scene)

        # Sort dates
        self.unique_dates = sorted(scenes_by_date.keys())
        self.scenes_by_date = {d: scenes_by_date[d] for d in self.unique_dates}

        # Count duplicates
        n_duplicates = sum(1 for d, s in self.scenes_by_date.items() if len(s) > 1)
        n_total_scenes = sum(len(s) for s in self.scenes_by_date.values())
        n_unique_dates = len(self.unique_dates)

        print(f"Scenes: {n_total_scenes} total, {n_unique_dates} unique dates")

        if self.start_year or self.end_year:
            year_range = ""
            if self.start_year and self.end_year:
                year_range = f"{self.start_year}-{self.end_year}"
            elif self.start_year:
                year_range = f"{self.start_year}+"
            elif self.end_year:
                year_range = f"≤{self.end_year}"

            n_filtered_total = n_filtered_start + n_filtered_end
            print(f"  Filtered to {year_range} ({n_filtered_total} scenes excluded)")

        if n_duplicates > 0:
            print(f"  {n_duplicates} dates have multiple scenes ('{self.duplicate_handling}')")

        # Show date range
        if self.unique_dates:
            print(f"  Date range: {self.unique_dates[0]} to {self.unique_dates[-1]}")

        self.dates = np.array([pd.Timestamp(d) for d in self.unique_dates])

    def _extract_composite_start_doys(self):
        """Extract the start DOY of each composite from the filename."""
        import re

        self.composite_start_doys = None

        if not self.use_doy_files:
            return

        start_doys = []

        # Pattern to extract start DOY from filename like HLS.M30.T18SUJ.2021011.2021020.2.0.EVI2.tif
        pattern = r'\.(\d{4})(\d{3})\.(\d{4})(\d{3})\.'

        for date in self.unique_dates:
            scene_list = self.scenes_by_date[date]
            scene = scene_list[0]  # Use first scene for this date

            match = re.search(pattern, scene.filepath.name)
            if match:
                start_doy = int(match.group(2))  # Second group is start DOY (001-365)
                start_doys.append(start_doy)
            else:
                # Fallback to scene DOY
                start_doys.append(scene.doy)

        self.composite_start_doys = np.array(start_doys)
        print(f"Extracted {len(self.composite_start_doys)} composite start DOYs")
        print(f"  Sample: {self.composite_start_doys[:5]}...")

    def _identify_valid_date_window(self) -> tuple[int, int]:
        """
        Compute the clamped [start_year, end_year] reader window and detect
        leading / trailing data edges.
        """
        available_years = sorted({s.year for s in self.scenes})
        first_data_year = available_years[0]
        last_data_year = available_years[-1]

        context_start = pd.Timestamp(f"{self.target_year}-01-01") - pd.DateOffset(months=self.context_months)
        context_end = pd.Timestamp(f"{self.target_year}-12-31") + pd.DateOffset(months=self.context_months)

        start_year = max(context_start.year, first_data_year)
        end_year = min(context_end.year, last_data_year)

        print("------------------------------\n")
        print(f"  Available data  : {first_data_year} – {last_data_year}")
        print(f"  Desired window  : {context_start} – {context_end}  "
              f"(target {self.target_year} ± {self.context_months} months)")
        print(f"  Window context to read in data : {start_year} – {end_year}")
        print("------------------------------\n")

        return start_year, end_year

    def _compute_chunk_slices(self):
        cy, cx = self.chunk_size
        self.chunk_slices = []

        for y_start in range(0, self.ny, cy):
            y_end = min(y_start + cy, self.ny)
            for x_start in range(0, self.nx, cx):
                x_end = min(x_start + cx, self.nx)
                self.chunk_slices.append((
                    slice(y_start, y_end),
                    slice(x_start, x_end)
                ))

        self.n_chunks = len(self.chunk_slices)

    def _estimate_memory(self):
        n_dates = len(self.unique_dates)
        cy, cx = self.chunk_size
        chunk_mem = n_dates * cy * cx * 4

        # Double if using DOY files
        if self.has_doy_files and self.use_doy_files:
            chunk_mem *= 2

        roi_str = " (clipped to ROI)" if self.roi is not None else ""
        print(f"Tile: {self.ny} x {self.nx}{roi_str}")
        print(f"Unique dates: {n_dates}")
        print(f"Chunks: {self.n_chunks} ({cy}x{cx})")
        print(f"Memory per chunk: {chunk_mem / 1e6:.1f} MB")

    def _load_scene_fast(self, scene) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Fast scene loading using pre-computed window.
        No clipping, no CRS checks, no rioxarray overhead.
        """
        if not scene.filepath.exists():
            return None, None

        try:
            with rio.open(scene.filepath) as src:
                if self._roi_window is not None:
                    evi_data = src.read(1, window=self._roi_window).astype(np.float32)
                else:
                    evi_data = src.read(1).astype(np.float32)

                nodata = self._file_nodata
                if nodata is not None:
                    evi_data[evi_data == nodata] = np.nan
        except Exception as e:
            print(f"  Error reading {scene.filepath.name}: {e}")
            return None, None

        # DOY file
        doy_data = None
        if self.use_doy_files and scene.doy_filepath is not None and scene.doy_filepath.exists():
            try:
                with rio.open(scene.doy_filepath) as src:
                    if self._roi_window is not None:
                        doy_data = src.read(1, window=self._roi_window).astype(np.float32)
                    else:
                        doy_data = src.read(1).astype(np.float32)
            except Exception as e:
                print(f"  Warning: DOY read failed {scene.doy_filepath.name}: {e}")

        return evi_data, doy_data

    def _load_date_fast(self, date) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Load and combine all scenes for a date using fast reader."""
        scene_list = self.scenes_by_date[date]

        if len(scene_list) == 1:
            return self._load_scene_fast(scene_list[0])

        evi_arrays = []
        doy_arrays = []
        for i, scene in enumerate(scene_list):
            evi_data, doy_data = self._load_scene_fast(scene)
            if evi_data is not None:
                evi_arrays.append(evi_data)
                if doy_data is not None:
                    doy_arrays.append(doy_data)

        if len(evi_arrays) == 0:
            return None, None

        if len(evi_arrays) == 1:
            return evi_arrays[0], doy_arrays[0] if doy_arrays else None

        # Combine
        stacked_evi = np.stack(evi_arrays, axis=0)
        if self.duplicate_handling == 'mean':
            combined_evi = np.nanmean(stacked_evi, axis=0)
        elif self.duplicate_handling == 'max':
            combined_evi = np.nanmax(stacked_evi, axis=0)
        else:
            combined_evi = np.nanmean(stacked_evi, axis=0)

        combined_doy = None
        if doy_arrays:
            stacked_doy = np.stack(doy_arrays, axis=0)
            combined_doy = np.nanmean(stacked_doy, axis=0)

        return combined_evi, combined_doy

    def load_chunk(self, chunk_idx: int):
        """Load a single chunk using fast windowed reads."""
        import time

        t_start = time.time()
        y_slice, x_slice = self.chunk_slices[chunk_idx]

        chunk_ny = y_slice.stop - y_slice.start
        chunk_nx = x_slice.stop - x_slice.start
        n_dates = len(self.unique_dates)

        shape = (n_dates, chunk_ny, chunk_nx)
        evi_data = np.full(shape, np.nan, dtype=np.float32)
        has_doy = self.has_doy_files and self.use_doy_files
        doy_data = np.full(shape, np.nan, dtype=np.float32) if has_doy else None

        for i, date in enumerate(self.unique_dates):
            if i % 25 == 0:
                print(f"  Loading date {i + 1}/{n_dates}")

            evi_arr, doy_arr = self._load_date_fast(date)

            if evi_arr is not None:
                evi_data[i] = evi_arr[y_slice, x_slice]
            if doy_data is not None and doy_arr is not None:
                doy_data[i] = doy_arr[y_slice, x_slice]

        # Create DataArrays
        evi_da = xr.DataArray(
            evi_data,
            dims=['time', 'y', 'x'],
            coords={
                'time': self.dates,
                'y': self.y_coords[y_slice],
                'x': self.x_coords[x_slice]
            }
        )

        doy_da = None
        if doy_data is not None:
            doy_da = xr.DataArray(
                doy_data,
                dims=['time', 'y', 'x'],
                coords={
                    'time': self.dates,
                    'y': self.y_coords[y_slice],
                    'x': self.x_coords[x_slice]
                }
            )

        elapsed = time.time() - t_start
        print(f"  Chunk {chunk_idx}: {n_dates} dates, {chunk_ny}x{chunk_nx} in {elapsed:.1f}s "
              f"({elapsed / n_dates:.3f}s/date)")

        if doy_da is not None:
            return evi_da, doy_da, self.composite_start_doys
        return evi_da

    def _pad_edge_year(
            self,
            evi_context: xr.DataArray,
            doy_context: xr.DataArray,
            comp_start_context: np.ndarray,
            is_first_year: bool,
            is_last_year: bool,
    ) -> tuple:
        """
        Pad edge years by duplicating target year data as synthetic context.
        Assumes 12-month context window.

        First year (e.g., 2018): [2018_shifted_back, 2018, 2019]
        Last year (e.g., 2025):  [2024, 2025, 2025_shifted_forward]
        """

        target_mask = evi_context.time.dt.year == self.target_year
        target_evi = evi_context.sel(time=target_mask)

        if len(target_evi.time) == 0:
            return evi_context, doy_context, comp_start_context

        if is_first_year:
            shift = -np.timedelta64(365, 'D')  # Shift backward
            label = "before"
        elif is_last_year:
            shift = np.timedelta64(365, 'D')  # Shift forward
            label = "after"
        else:
            return evi_context, doy_context, comp_start_context

        synthetic_times = target_evi.time.values + shift
        synthetic_evi = target_evi.copy(data=target_evi.values)
        synthetic_evi['time'] = synthetic_times
        evi_padded = xr.concat([evi_context, synthetic_evi], dim='time').sortby('time')

        print(f"    Edge pad ({self.target_year}): +{len(synthetic_times)} synthetic timesteps {label}")

        doy_padded = None
        if doy_context is not None:
            target_doy = doy_context.sel(time=target_mask)
            synthetic_doy = target_doy.copy(data=target_doy.values)
            synthetic_doy['time'] = synthetic_times
            doy_padded = xr.concat([doy_context, synthetic_doy], dim='time').sortby('time')

        comp_start_padded = None
        if comp_start_context is not None:
            target_comp_mask = target_mask.values
            synthetic_comp = comp_start_context[target_comp_mask]
            comp_start_unsorted = np.concatenate([comp_start_context, synthetic_comp])
            all_times = np.concatenate([evi_context.time.values, synthetic_times])
            sort_order = np.argsort(all_times)
            comp_start_padded = comp_start_unsorted[sort_order]

        return evi_padded, doy_padded, comp_start_padded

    def process_all_chunks_yearly(
            self,
            process_fn,
            chunks_in_memory: int = 16,
            context_months: int = 12,
            n_workers: int = 16,
            **process_kwargs
    ) -> dict[str, np.ndarray]:
        """
        Process all chunks, read all years in then process one year at a time.

        Args:
            process_fn: Function to apply to each chunk (e.g., annual_phenometrics_chunk)
            chunks_in_memory: Number of chunks to load in memory simultaneously
            **process_kwargs: Additional kwargs passed to process_fn
        Returns:
            Dict of full-size arrays (ny, nx) for each output metric
        """
        from rasterio.windows import Window

        if self.output_dir is None:
            raise ValueError("output_dir must be set to write per-metric GeoTIFFs on disk")

        print(f"\nProcess pool: {n_workers} workers — warm across all chunks")
        with Parallel(n_jobs=n_workers, prefer="processes", batch_size="auto") as pool:
            has_doy = self.has_doy_files and self.use_doy_files

            context_years = list(range(self.start_year, self.end_year + 1))
            print(f"Ingesting {len(context_years)} years: {context_years}")

            n_batches = (self.n_chunks + chunks_in_memory - 1) // chunks_in_memory
            # year_outputs = {}  # {year: {} for year in target_years}
            out_datasets = {}  # metric -> rasterio DatasetWriter
            metric_valid_counts = {}  # metric -> int
            os.makedirs(self.output_dir, exist_ok=True)

            for batch_idx in range(n_batches):
                start_chunk = batch_idx * chunks_in_memory
                end_chunk = min(start_chunk + chunks_in_memory, self.n_chunks)
                print(f"\nBatch {batch_idx + 1}/{n_batches} (chunks {start_chunk}-{end_chunk - 1})")

                for chunk_idx in range(start_chunk, end_chunk):
                    print(f"\n  Chunk {chunk_idx}/{self.n_chunks}")
                    result = self.load_chunk(chunk_idx)

                    if isinstance(result, tuple):
                        evi_da_full, doy_da_full, comp_start_full = result
                    else:
                        evi_da_full = result
                        doy_da_full = None
                        comp_start_full = None

                    y_slice, x_slice = self.chunk_slices[chunk_idx]
                    data_years = sorted(set(evi_da_full.time.dt.year.values))

                    print(f"    Loaded full time series: {evi_da_full.shape}")
                    print(f"    Date range: {evi_da_full.time.values[0]} to {evi_da_full.time.values[-1]}")
                    print(f"\n    Processing year {self.target_year}...")
                    start_context_year = pd.Timestamp(f"{self.start_year}-01-01")
                    end_context_year = pd.Timestamp(f"{self.end_year}-01-01")
                    evi_context = evi_da_full.sel(time=slice(start_context_year, end_context_year))
                    doy_context = (doy_da_full.sel(time=slice(start_context_year, end_context_year))
                                   if doy_da_full is not None else None)

                    if comp_start_full is not None:
                        time_mask = ((evi_da_full.time >= start_context_year) &
                                     (evi_da_full.time <= end_context_year)).values
                        comp_start_context = comp_start_full[time_mask]
                    else:
                        comp_start_context = None

                    # HANDLE FIRST AND LAST YEAR EDGE CASES FOR THE CONTEXT WINDOW BUFFERS
                    is_first_year = self.target_year == self.start_year
                    is_last_year = self.target_year == self.end_year
                    if is_first_year or is_last_year:
                        evi_context, doy_context, comp_start_context = self._pad_edge_year(
                            evi_context=evi_context,
                            doy_context=doy_context,
                            comp_start_context=comp_start_context,
                            is_first_year=is_first_year,
                            is_last_year=is_last_year,
                        )

                    if comp_start_context is not None and len(comp_start_context) != len(evi_context.time):
                        print(f"    WARNING: comp_start mismatch: {len(comp_start_context)} vs {len(evi_context.time)}")
                        comp_start_context = comp_start_context[:len(evi_context.time)]

                    # Processing pipeline logic for DOY composites or daily scenes
                    if doy_context is not None:
                        chunk_results = process_fn(
                            evi_context,
                            doy_data=doy_context,
                            composite_start_doys=comp_start_context,
                            target_year=self.target_year,
                            _pool=pool,
                            **process_kwargs
                        )
                    else:
                        chunk_results = process_fn(evi_context,
                                                   target_year=self.target_year,
                                                   _pool=pool,
                                                   **process_kwargs)

                    for metric, arr in chunk_results.items():
                        if metric.startswith('_') or not isinstance(arr, np.ndarray):
                            continue

                        if arr.ndim > 2:
                            arr = arr.squeeze()


                        preferred_block = int(min(self.chunk_size))
                        preferred_block = max(1, min(preferred_block, 512))
                        nx = int(self.nx)
                        ny = int(self.ny)
                        def largest_mult16_le(v):
                            if v < 16:
                                return 0
                            return (v // 16) * 16

                        # For tiled output we require multiples-of-16 and <= image dims
                        blockx = largest_mult16_le(min(preferred_block, nx))
                        blocky = largest_mult16_le(min(preferred_block, ny))
                        use_tiled = (blockx >= 16 and blocky >= 16)

                        if metric not in out_datasets:
                            profile = {
                                'driver': 'GTiff',
                                'height': self.ny,
                                'width': self.nx,
                                'count': 1,
                                'dtype': 'float32',
                                'crs': self.crs,
                                'transform': getattr(self, 'transform'),
                                'compress': 'DEFLATE',
                                # 'bigtiff': 'YES',
                            }
                            if use_tiled:
                                # tiled: provide both blockxsize and blockysize (multiples of 16)
                                profile.update({
                                    'tiled': True,
                                    'blockxsize': blockx,
                                    'blockysize': blocky,
                                })
                            else:
                                pass

                            if getattr(self, "_file_nodata", None) is not None:
                                profile['nodata'] = float(self._file_nodata)
                            else:
                                profile['nodata'] = -9999.0

                            out_path = os.path.join(self.output_dir, f"{metric}.tif")
                            if os.path.exists(out_path):
                                ds = rio.open(out_path, 'r+')
                                meta_ok = (
                                        ds.width == profile['width'] and
                                        ds.height == profile['height'] and
                                        ds.count == profile['count'] and
                                        ds.dtypes[0] == np.dtype(profile['dtype']).name and
                                        (ds.crs == profile['crs']) and
                                        (ds.transform == profile['transform'])
                                )
                                if not meta_ok:
                                    ds.close()
                                    os.remove(out_path)
                                    ds = rio.open(out_path, 'w', **profile)
                            else:

                                ds = rio.open(out_path, 'w', **profile)
                            out_datasets[metric] = ds
                            metric_valid_counts[metric] = 0

                        ds = out_datasets[metric]

                        # compute the raster window
                        y_slice, x_slice = self.chunk_slices[chunk_idx]
                        row_off = y_slice.start
                        col_off = x_slice.start
                        height = y_slice.stop - y_slice.start
                        width = x_slice.stop - x_slice.start
                        window = Window(col_off, row_off, width, height)
                        write_arr = arr.astype('float32')

                        # replace NaN with profile nodata sentinel so downstream tools see nodata
                        nod = ds.profile.get('nodata', None)
                        if nod is not None and np.isnan(nod) is False:
                            if np.isnan(write_arr).any():
                                write_arr = np.where(np.isnan(write_arr), nod, write_arr)

                        # Ensure dtype matches profile and NaNs are preserved for float
                        if write_arr.shape != (height, width):
                            raise RuntimeError(f"write array shape {write_arr.shape} != window {(height, width)}")

                        # Write into the file (single band)
                        ds.write(write_arr, 1, window=window)

                        # Update valid count
                        metric_valid_counts[metric] += np.sum(~np.isnan(write_arr))

                    del chunk_results
                    del evi_da_full
                    if doy_da_full is not None:
                        del doy_da_full
                    gc.collect()

            for metric, ds in out_datasets.items():
                ds.close()

            print(f"\n  Saving {self.target_year}: {len(metric_valid_counts)} metrics")
            for metric, n_valid in metric_valid_counts.items():
                print(f"    {metric}: {n_valid} valid pixels")

            # No large year_outputs remained in memory
            gc.collect()

    def enter_processing_stage(
            self,
            process_fn=None,
            chunks_in_memory: int = 16,
            min_evi: float = -0.2,
            max_evi: float = 1.0,
            interp_method: str = 'linear',
            skip_pixel_processing: bool = False,
            skip_timeseries: bool = False,
            context_months: int = 12,
            n_workers: int = 1,
            **process_kwargs,
    ):
        """
        pixel_results:   per-pixel annual phenometrics (dict of 2D arrays)
        """
        pixel_results = None
        if not skip_pixel_processing:
            if process_fn is None:
                raise ValueError("process_fn required when skip_pixel_processing=False")
            print("\n" + "=" * 60)
            print("Processing pixel-wise metrics")
            print("=" * 60)
            self.process_all_chunks_yearly(
                process_fn=process_fn,
                chunks_in_memory=chunks_in_memory,
                context_months=context_months,
                n_workers=n_workers,
                **process_kwargs,
            )
        else:
            print("  Pixel processing: skipped")

        # return pixel_results


# =============================================================================
# Output helper functions
# =============================================================================  
def save_results(
        pixel_results: Optional[dict],
        output_dir: Path,
        y_coords: np.ndarray = None,
        x_coords: np.ndarray = None,
        crs=None,
):
    """Save pixel GeoTIFFs and/or time series CSVs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pixel_results is not None:
        if y_coords is None or x_coords is None or crs is None:
            raise ValueError("y_coords, x_coords, crs required for GeoTIFFs")
        for name, arr in pixel_results.items():
            da = xr.DataArray(
                arr[np.newaxis, :, :], dims=["band", "y", "x"],
                coords={"band": [1], "y": y_coords, "x": x_coords})
            da = da.rio.write_crs(crs)
            da.rio.to_raster(output_dir / f"{name}.tif",
                             driver="GTiff", dtype="float32")
        print(f"  Saved {len(pixel_results)} GeoTIFFs → {output_dir}")
    else:
        print("  Pixel GeoTIFFs: skipped")
