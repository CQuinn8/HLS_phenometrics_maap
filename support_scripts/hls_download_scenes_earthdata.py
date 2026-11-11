#!/usr/bin/env python3

# Usage:
# python download_hls.py --tile=18SUJ --start_date="2024-01-01" --end_date="2024-03-30" --output_dir="./temp_out"

import os
import shutil
import argparse
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio as rio
import earthaccess
import backoff
from maap.maap import MAAP
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

# Timeout for processing a single scene after it's been downloaded
SCENE_TIMEOUT_SECONDS = 10 * 60

# --- HLS & Raster constants ---
SR_SCALE = 0.0001
QA_FILL = 255
IMAGE_SIZE = (3660, 3660)

# Map common band names to sensor-specific band IDs
COMMON_BANDS = ["Blue", "Green", "Red", "NIR_Narrow", "SWIR1", "SWIR2", "Fmask"]
L8_NAME_TO_INDEX = {
    "Blue": "B02", "Green": "B03", "Red": "B04",
    "NIR_Narrow": "B05", "SWIR1": "B06", "SWIR2": "B07", "Fmask": "Fmask",
}
S2_NAME_TO_INDEX = {
    "Blue": "B02", "Green": "B03", "Red": "B04",
    "NIR_Narrow": "B8A", "SWIR1": "B11", "SWIR2": "B12", "Fmask": "Fmask",
}

# Bit positions in the HLS Fmask layer
QA_BIT = {
    "cloud": 1,
    "adj_cloud": 2,
    "cloud shadow": 3,
    "snowice": 4,
    "water": 5,
}

# --- GDAL Configuration ---
# These settings help optimize reading cloud-optimized GeoTIFFs
os.environ.update({
    "CPL_TMPDIR": "/tmp",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "AWS_REQUEST_PAYER": "requester",
    "TQDM_DISABLE": "1",
    "PYTHONUNBUFFERED": "1",
})

# AUTHENTICATION
def configure_maap_credentials():
    """
    Configure Earthdata credentials for earthaccess.
      1. Use existing EARTHDATA_USERNAME/EARTHDATA_PASSWORD env vars if present.
      2. Try to fetch them from MAAP secrets.
      3. Fall back to ~/.netrc through earthaccess.
    """

    print("Configuring Earthdata credentials...")
    # 1. If env vars already exist
    env_user = os.environ.get("EARTHDATA_USERNAME")
    env_pass = os.environ.get("EARTHDATA_PASSWORD")

    if env_user and env_pass:
        print("Using existing EARTHDATA_USERNAME/EARTHDATA_PASSWORD environment variables.")
        earthaccess.login(strategy="environment")
        return

    # 2. Try MAAP secrets
    try:
        print("Trying to retrieve Earthdata credentials from MAAP secrets...")
        maap = MAAP(maap_host="api.maap-project.org")

        username = maap.secrets.get_secret("EARTHDATA_USERNAME")
        password = maap.secrets.get_secret("EARTHDATA_PASSWORD")

        if not isinstance(username, str):
            raise RuntimeError(
                f"MAAP secret EARTHDATA_USERNAME did not return a string. "
                f"Returned: {username}"
            )

        if not isinstance(password, str):
            raise RuntimeError(
                f"MAAP secret EARTHDATA_PASSWORD did not return a string. "
                f"Returned: {password}"
            )

        if not username or not password:
            raise RuntimeError("MAAP Earthdata username/password secrets are empty.")

        os.environ["EARTHDATA_USERNAME"] = username
        os.environ["EARTHDATA_PASSWORD"] = password

        print("Successfully configured Earthdata credentials from MAAP secrets.")
        earthaccess.login(strategy="environment")
        return

    except Exception as e:
        print("Could not retrieve Earthdata credentials from MAAP secrets.")
        print(f"MAAP credential error: {e}")

    # 3. Fall back to ~/.netrc
    try:
        print("Trying Earthdata login using ~/.netrc...")
        earthaccess.login(strategy="netrc")
        print("Successfully logged into Earthdata using ~/.netrc.")
        return

    except Exception as e:
        raise RuntimeError(
            "\nCould not configure Earthdata credentials.\n\n"
            "Options:\n"
            "  1. Export EARTHDATA_USERNAME and EARTHDATA_PASSWORD before running.\n"
            "  2. Create ~/.netrc with Earthdata credentials.\n"
            "  3. Re-authenticate your MAAP session so maap.secrets.get_secret works.\n\n"
            f"Final earthaccess/netrc error: {e}"
        )
        

def save_geotiff(filename, data, template_file, nodata=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with rio.open(template_file) as src:
        profile = src.profile.copy()
    
    profile.update({
        "driver": "GTiff",
        "dtype": data.dtype,
        "count": 1,
        "nodata": nodata,
        "compress": "lzw",
        "predictor": 2,
    })

    with rio.open(filename, "w", **profile) as dst:
        dst.write(data, 1)


def get_granule_id(granule):
    try:
        umm = granule["umm"]
        for key in ("GranuleUR", "ProducerGranuleId", "Id"):
            if key in umm:
                return umm[key]
    except Exception:
        pass

    try:
        meta = granule["meta"]
        for key in ("native-id", "concept-id", "concept-type"):
            if key in meta:
                return meta[key]
    except Exception:
        pass
    for attr in ("concept_id", "granule_id"):
        try:
            value = getattr(granule, attr)
            if callable(value):
                value = value()
            if value:
                return str(value)
        except Exception:
            pass

    return str(granule).splitlines()[0][:200]

@backoff.on_exception(backoff.expo, Exception, max_tries=5, jitter=backoff.full_jitter)
def download_granule_with_retry(granule, download_dir):
    granule_id = get_granule_id(granule)
    print(f"  [Download] Attempting: {granule_id}", flush=True)
    try:
        files = earthaccess.download([granule],local_path=download_dir,pqdm_kwargs={"disable": True})
    except TypeError:
        # Fallback for older earthaccess versions that do not support pqdm_kwargs
        files = earthaccess.download([granule],local_path=download_dir)

    return [str(f) for f in files]


def process_and_save_scene(scene_id, scene_files, tile, out_dir):
    try:
        # 1. Map local file paths to band names
        sat_type = scene_id.split('.')[1]
        name_to_index = L8_NAME_TO_INDEX if sat_type == "L30" else S2_NAME_TO_INDEX
        index_to_name = {v: k for k, v in name_to_index.items()}
        
        band_map = {}
        for f in scene_files:
            band_id = os.path.basename(f).split('.')[-2]
            if band_id in index_to_name:
                band_map[index_to_name[band_id]] = f
        
        # Ensure all required bands were downloaded
        if any(b not in band_map for b in COMMON_BANDS):
            missing = [b for b in COMMON_BANDS if b not in band_map]
            return f"SKIP  {scene_id} — missing band files: {missing}", None, None

        # 2. Read bands and Fmask into NumPy arrays
        bands_data = {}
        with rio.open(band_map["Fmask"]) as src:
            fmask = src.read(1)
        
        for band_name in COMMON_BANDS[:-1]: # Read all except Fmask
            with rio.open(band_map[band_name]) as src:
                bands_data[band_name] = src.read(1)

        # 3. Build quality mask
        bad_pixel_mask = (fmask & (1 << QA_BIT["cloud"])) > 0
        bad_pixel_mask |= (fmask & (1 << QA_BIT["adj_cloud"])) > 0
        bad_pixel_mask |= (fmask & (1 << QA_BIT["cloud shadow"])) > 0
        bad_pixel_mask |= (fmask & (1 << QA_BIT["water"])) > 0
        bad_pixel_mask |= (fmask & (1 << QA_BIT["snowice"])) > 0
        bad_pixel_mask |= (fmask == QA_FILL)
        
        # Also mask out negative reflectance values
        for data in bands_data.values():
            bad_pixel_mask |= data < 0

        # 4. Generate count masks for annual stats
        pre_mask_arr = (fmask != QA_FILL).astype(np.uint8)
        post_mask_arr = (~bad_pixel_mask).astype(np.uint8)
        
        # 5. Calculate EVI2
        red = bands_data["Red"].astype(np.float32) * SR_SCALE
        nir = bands_data["NIR_Narrow"].astype(np.float32) * SR_SCALE
        
        with np.errstate(divide='ignore', invalid='ignore'):
            evi2 = 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)        
        evi2[bad_pixel_mask] = np.nan
        evi2 = evi2.astype(np.float32)

        # 6. Save the final EVI2 GeoTIFF
        parts = scene_id.split(".")

        # scene_id example:
        # HLS.L30.T18SUJ.2019009T154611.v2.0
        sat_type = parts[1]          # L30 or S30
        tile_id = parts[2]           # T18SUJ
        date_julian = parts[3][:7]   # 2019009
        
        # Convert "v2.0" to "2.0"
        version = ".".join(parts[4:6]).removeprefix("v")        
        scene_date = datetime.strptime(date_julian, "%Y%j")
        scene_year_dir = os.path.join(out_dir, str(scene_date.year))        
        out_name = f"HLS.{sat_type}.{tile_id}.{date_julian}.{version}.EVI2.tif"
        out_path = os.path.join(scene_year_dir, out_name)        
        template_file = scene_files[0]        
        save_geotiff(out_path, evi2, template_file, nodata=np.nan)        
        return f"OK    {scene_id}", pre_mask_arr, post_mask_arr

    except Exception as e:
        return f"ERROR {scene_id} — unhandled exception: {e}", None, None


def process_hls_for_tile(tile, start_date, end_date, save_dir, n_workers):
    """Main function to download and process HLS data for a given tile and date range."""
    
    out_dir = os.path.join(save_dir, tile)
    download_dir = os.path.join(out_dir, "temp_downloads")
    os.makedirs(download_dir, exist_ok=True)

    try:
        # 1. SEARCH
        print("Searching for HLS granules...")
        earthaccess.login(strategy="environment")
        granules = earthaccess.search_data(
            short_name=["HLSL30", "HLSS30"],
            temporal=(start_date, end_date),
            granule_name=f"*T{tile}*",
        )
        if not granules:
            print(f"No granules found for tile {tile}. Creating empty indicator file.")
            Path(os.path.join(out_dir, "no_granules_found.txt")).touch()
            return

        # 2. DOWNLOAD
        print(f"Found {len(granules)} granules. Submitting to {n_workers} download workers...")
        local_files = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_granule_id = {}
        
            for g in granules:
                granule_id = get_granule_id(g)
                future = executor.submit(download_granule_with_retry, g, download_dir)
                future_to_granule_id[future] = granule_id
        
            for future in as_completed(future_to_granule_id):
                granule_id = future_to_granule_id[future]        
                try:
                    files = future.result()
                    files = [str(f) for f in files]
                    local_files.extend(files)
                    print(f"  [Download] SUCCESS: {granule_id} ({len(files)} files)")        
                except Exception as e:
                    print(f"  [Download] FAILED after all retries: {granule_id}. Final error: {e}")

        if not local_files:
            print("Download phase resulted in no files. Exiting.")
            return

        # 3. GROUP FILES AND PROCESS
        scenes = defaultdict(list)        
        for f in local_files:
            f = str(f)
            base = os.path.basename(f)        
            if not base.startswith("HLS."):
                continue        
            if not base.lower().endswith(".tif"):
                continue        
            parts = base.split(".")        
            if len(parts) < 8:
                continue        
            scene_id = ".".join(parts[:6])
            scenes[scene_id].append(f)
        
        print(f"\nProcessing {len(scenes)} scenes from local files using {n_workers} workers...")
        pre_accum = defaultdict(lambda: np.zeros(IMAGE_SIZE, dtype=np.uint16))
        post_accum = defaultdict(lambda: np.zeros(IMAGE_SIZE, dtype=np.uint16))
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_scene = {
                executor.submit(process_and_save_scene, sid, sfiles, tile, out_dir): sid
                for sid, sfiles in scenes.items()
            }
            n_done = n_ok = n_skip_err = 0
            for future in as_completed(future_to_scene):
                scene_id = future_to_scene[future]
                n_done += 1
                try:
                    status, pre_arr, post_arr = future.result(timeout=SCENE_TIMEOUT_SECONDS)
                    if status.startswith("OK"):
                        n_ok += 1
                        if pre_arr is not None and post_arr is not None:
                            scene_date = datetime.strptime(scene_id.split(".")[3][:7], "%Y%j")
                            pre_accum[scene_date.year] += pre_arr
                            post_accum[scene_date.year] += post_arr
                    else:
                        n_skip_err += 1
                    print(f"[{n_done}/{len(scenes)}] {status}")
                except TimeoutError:
                    n_skip_err +=1
                    print(f"[{n_done}/{len(scenes)}] TIMEOUT processing {scene_id}")
                except Exception as e:
                    n_skip_err += 1
                    print(f"[{n_done}/{len(scenes)}] FATAL ERROR processing {scene_id}: {e}")

        # 4. SAVE ANNUAL STATISTICS
        print("\nSaving annual quality statistics...")
        template_path = next(iter(scenes.values()))[0]
        for year, pre in sorted(pre_accum.items()):
            post = post_accum[year]
            year_dir = os.path.join(out_dir, str(year))
            
            with np.errstate(divide="ignore", invalid="ignore"):
                viable_pct = np.where(pre > 0, np.round((post / pre) * 100), 0).astype(np.uint8)

            save_geotiff(os.path.join(year_dir, f"HLS.T{tile}.{year}.PreMaskCount.tif"), pre, template_path, 65535)
            save_geotiff(os.path.join(year_dir, f"HLS.T{tile}.{year}.PostMaskCount.tif"), post, template_path, 65535)
            save_geotiff(os.path.join(year_dir, f"HLS.T{tile}.{year}.ViablePct.tif"), viable_pct, template_path, 255)
            print(f"  - Saved stats for {year}. Mean viable pixels: {viable_pct[pre > 0].mean():.1f}%")

        print(f"\nDone — {len(scenes)} scenes processed: {n_ok} saved, {n_skip_err} skipped or failed.")

    finally:
        # 5. CLEANUP 
        if os.path.exists(download_dir):
            print(f"Cleaning up temporary download directory: {download_dir}")
            shutil.rmtree(download_dir)

# =============================================================================
# SCRIPT ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process HLS data to EVI2.")
    parser.add_argument("--tile", required=True, help="MGRS tile ID (e.g., 18SUJ).")
    parser.add_argument("--start_date", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end_date", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the final outputs.")
    parser.add_argument("--N_WORKERS", default=4, type=int, help="Number of parallel workers for download/processing.")
    args = parser.parse_args()

    # Set up credentials (optional if env vars are already set)
    configure_maap_credentials()
    
    # Run the main processing function
    process_hls_for_tile(
        tile=args.tile,
        start_date=args.start_date,
        end_date=args.end_date,
        save_dir=args.output_dir,
        n_workers=args.N_WORKERS,
    )