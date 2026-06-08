#!/usr/bin/env python3

"""
DPS-ready Landsat -> HLS processing pipeline

Key DPS features:
- CLI arguments
- Uses /tmp for intermediate work
- Uses OUTPUT_DIR for DPS-collected outputs
- Uploads final outputs to S3
- Safe logging
- Proper failure handling
- Requester-pays compatible
"""

import os
import sys
import glob
import shutil
import logging
import argparse
from collections import defaultdict
from datetime import datetime

from urllib.parse import urlparse

import rasterio as rio

import boto3
import earthaccess
from maap.maap import MAAP
from rasterio.session import AWSSession
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed


# =============================================================================
# CONFIG
# =============================================================================

L8_NAME_TO_INDEX = {
    "Blue": "B02", "Green": "B03", "Red": "B04",
    "NIR_Narrow": "B05", "SWIR1": "B06", "SWIR2": "B07", "Fmask": "Fmask",
}

S2_NAME_TO_INDEX = {
    "Blue": "B02", "Green": "B03", "Red": "B04",
    "NIR_Narrow": "B8A", "SWIR1": "B11", "SWIR2": "B12", "Fmask": "Fmask",
}

COMMON_BANDS = ["Blue", "Green", "Red", "NIR_Narrow", "SWIR1", "SWIR2", "Fmask"]

REQUIRED_HLS_BANDS = {
    "L30": {"B02", "B03", "B04", "B05", "B06", "B07", "Fmask"},
    "S30": {"B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"},
}
# Bit positions in the HLS Fmask layer
QA_BIT = {
    "cloud": 1,
    "adj_cloud": 2,
    "cloud shadow": 3,
    "snowice": 4,
    "water": 5,
}

QA_FILL = 255
SR_SCALE = 0.0001
IMAGE_SIZE = (3660, 3660)

GDAL_CONFIG = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_TMPDIR": "/tmp",
    "AWS_REQUEST_PAYER": "requester",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".TIF,.tif,.vrt",
}
os.environ.update(GDAL_CONFIG)

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("hls_pipeline")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(message)s"
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# Attach handlers
logger.addHandler(console_handler)

# =============================================================================
# AWS / MAAP SETUP
# =============================================================================
def configure_requester_pays():
    logger.info("Configuring requester-pays credentials")

    maap = MAAP(maap_host="api.maap-project.org")

    os.environ["EARTHDATA_USERNAME"] = maap.secrets.get_secret("EARTHDATA_USERNAME")
    os.environ["EARTHDATA_PASSWORD"] = maap.secrets.get_secret("EARTHDATA_PASSWORD")

    credentials = maap.aws.requester_pays_credentials()
    boto3_session = boto3.Session(
        aws_access_key_id=credentials["aws_access_key_id"],
        aws_secret_access_key=credentials["aws_secret_access_key"],
        aws_session_token=credentials["aws_session_token"],
    )

    aws_session = AWSSession(
        boto3_session,
        requester_pays=True
    )

    return maap, aws_session


def cleanup_earthaccess_partials(download_dir):
    """
    Optionally, remove stale earthaccess partial_* files/directories from previous failed downloads.
    Necesary for local testing
    """
    if not os.path.exists(download_dir):
        return

    for p in glob.glob(os.path.join(download_dir, "partial_*")):
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
            logger.info(f"Removed stale earthaccess partial path: {p}")
        except Exception as e:
            logger.warning(f"Could not remove stale partial path {p}: {e}")


def is_required_hls_tif(url: str) -> bool:
    base = os.path.basename(urlparse(url).path)

    if not base.startswith("HLS."):
        return False

    if not base.lower().endswith(".tif"):
        return False

    parts = base.split(".")

    # Expected:
    # HLS.L30.T18SUJ.2024005T155218.v2.0.B04.tif
    # HLS.S30.T18SUJ.2024005T155639.v2.0.B8A.tif
    if len(parts) < 8:
        return False

    product = parts[1]   # L30 or S30
    band = parts[-2]     # B02, B8A, Fmask, etc.

    return product in REQUIRED_HLS_BANDS and band in REQUIRED_HLS_BANDS[product]


def get_required_hls_links(granule):
    try:
        links = granule.data_links(access="external")
    except TypeError:
        links = granule.data_links()
    except Exception:
        links = []

    required = []

    for link in links:
        if not isinstance(link, str):
            continue

        if not link.startswith("https://"):
            continue

        if is_required_hls_tif(link):
            required.append(link)

    return sorted(set(required))


def get_scene_id_from_link(url: str) -> str:
    base = os.path.basename(urlparse(url).path)
    return ".".join(base.split(".")[:6])


def get_granule_safe_id(granule, fallback_index):
    try:
        links = granule.data_links()
        for link in links:
            base = os.path.basename(urlparse(link).path)
            if base.startswith("HLS.") and ".tif" in base:
                return ".".join(base.split(".")[:6])
    except Exception:
        pass

    return f"granule_{fallback_index:05d}"


def download_one_granule_required_bands(granule, output_dir, index):
    links = get_required_hls_links(granule)

    if len(links) == 0:
        return []

    scene_id = get_scene_id_from_link(links[0])
    scene_dir = os.path.join(output_dir, "_hls_downloads", scene_id)
    os.makedirs(scene_dir, exist_ok=True)

    files = earthaccess.download(
        links,
        local_path=scene_dir,
        pqdm_kwargs={
            "n_jobs": 1,
            "disable": True,
        },
    )

    return [str(f) for f in files]


def download_hls_granule(mgrs_tile, start_date, end_date, output_dir, n_workers):
    # need to deal with EarthAccess credentials on DPS
    logger.info(f"Downloading HLS scenes for {mgrs_tile} from {start_date} to {end_date}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. SEARCH
    earthaccess.login(strategy="environment")
    results = earthaccess.search_data(
        short_name=["HLSL30", "HLSS30"],
        temporal=(start_date, end_date),
        granule_name=f"*T{mgrs_tile}*",
    )

    if len(results) == 0:
        raise RuntimeError(
            f"No HLS granule found for {mgrs_tile} {start_date}:{end_date}"
        )

    # 2. DOWNLOAD
    logger.info(f"Found {len(results)} HLS granules")
    download_root = os.path.join(output_dir, "_hls_downloads")
    os.makedirs(download_root, exist_ok=True)

    downloaded_files = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                download_one_granule_required_bands,
                granule,
                output_dir,
                i,
            ): i
            for i, granule in enumerate(results, start=1)
        }

        for future in as_completed(futures):
            i = futures[future]

            try:
                files = future.result()
                downloaded_files.extend(files)
                logger.info(f"[{i}/{len(results)}] downloaded {len(files)} target files")

            except Exception as e:
                logger.error(f"[{i}/{len(results)}] download failed: {e}")
                raise

    if len(downloaded_files) == 0:
        raise RuntimeError("Download phase resulted in no files")

    logger.info(f"Downloaded {len(downloaded_files)} files")

    return downloaded_files


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

# =============================================================================
# PROCESS HLS
# =============================================================================
def process_and_save_scene(scene_id, scene_files, out_dir):
    try:
        logger.info(f"Processing scene: {scene_id}")
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

        for band_name in COMMON_BANDS[:-1]:  # Read all except Fmask
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
        sat_type = parts[1]  # L30 or S30
        tile_id = parts[2]  # T18SUJ
        date_julian = parts[3][:7]  # 2019009

        # Convert "v2.0" to "2.0"
        version = ".".join(parts[4:6]).removeprefix("v")
        scene_date = datetime.strptime(date_julian, "%Y%j")
        scene_year_dir = os.path.join(out_dir, str(scene_date.year))
        os.makedirs(scene_year_dir, exist_ok=True)

        out_name = f"HLS.{sat_type}.{tile_id}.{date_julian}.{version}.EVI2.tif"
        out_path = os.path.join(scene_year_dir, out_name)
        print(f"Outpath:{out_path}")

        template_file = scene_files[0]
        save_geotiff(out_path, evi2, template_file, nodata=np.nan)
        return f"OK    {scene_id}", pre_mask_arr, post_mask_arr

    except Exception as e:
        return f"ERROR {scene_id} — unhandled exception: {e}", None, None


# =============================================================================
# per granule logger
# =============================================================================
def setup_logger(mgrs_tile, start_date, end_date):
    logger = logging.getLogger("hls_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    log_file = os.path.join(
        LOG_DIR,
        f"{mgrs_tile}_{start_date}_{end_date}_{timestamp}.log"
    )

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, log_file


def upload_all_outputs_to_s3(mgrs_tile, date):
    logger.info("Uploading outputs to S3")

    s3 = boto3.client("s3")

    uploaded = []

    # Create search pattern for files like: *18TWN*20231005*.tif
    julian_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%j")
    search_pattern = os.path.join(OUTPUT_DIR, f"*{mgrs_tile}*{julian_date}*.tif")
    # Find files
    tif_files = glob.glob(search_pattern)

    for fp in tif_files:
        key = (
            f"{S3_PREFIX}/"
            f"{mgrs_tile}_{date}/"
            f"{os.path.basename(fp)}"
        )

        logger.info(f"Uploading s3://{S3_BUCKET}/{key}")

        s3.upload_file(
            fp,
            S3_BUCKET,
            key
        )

        uploaded.append(f"s3://{S3_BUCKET}/{key}")

    return uploaded


def upload_to_s3(local_file, mgrs_tile, date):
    s3 = boto3.client("s3")
    key = (
        f"{S3_PREFIX}/"
        f"{mgrs_tile}_{date}/"
        f"{os.path.basename(local_file)}"
    )
    s3.upload_file(
        local_file,
        S3_BUCKET,
        key,
    )

    logger.info(f"Uploaded s3://{S3_BUCKET}/{key}")

# =============================================================================
# MAIN
# =============================================================================

def process_hls_tile(mgrs_tile, start_date, end_date, output_dir, n_workers = 1):
    try:
        # 0. Configure MAAP requester-pays
        maap, aws_session = configure_requester_pays()

        # 1. Download HLS reference granules
        hls_files = download_hls_granule(mgrs_tile, start_date, end_date, output_dir, n_workers)
        scenes = defaultdict(list)
        for f in hls_files:
            # Group files by their granule ID (e.g., HLS.L30.T18SUJ.2024005.v2.0)

            f = str(f)
            base = os.path.basename(f)
            if not base.startswith("HLS."):
                continue
            if not base.lower().endswith(".tif"):
                continue
            parts = base.split(".")
            # Expected:
            # HLS.L30.T18SUJ.2024005T155218.v2.0.B04.tif
            if len(parts) < 8:
                continue
            scene_id = ".".join(parts[:6])
            scenes[scene_id].append(f)

        print(f"\nProcessing {len(scenes)} scenes from local files using {1} workers...")
        pre_accum = defaultdict(lambda: np.zeros(IMAGE_SIZE, dtype=np.uint16))
        post_accum = defaultdict(lambda: np.zeros(IMAGE_SIZE, dtype=np.uint16))

        # 2. Run HLS workflow
        SCENE_TIMEOUT_SECONDS = 10 * 60
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_scene = {
                executor.submit(process_and_save_scene, sid, sfiles, output_dir): sid
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
                            date_julian = scene_id.split(".")[3][:7]
                            scene_date = datetime.strptime(date_julian, "%Y%j")
                            pre_accum[scene_date.year] += pre_arr
                            post_accum[scene_date.year] += post_arr
                    else:
                        n_skip_err += 1
                    print(f"[{n_done}/{len(scenes)}] {status}")
                except TimeoutError:
                    n_skip_err += 1
                    print(f"[{n_done}/{len(scenes)}] TIMEOUT processing {scene_id}")
                except Exception as e:
                    n_skip_err += 1
                    print(f"[{n_done}/{len(scenes)}] FATAL ERROR processing {scene_id}: {e}")

        # --- 4. SAVE ANNUAL STATISTICS ---
        print("\nSaving annual quality statistics...")
        print(pre_accum.keys())
        template_path = next(iter(scenes.values()))[0]
        for year, pre in sorted(pre_accum.items()):
            post = post_accum[year]
            year_dir = os.path.join(output_dir, str(year))
            print(year_dir)

            with np.errstate(divide="ignore", invalid="ignore"):
                viable_pct = np.where(pre > 0, np.round((post / pre) * 100), 0).astype(np.uint8)

            save_geotiff(os.path.join(year_dir, f"HLS.T{mgrs_tile}.{year}.PreMaskCount.tif"), pre, template_path, 65535)
            save_geotiff(os.path.join(year_dir, f"HLS.T{mgrs_tile}.{year}.PostMaskCount.tif"), post, template_path, 65535)
            save_geotiff(os.path.join(year_dir, f"HLS.T{mgrs_tile}.{year}.ViablePct.tif"), viable_pct, template_path, 255)
            print(f"  - Saved stats for {year}. Mean viable pixels: {viable_pct[pre > 0].mean():.1f}%")

        print(f"\nDone — {len(scenes)} scenes processed: {n_ok} saved, {n_skip_err} skipped or failed.")
        
        logger.info("PROCESSING COMPLETE")



    except Exception as e:
        logger.exception("PROCESSING FAILED")
        sys.exit(1)

    # cleanup()


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process HLS data to EVI2.")
    parser.add_argument("--mgrs_tile", required=True, help="MGRS tile ID (e.g., 18SUJ).")
    parser.add_argument("--start_date", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end_date", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the final outputs.")
    parser.add_argument("--n_workers", default=1, type=int, help="Number of parallel workers for download/processing.")

    args = parser.parse_args()

    mgrs_tile = args.mgrs_tile
    start_date = args.start_date
    end_date = args.end_date
    output_dir = args.output_dir

    LOG_DIR = f"{output_dir}/logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    logger, log_file = setup_logger(mgrs_tile, start_date, end_date)
    logger.info("=" * 60)
    logger.info(f"START PROCESSING {mgrs_tile} {start_date}:{end_date}")
    logger.info("=" * 60)

    process_hls_tile(
        mgrs_tile=mgrs_tile,
        start_date=start_date,
        end_date=end_date,
        output_dir=f"{output_dir}/{mgrs_tile}",
        n_workers=args.n_workers,
    )