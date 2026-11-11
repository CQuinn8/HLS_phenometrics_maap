# Usage:
# python download_hls.py --tile=18SUJ --start_date="2024-01-01" --end_date="2024-03-30" --output_dir="./temp_out"

#!/panfs/ccds02/nobackup/people/qzhou2/miniforge3/envs/hls_mamba/bin/python
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime

import argparse
import logging
import threading
import time
from typing import Any

import rasterio as rio
import earthaccess

import dask.array as da

from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import logging
from glob import glob

from maap.maap import MAAP


import boto3
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed


GDAL_CONFIG = {
    "CPL_TMPDIR": "/tmp",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "TIF,GPKG,SHP,SHX,PRJ,DBF,JSON,GEOJSON",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    # "GDAL_HTTP_MULTIPLEX": "YES",
    # "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MULTIPLEX": "No",
    "GDAL_HTTP_VERSION": "1.1",
    "PYTHONWARNINGS": "ignore",
    "GDAL_NUM_THREADS": "ALL_CPUS",
    "GDAL_HTTP_COOKIEFILE": str(Path.home() / "cookies.txt"),
    "GDAL_HTTP_COOKIEJAR": str(Path.home() / "cookies.txt"),
    "GDAL_HTTP_UNSAFESSL": "YES",
    "GDAL_HTTP_TIMEOUT": "60",          # total request timeout (seconds)
    "GDAL_HTTP_CONNECTTIMEOUT": "30",   # TCP connect timeout (seconds)
    "GDAL_HTTP_LOW_SPEED_TIME": "30",   # if transfer rate drops below...
    "GDAL_HTTP_LOW_SPEED_LIMIT": "1000",# 1KB/s for 30s abort
}

SCENE_TIMEOUT_SECONDS = 10 * 60 
NODATA = -9999
FMASK_NODATA = 255

sr_scale = 0.0001
SR_FILL = -9999
QA_FILL = 255

logger = logging.getLogger("hls_evi2")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

HLS_PRODUCTS = {
    "L30": "HLSL30",
    "S30": "HLSS30",
}

REQUIRED_HLS_BANDS = {
    "L30": ["B02", "B03", "B04", "B05", "B06", "B07", "Fmask"],
    "S30": ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"],
}


common_bands = ["Blue", "Green", "Red", "NIR_Narrow", "SWIR1", "SWIR2", "Fmask"]

L8_name2index = {
    "Blue": "B02", "Green": "B03", "Red": "B04",
    "NIR_Narrow": "B05", "SWIR1": "B06", "SWIR2": "B07", "Fmask": "Fmask",
}
S2_name2index = {
    "Blue": "B02", "Green": "B03", "Red": "B04",
    "NIR_Narrow": "B8A", "SWIR1": "B11", "SWIR2": "B12", "Fmask": "Fmask",
}

QA_BIT = {
    "cirrus": 0,
    "cloud": 1,
    "adj_cloud": 2,
    "cloud shadow": 3,
    "snowice": 4,
    "water": 5,
    "aerosol_l": 6,
    "aerosol_h": 7,
}

chunk_size = (1830, 1830)
image_size = (3660, 3660)

def configure_earthaccess():
    """
    Configure Earthdata authentication.

    On MAAP/DPS this tries to populate EARTHDATA_USERNAME/PASSWORD
    from MAAP secrets if they are not already present.

    Outside MAAP, you can use either environment variables or ~/.netrc.
    """

    if (
        ("EARTHDATA_USERNAME" not in os.environ or "EARTHDATA_PASSWORD" not in os.environ)
        and MAAP is not None
    ):
        try:
            maap = MAAP(maap_host="api.maap-project.org")
            os.environ["EARTHDATA_USERNAME"] = maap.secrets.get_secret("EARTHDATA_USERNAME")
            os.environ["EARTHDATA_PASSWORD"] = maap.secrets.get_secret("EARTHDATA_PASSWORD")
            logger.info("Loaded Earthdata credentials from MAAP secrets")
        except Exception as e:
            logger.warning(f"Could not load Earthdata credentials from MAAP secrets: {e}")

    if "EARTHDATA_USERNAME" in os.environ and "EARTHDATA_PASSWORD" in os.environ:
        logger.info("Logging into Earthdata using environment credentials")
        earthaccess.login(strategy="environment")
    else:
        logger.info("Logging into Earthdata using netrc credentials")
        earthaccess.login(strategy="netrc")


def parse_hls_filename(path: str):
    """
    Parse local HLS filename.

    Expected examples:
      HLS.L30.T18SUJ.2024001T160901.v2.0.B04.tif
      HLS.S30.T18SUJ.2024001T160901.v2.0.Fmask.tif
    """

    name = os.path.basename(str(path))
    parts = name.split(".")

    if len(parts) < 8:
        return None

    if parts[0] != "HLS":
        return None

    if not name.lower().endswith(".tif"):
        return None

    product = parts[1]          # L30 or S30
    tile_part = parts[2]        # T18SUJ
    date_part = parts[3]        # 2024001T...
    band = parts[-2]            # B04, B8A, Fmask, etc.

    if product not in HLS_PRODUCTS:
        return None

    if not tile_part.startswith("T"):
        return None

    try:
        date = datetime.strptime(date_part[:7], "%Y%j")
    except Exception:
        return None

    return {
        "product": product,
        "tile": tile_part[1:],
        "date": date,
        "band": band,
        "path": str(path),
    }


def is_hls_band_file(path: str, tile: str, band: str):
    info = parse_hls_filename(path)
    return (
        info is not None
        and info["tile"] == tile
        and info["band"] == band
    )


def configure_lpdaac_s3_session():
    """
    Configure a boto3 session for direct LPDAAC Earthdata Cloud S3 access.

    On MAAP, this uses MAAP's LPDAAC Earthdata S3 credentials endpoint.
    """

    configure_earthaccess()

    if MAAP is None:
        raise RuntimeError(
            "MAAP package is not available. For direct LPDAAC S3 download, "
            "run this in MAAP or modify this function to use your own Earthdata "
            "Cloud S3 credentials."
        )

    maap = MAAP(maap_host="api.maap-project.org")

    creds = maap.aws.earthdata_s3_credentials(
        "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials"
    )

    session = boto3.Session(
        aws_access_key_id=creds["accessKeyId"],
        aws_secret_access_key=creds["secretAccessKey"],
        aws_session_token=creds["sessionToken"],
        region_name="us-west-2",
    )

    return session


def granule_data_links(granule, access="direct"):
    """
    Return data links from an earthaccess DataGranule.

    access='direct' should return s3:// links for cloud-hosted LPDAAC data.
    """

    try:
        return granule.data_links(access=access)
    except TypeError:
        return granule.data_links()
    except Exception:
        links = []

        try:
            related = granule.get("umm", {}).get("RelatedUrls", [])
            for entry in related:
                url = entry.get("URL")
                if url:
                    links.append(url)
        except Exception:
            pass

        return links


def wanted_hls_asset(url: str, product: str):
    """
    Keep only the HLS TIF bands needed by the EVI2/QA workflow.
    """

    base = os.path.basename(urlparse(url).path)

    if not base.lower().endswith(".tif"):
        return False

    wanted_bands = REQUIRED_HLS_BANDS[product]

    return any(base.endswith(f".{band}.tif") for band in wanted_bands)


def download_one_s3_file(s3_client, s3_url: str, local_dir: str):
    """
    Download a single s3:// file to local_dir using boto3.

    Uses a temporary .part file and atomic rename to avoid partial/corrupt outputs.
    """

    parsed = urlparse(s3_url)

    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URL, got: {s3_url}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    local_path = os.path.join(local_dir, os.path.basename(key))

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        logger.info(f"Exists, skipping: {local_path}")
        return local_path

    tmp_path = f"{local_path}.part.{os.getpid()}.{threading.get_ident()}"

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    logger.info(f"Downloading {s3_url} -> {local_path}")

    try:
        s3_client.download_file(
            bucket,
            key,
            tmp_path,
            ExtraArgs={"RequestPayer": "requester"},
        )
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    os.replace(tmp_path, local_path)

    return local_path


def download_hls_data(
    tile: str,
    start_date: str,
    end_date: str,
    download_dir: str,
    products: tuple[str, ...] = ("L30", "S30"),
    download_workers: int = 4,
):
    """
    Search and download only required HLS L30/S30 bands for tile/date range.

    This avoids earthaccess.download(), which can fail with FileExistsError
    on large parallel downloads.
    """

    os.makedirs(download_dir, exist_ok=True)

    boto3_session = configure_lpdaac_s3_session()
    s3_client = boto3_session.client("s3", region_name="us-west-2")

    all_s3_urls = []

    for product in products:
        short_name = HLS_PRODUCTS[product]
        granule_pattern = f"HLS.{product}.T{tile}.*"

        logger.info(
            f"Searching Earthdata for {short_name}, tile={tile}, "
            f"dates={start_date} to {end_date}"
        )

        results = earthaccess.search_data(
            short_name=short_name,
            version="2.0",
            temporal=(start_date, end_date),
            granule_name=granule_pattern,
        )

        logger.info(f"Found {len(results)} {short_name} granules")

        for granule in results:
            links = granule_data_links(granule, access="direct")

            for link in links:
                if not link.startswith("s3://"):
                    continue

                if wanted_hls_asset(link, product):
                    all_s3_urls.append(link)

    all_s3_urls = sorted(set(all_s3_urls))

    if len(all_s3_urls) == 0:
        logger.info("No downloadable HLS S3 assets found")
        return []

    logger.info(f"Downloading {len(all_s3_urls)} required HLS assets")

    downloaded_files = []

    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        futures = {
            executor.submit(
                download_one_s3_file,
                s3_client,
                url,
                download_dir,
            ): url
            for url in all_s3_urls
        }

        for i, future in enumerate(as_completed(futures), start=1):
            url = futures[future]

            try:
                path = future.result()
                downloaded_files.append(path)
                logger.info(f"[{i}/{len(futures)}] downloaded: {os.path.basename(path)}")

            except Exception as e:
                logger.error(f"[{i}/{len(futures)}] failed: {url}")
                logger.error(str(e))

    logger.info(f"Downloaded/found {len(downloaded_files)} HLS files")

    return downloaded_files
    

def find_all_granules(
    tile: str,
    start_date: str,
    end_date: str,
    download_dir: str,
):
    """
    Download HLS data and return one row per granule, using the local Fmask path
    as the anchor path. The rest of the workflow derives band paths from Fmask.
    """
    
    downloaded_files = download_hls_data(
        tile=tile,
        start_date=start_date,
        end_date=end_date,
        download_dir=download_dir,
        products=("L30", "S30"),
    )

    # Use returned paths plus a fallback scan of download_dir in case files
    # already existed and earthaccess did not return all of them.
    candidate_paths = set(downloaded_files)

    for tif in Path(download_dir).rglob("*.tif"):
        candidate_paths.add(str(tif))

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    rows = []

    for path in sorted(candidate_paths):
        info = parse_hls_filename(path)

        if info is None:
            continue

        if info["tile"] != tile:
            continue

        if info["band"] != "Fmask":
            continue

        if not (start_dt <= info["date"] <= end_dt):
            continue

        rows.append(
            {
                "Date": info["date"],
                "Sat": info["product"],
                "granule_path": info["path"],
            }
        )

    if len(rows) == 0:
        logger.info("No HLS Fmask granules found after download")
        return pd.DataFrame(columns=["Date", "Sat", "granule_path"])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["granule_path"])
    df = df.sort_values(["Date", "Sat", "granule_path"]).reset_index(drop=True)

    return df


def band_path_from_fmask(fmask_path: str, band_key: str):
    """
    Convert a local Fmask filename to another local band filename.

    Example:
      HLS.S30.T18SUJ.2024001T160901.v2.0.Fmask.tif
    becomes:
      HLS.S30.T18SUJ.2024001T160901.v2.0.B8A.tif
    """

    p = Path(fmask_path)
    parts = p.name.split(".")

    if len(parts) < 8:
        raise ValueError(f"Unexpected HLS filename: {p.name}")

    parts[-2] = band_key
    candidate = p.with_name(".".join(parts))

    return str(candidate)

def saveGeoTiff(filename, data, template_file, nodata=None, scale=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    nband = 1 if data.ndim == 2 else data.shape[0]

    with rio.Env(**GDAL_CONFIG):
        with rio.open(template_file) as ds:
            profile = ds.profile.copy()
            profile.update({
                "dtype": data.dtype,
                "count": nband,
                "driver": "GTiff",
                "height": data.shape[-2],
                "width": data.shape[-1],
                "compress": "lzw",
                "nodata": nodata,
            })

        with rio.open(filename, "w", **profile) as dst:
            if nband == 1:
                dst.write(data, 1)
            else:
                for i in range(nband):
                    dst.write(data[i], i + 1)

            if scale is not None:
                dst._set_all_scales([scale] * nband)


def fetch_single_asset(asset_href: str, fill_value=SR_FILL):
    """
    Read one local HLS band into a dask array.

    asset_href is now a local path downloaded by earthaccess.
    """

    if not os.path.exists(asset_href):
        raise FileNotFoundError(asset_href)

    with rio.Env(**GDAL_CONFIG):
        with rio.open(asset_href) as src:
            return da.from_array(src.read(1), chunks=chunk_size)


def fetch_with_retry(
    asset_href,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    fill_value=SR_FILL,
    access_type="local",
):
    for attempt in range(max_retries):
        try:
            return fetch_single_asset(
                asset_href=asset_href,
                fill_value=fill_value,
            )
        except Exception as e:
            is_last = attempt == max_retries - 1

            if is_last:
                print(
                    f"All {max_retries} attempts failed for {asset_href}. "
                    f"Last error: {e}"
                )
                return None

            computed = min(base_delay * (2 ** attempt), max_delay)
            wait = computed * (0.5 + 0.5 * (time.time() % 1))

            print(
                f"Attempt {attempt + 1}/{max_retries} failed for "
                f"{os.path.basename(asset_href)}: {e} — retrying in {wait:.1f}s"
            )

            time.sleep(wait)


def process_scene(
    row,
    access_type: str = "local",
    bands: list[str] = common_bands,
) -> dict[str, da.Array | None]:

    name2index = L8_name2index if row.Sat == "L30" else S2_name2index
    scene_bands: dict[str, da.Array | None] = {}

    for band in bands:
        band_key = name2index[band]
        path = band_path_from_fmask(row.granule_path, band_key)

        arr = fetch_with_retry(path, access_type=access_type)

        if arr is None:
            print(f"[{row.Sat} | {row.Date}] band '{band}' fetch returned None: {path}")

        elif arr.shape != (image_size[0], image_size[1]):
            print(
                f"[{row.Sat} | {row.Date}] band '{band}' unexpected shape "
                f"{arr.shape}: {path}"
            )
            arr = None

        scene_bands[band] = arr

    return scene_bands

    
def process_and_save_scene(
    row, tile: str, out_dir: str, access_type: str
) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    """
    Returns (status, pre_mask_array, post_mask_array)
    pre_mask  : uint8 array — 1 where pixel was attempted (not fill/nodata)
    post_mask : uint8 array — 1 where pixel passed quality mask
    """
    scene_id = f"{row.Sat} {row.Date}"

    scene_bands = process_scene(row, access_type=access_type, bands=common_bands)
    missing = [b for b, a in scene_bands.items() if a is None]
    if missing:
        return f"SKIP  {scene_id} — missing bands: {missing}", None, None

    fmask = scene_bands["Fmask"]

    is_negative = (
        (scene_bands["Red"] < 0) | (scene_bands["NIR_Narrow"] < 0)
        | (scene_bands["Blue"] < 0) | (scene_bands["Green"] < 0)
        | (scene_bands["SWIR1"] < 0) | (scene_bands["SWIR2"] < 0)
    )
    bad_pixel_mask = (
        ((fmask & (1 << QA_BIT["cloud"])) > 0)
        | ((fmask & (1 << QA_BIT["adj_cloud"])) > 0)
        | ((fmask & (1 << QA_BIT["cloud shadow"])) > 0)
        | (fmask == QA_FILL)
        | is_negative
        | ((fmask & (1 << QA_BIT["water"])) > 0)
        | ((fmask & (1 << QA_BIT["snowice"])) > 0)
    )

    # Compute masks for count TIFs
    fmask_computed        = fmask.compute()
    bad_computed          = bad_pixel_mask.compute()

    pre_mask_arr  = (fmask_computed != QA_FILL).astype(np.uint8)    # valid data extent
    post_mask_arr = (~bad_computed).astype(np.uint8)                # passed QA

    # EVI2
    red     = scene_bands["Red"].astype(np.float32) * sr_scale
    nir     = scene_bands["NIR_Narrow"].astype(np.float32) * sr_scale
    evi2    = 2.5 * (nir - red) / (nir + 2.4 * red + 1)
    evi2_out = da.where(bad_pixel_mask, np.nan, evi2).compute().astype(np.float32)

    scene_date = datetime.strptime(os.path.basename(row.granule_path).split(".")[3][:7], "%Y%j")
    scene_dir = os.path.join(out_dir, scene_date.strftime("%Y"))
    os.makedirs(scene_dir, exist_ok=True)

    out_name  = f"HLS.{row.Sat}.T{tile}.{scene_date.year}{scene_date.strftime('%j')}.2.0.EVI2.tif"
    out_path  = os.path.join(scene_dir, out_name)
    saveGeoTiff(out_path, evi2_out, row.granule_path, nodata=np.nan)

    return f"OK    {scene_id}: {out_path}", pre_mask_arr, post_mask_arr

    
def process_hls(
    tile,
    start_date,
    end_date,
    save_dir,
    access_type="local",
    N_WORKERS=1,
    download_dir=None,
):
    out_dir = os.path.join(save_dir, tile)
    os.makedirs(out_dir, exist_ok=True)

    if download_dir is None:
        safe_start = start_date.replace("-", "")
        safe_end = end_date.replace("-", "")
        download_dir = os.path.join("/tmp", "hls_downloads", f"{tile}_{safe_start}_{safe_end}")

    os.makedirs(download_dir, exist_ok=True)

    print(f"HLS download directory: {download_dir}")

    granule_df = find_all_granules(
        tile=tile,
        start_date=start_date,
        end_date=end_date,
        download_dir=download_dir,
    )

    print(granule_df)

    if len(granule_df) == 0:
        print(f"No granules found for {tile}. Creating empty indicator file.")
        with open(os.path.join(out_dir, "No granules found"), "w") as f:
            pass
        return

    rows = list(granule_df.itertuples())
    n_scenes = len(rows)

    print(f"Submitting {n_scenes} scenes to {N_WORKERS} workers")

    from collections import defaultdict

    pre_accum = defaultdict(lambda: np.zeros(image_size, dtype=np.uint16))
    post_accum = defaultdict(lambda: np.zeros(image_size, dtype=np.uint16))

    template_path = rows[0].granule_path

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(
                process_and_save_scene,
                row,
                tile=tile,
                out_dir=out_dir,
                access_type=access_type,
            ): row
            for row in rows
        }

        n_done = n_ok = n_skip = n_err = 0

        for future in as_completed(futures):
            row = futures[future]
            n_done += 1

            try:
                status, pre_arr, post_arr = future.result(timeout=SCENE_TIMEOUT_SECONDS)

                if status.startswith("OK"):
                    n_ok += 1
                    scene_year = row.Date.year
                    pre_accum[scene_year] += pre_arr
                    post_accum[scene_year] += post_arr
                else:
                    n_skip += 1

                print(f"[{n_done}/{n_scenes}] {status}")

            except TimeoutError:
                n_err += 1
                print(f"[{n_done}/{n_scenes}] TIMEOUT {row.Sat} {row.Date}")

            except Exception as e:
                n_err += 1
                print(f"[{n_done}/{n_scenes}] ERROR {row.Sat} {row.Date}: {e}")

    for year, pre in sorted(pre_accum.items()):
        post = post_accum[year]
        year_dir = os.path.join(out_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        with np.errstate(divide="ignore", invalid="ignore"):
            viable_pct = np.where(
                pre > 0,
                np.round(post.astype(np.float32) / pre * 100).astype(np.uint8),
                np.uint8(0),
            )

        saveGeoTiff(
            os.path.join(year_dir, f"HLS.T{tile}.{year}.PreMaskCount.tif"),
            pre,
            template_path,
            nodata=65535,
        )

        saveGeoTiff(
            os.path.join(year_dir, f"HLS.T{tile}.{year}.PostMaskCount.tif"),
            post,
            template_path,
            nodata=65535,
        )

        saveGeoTiff(
            os.path.join(year_dir, f"HLS.T{tile}.{year}.ViablePct.tif"),
            viable_pct,
            template_path,
            nodata=255,
        )

        valid = viable_pct[pre > 0]
        mean_viable = valid.mean() if valid.size > 0 else np.nan

        print(
            f"  [{year}] quality TIFs saved — "
            f"mean viable: {mean_viable:.1f}%  "
            f"scenes: {int(pre.max())}"
        )

    print(f"Done — {n_scenes} scenes: {n_ok} saved, {n_skip} skipped, {n_err} errors")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tile", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--output_dir", required=True)

    # Kept for backward compatibility with your original CLI.
    # The new downloader processes local files after earthaccess download.
    parser.add_argument("--access_type", default="local")

    parser.add_argument("--N_WORKERS", default=1, type=int)

    parser.add_argument(
        "--download_dir",
        default=None,
        help=(
            "Optional local directory for downloaded HLS files. "
            "Default: /tmp/hls_downloads/<tile>_<start>_<end>"
        ),
    )

    args = parser.parse_args()

    os.environ.update(GDAL_CONFIG)

    process_hls(
        tile=args.tile,
        start_date=args.start_date,
        end_date=args.end_date,
        save_dir=args.output_dir,
        access_type=args.access_type,
        N_WORKERS=args.N_WORKERS,
        download_dir=args.download_dir,
    )