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

import geopandas
import rasterio as rio
import earthaccess

import dask.array as da

from maap.maap import MAAP
from pystac import Item
from rasterio.session import AWSSession
from rustac import DuckdbClient

from concurrent.futures import ThreadPoolExecutor, as_completed

GDAL_CONFIG = {
    "CPL_TMPDIR": "/tmp",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "TIF,GPKG,SHP,SHX,PRJ,DBF,JSON,GEOJSON",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
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

# LPCLOUD S3 CREDENTIAL REFRESH
CREDENTIAL_REFRESH_SECONDS = 50 * 60
SCENE_TIMEOUT_SECONDS = 10 * 60 

class CredentialManager:
    """Thread-safe credential manager for S3 access"""

    def __init__(self):
        self._lock = threading.Lock()
        self._credentials: dict[str, Any] | None = None
        self._fetch_time: float | None = None
        self._session: AWSSession | None = None

    def get_session(self) -> AWSSession:
        """Get current session, refreshing credentials if needed"""
        with self._lock:
            now = time.time()
            if (
                self._credentials is None
                or self._fetch_time is None
                or (now - self._fetch_time) > CREDENTIAL_REFRESH_SECONDS
            ):
                print("fetching/refreshing S3 credentials")
                self._credentials = self._fetch_credentials()
                self._fetch_time = now
                self._session = AWSSession(**self._credentials)
            return self._session

    @staticmethod
    def _fetch_credentials() -> dict[str, Any]:
        """Fetch new credentials from MAAP"""
        maap = MAAP(maap_host="api.maap-project.org")
        creds = maap.aws.earthdata_s3_credentials(
            "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials"
        )
        return {
            "aws_access_key_id": creds["accessKeyId"],
            "aws_secret_access_key": creds["secretAccessKey"],
            "aws_session_token": creds["sessionToken"],
            "region_name": "us-west-2",
        }


# Global credential manager instance
_credential_manager = CredentialManager()

HLS_COLLECTIONS = ["HLSL30_2.0", "HLSS30_2.0"]
HLS_STAC_GEOPARQUET_HREF = "s3://nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/{collection}/**/*.parquet"

URL_PREFIX = "https://data.lpdaac.earthdatacloud.nasa.gov/"
NODATA = -9999
FMASK_NODATA = 255

sr_scale = 0.0001
SR_FILL = -9999
QA_FILL = 255

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


def saveGeoTiff(filename, data, template_file, access_type="direct", nodata=None, scale=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    nband = 1 if data.ndim == 2 else data.shape[0]

    rasterio_env = {"session": _credential_manager.get_session()} if access_type == "direct" else {}

    with rio.Env(**rasterio_env):
        with rio.open(template_file) as ds:
            profile = ds.profile.copy()
            profile.update({
                "dtype": data.dtype,
                "count": nband,
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


def fetch_single_asset(asset_href: str, fill_value=SR_FILL, direct_bucket_access: bool = False):
    try:
        rasterio_env = {}
        if direct_bucket_access:
            rasterio_env["session"] = _credential_manager.get_session()
        with rio.Env(**rasterio_env):
            with rio.open(asset_href) as src:
                return da.from_array(src.read(1), chunks=chunk_size)
    except Exception as e:
        raise 


def fetch_with_retry(
    asset_href,
    max_retries: int = 5,          # more retries to absorb 503 bursts
    base_delay: float = 2.0,       # starting backoff in seconds
    max_delay: float = 60.0,       # cap backoff so it doesn't spiral
    fill_value=SR_FILL,
    access_type="external",
):

    for attempt in range(max_retries):
        try:
            return fetch_single_asset(
                asset_href=asset_href,
                fill_value=fill_value,
                direct_bucket_access=(access_type == "direct"),
            )
        except Exception as e:
            is_last = attempt == max_retries - 1
            err_str = str(e)
            is_503 = "503" in err_str or "ServiceUnavailable" in err_str or "SlowDown" in err_str

            if is_last:
                print(
                    f"All {max_retries} attempts failed for {asset_href}. Last error: {e}"
                )
                return None

            # Exponential backoff: 2^attempt * base, capped at max_delay
            computed = min(base_delay * (2 ** attempt), max_delay)
            # Full jitter: random value in [0, computed] — spreads retries across threads
            wait = computed * (0.5 + 0.5 * (time.time() % 1))  # cheap jitter without import random

            if is_503:
                # 503s are throttle signals — always back off, even on first attempt
                wait = max(wait, base_delay * 2)
                print(
                    f"503 throttle on attempt {attempt + 1}/{max_retries} for "
                    f"{os.path.basename(asset_href)} — backing off {wait:.1f}s"
                )
            else:
                print(
                    f"Attempt {attempt + 1}/{max_retries} failed for "
                    f"{os.path.basename(asset_href)}: {e} — retrying in {wait:.1f}s"
                )

            time.sleep(wait)


def find_tile_bounds(tile: str):
    gdf = geopandas.read_file(
        r"s3://maap-ops-workspace/shared/zhouqiang06/AuxData/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp"
    )
    bounds_list = [np.round(c, 3) for c in gdf[gdf["Name"] == tile].bounds.values[0]]
    return tuple(bounds_list)


def filter_url(url: str, tile: str, band: str):
    if (os.path.basename(url).split(".")[2][1:] == tile) and (url.endswith(f"{band}.tif")):
        return True
    return False


def get_HLS_data(tile: str, bandnum: int, start_date: str, end_date: str, access_type="external"):
    print("Searching HLS STAC Geoparquet archive for HLS data...")
    client = DuckdbClient(use_hive_partitioning=True)
    client.execute(
        """
        CREATE OR REPLACE SECRET secret (
            TYPE S3,
            PROVIDER CREDENTIAL_CHAIN
        );
        """
    )
    results = []
    for collection in HLS_COLLECTIONS:
        response = client.search(
            href=HLS_STAC_GEOPARQUET_HREF.format(collection=collection),
            datetime=f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            bbox=find_tile_bounds(tile),
        )
        results.extend(GetBandLists_HLS_STAC(response, tile, bandnum))
    if access_type == "direct":
        results = [r.replace(URL_PREFIX, "s3://") for r in results]
    return results


def GetBandLists_HLS_STAC(response, tile: str, bandnum: int):
    BandList = []
    for i in range(len(response)):
        product_type = response[i]["id"].split(".")[1]
        if product_type == "L30":
            bands = {2: "B02", 3: "B03", 4: "B04", 5: "B05", 6: "B06", 7: "B07", 8: "Fmask"}
        elif product_type == "S30":
            bands = {2: "B02", 3: "B03", 4: "B04", 5: "B8A", 6: "B11", 7: "B12", 8: "Fmask"}
        else:
            print("HLS product type not recognized: Must be L30 or S30.")
            os._exit(1)
        try:
            getBand = response[i]["assets"][bands[bandnum]]["href"]
            if filter_url(getBand, tile, bands[bandnum]):
                BandList.append(getBand)
        except Exception as e:
            print(e)
    return BandList


def find_all_granules(tile: str, start_date: str, end_date: str, access_type="external"):
    url_list = get_HLS_data(tile=tile, bandnum=8, start_date=start_date, end_date=end_date, access_type=access_type)
    if len(url_list) == 0:
        print("No granules found.")
        return pd.DataFrame()
    sat_list = [os.path.basename(g).split(".")[1] for g in url_list]
    date_list = [
        datetime.strptime(os.path.basename(g).split(".")[3][:7], "%Y%j") for g in url_list
    ]
    return pd.DataFrame({"Date": date_list, "Sat": sat_list, "granule_path": url_list})

def process_scene(
    row,
    access_type: str = "direct",
    bands: list[str] = common_bands,
) -> dict[str, da.Array | None]:

    name2index = L8_name2index if row.Sat in ("L30", "L10") else S2_name2index
    scene_bands: dict[str, da.Array | None] = {}

    for band in bands:
        band_key = name2index[band]
        url = row.granule_path.replace("Fmask", band_key)
        arr = fetch_with_retry(url, access_type=access_type)

        if arr is None:
            print(f"[{row.Sat} | {row.Date}] band '{band}' fetch returned None: {url}")
        elif arr.shape != (image_size[0], image_size[1]):
            print(
                f"[{row.Sat} | {row.Date}] band '{band}' unexpected shape {arr.shape}: {url}"
            )
            arr = None

        scene_bands[band] = arr

    return scene_bands


def process_and_save_scene(row, tile: str, out_dir: str, access_type: str) -> str:
    """
    Full pipeline for one scene: fetch → mask → EVI2 → save.

    Returns a short status string for the caller to log.
    """
    scene_id = f"{row.Sat} {row.Date}"

    # --- Fetch all bands ---
    scene_bands = process_scene(row, access_type=access_type, bands=common_bands)

    missing = [b for b, a in scene_bands.items() if a is None]
    if missing:
        return f"SKIP  {scene_id} — missing bands: {missing}"

    fmask = scene_bands["Fmask"]

    # --- Pixel quality mask ---
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
    )
    bad_pixel_mask = bad_pixel_mask | (
        ((fmask & (1 << QA_BIT["water"])) > 0)
        | ((fmask & (1 << QA_BIT["snowice"])) > 0)
    )

    # --- EVI2 ---
    red = scene_bands["Red"].astype(np.float32) * sr_scale
    nir = scene_bands["NIR_Narrow"].astype(np.float32) * sr_scale
    evi2 = 2.5 * (nir - red) / (nir + 2.4 * red + 1)
    evi2_out = da.where(bad_pixel_mask, np.nan, evi2).compute().astype(np.float32)

    # --- Output path ---
    scene_date = datetime.strptime(
        os.path.basename(row.granule_path).split(".")[3][:7], "%Y%j"
    )
    scene_dir = os.path.join(out_dir, scene_date.strftime("%Y"))
    os.makedirs(scene_dir, exist_ok=True)

    out_name = f"HLS.{row.Sat}.T{tile}.{scene_date.year}{scene_date.strftime('%j')}.2.0.EVI2.tif"
    out_path = os.path.join(scene_dir, out_name)
    saveGeoTiff(out_path, evi2_out, row.granule_path, nodata=np.nan)

    return f"OK    {scene_id}: {out_path}"


def process_hls(tile, start_date, end_date, save_dir, access_type="direct", N_WORKERS=1):
    out_dir = os.path.join(save_dir, tile)
    os.makedirs(out_dir, exist_ok=True)

    granule_df = find_all_granules(
        tile=tile, start_date=start_date, end_date=end_date, access_type=access_type
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
        n_done = 0
        n_ok = 0
        n_skip = 0
        n_err = 0
        for future in as_completed(futures):
            row = futures[future]
            n_done += 1
            try:
                status = future.result(timeout=SCENE_TIMEOUT_SECONDS)
                if status.startswith("OK"):
                    n_ok += 1
                else:
                    n_skip += 1
                print(f"[{n_done}/{n_scenes}] {status}")
            except TimeoutError:
                n_err += 1
                print(f"[{n_done}/{n_scenes}] TIMEOUT {row.Sat} {row.Date} — scene took >{SCENE_TIMEOUT_SECONDS}s")
            except Exception as e:
                n_err += 1
                print(f"[{n_done}/{n_scenes}] ERROR {row.Sat} {row.Date}: {e}")

    print(
        f"Done — {n_scenes} scenes: {n_ok} saved, {n_skip} skipped, {n_err} errors"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--access_type", default="direct")
    parser.add_argument("--N_WORKERS", default=1, type=int)
    args = parser.parse_args()

    os.environ.update(GDAL_CONFIG)
    process_hls(
        tile=args.tile,
        start_date=args.start_date,
        end_date=args.end_date,
        save_dir=args.output_dir,
        access_type=args.access_type,
        N_WORKERS=args.N_WORKERS,
    )