# Usage:
# composite: python download_hls.py --tile=18SUJ --start_date="2024-01-01" --end_date="2024-03-30" --output_dir="./temp_out" --stat="max"
# scene-level only: python download_hls.py --tile=18SUJ --start_date="2024-01-01" --end_date="2024-03-30" --output_dir="./temp_out" --scene_only=True --mask_water=True

#!/panfs/ccds02/nobackup/people/qzhou2/miniforge3/envs/hls_mamba/bin/python
# import platform
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime

import argparse
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import geopandas
import rasterio as rio
import rioxarray as rxr
import earthaccess

import dask.array as da

from maap.maap import MAAP
from pystac import Asset, Catalog, CatalogType, Item
from rasterio.session import AWSSession
from rustac import DuckdbClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logging.getLogger("botocore").setLevel(logging.WARNING)
logger = logging.getLogger("HLSComposite")

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

}

# LPCLOUD S3 CREDENTIAL REFRESH
CREDENTIAL_REFRESH_SECONDS = 50 * 60

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

            # Check if credentials need refresh
            if (
                self._credentials is None
                or self._fetch_time is None
                or (now - self._fetch_time) > CREDENTIAL_REFRESH_SECONDS
            ):
                logger.info("fetching/refreshing S3 credentials")
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
DTYPE = "int16"
FMASK_DTYPE = "uint8"
NODATA = -9999
FMASK_NODATA = 255

sr_scale = 0.0001
ang_scale = 0.01
SR_FILL = -9999
QA_FILL = 255 #FMASK_FILL

common_bands = ["Blue","Green","Red","NIR_Narrow","SWIR1", "SWIR2", "Fmask"]

L8_bandname = {"B01":"Coastal_Aerosol", "B02":"Blue", "B03":"Green", "B04":"Red", 
               "B05":"NIR_Narrow", "B06":"SWIR 1", "B07":"SWIR 2", "B09":"Cirrus"}
S2_bandname = {"B01":"Coastal_Aerosol", "B02":"Blue", "B03":"Green", "B04":"Red", 
               "B8A":"NIR_Narrow", "B11":"SWIR1", "B12":"SWIR2", "B10":"Cirrus"}

L8_name2index = {'Coastal_Aerosol': 'B01', 'Blue': 'B02', 'Green': 'B03', 'Red': 'B04',
                 'NIR_Narrow': 'B05', 'SWIR1': 'B06', 'SWIR2': 'B07', 'Fmask': 'Fmask'}
S2_name2index = {'Coastal_Aerosol': 'B01', 'Blue': 'B02', 'Green': 'B03', 'Red': 'B04', 
                 'NIR_Edge1': 'B05', 'NIR_Edge2': 'B06', 'NIR_Edge3': 'B07', 
                  'NIR_Broad': 'B08', 'NIR_Narrow': 'B8A', 'SWIR1': 'B11', 'SWIR2': 'B12', 'Fmask': 'Fmask'}

BAND_MAPPING = {
    "HLSL30_2.0": {
        "coastal_aerosol": "B01",
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir_narrow": "B05",
        "swir_1": "B06",
        "swir_2": "B07",
        "cirrus": "B09",
        "thermal_infrared_1": "B10",
        "thermal": "B11",
        "Fmask": "Fmask",
    },
    "HLSS30_2.0": {
        "coastal_aerosol": "B01",
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "red_edge_1": "B05",
        "red_edge_2": "B06",
        "red_edge_3": "B07",
        "nir_broad": "B08",
        "nir_narrow": "B8A",
        "water_vapor": "B09",
        "cirrus": "B10",
        "swir_1": "B11",
        "swir_2": "B12",
        "Fmask": "Fmask",
    },
}
# these are the ones that we are going to use
DEFAULT_BANDS = [
    "red",
    "green",
    "blue",
    "nir_narrow",
    "swir_1",
    "swir_2",
    "Fmask",
]
DEFAULT_RESOLUTION = 30


QA_BIT = {'cirrus': 0,
'cloud': 1,
'adj_cloud': 2,
'cloud shadow':3,
'snowice':4,
'water':5,
'aerosol_l': 6,
'aerosol_h': 7
}

chunk_size = dict(band=1, x=1830, y=1830)
image_size = (3660, 3660)

def mask_hls(qa_arr, mask_list=['cloud', 'adj_cloud', 'cloud shadow']):
    # This function takes the HLS QA array as input and exports the cloud mask array. 
    # The mask_list assigns the QA conditions you would like to mask.
    msk = np.zeros_like(qa_arr)#.astype(bool)
    for m in mask_list:
        if m in QA_BIT.keys():
            msk += (qa_arr & 1 << QA_BIT[m]) > 0
        if m == 'aerosol_high':
            msk += ((qa_arr & (1 << QA_BIT['aerosol_h'])) > 0) * ((qa_arr & (1 << QA_BIT['aerosol_l'])) > 0)
        if m == 'aerosol_moderate':
            msk += ((qa_arr & (1 << QA_BIT['aerosol_h'])) > 0) * ((qa_arr | (1 << QA_BIT['aerosol_l'])) != qa_arr)
        if m == 'aerosol_low':
            msk += ((qa_arr | (1 << QA_BIT['aerosol_h'])) != qa_arr) * ((qa_arr & (1 << QA_BIT['aerosol_l'])) > 0)
    return msk > 0


def get_geo(filepath):
    # ds = gdal.Open(filepath)
    # return ds.GetGeoTransform(), ds.GetProjection()
    with rio.open(filepath) as ds:
        return ds.transform, ds.crs


def saveGeoTiff(filename, data, template_file, access_type="direct", nodata=None, scale=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    nband = 1 if data.ndim == 2 else data.shape[0]
    
    rasterio_env = {"session": _credential_manager.get_session()} if access_type == "direct" else {}
    
    with rio.Env(**rasterio_env):
        with rio.open(template_file) as ds:
            profile = ds.profile.copy()
            profile.update({
                'dtype': data.dtype,
                'count': nband,
                'height': data.shape[-2],
                'width': data.shape[-1],
                'compress': 'lzw',
                'nodata': nodata
            })
            
        with rio.open(filename, 'w', **profile) as dst:
            if nband == 1:
                dst.write(data, 1)
            else:
                for i in range(nband):
                    dst.write(data[i], i + 1)
            
            # Set metadata tags for scale factor
            if scale is not None:
                dst._set_all_scales([scale] * nband)


def get_stac_items(
    mgrs_tile: str, start_datetime: datetime, end_datetime: datetime
) -> list[Item]:
    logger.info("querying HLS archive")
    client = DuckdbClient(use_hive_partitioning=True)
    client.execute(
        """
        CREATE OR REPLACE SECRET secret (
             TYPE S3,
             PROVIDER CREDENTIAL_CHAIN
        );
        """
    )

    items = []
    for collection in HLS_COLLECTIONS:
        items.extend(
            client.search(
                href=HLS_STAC_GEOPARQUET_HREF.format(collection=collection),
                datetime="/".join(
                    dt.isoformat() for dt in [start_datetime, end_datetime]
                ),
                filter={
                    "op": "and",
                    "args": [
                        {
                            "op": "like",
                            "args": [{"property": "id"}, f"%.T{mgrs_tile}.%"],
                        },
                        {
                            "op": "between",
                            "args": [
                                {"property": "year"},
                                start_datetime.year,
                                end_datetime.year,
                            ],
                        },
                    ],
                },
            )
        )

    logger.info(f"found {len(items)} items")

    return [Item.from_dict(item) for item in items]


def fetch_single_asset(
    asset_href: str,
    fill_value=SR_FILL,
    direct_bucket_access: bool = False,
):
    """
    Fetch data from a single asset.
    """
    try:
        # Get session from credential manager if using direct bucket access
        rasterio_env = {}
        if direct_bucket_access:
            rasterio_env["session"] = _credential_manager.get_session()

        with rio.Env(**rasterio_env):
            # return rxr.open_rasterio(asset_href, lock=False, chunks=chunk_size, driver='GTiff').squeeze()
            with rio.open(asset_href) as src:
                return da.from_array(src.read(1), chunks=chunk_size)


    except Exception as e:
        logger.warning(f"Failed to read {asset_href}: {e}")
        return None # np.full((image_size[0], image_size[1]), fill_value)


def fetch_with_retry(asset_href: Path, max_retries: int = 3, delay: int = 3, fill_value=SR_FILL, access_type="external"):
    for attempt in range(max_retries):
        try:
            return fetch_single_asset(
                asset_href=asset_href,
                fill_value=fill_value,
                direct_bucket_access=(access_type == "direct"),
            )
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = delay #* (2**attempt)
                logger.warning(
                    f"Link {asset_href} attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"All {max_retries} attempts failed for {asset_href}. Last error: {e}"
                )
                return None # np.full((image_size[0], image_size[1]), fill_value)


def apply_fmask(data: np.ndarray, fmask: np.ndarray) -> np.ma.masked_array:
    return np.ma.masked_array(data, fmask)


def apply_union_of_masks(bands: list[np.ma.masked_array]) -> list[np.ma.masked_array]:
    mask = np.ma.nomask
    for band in bands:
        mask = np.ma.mask_or(mask, band.mask)

    for band in bands:
        band.mask = mask

    return bands


def get_meta(file_path: str):
    return rio.open(file_path).meta


def find_tile_bounds(tile: str):
    gdf = geopandas.read_file(r"s3://maap-ops-workspace/shared/zhouqiang06/AuxData/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp")
    # gdf = geopandas.read_file(r"/projects/my-public-bucket/AuxData/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp")
    bounds_list = [np.round(c, 3) for c in gdf[gdf["Name"]==tile].bounds.values[0]]
    return tuple(bounds_list)


def get_HLS_data(tile:str, bandnum:int, start_date:str, end_date:str, access_type="external"):
    print("Searching HLS STAC Geoparquet archive for HLS data...")
    # from rustac import DuckdbClient
    client = DuckdbClient(use_hive_partitioning=True)
    # configure duckdb to find S3 credentials
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
        results.extend(
            GetBandLists_HLS_STAC(response, tile, bandnum)
            )
    if access_type=="direct":
        results = [r.replace(URL_PREFIX, "s3://") for r in results]
    return results


def GetBandLists_HLS_STAC(response, tile:str, bandnum:int):
    BandList = []
    for i in range(len(response)):
        product_type = response[i]['id'].split('.')[1]
        if product_type=='L30':
            bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B05', 6:'B06', 7:'B07',8:'Fmask'})
        elif product_type=='S30':
            bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B8A', 6:'B11', 7:'B12',8:'Fmask'})
        else:
            print("HLS product type not recognized: Must be L30 or S30.")
            os._exit(1)
            
        try:
            getBand = response[i]['assets'][bands[bandnum]]['href']
            if filter_url(getBand, tile, bands[bandnum]):
                BandList.append(getBand)
        except Exception as e:
            print(e)
    return BandList


def filter_url(url: str, tile: str, band: str):
    if (os.path.basename(url).split('.')[2][1:]==tile) & (url.endswith(f"{band}.tif")):
        return True
    return False   


def get_tile_urls(tile:str, bandnum:int, start_date:str, end_date:str, access_type="external"):
    print("Searching EarthAccess for HLS data...")
    url_list = []
    try:
        results = earthaccess.search_data(short_name=f"HLSL30",
                                        cloud_hosted=True,
                                        temporal = (start_date, end_date), #"2022-07-17","2022-07-31"
                                        bounding_box = find_tile_bounds(tile), #bounding_box = (-51.96423,68.10554,-48.71969,70.70529)
                                        )
    except Exception as e:
        print(f"An error occurred searching HLSL30: {e}")
        results = []
    if len(results) > 0:
        bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B05', 6:'B06', 7:'B07',8:'Fmask'})
        for rec in results:
            for url in rec.data_links(access=access_type):
                if filter_url(url, tile, bands[bandnum]):
                    url_list.append(url)
    try:
        results = earthaccess.search_data(short_name=f"HLSS30",
                                        cloud_hosted=True,
                                        temporal = (start_date, end_date), #"2022-07-17","2022-07-31"
                                        bounding_box = find_tile_bounds(tile), #bounding_box = (-51.96423,68.10554,-48.71969,70.70529)
                                        )
    except Exception as e:
        print(f"An error occurred searching HLSS30: {e}")
        results = []
    if len(results) > 0:
        bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B8A', 6:'B11', 7:'B12',8:'Fmask'})
        for rec in results:
            for url in rec.data_links(access=access_type):
                if filter_url(url, tile, bands[bandnum]):
                    url_list.append(url)
    return url_list


def find_all_granules(tile: str, bandnum: int, start_date: str, end_date: str, search_source="STAC", access_type="external"):
    if search_source.lower() == "stac":
        url_list = get_HLS_data(tile=tile, bandnum=bandnum, start_date=start_date, end_date=end_date, access_type=access_type)
    elif search_source.lower() == "earthaccess":
        url_list = get_tile_urls(tile=tile, bandnum=bandnum, start_date=start_date, end_date=end_date, access_type=access_type) 
    else:
        print("search_source not recognized. Must be 'STAC' or 'earthaccess'.")
        # os._exit(1)
    if len(url_list) == 0:
        print("No granules found.")
        return pd.DataFrame()
    sat_list = [os.path.basename(g).split('.')[1] for g in url_list] # sat from HLS.L10.T18SUJ.2020010T155225.v2.0
    date_list = [datetime.strptime(os.path.basename(g).split('.')[3][:7], "%Y%j") for g in url_list] # date from HLS.L10.T18SUJ.2020010T155225.v2.0
    return pd.DataFrame({"Date": date_list, "Sat": sat_list, "granule_path": url_list})


def process_hls(tile, start_date, end_date, stat, save_dir, access_type="direct", scene_only=False, mask_water=False):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    granule_df = find_all_granules(tile=tile, bandnum=8, start_date=start_date, end_date=end_date, access_type=access_type)
    print(granule_df)
    if scene_only:
        out_dir = os.path.join(save_dir, tile)
    else:    
        out_dir = os.path.join(save_dir, tile, start_date[:4], f"HLS.M30.T{tile}.{start_dt.strftime('%Y%j')}.{end_dt.strftime('%Y%j')}.2.0")
    os.makedirs(out_dir, exist_ok=True)
    
    if len(granule_df) == 0: 
        logger.warning(f"No granules found for {tile}. Creating empty indicator file.")
        with open(os.path.join(out_dir, "No granules found"), 'w') as f:
            pass
        return
    
    valid_indices = []
    fmask_urls = [row.granule_path for row in granule_df.itertuples()]    
    for i, url in enumerate(fmask_urls):
        result = fetch_with_retry(url, access_type=access_type)
        if result is None:
            logger.warning(f"Skipping granule {i} — fetch returned None: {url}")
        elif result.shape == ():
            logger.warning(f"Skipping granule {i} — empty array shape: {url}")
        elif result.shape != (image_size[0], image_size[1]):
            logger.warning(f"Skipping granule {i} — unexpected shape {result.shape}: {url}")
        else:
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        logger.warning(f"No valid granules for {tile} after fetch validation.")
        with open(os.path.join(out_dir, "No valid granules"), 'w') as f:
            pass
        return
    
    granule_df = granule_df.iloc[valid_indices].reset_index(drop=True)
    logger.info(f"{len(granule_df)} valid granules after filtering")
    
    # Build band stacks using only valid granules
    band_stack_dict = {}
    # for band in common_bands:
    #     print(band)
    #     urls = [row.granule_path.replace("Fmask", (L8_name2index if row.Sat in ["L30", "L10"] else S2_name2index)[band]) 
    #             for row in granule_df.itertuples()]
    #     results = [fetch_with_retry(u, access_type=access_type) for u in urls]
        
    #     # Secondary filter in case any non-Fmask bands fail
    #     valid_results = [(r, i) for i, r in enumerate(results) if r is not None and r.shape != ()]
        
    #     if len(valid_results) == 0:
    #         logger.warning(f"No valid arrays for band {band}, skipping tile.")
    #         return
        
    #     if len(valid_results) < len(results):
    #         logger.warning(f"Band {band}: {len(results) - len(valid_results)} arrays dropped")
        
    #     band_stack_dict[band] = da.stack([r for r, _ in valid_results], axis=0)
    for band in common_bands:
        print(band)
        urls = [row.granule_path.replace("Fmask", (L8_name2index if row.Sat in ["L30", "L10"] else S2_name2index)[band]) 
                for row in granule_df.itertuples()]
        band_stack_dict[band] = da.stack([fetch_with_retry(u, access_type=access_type) for u in urls], axis=0)

    fmask_stack = band_stack_dict["Fmask"]
    
    # 1. Negative Values Check (Exclude in all cases)
    is_negative = (band_stack_dict["Red"] < 0) | (band_stack_dict["NIR_Narrow"] < 0) | \
                  (band_stack_dict["Blue"] < 0) | (band_stack_dict["Green"] < 0) | \
                  (band_stack_dict["SWIR1"] < 0) | (band_stack_dict["SWIR2"] < 0)

    # 2. Basic Quality Mask (Cloud, Shadow, No Data, or Negative)
    basic_mask = (((fmask_stack & (1 << QA_BIT['cloud'])) > 0) | 
                  ((fmask_stack & (1 << QA_BIT['adj_cloud'])) > 0) | 
                  ((fmask_stack & (1 << QA_BIT['cloud shadow'])) > 0) |
                  (fmask_stack == QA_FILL) |
                  is_negative)

    # 3. Aerosol Identification
    # High Aerosol: both bits 6 and 7 are set
    is_high_aerosol = ((fmask_stack & (1 << QA_BIT['aerosol_h'])) > 0) & ((fmask_stack & (1 << QA_BIT['aerosol_l'])) > 0)
    
    # Low/Moderate Aerosol: Neither cloud/shadow/negative AND not high aerosol
    is_low_mod_aerosol = ~is_high_aerosol & ~basic_mask
    
    # Check if ANY low/moderate aerosol observations exist for each pixel in the stack
    any_low_mod_available = da.any(is_low_mod_aerosol, axis=0)

    # 4. Optional water/ice mask
    if mask_water:
        water_snowice_mask = (((fmask_stack & (1 << QA_BIT['water'])) > 0) | 
                              ((fmask_stack & (1 << QA_BIT['snowice'])) > 0))
        basic_mask = basic_mask | water_snowice_mask
        
    # 4. Final Composite Masking Logic:
    # Always exclude basic_mask (clouds, shadows, negatives).
    # Exclude high aerosol observations IF there is at least one low/mod observation available.
    # The logic ensures that high aerosol pixels are treated as "bad" (masked out) as long as there is at least one "good" pixel (clear and low/mod aerosol) available in the temporal stack. 
    # If no low/mod aerosol pixels exist for that coordinate, the high aerosol pixel is allowed through to prevent data holes.
    bad_pixel_mask = basic_mask | (is_high_aerosol & any_low_mod_available)
    
    all_nan_mask = da.all(bad_pixel_mask, axis=0)

    # Calculate EVI2 for selection
    red = band_stack_dict["Red"].astype(np.float32) * sr_scale
    nir = band_stack_dict["NIR_Narrow"].astype(np.float32) * sr_scale
    evi2_stack = 2.5 * (nir - red) / (nir + 2.4 * red + 1)

    if scene_only:
        # process and save EVI2
        for i, row in enumerate(granule_df.itertuples()):
            scene_date = datetime.strptime(os.path.basename(row.granule_path).split('.')[3][:7], "%Y%j")
            scene_id = os.path.basename(row.granule_path).replace("Fmask", "").rstrip(".")
    
            # outdir: TILEID/YEAR/<.tifs>
            scene_dir = os.path.join(out_dir, scene_date.strftime('%Y'))
            os.makedirs(scene_dir, exist_ok=True)
    
            # Get this scene's mask and EVI2
            scene_mask = bad_pixel_mask[i]
            scene_evi2 = da.where(scene_mask, np.nan, evi2_stack[i])
            evi2_out = scene_evi2.compute().astype(np.float32)

            # Save
            out_path = os.path.join(scene_dir, f"{scene_id}.EVI2.tif")
            saveGeoTiff(out_path, evi2_out, row.granule_path, nodata=np.nan)
            logger.info(f"Saved: {out_path}")

    else:    
        def _get_best_index(evi, mask, all_nan, stat_type):
            evi_f = evi.copy()
            evi_f[mask] = np.nan
            with np.errstate(all='ignore'):
                if stat_type == 'max': 
                    target = np.nanmax(evi_f, axis=0)
                elif stat_type == 'median': 
                    target = np.nanmedian(evi_f, axis=0) 
                else: 
                    return
                
                diff = np.abs(evi_f - target)
                diff[np.isnan(diff)] = 1e9
                
                idx = np.argmin(diff, axis=0)
                idx[all_nan] = 0
                return idx.astype(np.int16)
    
        best_idx = da.map_blocks(_get_best_index, evi2_stack, bad_pixel_mask, all_nan_mask, 
                                 stat_type=stat, drop_axis=0, dtype=np.int16)
    
        template_path = granule_df.iloc[0]["granule_path"]
    
        for band in common_bands:
            logger.info(f"Processing: {band}")
            current_stack = band_stack_dict[band]
            comp_result = da.where(all_nan_mask, (QA_FILL if band=="Fmask" else SR_FILL), da.choose(best_idx, current_stack))
    
            if band != "Fmask":
                def _safe_std(data, mask, all_nan):
                    data_f = data.astype(np.float32)
                    data_f[mask] = np.nan
                    with np.errstate(all='ignore'):
                        res = np.nanstd(data_f, axis=0)
                        res[all_nan] = 0
                        return res
                
                std_lazy = da.map_blocks(_safe_std, current_stack, bad_pixel_mask, all_nan_mask, drop_axis=0, dtype=np.float32)
                comp_out, std_out = da.compute(comp_result, std_lazy)
                
                saveGeoTiff(os.path.join(out_dir, f"{os.path.basename(out_dir)}.{band}.tif"), 
                            comp_out.astype(np.int16), template_path, nodata=SR_FILL, scale=sr_scale)
                saveGeoTiff(os.path.join(out_dir, f"{os.path.basename(out_dir)}.{band}.std.tif"), 
                            std_out.round().astype(np.int16), template_path, nodata=SR_FILL, scale=sr_scale)
            else:
                saveGeoTiff(os.path.join(out_dir, f"{os.path.basename(out_dir)}.{band}.tif"), 
                            comp_result.compute().astype(np.uint8), template_path, nodata=QA_FILL)
    
        # Valid Count and DOY
        valid_count = da.sum(~bad_pixel_mask, axis=0).astype(np.uint8).compute()
        saveGeoTiff(os.path.join(out_dir, f"{os.path.basename(out_dir)}.ValidCount.tif"), valid_count, template_path)
        
        doy_vals = np.array([int(datetime.strptime(os.path.basename(p).split('.')[3][:7], "%Y%j").strftime("%j")) for p in granule_df.granule_path])
        doy_stack = da.from_array(doy_vals[:, None, None], chunks=(len(doy_vals), 1830, 1830))
        best_doy = da.choose(best_idx, doy_stack)
        rel_doy = da.where(all_nan_mask, 0, (best_doy - int(start_dt.strftime("%j")) + 1)).astype(np.uint8).compute()
        saveGeoTiff(os.path.join(out_dir, f"{os.path.basename(out_dir)}.DOY.tif"), rel_doy, template_path)


def run(**kwargs):
    for i in range(3):
        try:
            process_hls(**kwargs)
            return
        except Exception as e:
            logger.warning(f"Attempt {i+1} failed: {e}")
            time.sleep(5)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--stat", default="max")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--access_type", default="direct")
    parser.add_argument("--scene_only", default=False)
    parser.add_argument("--mask_water", default=False)
    args = parser.parse_args()

    os.environ.update(GDAL_CONFIG)
    run(tile=args.tile, start_date=args.start_date, end_date=args.end_date, stat=args.stat, 
        save_dir=args.output_dir, access_type=args.access_type, scene_only=args.scene_only,
       mask_water=args.mask_water)