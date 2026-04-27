from pathlib import Path
import os
import re
import logging
import argparse
import rasterio as rio
import rioxarray as rxr
import dask
import dask.array as da
from dask.distributed import LocalCluster, Client
import xarray as xr
import numpy as np

import gc

common_bands = ["Blue","Green","Red","NIR_Narrow","SWIR1", "SWIR2", "Fmask"]

L8_bandnames = {#"B01":"Coastal_Aerosol", 
               "B02":"Blue", 
               "B03":"Green", 
               "B04":"Red", 
               "B05":"NIR_Narrow", 
               "B06":"SWIR1", 
               "B07":"SWIR2", 
               #"B09":"Cirrus",
              "Fmask":"Fmask"}

S2_bandnames = {#"B01":"Coastal_Aerosol", 
               "B02":"Blue", 
               "B03":"Green", 
               "B04":"Red", 
               "B8A":"NIR_Narrow", 
               "B11":"SWIR1", 
               "B12":"SWIR2", 
               #"B10":"Cirrus",
              "Fmask":"Fmask"}
QA_BIT = {'cirrus': 0,
          'cloud': 1,
          'adj_cloud': 2,
          'cloud shadow':3,
          'snowice':4,
          'water':5,
          'aerosol_l': 6,
          'aerosol_h': 7
         }
QA_FILL = 255
SR_FILL = -9999
sr_scale = 0.0001
chunks=dict(band=1, x=512, y=512)


def saveGeoTiff(filename, data, template_file, access_type="direct", nodata=None, scale=None):
    
    nband = 1 if data.ndim == 2 else data.shape[0]
    
    rasterio_env = {"session": _credential_manager.get_session()} if access_type == "direct" else {}
    
    with rio.Env(**rasterio_env):
        with rio.open(template_file) as ds:
            profile = ds.profile.copy()
            profile.update({
                'driver': 'GTiff',
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


def gen_path_prefix_output(filepath, outdir, tile):
    # prefix/path   
    year = filepath.parts[2]
    granule_name = filepath.parts[7]
    new_granule = granule_name.replace("L30", "M30").replace("S30", "M30")
    new_granule = re.sub(r'T\d{6}', '', new_granule)    
    new_prefix = Path(outdir / f'{tile}/{year}/{new_granule}/')

    output_dir = Path(outdir, tile, year)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return new_prefix


def gen_base_filename(filename, custom_band_name = None, base_only = False):
    fileparts = filename.split('.')
    new_date_str = re.sub(r'T\d{6}', '', fileparts[3]) 

    if not base_only:
        if custom_band_name:
            band_name = custom_band_name
        else:
            if fileparts[1] == "L30":# and fileparts[-2] in ["B04", "B05", "Fmask"]:
                band_name = L8_bandnames[fileparts[-2]]
            elif fileparts[1] == "S30":# and fileparts[-2] in ["B04", "B8A", "Fmask"]:
                band_name = S2_bandnames[fileparts[-2]]
            
            if band_name not in common_bands: 
                band_name = "NOTPROCESSING"        
        # for now preserve L30 and S30 because I am not mosaicing images yet
        # new_filename = Path(f'{parts[0]}.M30.{parts[2]}.{new_date_str}.2.0.{band_name}.tif.')
        new_filename = Path(f'{fileparts[0]}.{fileparts[1]}.{fileparts[2]}.{new_date_str}.2.0.{band_name}.tif')
    else:
        new_filename = Path(f'{fileparts[0]}.{fileparts[1]}.{fileparts[2]}.{new_date_str}.2.0.')
    
    return new_filename


def safe_read_raster(file_path):
    """Safely read a raster file with error handling"""
    try:
        # Attempt to read the file
        data = rxr.open_rasterio(file_path, chunks=chunks, masked_and_scale=True).squeeze('band', drop=True)
        return data, None  # Return data and no error
        
    except Exception as e:
        # Handle various rasterio errors
        error_msg = f"Failed to read {file_path}: {str(e)}"
        #print(f"WARNING: {error_msg}")
        return None, error_msg    


@dask.delayed
def process_granule(dirpath, tif_files, tile, outdir):
    """Process a single granule directory and write EVI2 tif"""
    try:
        print(f"Target granule: {dirpath}", flush=True)

        # Subset by Landsat or Sentinel
        sensor = tif_files[0].split('.')[1]
        if sensor in ["L30", "L10"]:
            target_bands = L8_bandnames
        elif sensor in ["S30", "S20"]:
            target_bands = S2_bandnames
        else:
            print(f"Unknown sensor {sensor}, skipping", flush=True)
            return None

        files_by_band = {}
        for f in tif_files:
            band = f.split('.')[-2]
            if band in target_bands:
                files_by_band[band] = Path(dirpath) / f

        output_path = gen_path_prefix_output(next(iter(files_by_band.values())), outdir, tile)
        out_file_base = gen_base_filename(next(iter(files_by_band.values())).name, base_only=True)

        print("OUTPUT_PATH:", output_path, flush=True)
        print("out_file_base:", out_file_base, flush=True)

        if not output_path.with_name(f'{out_file_base}EVI2.tif').exists():
            print(f"Target granule: {dirpath}", flush=True)
            if "Fmask" in files_by_band and len(files_by_band) > 1:
                band_stack_dict = {}
                ds = None
                error = None
                for band_code, file_path in files_by_band.items():
                    band_name = target_bands.get(band_code)
                    ds, error = safe_read_raster(file_path)
                    band_stack_dict[band_name] = ds

                    if error:
                        break

                print(f"Loaded bands: {list(band_stack_dict.keys())}", flush=True)

            print(f"Successfully loaded: {ds.shape}", flush=True)

            # FOLLOW QIANG's composite QA/QC
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

            # 3. Water/snow-ice mask
            water_snowice_mask = (((fmask_stack & (1 << QA_BIT['water'])) > 0) | 
                                  ((fmask_stack & (1 << QA_BIT['snowice'])) > 0))

            bad_pixel_mask = basic_mask | water_snowice_mask

            template_path = list(files_by_band.values())[0]

            # EVI2 calculation
            red_band = xr.where(bad_pixel_mask, SR_FILL, band_stack_dict["Red"].drop_vars("spatial_ref"))
            nir_band = xr.where(bad_pixel_mask, SR_FILL, band_stack_dict["NIR_Narrow"].drop_vars("spatial_ref"))

            na_mask = (red_band < 0) | (nir_band < 0)
            red_band = red_band.where(~na_mask) * sr_scale
            nir_band = nir_band.where(~na_mask) * sr_scale

            pos_mask = (red_band > 1) | (nir_band > 1)
            red_band = red_band.where(~pos_mask)
            nir_band = nir_band.where(~pos_mask)

            evi2 = 2.5 * (nir_band - red_band) / (nir_band + 2.4 * red_band + 1)
            
            out_file_evi2 = output_path.with_name(f'{out_file_base}EVI2.tif')
            saveGeoTiff(out_file_evi2,
                        evi2.compute().values,
                        template_path,
                        access_type="local",
                        nodata=SR_FILL,
                        scale=None)
            print(f"Saved: {out_file_evi2}", flush=True)
            gc.collect()
            return str(out_file_evi2)

    except Exception as e:
        print(f"ERROR processing {dirpath}: {e}", flush=True)
        return None


def run_calculation(tile, indir, outdir, n_workers=1):
    print(f"Calculating EVI2 with n_workers={n_workers}", flush=True)

    # Collect all granule directories
    delayed_tasks = []
    for dirpath, dirnames, filenames in os.walk(indir):
        tif_files = [f for f in filenames if f.endswith(".tif")]
        if not tif_files:
            continue
        delayed_tasks.append(process_granule(dirpath, tif_files, tile, outdir))

    print(f"Found {len(delayed_tasks)} granules to process", flush=True)

    # Run with local dask cluster
    with LocalCluster(n_workers=n_workers, threads_per_worker=1) as cluster, Client(cluster) as client:
        print(f"Dask dashboard: {client.dashboard_link}", flush=True)
        results = dask.compute(*delayed_tasks)

    completed = [r for r in results if r is not None]
    print(f"Completed: {len(completed)}/{len(delayed_tasks)} granules", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile",      required=True)
    parser.add_argument("--indir",     required=True)
    parser.add_argument("--outdir",    required=True)
    parser.add_argument("--n_workers", default=1, type=int)
    args = parser.parse_args()

    run_calculation(
        tile=args.tile,
        indir=Path(args.indir),
        outdir=Path(args.outdir),
        n_workers=args.n_workers
    )