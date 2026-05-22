#!/usr/bin/env python3

import argparse
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

COLLECTION_SHORT_NAME = "MSLSP30NA"
VERSION = "011"


def search_granules(tile_id):
    page = 1
    all_urls = []
    while True:
        params = {
            "short_name": "MSLSP30NA",
            "version": "011",
            "page_size": 2000,
            "page_num": page,
        }
        
        r = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json", params=params)
        r.raise_for_status()
        entries = r.json()["feed"]["entry"]

        if len(entries) == 0:
            break

        for entry in entries:
            granule_id = entry["producer_granule_id"]            
            if tile_id not in granule_id:
                continue

            for link in entry.get("links", []):
                href = link.get("href", "")
                if href.endswith(".nc"):
                    all_urls.append(href)

        page += 1

    return sorted(all_urls)


def download_file(url, outdir):
    fname = url.split("/")[-1]
    outfile = outdir / fname

    if outfile.exists():
        return

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(outfile, "wb") as f, tqdm(
            total=total,
            desc=fname,
            unit="B",
            unit_scale=True,
            leave=False,
        ) as pbar:

            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tiles",
        nargs="+",
        required=True,
        help="MGRS tile IDs"
    )

    parser.add_argument(
        "--outdir",
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    print(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_downloads = []

    for tile in args.tiles:
        print(f"\nSearching tile {tile}")
        urls = search_granules(tile)
        
        print(f"Found {len(urls)} granules")
        tile_dir = outdir / tile
        tile_dir.mkdir(exist_ok=True)

        for url in urls:
            all_downloads.append((url, tile_dir))

    print(f"\nTotal files: {len(all_downloads)}")
    with ThreadPoolExecutor(max_workers=args.workers) as exe:

        futures = [
            exe.submit(download_file, url, tile_dir)
            for url, tile_dir in all_downloads
        ]

        for f in futures:
            f.result()


if __name__ == "__main__":
    main()