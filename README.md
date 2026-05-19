# HLS Phenometric Development Algorithm
This repository contains the code associated with generating HLS-derived phenological metrics on the MAAP Platform. It is registerd as ``hls-phenometrics:main`` and is set up to run from two user provided parameters:
- MGRS tile ID: "18SUJ"
- Target year: "2023"

For the tile and target year parameters the algorithm will:
1) Download HLS spectral data from NASA EarthData including +/- 12 months around the target year, referred to as context months.
2) Calculate and save EVI2 tifs for every valid scene in the 36 month time window.
3) Use EVI2 scenes to calculate pixel-wise phenological metrics following a similar approach used by Bolton et al., 2020 with modifications to python and our implementation.

## Phenological 