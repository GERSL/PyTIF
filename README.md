# PyTIF

PyTIF is a Python implementation of time-series-based image fusion (TIF) for
Harmonized Landsat and Sentinel-2 (HLS) observations. It pairs Landsat (L30)
and Sentinel-2 (S10) surface-reflectance observations by date, clusters the
matched observations, and fits robust per-band regression models. Processing
is organized into spatial chips.

## Requirements

Use Python with the following packages installed:

- Click
- NumPy
- pandas
- Rasterio
- scikit-learn

For example:

```bash
python -m pip install click numpy pandas rasterio scikit-learn
```

Rasterio depends on GDAL; use an installation compatible with your operating
system and Python environment.

## Quick check with included data

The repository includes a single-pixel example dataset in `testdata/`. Run the
debug workflow from the project directory:

```bash
python coeff_generate.py --debug true --output ./output
```

This writes `output/tif_debug_results_T18TYM.csv`. Debug mode uses the included
data and does not search for HLS imagery. Its band order is fixed to
`Blue,Green,Red,NIR,SWIR1,SWIR2`.

To see all command-line options:

```bash
python coeff_generate.py --help
```

## Process HLS imagery

Provide separate roots for L30 and S10 HLS scenes. The code searches the root
and one directory level below it for scene directories named like
`HLS.L30.*.v*` or `HLS.S10.*.v*`. Each scene directory must contain the
band and `Fmask` GeoTIFFs named for that scene, with reflectance
`scale_factor` and `add_offset` tags.

Example:

```bash
python coeff_generate.py \
  --tile T18TYM \
  --l30 /data/HLS/L30/T18TYM \
  --s10 /data/HLS/S10/T18TYM \
  --output ./output \
  --imagesize 10980 \
  --chipsize 60 \
  --ji 1 \
  --jn 1 \
  --cp 1
```

`--imagesize` and `--chipsize` are pixel counts on the 10 m grid. The chip size
must be a multiple of 60 and divide the image size evenly. `--ji` is a
one-based job index, and `--jn` is the total number of jobs; together they
divide the chip workload. `--cp` controls concurrent scene reads.

By default, the script processes `Blue,Green,Red,NIR,SWIR1,SWIR2`. You can
provide a comma-separated band list with `--bands`; supported names are
`Coastal`, `Blue`, `Green`, `Red`, `NIR`, `SWIR1`, and `SWIR2`.

The standard workflow writes a CSV table of per-pixel, per-band model
coefficients and quality fields for each processed chip. Chip output filenames
currently use the `.cvs` extension.

## Reference

Song, K., Zhu, Z., Qiu, S., Olofsson, P., Neigh, C. S., Ju, J., & Zhou, Q.
(2025). TIF: A time-series-based image fusion algorithm. *Remote Sensing of
Environment, 331*, 115035.
[https://doi.org/10.1016/j.rse.2025.115035](https://doi.org/10.1016/j.rse.2025.115035)
