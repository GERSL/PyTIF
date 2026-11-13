"""Create 10 m S10-like images by applying TIF coefficients to L30 scenes."""

import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.windows import Window
import time

from hls import HSL_BANDS, find_hls_images, read_image
from utils import generate_tile_chips


NODATA = -9999.0

def load_coefficient_chip(dir_coeff, chip_name, missing_ok=True):
    """
    Load the coefficient table for a specific chip in a tile. Each table is expected to be a CSV file with columns:
    - row: the row index of the chip in the tile
    - col: the column index of the chip in the tile
    - cluster: the cluster index for the regression model
    - band: the band name (e.g., "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2")
    - centroid_l30: the centroid value of the L30 reflectance for the cluster
    - intercept: the intercept of the regression model for the cluster
    - slope: the slope of the regression model for the cluster
    - QA: the quality assurance flag for the regression model
    
    Args:
        dir_coeff (str or Path): The directory containing the coefficient files.
        chip_name (str): The name of the chip to load coefficients for.
        missing_ok (bool): If True, missing coefficient files will be ignored; if False, a FileNotFoundError will be raised for any missing files.
        
    Returns:
        pd.DataFrame: A DataFrame containing the coefficients for the specified chip.
    """
    dir_chips = Path(dir_coeff)
    chip_filename = f"{chip_name}.parquet"
    chip_filepath = dir_chips.joinpath(chip_filename)
    
    if not chip_filepath.is_file():
        if missing_ok:
            return pd.DataFrame()  # Return an empty DataFrame if missing is allowed
        else:
            raise FileNotFoundError(f"Missing expected coefficient file: {chip_filepath}")
    
    coeff_table_chip = pd.read_parquet(chip_filepath)
    
    return coeff_table_chip

def read_reflectance(path, window):
    """Read HLS scaled reflectance and its valid-data mask."""
    with rasterio.open(path) as src:
        pixels = src.read(1, window=window, masked=True)
        scale = float(src.tags().get("scale_factor", 1.0))
        offset = float(src.tags().get("add_offset", 0.0))
    return (np.ma.getdata(pixels).astype(np.float32) * scale + offset,
            ~np.ma.getmaskarray(pixels))


def fuse_chip(table, values, valid, bands, start_row, start_col):
    """Select the nearest multi-band L30 centroid and evaluate its model."""
    height, width = values.shape[1:]
    output = np.full((len(bands), height, width), NODATA, dtype=np.float32)
    # QA 3 is the converged robust fit emitted by coeff_generate.py.
    table = table[(table["QA"] == 3) & np.isfinite(table["cluster"])].copy()
    table["band"] = table["band"].str.lower()
    wanted = [band.lower() for band in bands]
    table = table[table["band"].isin(wanted)]
    if table.empty:
        return output

    # Use the filename's chip origin, rather than the first valid model row:
    # a leading pixel can legitimately have QA=0 and be absent after filtering.
    table["r"] = table["row"].astype(int) - start_row
    table["c"] = table["col"].astype(int) - start_col
    table = table[table.r.between(0, height - 1) & table.c.between(0, width - 1)]
    clusters = np.sort(table.cluster.unique())
    n_pixels, n_clusters, n_bands = height * width, len(clusters), len(bands)
    centroid = np.full((n_pixels, n_clusters, n_bands), np.nan, dtype=np.float32)
    intercept = centroid.copy()
    slope = centroid.copy()
    cluster_ids = {cluster: index for index, cluster in enumerate(clusters)}
    band_ids = {band: index for index, band in enumerate(wanted)}
    for item in table[["r", "c", "cluster", "band", "centroid_l30", "intercept", "slope"]].itertuples(index=False):
        r, c, cluster, band, center, offset, gain = item
        pixel, k, b = int(r) * width + int(c), cluster_ids[cluster], band_ids[band]
        centroid[pixel, k, b], intercept[pixel, k, b], slope[pixel, k, b] = center, offset, gain

    observed = np.moveaxis(values, 0, -1).reshape(n_pixels, n_bands)
    finite_model = np.isfinite(centroid).all(2) & np.isfinite(intercept).all(2) & np.isfinite(slope).all(2)
    distance = np.sum((centroid - observed[:, None]) ** 2, axis=2)
    distance[~finite_model] = np.inf
    chosen = np.argmin(distance, axis=1)
    rows = np.arange(n_pixels)
    prediction = intercept[rows, chosen] + slope[rows, chosen] * observed
    usable = valid.all(0).reshape(n_pixels) & np.isfinite(distance[rows, chosen])
    prediction[~usable] = NODATA
    return np.moveaxis(prediction.reshape(height, width, n_bands), -1, 0).astype(np.float32)

def fuse_image(tile, dir_image, dir_coeff, bands, imagesize, chipsize, output):
    """Fuse L30 bands with coefficients to create S10-like 10m resolution images.
    
    Args:
        dir_image: Directory containing the L30 HLS scene.
        dir_coeff: Directory containing the coefficient files.
        bands: List of band names to fuse.
        imagesize: Full image size in pixels (e.g., 10980 for 10980x10980).
        chipsize: Chip size for processing in pixels (e.g., 180).
        output: Output directory to save the fused image.
    """
    # Read L30 bands at once
    image_filename = Path(dir_image).name
    image_data, image_qa, profile, transform, tags = read_image(dir_image, 'L30', bands, resolution=10, qa_included=True)
    
    image_chips  = generate_tile_chips(imagesize, chipsize)

    # Process each chip
    for chip_name, _,_,_,_ in image_chips:
        # Load the coeff table for this chip
        chip_coeff_table = load_coefficient_chip(dir_coeff, f"{tile}_{chip_name}", missing_ok=True)
        if len(chip_coeff_table) == 0:
            continue
        # band by band
        for ib, band in enumerate(bands):
            # Get the coefficients for this band
            mask = chip_coeff_table["band"].str.lower() == band.lower()
            chip_coeff = chip_coeff_table[mask]
            
            if chip_coeff.empty:
                continue
                
            # Safely update using .loc to avoid SettingWithCopyWarning
            chip_coeff_table.loc[chip_coeff.index, "l30"] = image_data[
                ib, chip_coeff["row"].values, chip_coeff["col"].values
            ]
        # select cluster for same row and col, by finding the nearest centroid, l30 to centroid_l30 in distance in bands
        # 1. Calculate squared difference for each band
        chip_coeff_table["diff_sq"] = (
            chip_coeff_table["l30"] - chip_coeff_table["centroid_l30"]
        ) ** 2

        # 2. Sum across bands per (row, col, cluster) and take the square root for Euclidean distance
        cluster_distances = (
            chip_coeff_table.groupby(["row", "col", "cluster"])["diff_sq"]
            .sum()
            .apply(np.sqrt)
            .reset_index(name="euclidean_dist")
        )

        # 3. Find the cluster with the minimum Euclidean distance for each (row, col) location
        idx_min = cluster_distances.groupby(["row", "col"])["euclidean_dist"].idxmin()
        best_clusters = cluster_distances.loc[idx_min, ["row", "col", "cluster", "euclidean_dist"]]

        # 4. Filter chip_coeff_table to keep only the winning cluster for each location
        chip_coeff_table = chip_coeff_table.merge(
            best_clusters, on=["row", "col", "cluster"], how="inner"
        )
        
        # replace the image_data with the selected cluster's model for this chip, band by band
        for ib, band in enumerate(bands):
            mask = chip_coeff_table["band"].str.lower() == band.lower()
            chip_coeff = chip_coeff_table[mask]
            if chip_coeff.empty:
                continue
            # Updating the image data array
            image_data[
                ib, chip_coeff["row"].values, chip_coeff["col"].values
            ] = (
                chip_coeff["intercept"].values
                + chip_coeff["slope"].values * image_data[
                    ib, chip_coeff["row"].values, chip_coeff["col"].values
                ]
            )

    # save the fused image to output directory, according to the bands
    image_filename_l10 = image_filename.replace("HLS.L30", "HLS.S10")
    Path(output).joinpath(image_filename_l10+".part").mkdir(parents=True, exist_ok=True)
    for ib, band in enumerate(bands):
        # Get the original band file name from the L30 image directory
        band_postfix = HSL_BANDS[band][0]  # Assuming L30 band code is the first element
        band_output_path = Path(output).joinpath(image_filename_l10+".part").joinpath(f"{image_filename_l10}.{band_postfix}.tif")
        band_output_path.parent.mkdir(parents=True, exist_ok=True)
        band_output_path_part = band_output_path.with_suffix(".part")
        with rasterio.open(
            band_output_path_part,
            "w",
            driver="GTiff",
            height=image_data.shape[1],
            width=image_data.shape[2],
            count=1,
            dtype=np.int16,
            crs=profile["crs"],
            transform=transform,
            nodata=NODATA,
        ) as dst:
            # Update metadata tags safely
            dst.update_tags(**tags)
            
            # Convert floating-point image_data back to scaled integer format
            # Inverse of: loaded = float * scale_factor + add_offset
            scaled_data = (image_data[ib] - float(tags["add_offset"])) / float(tags["scale_factor"])
            
            # Clip/handle nodata and cast to int16
            output_band_data = np.where(
                np.isnan(image_data[ib]), 
                NODATA, 
                np.round(scaled_data)
            ).astype(np.int16)
            
            dst.write(output_band_data, 1)
        os.rename(band_output_path_part, band_output_path)
    os.rename(Path(output).joinpath(image_filename_l10+".part"), Path(output).joinpath(image_filename_l10))
        
@click.command()
@click.option("--tile", default="T18TYM", help="Tile name to filter L30 scenes.")
@click.option('--bands', default="Blue,Green,Red,NIR,SWIR1,SWIR2", help='Comma-separated list of bands to process')
@click.option('--imagesize', default=10980, help='Size of the images to process, e.g., 10980 for 10980 by 10980 pixels in 10 meters')
@click.option('--chipsize', default=180, help='Chip size for processing the images per core, e.g., 180 for 180 by 180 pixels in 10 meters per core')
@click.option('--l30', default="/gpfs/sharedfs1/zhulab/Shi/ProjectHLS10/TIF/Data/L30/T18TYM", help='Directory containing L30 images')
@click.option("--depth", default=2, show_default=True, help="L30 scene search depth.")
@click.option('--coeff', default="/gpfs/sharedfs1/zhulab/Shi/ProjectHLS10/TIF/Output/TIFCoeff/T18TYM", help='Coefficient output directory')
@click.option("--output", default="/gpfs/sharedfs1/zhulab/Shi/ProjectHLS10/TIF/Output/TIFImages/T18TYM", help='Image output directory')
def main(tile, bands, imagesize, chipsize, l30, depth, coeff, output):
    """Generate a 10 m S10-like image for every L30 image."""
    
    timestart = time.time()
    print(f">>> Start processing {tile}")
    
    bands = [band.strip() for band in bands.split(",") if band.strip()]

    images, _ = find_hls_images(l30, "L30", depth=depth)
    print(f">>> Starting processing {len(images)} L30 images")
    Path(output).mkdir(parents=True, exist_ok=True)
    for image in images:
        # dir_image, dir_coeff, bands, imagesize, chipsize, output
        fuse_image(tile, image, coeff, bands, imagesize, chipsize, output)


if __name__ == "__main__":
    main()
