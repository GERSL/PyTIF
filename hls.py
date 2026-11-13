"""

"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from datetime import datetime
import glob
import os

# Paried bands for L30 and S10, with band names saved
HSL_BANDS = {
    "Coastal": ("B01", "B01"),
    "Blue": ("B02", "B02"),
    "Green": ("B03", "B03"),
    "Red": ("B04", "B04"),
    "NIR": ("B05", "B08"), # NIR Narrow
    "SWIR1": ("B06", "B11"),
    "SWIR2": ("B07", "B12"),
}

def _image_band_file(scene, band):
    scene = Path(scene)
    return scene / f"{scene.name}.{band}.tif"


def _hls_fmask_qa(src, window):
    """Decode HLS Fmask QA into clear, water, shadow, snow, cloud, or no-data classes.

    When class flags
    overlap, cloud takes precedence over snow, shadow, and water.
    """

    fmask = src.read(1, window=window, masked=True)
    flags = np.ma.getdata(fmask)
    classes = np.zeros(flags.shape, dtype=np.uint8)
    classes[(flags & (1 << 5)) != 0] = 1  # Water
    classes[(flags & (1 << 3)) != 0] = 2  # Cloud shadow
    classes[(flags & (1 << 4)) != 0] = 3  # Snow/ice
    classes[(flags & (1 << 1)) != 0] = 4  # Cloud
    classes[(flags & (1 << 2)) != 0] = 4 # adjacent cloud and shadow
    no_data = np.ma.getmaskarray(fmask) | (flags == 255)
    classes[no_data] = 255
    return classes


def _read_sensor(images, sensor, bands, window, cp=1):
    shape = (len(images), len(bands) + 1, int(window.height), int(window.width)) # +1 for cloud mask
    cube = np.zeros(shape, dtype=np.float32)
    def read_scene(indexed_image):
        time_index, img = indexed_image
        with rasterio.open(_image_band_file(img, "Fmask")) as src:
            fmask = _hls_fmask_qa(src, window)

        cube[time_index, -1] = fmask.astype(np.float32)
        for band_index, band in enumerate(bands):
            with rasterio.open(_image_band_file(img, band)) as src:
                tags = src.tags()
                pixels = src.read(1, window=window, masked=True)
                cube[time_index, band_index] = (
                    np.ma.getdata(pixels).astype(np.float32) * float(tags["scale_factor"]) + float(tags["add_offset"])
                )
    if cp < 2 or len(images) < 2:
        for indexed_image in enumerate(images):
            read_scene(indexed_image)
    else:
        with ThreadPoolExecutor(max_workers=min(cp, len(images))) as executor:
            for _ in executor.map(read_scene, enumerate(images)):
                pass
    return cube

def read_image(dir_image, sensor, bands, resolution=10, qa_included=True):
    """Read HLS image data from a directory.
    
    Args:
        dir_image: Directory containing the HLS scene files.
        sensor: Sensor type ("L30" for Landsat-30 or "S10" for Sentinel-10).
        bands: List of band names (e.g., ["Red", "NIR", "Blue"]).
        resolution: Target resolution in meters (10 or 30). Default is 10.
        qa_included: Whether to include the Fmask QA band. Default is True.
    
    Returns:
        numpy array of shape (num_bands [+ 1 if qa_included], height, width) with float32 data.
    """
    # Map band names to file band codes for the sensor
    band_files = [HSL_BANDS[band][0 if sensor == "L30" else 1] for band in bands]
    
    # Determine the image size from the first band file
    first_band_file = _image_band_file(dir_image, band_files[0])
    with rasterio.open(first_band_file) as src:
        height = src.height
        width = src.width
        tags = src.tags()
        scale_factor = float(tags["scale_factor"])
        add_offset = float(tags["add_offset"])
    
    # Adjust dimensions for resolution mismatch
    # L30 bands are 30m native, S10 bands are 10m native
    native_resolution = 30 if sensor == "L30" else 10
    if resolution != native_resolution:
        scale = native_resolution // resolution
        height *= scale
        width *= scale
    
    # Allocate output array
    image_data = np.zeros((len(bands), height, width), dtype=np.float32)
    
    # Load the image data from each band file
    for band_idx, (band_name, band_file) in enumerate(zip(bands, band_files)):
        with rasterio.open(_image_band_file(dir_image, band_file)) as src:
            pixels = src.read(1, masked=True, out_shape=(height, width))
            image_data[band_idx] = np.ma.getdata(pixels).astype(np.float32) * scale_factor + add_offset
    
    # Load QA/Fmask if requested
    if qa_included:
        with rasterio.open(_image_band_file(dir_image, "Fmask")) as src:
            image_qa = src.read(1, masked=True, out_shape=(height, width))
    else:
        image_qa = []
    
    return image_data, image_qa


def find_hls_images(root, sensor, depth=2):
    """Find image directories for a sensor and return them sorted.

    Args:
        root: Root directory to search.
        sensor: Sensor type to match in directory names.
        depth: Number of directory levels to search, starting at root. The default is 2, which means it will search in root (e.g., all images in root), root/* (e.g., all images in <year> of the root).
    """
    images = set()
    for level in range(depth):
        pattern = os.path.join(
            root, *(["*"] * level), f"HLS.{sensor}.*.v*"
        )
        images.update(
            path for path in glob.glob(pattern) if os.path.isdir(path)
        )

    # Get image list acquisition dates
    # Extract raw timestamp strings
    image_dates = [os.path.basename(img).split(".")[3] for img in images]

    # Convert raw timestamp strings to datetime objects, following the format "%Y%jT%H%M%S" (e.g., "2021001T123456" for January 1, 2021, at 12:34:56)
    image_dates = [datetime.strptime(d, "%Y%jT%H%M%S") for d in image_dates]
    
    # Sort both lists simultaneously based on the Julian date values
    images, image_dates = zip(
        *sorted(zip(images, image_dates), key=lambda pair: pair[1])
    )

    return images, np.array(image_dates) # to be indexed with the pixel series to get the corresponding dates

def read_hls_cube(
    bands,
    image_list_l30,
    image_list_s10,
    start_row,
    start_col,
    end_row,
    end_col,
    cp: int = 1,
):
    """Read the selected HLS chip using up to cp concurrent scene reads."""
    if not isinstance(cp, int) or isinstance(cp, bool) or cp < 1:
        raise ValueError("cp must be a positive integer")
    if any(index % 3 for index in (start_row, start_col, end_row, end_col)):
        raise ValueError("10 m window bounds must align with 30 m pixels (multiples of 3)")

    bands_l30 = [HSL_BANDS[band][0] for band in bands if band in HSL_BANDS]
    bands_s10 = [HSL_BANDS[band][1] for band in bands if band in HSL_BANDS]

    l30_window = Window(
        start_col // 3,
        start_row // 3,
        (end_col - start_col) // 3,
        (end_row - start_row) // 3,
    )
    s10_window = Window(
        start_col, start_row, end_col - start_col, end_row - start_row
    )
    cube_l30 = _read_sensor(
        image_list_l30, "L30", bands_l30, l30_window, cp,
    )
    cube_s10 = _read_sensor(
        image_list_s10, "S10", bands_s10, s10_window, cp,
    )
    return cube_l30, cube_s10