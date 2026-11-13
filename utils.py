def generate_tile_chips(image_size, chip_size):
    """Return chips as (name, start_row, start_col, end_row, end_col).

    End coordinates are exclusive. Edge chips may be smaller than chip_size.
    """
    return [
        (
            f"chip_{row:06d}_{col:06d}_{min(row + chip_size, image_size):06d}_{min(col + chip_size, image_size):06d}",
            row,
            col,
            min(row + chip_size, image_size),
            min(col + chip_size, image_size),
        )
        for row in range(0, image_size, chip_size)
        for col in range(0, image_size, chip_size)
    ]
