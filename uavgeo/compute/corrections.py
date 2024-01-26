import numpy as np
import xarray as xr


def scale_to_uint8(arr,bins=256,dtype=np.uint8):
    """
    Scales an input array to the range of proposed number of bins: uint8 values (0-255) bins = 256, uint16 (0-65535) bins  = 65536 etc.
    

    Parameters:
        arr (array-like): Input array.
        bins (int): Number of bins for scaling (default: 256).
        dtype (numpy.dtype): Data type for the output array (default: np.uint8).

    Returns:
        ndarray: Scaled array with values in the range of uint8.
    """
    return (arr// bins).astype(dtype)

def scale_to_01(arr,max=255):
    """
    Scales an input array to the range of 0-1 by dividing it by the maximum value.

    Parameters:
        arr (array-like): Input array.
        max_value (numeric): Maximum value to scale the array (default: 255).

    Returns:
        ndarray: Scaled array with values in the range of 0-1.
    """

    return arr.astype(np.float16)/max

def scale_band_to_min_max(band, min,max, clip =True):
    """
    Scale a band array to a specified minimum and maximum value range.

    This function scales a given band array (`band`) to a specified minimum and maximum value range
    (`min_value` to `max_value`). Optionally, the scaled values can be clipped to ensure they fall
    within the specified range.

    Args:
        band (np.ndarray): The input band array to be scaled.
        min_value (float): The desired minimum value of the scaled array.
        max_value (float): The desired maximum value of the scaled array.
        clip (bool, optional): If True, the scaled values will be clipped to the specified range.
            Default is True.

    Returns:
        np.ndarray: A scaled band array with values transformed to fit within the specified
        minimum and maximum value range.

    Note:
        The input array `band` is not modified. The function returns a new scaled array.
    """

    if clip:
        band = band.clip(min = min, max=max)
        
    band = band.astype(float)
    band = (band-min) * (255/(max-min))
    return band.astype(np.uint8)