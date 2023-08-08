import numpy as np
import xarray as xr


"""
Mostly taken from the following GitHub and paper:
https://github.com/sentinel-hub/natural-color
B. Sovdat, M. Kadunc, M. Batič, G. Milčinski, Natural color representation of Sentinel-2 data.
And converted to JS to Python code using chatGPT
Can probably be optimized a bunch, based on usig more numpy functions
"""

def zipper(a, b, f):
    """
    Applies the function `f` element-wise to the corresponding elements of arrays `a` and `b`, and returns the result as a NumPy array.

    Parameters:
        a (array-like): First input array.
        b (array-like): Second input array.
        f (callable): Function to be applied to each pair of elements from `a` and `b`.

    Returns:
        ndarray: NumPy array containing the results of applying `f` to each pair of elements from `a` and `b`.
    """
    return np.array([f(ai, bi) for ai, bi in zip(a, b)])

def mapConst(arr, c, f):
    """
    Applies the function `f` to each element of array `arr` along with a constant `c`, and returns the result as a NumPy array.

    Parameters:
        arr (array-like): Input array.
        c (numeric): Constant value.
        f (callable): Function to be applied to each element of `arr` along with `c`.

    Returns:
        ndarray: NumPy array containing the results of applying `f` to each element of `arr` along with `c`.
    """
    return np.array([f(ai, c) for i, ai in enumerate(arr)])

def dotSS(a, b):
    """
    Computes the element-wise multiplication of two input values.

    Parameters:
        a (numeric): First input value.
        b (numeric): Second input value.

    Returns:
        numeric: Element-wise product of `a` and `b`.
    """
    return a * b

def dotVS(v, s):
    """
    Computes the element-wise multiplication of an input array `v` with a scalar `s`.

    Parameters:
        v (array-like): Input array.
        s (numeric): Scalar value.

    Returns:
        ndarray: NumPy array containing the element-wise product of `v` and `s`.
    """
    return mapConst(v, s, dotSS)

def dotVV(a, b):
    """
    Computes the dot product of two input arrays `a` and `b` by applying `dotSS` element-wise using `zipper`, and then summing the resulting array.

    Parameters:
        a (array-like): First input array.
        b (array-like): Second input array.

    Returns:
        numeric: Dot product of `a` and `b`.
    """
    return np.sum(zipper(a, b, dotSS))

def dotMV(A, v):
    """
    Computes the matrix-vector product of input matrix `A` and input vector `v` using `dotVV` and `mapConst`.

    Parameters:
        A (array-like): Input matrix.
        v (array-like): Input vector.

    Returns:
        ndarray: NumPy array representing the matrix-vector product of `A` and `v`.
    """
    return mapConst(A, v, dotVV)

def adj(C):
    """
    Adjusts a given input value `C` based on certain conditions.

    Parameters:
        C (numeric): Input value.

    Returns:
        numeric: Adjusted value according to the conditions.
    """
    if C < 0.0031308:
        return 12.92 * C
    else:
        return 1.055 * pow(C, 0.41666) - 0.055

def labF(t):
    """
    Computes the result of a function `f` on an input value `t` based on certain conditions.

    Parameters:
        t (numeric): Input value.

    Returns:
        numeric: Result of applying the function `f` on `t` according to the conditions.
    """
    if t > 0.00885645:
        return pow(t, 1.0/3.0)
    else:
        return 0.137931 + 7.787 * t

def invLabF(t):
    """
    Computes the result of a function `f` on an input value `t` based on certain conditions.

    Parameters:
        t (numeric): Input value.

    Returns:
        numeric: Result of applying the function `f` on `t` according to the conditions.
    """
    if t > 0.2069:
        return t * t * t
    else:
        return 0.12842 * (t - 0.137931)

def XYZ_to_Lab(XYZ):
    """
    Converts an input XYZ color value to Lab color space using the functions `labF` and `np.array`.

    Parameters:
        XYZ (array-like): Input XYZ color value.

    Returns:
        ndarray: NumPy array representing the corresponding Lab color value.
    """
    lfY = labF(XYZ[1])
    return np.array([(116.0 * lfY - 16)/100,
                     5 * (labF(XYZ[0]) - lfY),
                     2 * (lfY - labF(XYZ[2]))])

def Lab_to_XYZ(Lab):
    """
    Converts an input Lab color value to XYZ color space using the function `invLabF` and `np.array`.

    Parameters:
        Lab (array-like): Input Lab color value.

    Returns:
        ndarray: NumPy array representing the corresponding XYZ color value.
    """
    YL = (100 * Lab[0] + 16) / 116
    return np.array([invLabF(YL + Lab[1] / 5.0),
                     invLabF(YL),
                     invLabF(YL - Lab[2] / 2.0)])

def XYZ_to_sRGBlin(xyz):
    """
    Converts an input XYZ color value to linear sRGB color space by performing a matrix-vector multiplication using `dotMV`.

    Parameters:
        xyz (array-like): Input XYZ color value.

    Returns:
        ndarray: NumPy array representing the corresponding linear sRGB color value.
    """
    T = np.array([[3.240, -1.537, -0.499],
                  [-0.969, 1.876, 0.042],
                  [0.056, -0.204, 1.057]])
    return dotMV(T, xyz)

def XYZ_to_sRGB(xyz):
    """
    Converts an input XYZ color value to sRGB color space by applying `adj` to each element of the array returned by `XYZ_to_sRGBlin`.

    Parameters:
        xyz (array-like): Input XYZ color value.

    Returns:
        ndarray: NumPy array representing the corresponding sRGB color value.
    """
    return np.array(list(map(adj, XYZ_to_sRGBlin(xyz))))

def Lab_to_sRGB(Lab):
    """
    Converts an input Lab color value to sRGB color space by first converting it to XYZ color space using `Lab_to_XYZ`, and then to sRGB using `XYZ_to_sRGB`.

    Parameters:
        Lab (array-like): Input Lab color value.

    Returns:
        ndarray: NumPy array representing the corresponding sRGB color value.
    """
    return XYZ_to_sRGB(Lab_to_XYZ(Lab))

def S2_to_XYZ(rad, T, gain):
    """
    Converts an input radiance value to XYZ color space using `S2_to_XYZ` by performing a matrix-vector multiplication and scaling with `gain`.

    Parameters:
        rad (array-like): Input radiance value.
        T (array-like): Transformation matrix.
        gain (numeric): Scaling factor.

    Returns:
        ndarray: NumPy array representing the corresponding XYZ color value.
    """
    return dotVS(dotMV(T, rad), gain)

def ProperGamma_S2_to_sRGB(rad,gg,gamma,gL):
    """
    Converts an input radiance value to sRGB color space using `S2_to_XYZ`, `XYZ_to_Lab`, `np.power`, and `Lab_to_sRGB`.

    Parameters:
        rad (array-like): Input radiance value.
        gg (numeric): Gain factor.
        gamma (numeric): Gamma factor.
        gL (numeric): Luminance factor.

    Returns:
        ndarray: NumPy array representing the corresponding sRGB color value.
    """

    T = np.array([[0.268, 0.361, 0.371],
              [0.240, 0.587, 0.174],
              [1.463, -0.427, -0.043]])

    XYZ = S2_to_XYZ(rad, T, gg)
    Lab = XYZ_to_Lab(XYZ)
    L = np.power(gL * Lab[0], gamma)
    return Lab_to_sRGB([L, Lab[1], Lab[2]])

def calc_srgb_from_msrgb(ms_rgb, gg, gamma, gL):
    """
    Converts multispectral RGB bands in xarray format to sRGB color space using the ProperGamma_S2_to_sRGB function.

    Parameters:
        ms_rgb (xarray.DataArray): Multispectral RGB values (0-1 scaling).
        gg (numeric): Gain factor.
        gamma (numeric): Gamma factor.
        gL (numeric): Luminance factor.

    Returns:
        xarray.DataArray: Converted sRGB values in sRGB color space.
    """
    m = ms_rgb.max()
    if m>1:
        raise ValueError("Bands should be scaled between 0 and 1 reflectance values. You can use the scale_to_01 function for this.")
    
    # reshape input data to fit [[r,g,b],[r,g,b]] format
    reshaped = ms_rgb.values.reshape(-1,3)
    #this can take a while, depending on data size format (around 1` min for a 1024x1024 chip`)
    result = np.apply_along_axis(ProperGamma_S2_to_sRGB, axis = 1, arr = reshaped, gg=gg,gamma=gamma,gL=gL)
    result = result.reshape(ms_rgb.shape)

    return xr.DataArray(result, coords=ms_rgb.coords, attrs=ms_rgb.attrs)


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