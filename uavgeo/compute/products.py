import xarray as xr
import numpy as np
import math
    

def rescale_floats(arr, scaling=255, dtype ='uint8') -> xr.DataArray:
    """
    rescales the input values to fit within min-mac of 0 to scaling (255)m therefore converted to UInt8 (0-255)
    """
    return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * scaling)).astype(dtype,casting = "unsafe")

def calc_chm(dtm: xr.DataArray, dsm: xr.DataArray, rescale=False) -> xr.DataArray:
    """
    Calculates the Canopy Height model from inputs Surface (dsm) and Terrain model (dtm)

    from https://www.earthdatascience.org:
    The canopy height model (CHM) represents the HEIGHT of the trees. This is not an elevation value, rather it’s the height or distance between the ground and the top of the trees (or buildings or whatever object that the lidar system detected and recorded).
    Some canopy height models also include buildings, so you need to look closely at your data to make sure it was properly cleaned before assuming it represents all trees!
    """
    dtm_a = dtm.astype(float)
    dsm_a = dsm.astype(float)
    chm = dsm_a-dtm_a
    chm.name = "chm"
    if rescale:
        chm = rescale_floats(chm)
    return chm

def calc_vineyard_lai() -> xr.DataArray:
    """
    Based on https://oeno-one.eu/article/view/4639
    Velez et al. 2021 proposed a LAI method specifically for Vineyards and UAVs.

    """
    raise NotImplementedError()

def calc_lai(bandstack:xr.DataArray, chm:xr.DataArray,ndvi_v, ndvi_s, k, red_id=1, nir_id=2) -> xr.DataArray:
    """
    Based on https://link.springer.com/article/10.1007/s11119-023-09993-9#Sec17
    Furlanetto et al. 2023 proposed a LAI method specifically for Maize crop

    """

    ndvi = calc_ndvi(bandstack, red_id, nir_id, rescale = False)
    if ndvi_s is None:
        ndvi_s = -0.1

    if ndvi_v is None:
        ndvi_v = max(ndvi)
    
    fvc = (ndvi-ndvi_s)/(ndvi_v-ndvi_s)
    lai = (-ln(1-fvc))/k

    raise NotImplementedError()

