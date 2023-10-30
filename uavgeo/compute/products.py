import xarray as xr
import numpy as np
import math
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata, rasterize_points_radial
from functools import partial
import uavgeo as ug
from tqdm.autonotebook import tqdm

def calc_dem_from_dsm(dsm: xr.DataArray, pixel_size, sampling_meters):
    """
    Calculate a Digital Elevation Model (DEM) from a Digital Surface Model (DSM) using sampling parameters.

    This function takes a Digital Surface Model (DSM), a pixel size, and a sampling distance in meters
    as input and generates a Digital Elevation Model (DEM) based on these parameters.

    Parameters:
        dsm (xr.DataArray): A Digital Surface Model represented as an xarray DataArray.
        pixel_size (float): The pixel size in the same unit as the DSM data.
        sampling_meters (float): The distance in meters to define the sampling grid for DEM creation.

    Returns:
        xr.DataArray: A Digital Elevation Model (DEM) as an xarray DataArray containing elevation values.

    The function works by creating a grid of points at the specified sampling distance and then finding
    the lowest elevation value within each grid cell in the DSM. The resulting DEM is based on these minimum
    elevation values.

    Note:
        - This function uses the rasterio library for Geographic Information System (GIS) operations.
        - The DSM and DEM should have the same coordinate reference system (CRS) for accurate results.
        - Depending on the resolution of the DSM, this process can be time-consuming.

    Example Usage:
        dsm_data =  rxr.open_rasterio("dsm.tif",mode = "w")  # Load DSM data from a GeoTIFF file
        pixel_size = 1.0  # Specify the pixel size in meters
        sampling_distance = 10.0  # Define the sampling distance in meters
        dem = calc_dem_from_dsm(dsm_data, pixel_size, sampling_distance)  # Calculate the DEM
    """
        
    tqdm.pandas()
    
    xy_dims = int(sampling_meters/pixel_size)

    input_dims = {"x": xy_dims, "y":xy_dims}
    input_overlap = {"x": 0, "y":0}

    #shape (resolution) of input image
    shape = dsm.shape
    c = dsm.rio.crs
    #create a grid based on the resolution of the chips
    sampling_grid = ug.compute.create_chip_bounds_gdf(input_dims = input_dims, 
                                                   input_overlap=input_overlap, 
                                                   shape_x = shape[2], 
                                                   shape_y = shape[1], 
                                                   crs = c)
    sampling_grid["crs_geom"] = sampling_grid["geometry"].apply(lambda x: ug.compute.imgref_to_crs(dsm, x)) 
    sampling_grid = sampling_grid.set_geometry(sampling_grid["crs_geom"])

    # find lowest points in grid THIS TAKES A WHILE, depending on image resolution etc., perhaps parallelize it?
    
    # progress_apply might need a from tqdm.notebook import tqdm, otherwise just import tqdm might work
   
    sampling_grid["h"] = sampling_grid.progress_apply(lambda x: np.nanmin(dsm.rio.clip_box(minx = x.geometry.bounds[0], 
                                   miny =x.geometry.bounds[1] ,
                                   maxx =x.geometry.bounds[2] , 
                                   maxy= x.geometry.bounds[3]),), axis=1)
    
    sampling_grid = sampling_grid.dropna().set_geometry(sampling_grid.geometry.centroid)

    dem = make_geocube(vector_data = sampling_grid,
                        measurements = ["h"],
                        like = dsm,
                        rasterize_function=partial(rasterize_points_griddata, method="linear"),)
    return dem.h

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

