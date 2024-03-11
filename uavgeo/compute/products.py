import xarray as xr
import numpy as np
import math
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata, rasterize_points_radial
from functools import partial
import uavgeo as ug
from tqdm.autonotebook import tqdm
import geopandas as gpd
import rioxarray as rxr
import pandas as pd

def calc_dtm_from_dsm(dsm: xr.DataArray, pixel_size, sampling_meters):
    """
    Calculate a Digital Terrain Model (DTM) from a Digital Surface Model (DSM) using sampling parameters.

    This function takes a Digital Surface Model (DSM), a pixel size, and a sampling distance in meters
    as input and generates a DTM based on these parameters.

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

    dtm = make_geocube(vector_data = sampling_grid,
                        measurements = ["h"],
                        like = dsm,
                        rasterize_function=partial(rasterize_points_griddata, method="linear"),)
    return dtm.h

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

def calc_vineyard_shadows(xr,band_id=1 ) -> xr.DataArray:
    """
    Calculate vineyard shadows using a method proposed by Velez et al. (2021) for vineyards and UAVs. Based on Kmeans.

    Parameters:
    - xr (xr.DataArray): An xarray DataArray containing the relevant bands, where band 1 represents
      the red channel used for shadow detection.

    Returns:
    - xr.DataArray: A binary mask representing vineyard shadows, where 1 indicates shadow pixels.

    Reference:
    - Velez et al. (2021). "A New Leaf Area Index Methodology Based on Unmanned Aerial Vehicle Imagery for Vineyards."
      URL: https://oeno-one.eu/article/view/4639

    Example:
    >>> vineyard_data = xr.open_rasterio('path/to/vineyard_image.tif')
    >>> shadows_mask = calc_vineyard_shadows(vineyard_data)

    Note:
    - The function applies a shadow detection method based on k-means clustering.
    - The resulting binary mask represents vineyard shadows.
    """
    # Shadows/LAI implementation
    shadows = xr.sel(band=[band_id])

    #flatten array
    flat_red = shadows.values.reshape(-1, 1)
    from sklearn.cluster import KMeans
    #Run k-means
    kmeans = KMeans(n_clusters = 5, random_state = 10, n_init='auto')
    kmeans.fit(flat_red)
    #extract label values
    labels = kmeans.labels_
    # find class with lowest median (==leaf shadow pixels) using cluster centroids 
    centroids = kmeans.cluster_centers_
    shadow_label = np.where(centroids == min(centroids))[0][0]
    # 0 and 1 encoded
    shadows_classified = (labels==shadow_label) *1
    # reconstruct array back into original shapes
    shadow_values= shadows_classified.reshape(shadows.values.shape)
    shadows.values  = shadow_values.astype(float)
    shadows = shadows.where(shadows > 0)

    return shadows

def calc_lai(bandstack:xr.DataArray, chm:xr.DataArray,ndvi_v, ndvi_s, k, red_id=1, nir_id=2) -> xr.DataArray:
    """
    Based on https://link.springer.com/article/10.1007/s11119-023-09993-9#Sec17
    Furlanetto et al. 2023 proposed a LAI method specifically for Maize crop

    """
    raise NotImplementedError()
    ndvi = calc_ndvi(bandstack, red_id, nir_id, rescale = False)
    if ndvi_s is None:
        ndvi_s = -0.1

    if ndvi_v is None:
        ndvi_v = max(ndvi)
    
    fvc = (ndvi-ndvi_s)/(ndvi_v-ndvi_s)
    lai = (-ln(1-fvc))/k

def extract_features(gpd_df, xr:xr.DataArray, stats = ["mean"], prefix =""):
    """
    Extracts statistical features from a raster dataset within the regions defined by a GeoDataFrame.

    Parameters:
    - gpd_df (geopandas.GeoDataFrame): The GeoDataFrame defining the regions of interest.
    - xr (xarray.DataArray): The raster dataset from which features will be extracted.
    - stats (list, optional): A list of statistical measures to calculate for each region.
      Valid options include ["mean", "min", "max", "median", "std", "sum", "count", "percentile_xx"].
      Default is ["mean"].
    - prefix (str, optional): A prefix to be added to the names of the extracted features.
      Default is an empty string.

    Returns:
    - geopandas.GeoDataFrame: A GeoDataFrame containing the extracted features as properties.

    Example:
    >>> gdf = gpd.read_file('path/to/shapefile.shp')
    >>> raster_data = xr.open_rasterio('path/to/raster.tif')
    >>> features = extract_features(gdf, raster_data, stats=["mean", "std"], prefix="landcover_")

    Note:
    - The function uses the 'rasterstats' library for zonal statistics.
    - The resulting GeoDataFrame will have columns named according to the specified 'stats' and 'prefix'.
    """
    from rasterstats import zonal_stats
    feat_t = zonal_stats(vectors = gpd_df, 
                         raster = xr.values[0,:,:], 
                         affine = xr.rio.transform(),
                         nodata = np.nan,
                         raster_out=False,
                         geojson_out=True,
                         stats=stats, 
                         prefix= prefix)
    return gpd.GeoDataFrame([item["properties"] for item in feat_t])


# define tertiles
# similar to method 2 in padua2019
def tertiler(df,min=-1,max=10000):
    
    # Get tertile positions
    tertiles = df['ndvi_mean'].quantile([1/3,2/3]).tolist()
    # Add a lower and upper range for the bins in pd.cut
    tertiles = [min] + tertiles + [max]
    print(tertiles)
    # Add a new tertile column to the data frame based on the tertile cut
    df['vigor_class'] = pd.cut(df['ndvi_mean'].fillna(0), bins=tertiles, labels=['low', 'medium', 'high'])
    
    return df
