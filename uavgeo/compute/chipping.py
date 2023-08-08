from shapely import box, Polygon
import geopandas as gpd
import xarray as xr
import numpy as np
from rioxarray import merge

def create_chip_bounds_gdf(input_dims:dict, shape_x:int, shape_y:int, input_overlap=None, crs = None):
    """
    Create a GeoDataFrame containing bounding boxes (chips) covering a specified area.

    This function generates a GeoDataFrame with bounding boxes that cover the given 2D area
    defined by `shape_x` and `shape_y`. The bounding boxes (chips) are created based on the
    dimensions specified in `input_dims`. Optionally, an `input_overlap` can be provided to
    specify overlapping areas between consecutive chips. The generated GeoDataFrame can be
    assigned a coordinate reference system (CRS) using the `crs` parameter.

    Args:
        input_dims (dict): A dictionary containing the dimensions of a single chip. e.g. {"x":128, "y":128}
        shape_x (int): The width of the overall area to cover with chips.
        shape_y (int): The height of the overall area to cover with chips.
        input_overlap (dict, optional): A dictionary specifying overlap along each dimension. similar format to input_dims
            If not provided, no overlap is used (default). Each dimension's overlap reduces
            the respective chip size by the specified amount.
        crs (CRS, optional): The coordinate reference system to assign to the GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing bounding boxes representing chips. Each
        row contains a 'box_id' and a geometry column specifying the bounding box's shape.
    """
    

    if input_overlap is None:
        input_stepover = input_dims
    
    else: 
        input_stepover = { key: input_dims[key]-input_overlap[key] for key in input_dims}
        
    maxy = ((shape_y-input_dims["y"])//input_stepover["y"]) + 1
    maxx = ((shape_x-input_dims["x"]) //input_stepover["x"]) + 1

    start_x = 0
    start_y = 0
    end_x = input_dims["x"]
    end_y = input_dims["y"]
    
    geom = []
    
    z = []
    k=0
    for i in range(maxy):    
        for j in range(maxx):
            geom.append(box(start_x,start_y,end_x, end_y))

            z.append(k)
            start_x+=input_stepover["x"]
            end_x+=input_stepover["x"]
            k+=1
    
        start_x = 0
        end_x = input_dims["x"]    
        start_y+=input_stepover["y"]
        end_y+=input_stepover["y"]
        
    return gpd.GeoDataFrame({"box_id":z}, geometry = geom,crs=crs)
        
def imgref_to_crsref_boxes(raster:xr.DataArray, gdf: gpd.GeoDataFrame):
    """
    Convert image-reference bounding boxes to CRS-reference bounding boxes.

    This function takes a GeoDataFrame containing image-reference bounding boxes and a
    corresponding raster DataArray (`raster`) that serves as the reference grid for the
    image. It converts the image-reference bounding boxes to CRS (coordinate reference system)
    reference bounding boxes by mapping the image reference coordinates to CRS coordinates.

    Args:
        raster (xr.DataArray): A raster DataArray providing the reference grid for the image.
        gdf (gpd.GeoDataFrame): A GeoDataFrame containing image-reference bounding boxes.
            The 'geometry' column should contain the image-reference bounding box geometries.

    Returns:
        gpd.GeoDataFrame: A modified GeoDataFrame with an additional 'c_geom' column containing
        the CRS-reference bounding box geometries.

    Note:
        The provided GeoDataFrame ('gdf') is modified in-place by adding the 'c_geom' column
        with the CRS-reference bounding box geometries.
    """

    c_geom = []
    for geom in gdf.geometry:

        xmin, ymin, xmax, ymax = geom.bounds
        
        x = raster.x[int(xmin):int(xmax)]
        y= raster.y[int(ymin):int(ymax)]
        
        c_xmin = float(x[0])
        c_xmax = float(x[-1])
        c_ymin = float(y[0])
        c_ymax = float(y[-1])
        c_bbox = box(c_xmin, c_ymin, c_xmax, c_ymax)
        c_geom.append(c_bbox)
    gdf["c_geom"] = c_geom
    return gdf
    
def imgref_to_crs(raster:xr.DataArray, row:Polygon):
    """
    Convert image-reference bounding box to CRS-reference bounding box.

    This function takes a single Polygon representing an image-reference bounding box and
    a corresponding raster DataArray (`raster`) that serves as the reference grid for the
    image. It converts the image-reference bounding box to a CRS (coordinate reference system)
    reference bounding box by mapping the image reference coordinates to CRS coordinates.

    Args:
        raster (xr.DataArray): A raster DataArray providing the reference grid for the image.
        row (Polygon): A Polygon representing the image-reference bounding box.

    Returns:
        box: A bounding box in CRS reference coordinates.

    Note:
        This function does not modify the original 'row' object. It returns a new bounding box
        in CRS reference coordinates.
    """

    xmin, ymin, xmax, ymax = row.bounds
    
    x = raster.x[int(xmin):int(xmax)]
    y= raster.y[int(ymin):int(ymax)]
    
    c_xmin = float(x[0])
    c_xmax = float(x[-1])
    c_ymin = float(y[0])
    c_ymax = float(y[-1])
    c_bbox = box(c_xmin, c_ymin, c_xmax, c_ymax)
    return c_bbox

def apply_geom_crs(gdf:gpd.GeoDataFrame, raster:xr.DataArray):
    """
    Apply coordinate transformation to geometries in a GeoDataFrame using a raster reference.

    This function takes a GeoDataFrame (`gdf`) containing geometries and a raster DataArray
    (`raster`) that serves as the reference grid for the coordinate transformation. The function
    applies the `imgref_to_crs` function to each geometry in the GeoDataFrame using the provided
    raster, creating a new geometry column 'x_geom' with the transformed geometries. The original
    geometries are retained in a new column 'chip_geom'. The GeoDataFrame is then updated to use
    the 'x_geom' column as its geometry.

    Args:
        gdf (gpd.GeoDataFrame): A GeoDataFrame containing geometries to transform.
        raster (xr.DataArray): A raster DataArray providing the reference grid for the transformation.

    Returns:
        gpd.GeoDataFrame: A modified GeoDataFrame with the 'x_geom' column containing transformed
        geometries and the 'chip_geom' column containing the original geometries. The GeoDataFrame
        uses the 'x_geom' column as its geometry.

    Note:
        This function modifies the original 'gdf' object in place. It returns the modified GeoDataFrame.
    """
    gdf["x_geom"] = gdf["geometry"].apply(lambda x: imgref_to_crs(raster,x))
    gdf["chip_geom"] = gdf["geometry"]
    return gdf.set_geometry(gdf["x_geom"])

def np_chip_to_rxr(darray:xr.DataArray, geom:gpd.GeoDataFrame, crs):
    """
    Convert a NumPy chip array to a raster DataArray with defined coordinates and CRS.

    This function takes a NumPy array (`darray`) representing a chip and a GeoDataFrame
    (`geom`) containing the geometry information for the chip's spatial extent. The function
    creates a raster DataArray using the input array, with coordinates derived from the
    geometry of the chip and assigns the specified CRS to the DataArray.

    Args:
        darray (np.ndarray): A NumPy array representing the chip data.
        geom (gpd.GeoDataFrame): A GeoDataFrame containing the chip's geometry information.
            The 'geometry' column should contain the bounding box geometry.
        crs (CRS): The coordinate reference system to assign to the DataArray.

    Returns:
        xr.DataArray: A raster DataArray with the provided chip data, coordinates, and CRS.
            The DataArray is transposed to have dimensions ('band', 'y', 'x').

    Note:
        This function does not modify the original 'darray' or 'geom' objects. It returns a new
        raster DataArray with assigned coordinates and CRS.
    """

    min_x, min_y, max_x, max_y = geom.geometry.bounds
    
    x_coords = np.linspace(min_x, max_x, darray.shape[1])
    #the array reversal might have to do with negative y coords in my exploration data

    y_coords =np.linspace(min_y, max_y, darray.shape[0])[::-1]
    #darray = np.rot90(darray)
    return xr.DataArray(darray, dims=("y", "x", "band"), coords={"x": x_coords, "y": y_coords}).rio.write_crs(crs).transpose('band', 'y', 'x')

def chips_to_single(chiplist:list, empty_raster:xr.DataArray, chip_geoms:gpd.GeoDataFrame, single_band:int = None, clip:bool = False):
    """
    Convert a list of chip arrays to a single raster DataArray with defined coordinates and CRS.

    This function takes a list of chip arrays (`chiplist`), an empty reference raster DataArray
    (`empty_raster`), and a GeoDataFrame containing geometry information for each chip
    (`chip_geoms`). The function creates a single raster DataArray by combining the chip arrays
    and aligning them with the empty reference raster. Optionally, a single band can be selected
    using the `single_band` parameter. The result can also be clipped to the bounding box of the
    chip geometries using the `clip` parameter.

    Args:
        chiplist (list): A list of NumPy arrays representing chip data.
        empty_raster (xr.DataArray): An empty reference raster DataArray to align the chips with. (should have same x, y, and band (and order))
        chip_geoms (gpd.GeoDataFrame): A GeoDataFrame containing geometry information for each chip.
            The 'geometry' column should contain the bounding box geometry.
        single_band (int, optional): If specified, selects a single band from each chip array.
        clip (bool, optional): If True, clips the output raster to the extent of the chip geometries.

    Returns:
        xr.DataArray: A raster DataArray containing the combined chip data aligned with the
        reference raster. If `single_band` is specified, only that band will be included.

    Note:
        This function does not modify the original objects. It returns a new raster DataArray
        containing the combined chip data.

    Note:
        If `single_band` is not specified, the output DataArray will have dimensions
        ('band', 'y', 'x'). If `single_band` is specified, the output DataArray will have
        dimensions ('y', 'x').
    """

    result = []
    for i, row in chip_geoms.iterrows():
        result.append(np_chip_to_rxr(chiplist[i],row,chip_geoms.crs))

    #if it is just an NDVI isngle band raster or something
    single_result = []
    if single_band is not None:
        for raster in result:
            single_result.append(raster.sel(band=single_band))
        result = single_result
    
    #merge all the individual raster values with the empty raster

    output = merge.merge_arrays([empty_raster]+result)

    if clip:
        output = output.rio.clip_box(minx = chip_geoms.bounds.min().minx, miny =chip_geoms.bounds.min().miny ,maxx =chip_geoms.bounds.max().maxx , maxy= chip_geoms.bounds.max().maxy)

    return output
    