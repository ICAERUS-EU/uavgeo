"""
DataPipes for :doc:`geopandas <geopandas:index>`.
"""
from typing import Any, Dict, Iterator, Optional, Union, Hashable

try:
    import pandas as pd
except ImportError:
    pd = None
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, StreamReader, FileOpener, IterableWrapper
import torchdata
try:
    import geopandas as gpd
except ImportError:
    gpd = None

from shapely import box
def start_pipe(iterable):
    return IterableWrapper(iterable=iterable)
import torch
import os

@functional_datapipe("parse_yolo")
class YoloLoaderIterDataPipe(IterDataPipe):
    def __init__(
        self,
        source_datapipe: IterDataPipe,
        **kwargs: Optional[Dict[str, Any]]) -> None:
        if pd is None:
            raise ModuleNotFoundError(
                "Package `pandas` is required to be installed to use this datapipe. "
                "Please use `pip install pandas` or "
                "`conda install -c conda-forge pandas` "
                "to install the package"
                )
        self.source_datapipe: IterDataPipe = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator:

        l = list(self.source_datapipe)
        df = pd.DataFrame() 

        for fn, yolobox in l:
            d = {}
            d["file"]= fn
            d["c"] = int(yolobox[0])
            d["x"] = float(yolobox[1]) 
            d["y"] = float(yolobox[2]) 
            d["w"] = float(yolobox[3]) 
            d["h"] = float(yolobox[4]) 
            a = pd.DataFrame([d])
            df = pd.concat([df,a])
        unique_files = df["file"].unique()
        for item in unique_files:
            yield df[df["file"]==item].copy()

        
@functional_datapipe("yolobox_to_gpd")
class YoloBoxToTorchIterDataPipe(IterDataPipe):
    def __init__(
        self,
        source_datapipe: IterDataPipe,
        image_datapipe: IterDataPipe[Union[xr.DataArray, xr. Dataset]],
        **kwargs: Optional[Dict[str, Any]]) -> None:
        if pd is None:
            raise ModuleNotFoundError(
                "Package `pandas` is required to be installed to use this datapipe. "
                "Please use `pip install pandas` or "
                "`conda install -c conda-forge pandas` "
                "to install the package"
                )
        self.source_datapipe: IterDataPipe = source_datapipe
        self.image_datapipe: IterDataPipe = image_datapipe

        self.kwargs = kwargs

    def __iter__(self) -> Iterator:

        for img, label in zip(self.image_datapipe, self.source_datapipe):

            torch_tensor = torch.tensor(label[["x", "y", "w", "h"]].values)
            torch_boxes = self.yolo_to_torch_boxes(torch_tensor, img.shape[2], img.shape[1])
            df2 =  pd.DataFrame(torch_boxes.numpy(), columns = ["xmin", "ymin", "xmax", "ymax"])

            #make the coordinates into a geometry item
            geom = [box(xmin =x1,ymin=y1 , xmax=x2,ymax=y2) for x1,y1,x2,y2 in zip(df2["xmin"], df2["ymin"], df2["xmax"], df2["ymax"])]
            #drop existing coords
            df2 = df2.drop(['xmin','ymin','xmax','ymax'], axis=1)
            #convert into geodataframe with yolo labels as geom
            geodf = gpd.GeoDataFrame(df2, geometry = geom)
            geodf["c"] = list(label["c"])
            yield img, geodf


    def yolo_to_torch_boxes(self, yolo_boxes, image_width, image_height):
        yolo_boxes = yolo_boxes.clone()

        # Convert center coordinates to absolute coordinates
        yolo_boxes[:, 0] = yolo_boxes[:, 0] * image_width
        yolo_boxes[:, 1] = yolo_boxes[:, 1] * image_height

        # Convert width and height to absolute coordinates
        yolo_boxes[:, 2] = yolo_boxes[:, 2] * image_width
        yolo_boxes[:, 3] = yolo_boxes[:, 3] * image_height

        # Convert to PyTorch bounding box format (xmin, ymin, xmax, ymax)
        torch_boxes = torch.zeros_like(yolo_boxes)
        torch_boxes[:, 0] = yolo_boxes[:, 0] - yolo_boxes[:, 2] / 2
        torch_boxes[:, 1] = yolo_boxes[:, 1] - yolo_boxes[:, 3] / 2
        torch_boxes[:, 2] = yolo_boxes[:, 0] + yolo_boxes[:, 2] / 2
        torch_boxes[:, 3] = yolo_boxes[:, 1] + yolo_boxes[:, 3] / 2

        return torch_boxes

@functional_datapipe("chip_image_and_label")
class GPDGeomRectangleClipperIterDataPipe(IterDataPipe):
    """
    Takes vector :py:class:`geopandas.GeoSeries` or
    :py:class:`geopandas.GeoDataFrame` geometries and clips them with the
    rectangular extent of an :py:class:`xarray.DataArray` or
    :py:class:`xarray.Dataset` grid to yield tuples of spatially subsetted
    :py:class:`geopandas.GeoSeries` or :py:class:`geopandas.GeoDataFrame`
    vectors and the correponding :py:class:`xarray.DataArray` or
    :py:class:`xarray.Dataset` raster object used as the clip mask (functional
    name: ``clip_vector_with_rectangle``).

    Uses the rectangular clip algorithm of :py:func:`geopandas.clip`, with the
    bounding box rectangle (minx, miny, maxx, maxy) derived from input raster
    mask's bounding box extent.

    Note
    ----
    If the input vector's coordinate reference system (``crs``) is different to
    the raster mask's coordinate reference system (``rio.crs``), the vector
    will be reprojected using :py:meth:`geopandas.GeoDataFrame.to_crs` to match
    the raster's coordinate reference system.

    Parameters
    ----------
    source_datapipe : IterDataPipe[geopandas.GeoDataFrame]
        A DataPipe that contains :py:class:`geopandas.GeoSeries` or
        :py:class:`geopandas.GeoDataFrame` vector geometries with a
        :py:attr:`.crs <geopandas.GeoDataFrame.crs>` property.

    mask_datapipe : IterDataPipe[xarray.DataArray]
        A DataPipe that contains :py:class:`xarray.DataArray` or
        :py:class:`xarray.Dataset` objects with a
        :py:attr:`.rio.crs <rioxarray.rioxarray.XRasterBase.crs>` property and
        :py:meth:`.rio.bounds <rioxarray.rioxarray.XRasterBase.bounds>` method.

    kwargs : Optional
        Extra keyword arguments to pass to :py:func:`geopandas.clip`.

    Yields
    ------
    paired_obj : Tuple[geopandas.GeoDataFrame, xarray.DataArray]
        A tuple consisting of the spatially subsetted
        :py:class:`geopandas.GeoSeries` or :py:class:`geopandas.GeoDataFrame`
        vector, and the corresponding :py:class:`xarray.DataArray` or
        :py:class:`xarray.Dataset` raster used as the clip mask.

    Raises
    ------
    ModuleNotFoundError
        If ``geopandas`` is not installed. See
        :doc:`install instructions for geopandas <geopandas:getting_started/install>`
        (e.g. via ``pip install geopandas``) before using this class.

    NotImplementedError
        If the length of the vector ``source_datapipe`` is not 1. Currently,
        all of the vector geometries have to be merged into a single
        :py:class:`geopandas.GeoSeries` or :py:class:`geopandas.GeoDataFrame`.
        Refer to the section on Appending under geopandas'
        :doc:`geopandas:docs/user_guide/mergingdata` docs.

    Example
    -------
    >>> import pytest
    >>> import rioxarray
    >>> gpd = pytest.importorskip("geopandas")
    ...
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> from zen3geo.datapipes import GeoPandasRectangleClipper
    ...
    >>> # Read in a vector polygon data source
    >>> geodataframe = gpd.read_file(
    ...     filename="https://github.com/geopandas/geopandas/raw/v0.11.1/geopandas/tests/data/overlay/polys/df1.geojson",
    ... )
    >>> assert geodataframe.crs == "EPSG:4326"  # latitude/longitude coords
    >>> dp_vector = IterableWrapper(iterable=[geodataframe])
    ...
    >>> # Get list of raster grids to cut up the vector polygon later
    >>> dataarray = rioxarray.open_rasterio(
    ...     filename="https://github.com/rasterio/rasterio/raw/1.3.2/tests/data/world.byte.tif"
    ... )
    >>> assert dataarray.rio.crs == "EPSG:4326"  # latitude/longitude coords
    >>> dp_raster = IterableWrapper(
    ...     iterable=[
    ...         dataarray.sel(x=slice(0, 2)),  # longitude 0 to 2 degrees
    ...         dataarray.sel(x=slice(2, 4)),  # longitude 2 to 4 degrees
    ...     ]
    ... )
    ...
    >>> # Clip vector point geometries based on raster masks
    >>> dp_clipped = dp_vector.clip_vector_with_rectangle(
    ...     mask_datapipe=dp_raster
    ... )
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_clipped)
    >>> geodataframe0, raster0 = next(it)
    >>> geodataframe0
       col1                                           geometry
    0     1  POLYGON ((0.00000 0.00000, 0.00000 2.00000, 2....
    >>> raster0
    <xarray.DataArray (band: 1, y: 1200, x: 16)>
    array([[[0, 0, ..., 0, 0],
            [0, 0, ..., 0, 0],
            ...,
            [1, 1, ..., 1, 1],
            [1, 1, ..., 1, 1]]], dtype=uint8)
    Coordinates:
      * band         (band) int64 1
      * x            (x) float64 0.0625 0.1875 0.3125 0.4375 ... 1.688 1.812 1.938
      * y            (y) float64 74.94 74.81 74.69 74.56 ... -74.69 -74.81 -74.94
        spatial_ref  int64 0
    ...
    >>> geodataframe1, raster1 = next(it)
    >>> geodataframe1
       col1                                           geometry
    1     2  POLYGON ((2.00000 2.00000, 2.00000 4.00000, 4....
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        input_dims: Dict[Hashable, int],

        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        if gpd is None:
            raise ModuleNotFoundError(
                "Package `geopandas` is required to be installed to use this datapipe. "
                "Please use `pip install geopandas` or "
                "`conda install -c conda-forge geopandas` "
                "to install the package"
            )
        self.source_datapipe: IterDataPipe = source_datapipe
        self.input_dims: Dict[Hashable, int] = input_dims
        self.kwargs = kwargs

    def __iter__(self) -> Iterator:

        for raster, df in self.source_datapipe:
            for chip in raster.batch.generator(input_dims=self.input_dims, **self.kwargs):
                bounds = chip.rio.bounds()
                clipped_gdf = df.clip(mask=bounds)
                # Now that all the chips are in their image-coordinates (e.g. chip 2 has bounds box of (256, 768,  0, 512)) 
                # We will reset them to all be in their own image coordinates (0,512,0,512), using an Affine translation
                # get the transform of raster
                tf = chip.rio.transform()
                # inverse those values
                inv_xoff = -1 * tf.xoff
                inv_yoff = -1 * tf.yoff
                # translate the bounding box geometry:
                clipped_gdf.geometry = clipped_gdf.translate(xoff = inv_xoff , yoff = inv_yoff )
                # translate the raster/chip coordinates:
                chip = chip.assign_coords(x= (chip.x+inv_xoff), y= (chip.y+inv_yoff))
                yield chip, clipped_gdf 

@functional_datapipe("save_image_and_label")
class ImageLabelSaverIterDataPipe(IterDataPipe):
    """
    Takes vector :py:class:`geopandas.GeoSeries` or
    :py:class:`geopandas.GeoDataFrame` geometries and clips them with the
    rectangular extent of an :py:class:`xarray.DataArray` or
    :py:class:`xarray.Dataset` grid to yield tuples of spatially subsetted
    :py:class:`geopandas.GeoSeries` or :py:class:`geopandas.GeoDataFrame`
    vectors and the correponding :py:class:`xarray.DataArray` or
    :py:class:`xarray.Dataset` raster object used as the clip mask (functional
    name: ``clip_vector_with_rectangle``).

    Uses the rectangular clip algorithm of :py:func:`geopandas.clip`, with the
    bounding box rectangle (minx, miny, maxx, maxy) derived from input raster
    mask's bounding box extent.

    Note
    ----
    If the input vector's coordinate reference system (``crs``) is different to
    the raster mask's coordinate reference system (``rio.crs``), the vector
    will be reprojected using :py:meth:`geopandas.GeoDataFrame.to_crs` to match
    the raster's coordinate reference system.

    Parameters
    ----------
    source_datapipe : IterDataPipe[geopandas.GeoDataFrame]
        A DataPipe that contains :py:class:`geopandas.GeoSeries` or
        :py:class:`geopandas.GeoDataFrame` vector geometries with a
        :py:attr:`.crs <geopandas.GeoDataFrame.crs>` property.

    mask_datapipe : IterDataPipe[xarray.DataArray]
        A DataPipe that contains :py:class:`xarray.DataArray` or
        :py:class:`xarray.Dataset` objects with a
        :py:attr:`.rio.crs <rioxarray.rioxarray.XRasterBase.crs>` property and
        :py:meth:`.rio.bounds <rioxarray.rioxarray.XRasterBase.bounds>` method.

    kwargs : Optional
        Extra keyword arguments to pass to :py:func:`geopandas.clip`.

    Yields
    ------
    paired_obj : Tuple[geopandas.GeoDataFrame, xarray.DataArray]
        A tuple consisting of the spatially subsetted
        :py:class:`geopandas.GeoSeries` or :py:class:`geopandas.GeoDataFrame`
        vector, and the corresponding :py:class:`xarray.DataArray` or
        :py:class:`xarray.Dataset` raster used as the clip mask.

    Raises
    ------
    ModuleNotFoundError
        If ``geopandas`` is not installed. See
        :doc:`install instructions for geopandas <geopandas:getting_started/install>`
        (e.g. via ``pip install geopandas``) before using this class.

    NotImplementedError
        If the length of the vector ``source_datapipe`` is not 1. Currently,
        all of the vector geometries have to be merged into a single
        :py:class:`geopandas.GeoSeries` or :py:class:`geopandas.GeoDataFrame`.
        Refer to the section on Appending under geopandas'
        :doc:`geopandas:docs/user_guide/mergingdata` docs.

    Example
    -------
    >>> import pytest
    >>> import rioxarray
    >>> gpd = pytest.importorskip("geopandas")
    ...
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> from zen3geo.datapipes import GeoPandasRectangleClipper
    ...
    >>> # Read in a vector polygon data source
    >>> geodataframe = gpd.read_file(
    ...     filename="https://github.com/geopandas/geopandas/raw/v0.11.1/geopandas/tests/data/overlay/polys/df1.geojson",
    ... )
    >>> assert geodataframe.crs == "EPSG:4326"  # latitude/longitude coords
    >>> dp_vector = IterableWrapper(iterable=[geodataframe])
    ...
    >>> # Get list of raster grids to cut up the vector polygon later
    >>> dataarray = rioxarray.open_rasterio(
    ...     filename="https://github.com/rasterio/rasterio/raw/1.3.2/tests/data/world.byte.tif"
    ... )
    >>> assert dataarray.rio.crs == "EPSG:4326"  # latitude/longitude coords
    >>> dp_raster = IterableWrapper(
    ...     iterable=[
    ...         dataarray.sel(x=slice(0, 2)),  # longitude 0 to 2 degrees
    ...         dataarray.sel(x=slice(2, 4)),  # longitude 2 to 4 degrees
    ...     ]
    ... )
    ...
    >>> # Clip vector point geometries based on raster masks
    >>> dp_clipped = dp_vector.clip_vector_with_rectangle(
    ...     mask_datapipe=dp_raster
    ... )
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_clipped)
    >>> geodataframe0, raster0 = next(it)
    >>> geodataframe0
       col1                                           geometry
    0     1  POLYGON ((0.00000 0.00000, 0.00000 2.00000, 2....
    >>> raster0
    <xarray.DataArray (band: 1, y: 1200, x: 16)>
    array([[[0, 0, ..., 0, 0],
            [0, 0, ..., 0, 0],
            ...,
            [1, 1, ..., 1, 1],
            [1, 1, ..., 1, 1]]], dtype=uint8)
    Coordinates:
      * band         (band) int64 1
      * x            (x) float64 0.0625 0.1875 0.3125 0.4375 ... 1.688 1.812 1.938
      * y            (y) float64 74.94 74.81 74.69 74.56 ... -74.69 -74.81 -74.94
        spatial_ref  int64 0
    ...
    >>> geodataframe1, raster1 = next(it)
    >>> geodataframe1
       col1                                           geometry
    1     2  POLYGON ((2.00000 2.00000, 2.00000 4.00000, 4....
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        output_path: str,
        skip_empty: bool = True,
        img_ext: str = ".png",
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        if gpd is None:
            raise ModuleNotFoundError(
                "Package `geopandas` is required to be installed to use this datapipe. "
                "Please use `pip install geopandas` or "
                "`conda install -c conda-forge geopandas` "
                "to install the package"
            )
        self.source_datapipe: IterDataPipe = source_datapipe
        self.output_path = output_path
        self.skip_empty = skip_empty
        self.img_ext = img_ext
        self.kwargs = kwargs


    def __iter__(self) -> Iterator:
        i = 0
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        r_filepath = os.path.join(self.output_path, "images" )
        if not os.path.exists(r_filepath):
            os.mkdir(r_filepath)

        l_filepath = os.path.join(self.output_path, "labels" )
        if not os.path.exists(l_filepath):
            os.mkdir(l_filepath)

        for raster, df in self.source_datapipe:
            number = "{:07d}" .format(i)

            r_filename = os.path.join(r_filepath, number+self.img_ext)
            l_filename = os.path.join(l_filepath, number+".txt" )
            
            if len(df)<1 and self.skip_empty:
                continue
            
            raster.rio.to_raster(r_filename)
            self.save_gdf_to_yolo(gdf = df, path = l_filename, shape = raster.shape)
            i+=1
            yield raster, df

    def save_gdf_to_yolo(self, gdf,path, shape):

        height = shape[2]
        width = shape[1]
        
        boxs = gdf.geometry.bounds
        boxs = torch.tensor(boxs.values)
        
        yolo_boxes = self.convert_to_yolo_batch(boxs, img_width = width, img_height = height) 
        labels = list(gdf["c"])

        self.write_yolo_boxes_to_file(boxes = yolo_boxes, labels = labels, file_path = path)
    

    def convert_to_yolo_batch(self, boxes, img_width, img_height):
        # Calculate widths and heights
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        # Calculate center points
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2

        # Normalize values 
        normalized_cx = cx / img_width
        normalized_cy = cy / img_height
        normalized_w = widths / img_width
        normalized_h = heights / img_height

        # Create YOLO tensors
        yolo_tensors = torch.stack([normalized_cx, normalized_cy, normalized_w, normalized_h], dim=1)
        return yolo_tensors

    def write_yolo_boxes_to_file(self, boxes, labels, file_path):
        with open(file_path, 'w') as f:
            for i in range(boxes.size(0)):
                yolo_box = boxes[i]
                label = labels[i]
                line = f"{label} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n"
                f.write(line)