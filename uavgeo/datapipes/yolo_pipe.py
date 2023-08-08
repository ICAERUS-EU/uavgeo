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
import torch
import os
try:
    import xbatcher
except ImportError:
    xbatcher = None 
try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import ultralytics
except ImportError:
    ultralytics = None
import numpy as np

import uavgeo as ug


def start_pipe(iterable):
    return IterableWrapper(iterable=iterable)

@functional_datapipe("parse_yolo")
class YoloLoaderIterDataPipe(IterDataPipe):
    """
    DataPipe class for loading YOLO detection results from strings (such as loaded from parse_csv()).

    Args:
        source_datapipe (IterDataPipe): The source data pipe providing filenames and YOLO bounding box data.
        **kwargs: Optional keyword arguments.

    Raises:
        ModuleNotFoundError: If the `pandas` package is not installed.

    Yields:
        pandas.DataFrame: A DataFrame containing the YOLO detection results for each unique file.

    Examples:
        >>> source_datapipe = ...
        >>> yolo_loader = YoloLoaderIterDataPipe(source_datapipe) # or parse_yolo(source_datapipe)
        >>> for data in yolo_loader:
        >>>     print(data)
    """

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
        """
        Iterate over the YOLO detection results.

        Yields:
            pandas.DataFrame: A DataFrame containing the YOLO detection results for each unique file.
        """

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
    """
    DataPipe class for converting YOLO bounding boxes to GeoPandas GeoDataFrames.

    Args:
        source_datapipe (IterDataPipe): The source data pipe providing YOLO bounding box labels.
        image_datapipe (IterDataPipe): The data pipe providing the input images.
        **kwargs: Optional keyword arguments.

    Raises:
        ModuleNotFoundError: If the `geopandas` package is not installed.

    Yields:
        Tuple[xr.DataArray, geopandas.GeoDataFrame]: A tuple containing the input image and corresponding GeoPandas GeoDataFrame.

    Examples:
        >>> source_datapipe = ...
        >>> image_datapipe = ...
        >>> yolo_to_gpd = YoloBoxToTorchIterDataPipe(source_datapipe, image_datapipe) # or yolobox_to_gpd(source_datapipe, image_datapipe)
        >>> for img, geodf in yolo_to_gpd:
        >>>     print(img, geodf)
    """

 
    def __init__(
        self,
        source_datapipe: IterDataPipe,
        image_datapipe: IterDataPipe[Union[xr.DataArray, xr. Dataset]],
        **kwargs: Optional[Dict[str, Any]]) -> None:
        if gpd is None:
            raise ModuleNotFoundError(
                "Package `geopandas` is required to be installed to use this datapipe. "
                "Please use `pip install geopandas` or "
                "`conda install -c conda-forge geopandas` "
                "to install the package"
                )
        self.source_datapipe: IterDataPipe = source_datapipe
        self.image_datapipe: IterDataPipe = image_datapipe

        self.kwargs = kwargs

    def __iter__(self) -> Iterator:
        """
        Iterate over the converted YOLO bounding boxes and images.

        Yields:
            Tuple[xr.DataArray, geopandas.GeoDataFrame]: A tuple containing the input image and corresponding GeoPandas GeoDataFrame.
        """

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
    xbatcher is uised to clip the input raster to the correct amount of chips.

    Uses the rectangular clip algorithm of :py:func:`geopandas.clip`, with the
    bounding box rectangle (minx, miny, maxx, maxy) derived from input raster
    mask's bounding box extent.

    Then transforms all the resulting coordinates to the local-image coordinates:
    # Now that all the chips are in their image-coordinates (e.g. chip 2 has bounds box of (256, 768,  0, 512)) 
    # We will reset them to all be in their own image coordinates (0,512,0,512), using an Affine translation

    Args:
        source_datapipe (IterDataPipe): The source data pipe providing raster and GeoDataFrame pairs: (raster, gdf)
        input_dims (Dict[Hashable, int]): The input dimensions for generating chips.
        **kwargs: Optional keyword arguments.

    Raises:
        ModuleNotFoundError: If the `geopandas` or `xbatcher` package is not installed.

    Yields:
        Tuple[xr.DataArray, geopandas.GeoDataFrame]: A tuple containing the clipped chip and corresponding clipped GeoDataFrame.

    Examples:
        >>> source_datapipe = ...
        >>> input_dims = ...
        >>> chip_clipper = ChipClipperIterDataPipe(source_datapipe, input_dims)
        >>> for chip, clipped_gdf in chip_clipper:
        >>>     print(chip, clipped_gdf)
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
        if xbatcher is None:
            raise ModuleNotFoundError(
                "Package `xbatcher` is required to be installed to use this datapipe. "
                "Please use `pip install xbatcher` or "
                "`conda install -c conda-forge xbatcher` "
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
    DataPipe class for saving images and associated labels.

    Args:
        source_datapipe (IterDataPipe): The source data pipe providing raster and GeoDataFrame pairs.
        output_path (str): The output path to save the images and labels.
        skip_empty (bool, optional): Flag indicating whether to skip empty GeoDataFrames. Default is True.
        img_ext (str, optional): The extension of the image files. Default is ".png".
        **kwargs: Optional keyword arguments.

    Raises:
        ModuleNotFoundError: If the `geopandas` or `rasterio` package is not installed.

    Yields:
        Tuple[xr.DataArray, geopandas.GeoDataFrame]: A tuple containing the original raster and corresponding GeoDataFrame.

    Examples:
        >>> source_datapipe = ...
        >>> output_path = ...
        >>> image_label_saver = ImageLabelSaverIterDataPipe(source_datapipe, output_path)
        >>> for raster, df in image_label_saver:
        >>>     print(raster, df)
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
        if rasterio is None:
            raise ModuleNotFoundError(
                "Package `rasterio` is required to be installed to use this datapipe. "
                "Please use `pip install rasterio` or "
                "`conda install -c conda-forge rasterio` "
                "to install the package"
            )
        self.source_datapipe: IterDataPipe = source_datapipe
        self.output_path = output_path
        self.skip_empty = skip_empty
        self.img_ext = img_ext
        self.kwargs = kwargs


    def __iter__(self) -> Iterator:
        """
        Iterate over the images and labels, and save them to disk.

        Yields:
            Tuple[xr.DataArray, geopandas.GeoDataFrame]: A tuple containing the original raster and corresponding GeoDataFrame.
        """
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
        """
        Convert bounding box coordinates to YOLO format.

        Args:
            boxes (torch.Tensor): The bounding box coordinates.
            img_width (int): The width of the image.
            img_height (int): The height of the image.

        Returns:
            torch.Tensor: The bounding box coordinates in YOLO format.
        """
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
        """
        Write YOLO bounding box coordinates and labels to a text file.

        Args:
            boxes (torch.Tensor): The bounding box coordinates in YOLO format.
            labels (List[int]): The labels corresponding to each bounding box.
            file_path (str): The path to save the text file.
        """
        with open(file_path, 'w') as f:
            for i in range(boxes.size(0)):
                yolo_box = boxes[i]
                label = labels[i]
                line = f"{label} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n"
                f.write(line)



@functional_datapipe("predict_yolo")
class YoloPredictIterDataPipe(IterDataPipe):
    """
    An iterative data pipe that applies YOLO object detection using the ultralytics.YOLO model to input data.

    Args:
        source_datapipe (IterDataPipe): The source data pipeline providing input data.
        input_dims (Dict[Hashable, int]): A dictionary mapping hashable keys to integer values representing the input dimensions for the YOLO model.
        model (ultralytics.YOLO): An instance of the ultralytics.YOLO model.
        **kwargs (Optional[Dict[str, Any]]): Additional keyword arguments to be passed to the YOLO model.

    Raises:
        ModuleNotFoundError: If the `ultralytics` package is not installed.

    Yields:
        ultralytics.yolo.engine.results.Results: Predicted results from the YOLO model.

    Examples:
        # Load a YOLO model:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        # Create a YoloPredictIterDataPipe instance
        source_datapipe = #some type of preprocessed datapipe with xarray rasters ready for prediction
        data_pipe = YoloPredictIterDataPipe(source_datapipe, model) # or source_datapipe.predict_yolo()
        # Iterate over the predicted results
        for prediction in data_pipe:
            # Process the prediction
            ...
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,

        model: ultralytics.YOLO,
        tf_required: bool = True,

        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        if ultralytics is None:
            raise ModuleNotFoundError(
                "Package `ultralytics` is required to be installed to use this datapipe. "
                "Please use `pip install ultralytics` "
                "to install the package"
            )
        if gpd is None:
            raise ModuleNotFoundError(
                "Package `geopandas` is required to be installed to use this datapipe. "
                "Please use `pip install geopandas` "
                "or `conda install -c conda-forge geopandas`"
                "to install the package"
            )
        self.source_datapipe: IterDataPipe = source_datapipe
        self.model = model
        self.kwargs = kwargs
        self.tf_required = tf_required
    def __iter__(self) -> Iterator:

        for raster in self.source_datapipe:
            
            #make sure yolo can read it properly:
            shape = raster.shape
            #invert the shape from e.g. (3,512,512) to (512,512,3) YOLO expects numpy HWC - BGR
            reshaped = np.transpose(raster.values, (1,2,0))
            #shape is to force the ultimate image resolution, instead of default values

            #run it through the model: YOLO in this case
            results = self.model(reshaped,imgsz = max(shape), **self.kwargs)
            results = results[0]

            names = results.names
            # get minx miny maxx maxy from the result
            xyxy = np.array(results.boxes.xyxy)
            #get individual result-classes
            cls = np.array(results.boxes.cls).astype(int)
            #get the class-ids to string-format 0: "person" cls_names = ["person"]
            cls_names = [names[item] for item in cls]

            #instantiate the geodataframe
            gdf = gpd.GeoDataFrame({"class":cls, "class_names":cls_names, "xmin":xyxy[:,0],"ymin":xyxy[:,1],"xmax":xyxy[:,2],"ymax":xyxy[:,3]})
            
            # set the bounding box coordinates as the geometry
            geom = [box(xmin =x1,ymin=y1 , xmax=x2,ymax=y2) for x1,y1,x2,y2 in zip(gdf["xmin"], gdf["ymin"], gdf["xmax"], gdf["ymax"])]
            gdf = gdf.set_geometry(geom)
            # apply transform geometry of the y-axis due to chipping
            if self.tf_required:
                # geom2= gdf.translate(xoff = raster.rio.transform().xoff , yoff = raster.rio.transform().yoff )
                geom2 = [self.invert_y_coordinates_in_image(boxy, yoff = max(shape)) for boxy in geom]
                gdf = gdf.set_geometry(geom2)

            yield gdf

    def invert_y_coordinates_in_image(self, box_geometry, yoff):
        # Get the original box coordinates
        minx, miny, maxx, maxy = box_geometry.bounds

        # Invert the Y coordinates
        inverted_miny = yoff - maxy
        inverted_maxy = yoff - miny



        # Create a new box geometry with inverted Y coordinates
        inverted_box_geometry = box(minx, inverted_miny, maxx, inverted_maxy)

        return inverted_box_geometry



@functional_datapipe("yoloprediction_to_gpd")
class YoloResultToGPDIterDataPipe(IterDataPipe):
 
    def __init__(
        self,
        source_datapipe: IterDataPipe,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        if ultralytics is None:
            raise ModuleNotFoundError(
                "Package `ultralytics` is required to be installed to use this datapipe. "
                "Please use `pip install ultralytics` "
                "to install the package"
            )

        
        self.source_datapipe: IterDataPipe = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator:

        for results in self.source_datapipe:
            #get all class names
            names = results.names
            # get xyxy (minminmaxmax) from the result
            xyxy = np.array(results.boxes.xyxy)
            #get individual result-classes
            cls = np.array(results.boxes.cls).astype(int)
            #get the class-ids to string-format 0: "person" cls_names = ["person"]
            cls_names = [names[item] for item in cls]

            #instantiate the geodataframe
            gdf = gpd.GeoDataFrame({"class":cls, "class_names":cls_names, "xmin":xyxy[:,0],"ymin":xyxy[:,1],"xmax":xyxy[:,2],"ymax":xyxy[:,3]})
            
            # set the bounding box coordinates as the geometry
            geom = [box(xmin =x1,ymin=y1 , xmax=x2,ymax=y2) for x1,y1,x2,y2 in zip(gdf["xmin"], gdf["ymin"], gdf["xmax"], gdf["ymax"])]
            gdf = gdf.set_geometry(geom)
            # all in image-coordinates ofcourse!
            yield gdf 


"""
Image chipping pipes, based around geopandas boxes and matrices:
To help support georeferenced bounding box results: which are currently a pain in the * to use

"""


@functional_datapipe("gdf_image_chipper")
class ImgChipGDFInitIterDataPipe(IterDataPipe):
 
    def __init__(
        self,
        source_datapipe: IterDataPipe,
        input_dims: Dict,
        set_crs: bool = False,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        if gpd is None:
            raise ModuleNotFoundError(
                "Package `geopandas is required to be installed to use this datapipe. "
                "Please use `pip install geopandas` "
                "to install the package"
            )

        
        self.source_datapipe: IterDataPipe = source_datapipe
        self.input_dims = input_dims
        self.set_crs=set_crs
        self.kwargs = kwargs


    def __iter__(self) -> Iterator:

        for raster in self.source_datapipe:
            shape = raster.shape
            if self.set_crs:
                s_crs = raster.rio.crs
            else: 
                s_crs = None

            gdf = ug.compute.create_chip_bounds_gdf(input_dims = self.input_dims, shape_x = shape[2], shape_y=shape[1], crs = s_crs,**self.kwargs )


            yield raster, gdf     

@functional_datapipe("img_gdf_coords_to_crs_coords")
class ImgGDFRefToCRSGDFRefIterDataPipe(IterDataPipe):
 
    def __init__(
        self,
        source_datapipe: IterDataPipe,

        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        if gpd is None:
            raise ModuleNotFoundError(
                "Package `geopandas is required to be installed to use this datapipe. "
                "Please use `pip install geopandas` "
                "to install the package"
            )

        
        self.source_datapipe: IterDataPipe = source_datapipe
        self.kwargs = kwargs


    def __iter__(self) -> Iterator:

        for raster, gdf  in self.source_datapipe:

            gdf["x_geom"] = gdf["geometry"].apply(lambda x: ug.compute.imgref_to_crs(raster,x))
            gdf = gdf.set_geometry(gdf["x_geom"])       

            yield raster, gdf 

        
@functional_datapipe("chip_raster_from_gdf")
class ChipRasterFromGDFBoxesIterDataPipe(IterDataPipe):
 
    def __init__(
        self,
        source_datapipe: IterDataPipe,
        force_crs = False,

        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        if gpd is None:
            raise ModuleNotFoundError(
                "Package `geopandas is required to be installed to use this datapipe. "
                "Please use `pip install geopandas` "
                "to install the package"
            )

        
        self.source_datapipe: IterDataPipe = source_datapipe
        self.force_crs = force_crs

        self.kwargs = kwargs



    def __iter__(self) -> Iterator:

        for raster, gdf  in self.source_datapipe:
            if self.force_crs:
                try:
                    # make it fit the rio formatting: might fail due to x and y not existing
                    raster = raster.assign_coords(x=raster.x, y = raster.y, spatial_ref = 0) 
                    raster = raster.write_crs(gdf.crs)
                except: pass
                
            for row in gdf.iterrows():

                chip = raster.rio.clip_box(minx = row.geometry.bounds[0], miny =row.geometry.bounds[1] ,maxx =row.geometry.bounds[2] , maxy= row.geometry.bounds[3])  

            yield chip, row     
