<div align="center">
  <p>
    <a href="https://icaerus.eu" target="_blank">
      <img width="50%" src="https://icaerus.eu/wp-content/uploads/2022/09/ICAERUS-logo-white.svg"></a>
    <h3 align="center">uavgeo ⛰️</h3>
    
   <p align="center">
    A UAV-specific Python image processing library built upon <i>xarray</i> and <i>geopandas</i>.
    <br/>
    <br/>
    <a href="https://github.com/icaerus-eu/uavgeo/wiki"><strong>Explore the wiki »</strong></a>
    <br/>
    <br/>
    <a href="https://github.com/icaerus-eu/uavgeo/issues">Report Bug</a>
    .
    <a href="https://github.com/icaerus-eu/uavgeo/issues">Request Feature</a>
  </p>
</p>
</div>

![Downloads](https://img.shields.io/github/downloads/icaerus-eu/uavgeo/total) ![Contributors](https://img.shields.io/github/contributors/icaerus-eu/uavgeo?color=dark-green) ![Forks](https://img.shields.io/github/forks/icaerus-eu/uavgeo?style=social) ![Stargazers](https://img.shields.io/github/stars/icaerus-eu/uavgeo?style=social) ![Issues](https://img.shields.io/github/issues/icaerus-eu/uavgeo) ![License](https://img.shields.io/github/license/icaerus-eu/uavgeo) 

## Table Of Contents

* [Summary](#summary)
* [Features](#features)
  * [Index calculation](#index-calculations)
  * [Dataset chipping](#dataset-chipping)
  * [DTM creation](#creating-a-dtm-with-the-calc_dem_from_dsm-function)
  * [Dataset shadows](#shadow-calculations)
* [Installation](#installation)
  
## Summary
UAV image analysis is a powerful tool to gain valuable insights into the rural, urban and natural environment. Especially in conjunction with Deep Learning, large strides can be made. The problem however is that there is little standardization and a lot of boilerplate code to be written for image analysis. This package serves to bridge the gap in image processing and machine learning in UAV applications. It builds upon the efforts from `xarray`, `rasterio/rioxarray` and `geopandas`. 
Importing should be done through rioxarray functions. The currently implemented functions in `uavgeo` examples cover index calculation (see below) and data/rastser chipping and reconstruction.

## Features
The `uavgeo` package can be installed through `pip`. Additionally, a docker container with jupyterlab can be used. See the Installation section for more information.

### Dataset chipping
![chipped image](https://github.com/icaerus-eu/uavgeo/blob/main/docs/images/chipped.png?height=200)

Chipping is a prerequisite for geographic raster data to be processed for ML/DL models.
This library implements it as follows:
1. creating a chips-geodataframe based on wanted dimensions, overlap and raster shape
2. chipping the input raster into a list of chips
3. reset the coordinates from crs to image pixels (numpy assumed dimensions)
4. export the list of images to file (or do whatever)
5. (optional): perform the ML modelling on the chips
6. (optional): reconstruct the images back to the original raster and crs

This whole pipeline and functions are presented in an [example notebook](https://github.com/icaerus-eu/uavgeo/blob/main/notebooks/chipping_examples.ipynb)

### Creating a DTM with the `calc_dtm_from_dsm` function
![dtmdsmchm](https://github.com/icaerus-eu/uavgeo/blob/main/docs/images/dsmdtmchm.png?height=200)
The `calc_dtm_from_dsm` function is a utility to create a Digital Terrain Model (DTM) from a Digital Surface Model (DSM) using specified sampling parameters. It operates on data represented as xarray DataArray and relies on the rasterio library for Geographic Information System (GIS) operations. The resulting DTM is created by sampling and extracting the minimum elevation values from the DSM at a user-defined grid, built upon the chipping presented above.

An example can be found in an [example notebook](https://github.com/icaerus-eu/uavgeo/blob/main/notebooks/create_dem_dtm_chm_examples.ipynb)

Inputs:
- `dsm (xr.DataArray)`: The input Digital Surface Model as an xarray DataArray.
- `pixel_size (float)`: The pixel size in the same unit as the DSM data.
- `sampling_meters (float)`: The distance in meters that defines the sampling grid for DEM creation.

```python
import uavgeo as ug
import rioxarrary as rxr

dsm_data = rxr.open_rasterio(('dsm.tif')  # Load DSM data from a GeoTIFF file
pixel_size = 1.0  # Specify the pixel size in meters
sampling_distance = 10.0  # Define the sampling distance in meters
dtm = ug.compute.calc_dtm_from_dsm(dsm_data, pixel_size, sampling_distance)  # Calculate the DEM
chm = dtm-dsm_data # subtract the two, and you have a Canopy Height Model too
```

### Index calculations
You can use it to calculate a variety of indices from your imagery:
```python
# assuming you already loaded your data as ortho:
import uavgeo as ug

savi  = ug.compute.calc_savi(bandstack = ortho, red_id=1, nir_id=4, l = 0.51)
savi.plot.imshow(cmap = "greens")
```

#### Implemented indices:
Based on the list from [FieldImageR](https://www.opendronemap.org/fieldimager/). With some additional indices added.
They can be accesses through the `uavgeo.compute` module. All functions expect a `bandstack`, which is an `xarray.DataArray`wityh multiple bands as`bands` data. And the required bands ids, eg.: `red_id=1`. By default the functions rescale the output floats back to uint8 (0-255). This behaviour can be turned of with the `rescale = False` parameter.

| Index | calc_indexname | Description | Formula | Related Traits | References |
|-------|----------------|-------------|---------|----------------|------------|
| BI    | `calc_bi`        | Brightness Index | sqrt((RA^2+GA^2+B^2)/3) | Vegetation coverage, water content | Richardson and Wiegand (1977) |
| SCI   | `calc_sci`       | Soil Color Index | (R-G)/(R+G) | Soil color | Mathieu et al. (1998) |
| GLI   | `calc_gli`       | Green Leaf Index | (2 * G-R-B)/(2 * G+R+B) | Chlorophyll | Louhaichi et al. (2001) |
| HI    | `calc_hi`        | Hue Index | (2*R-G-B)/(G-B) | Soil color | Escadafal et al. (1994) |
| NGRDI | `calc_ngrdi`     | Normalized Green Red Difference Index | (G-R)/(G+R) | Chlorophyll, biomass, water content | Tucker (1979) |
| SI    | `calc_si`        | Saturation Index | (R-B)/(R+B) | Soil color | Escadafal et al. (1994) |
| VARI  | `calc_vari`      | Visible Atmospherically Resistant Index | (G-R)/(G+R-B) | Canopy, biomass, chlorophyll | Gitelson et al. (2002) |
| HUE   | `calc_hue`       | Overall Hue Index# | atan(2*(B-G-R)/30.5*(G-R)) | Soil color | Escadafal et al. (1994) |
| BGI   | `calc_bgi`      | Blue Green Pigment Index | B/G | Chlorophyll | Zarco-Tejada et al. (2005) |
| PSRI  | `calc_psri`      | Plant Senescence Reflectance Index | (R-G)/(RE) | Chlorophyll, LAI | Merzlyak et al. (1999) |
| NDVI  | `calc_ndvi`      | Normalized Difference Vegetation Index | (NIR-R)/(NIR+R) | Chlorophyll, nitrogen, maturity | Rouse et al. (1974) |
| GNDVI | `calc_gndvi`     | Green Normalized Difference Vegetation Index | (NIR-G)/(NIR+G) | Chlorophyll, LAI, biomass, yield | Gitelson et al. (1996) |
| RVI   | `calc_rvi`       | Ratio Vegetation Index | NIR/R | Chlorophyll, LAI, nitrogen, protein content, water content | Pearson and Miller (1972) |
| NDRE  | `calc_ndre`      | Normalized Difference Red Edge Index | (NIR-RE)/(NIR+RE) | Biomass, water content, nitrogen | Gitelson and Merzlyak (1994) |
| TVI   | `calc_tvi`       | Triangular Vegetation Index | 0.5 * (120 * (NIR — G)-200 * (R — G)) | Chlorophyll | Broge and Leblanc (2000) |
| CVI   | `calc_cvi`       | Chlorophyll Vegetation Index | (NIR * R)/(GA^2) | Chlorophyll | Vincini et al. (2008) |
| EVI   | `calc_evi`       | Enhanced Vegetation Index | 2.5  *(NIR — R)/(NIR + 6 * R — 7.5 * B) | Nitrogen, chlorophyll | Huete et al. (2002) |
| CIG   | `calc_cig`       | Chlorophyll Index — Green | (NIR/G) — 1 | Chlorophyll | Gitelson et al. (2003) |
| CIRE  | `calc_cire`      | Chlorophyll Index — Red Edge | (NIR/RE) — 1 | Chlorophyll | Gitelson et al. (2003) |
| DVI   | `calc_dvi`       | Difference Vegetation Index | NIR-RE | Nitrogen, chlorophyll | Jordan (1969) |
| RGBVI   | `calc_rgbvi`       | RGB Vegetation Index | ((G^2)-(R * B)/(G^2)+(R * B)) | Nitrogen, chlorophyll | Bendig et al. (2015) |
| SAVI  | `calc_savi`      | Soil Adjusted Vegetation Index | (NIR-R)/(NIR+R+l)*(1+l) | Vegetation coverage, LAI | Huete (1988) |
|-------|----------------|-------------|---------|----------------|------------|
| NDWI  | `calc_ndwi`      | Normalized Difference Water Index | (G-NIR)/(G+NIR) | Water coverage, water content| McFeeters (1996) |
| MNDWI | `calc_mndwi`     | Modified Normalized Difference Water Index | (G-SWIR)/(GREEN+SWIR) | Water coverage, water content| McFeeters (1996) |
| AWEIsh | `calc_aweish`     | Automated water extraction index (sh) | B + 2.5 * G - 1.5 * (NIR-SWIR1) - 0.25 * SWIR2 | Water coverage, water content| Fayeisha (2014) |
| AWEInsh | `calc_aweinsh`     | Automated water extraction index (nsh) | 4 * (G - SWIR1) - (0.25 * NIR + 2.75* SWIR1) | Water coverage, water content| Fayeisha (2014) |

#### Custom/other spectral index:
You could also write your own index calculators, according to the following template:

```python
from uavgeo.compute import rescale_floats

def calc_custom(bandstack:xr.DataArray, band_a=1, band_b=2, rescale=True):
    
    ds_b = bandstack.astype(float)
    a: xr.DataArray = ds_b.sel(band=band_a)
    b: xr.DataArray = ds_b.sel(band=band_b)
    
    custom = a/b+1
    custom.name = "custom index"
    if rescale:
        custom = rescale_floats(custom)
    return custom
```

### Shadow calculations:
Calculate vineyard shadows using a method proposed by Velez et al. (2021) for vineyards and UAVs. Based on Kmeans. Velez et al. (2021). "A New Leaf Area Index Methodology Based on Unmanned Aerial Vehicle Imagery for Vineyards." https://oeno-one.eu/article/view/4639

```python
from uavgeo.compute import calc_vineyard_shadows

vineyard_data = xr.open_rasterio('path/to/vineyard_image.tif')
shadows_mask = calc_vineyard_shadows(vineyard_data)
# can also be used with a band_id to base the shadows from (NIR or Red are usally best)
shadows_mask = calc_vineyard_shadows(vineyard_data, band_id =4) 
```


## Installation:

It is built upon the work of `rioxarray`,  `geopandas`, `shapely` and a few more: see requirements.txt.
Additionally, when working with the object detection part, the `ultralytics` and `torch` libraries (`torch`, `torchvision`, `torchdata`) is also a prerequisite.
You can choose to install everything in a Python virtual environment or directly run a jupyterlab docker:

##### Option A: Setup directly in python:
0. Create a new environment (optional but recommended):
   
   ```bash
   conda create -n uavgeo_env python=3.10
   conda activate uavgeo_env
   ```
1.   Install the required dependencies:

        Using conda (not recommended):

        ```bash
        conda install -c conda-forge rioxarray geopandas shapely
        ```
        Using pip:
        ```bash
        pip install -f rioxarray geopandas shapely
        ```
2. Install this package (for now: pip only)
   ```bash
       pip install uavgeo
   ```

##### Option B: Setup through Docker:
This starts a premade jupyter environment with everything preinstalled, based around a nvidia docker image for DL support.
* Linux/Ubuntu:
  ```bash
  docker run --rm -it --runtime=nvidia -p 8888:8888 --gpus 1 --shm-size=5gb --network=host -v /path_to_local/dir:/home/jovyan jurrain/drone-ml:gpu-torch11.8-uavgeoformers
  ```

`--network=host` flag whether you want to run it on a different machine in the same network, and want to access the notebook. (does not run locally)

`-v` flag makes sure that once downloaded, it stays in that folder, accessible from the PC, and when restarting, all the weights etc. remain in that folder. `path_to_local/dir` is thew path to your working dir where you want to access the notebook from. can be `.` if you already `cd`ed into it.

` --runtime=nvidia` can be skipped when working on WSL2



* Windows requires WSL2 and NVIDIA drivers, WSL2 should also have the nvidia toolkit (for deep learning)
