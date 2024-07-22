import xarray as xr
import math
import numpy as np    
   
def rescale_floats(arr) -> xr.DataArray:
    """
    rescales the input values to fit within min-mac of 0 to 255
    """
    return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8',casting = "unsafe")

def rescale_index(arr) -> xr.DataArray:
    """
    rescales the input values to fit within -1 and 1 to 0 to 255
    """
    min = -1
    max= 1
    return ((arr - min) * (1/(max - min) * 255)).astype('uint8',casting = "unsafe")

def calc_ndvi(bandstack:xr.DataArray,red_id=1, nir_id=3, rescale =True):
    """
    Normalized Difference Vegetation Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, appending the band ndvi.
    Assuming band 1 is RED, band 2 is NIR
    Rescale sets the min-max to 0-255
    """


    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=red_id)
    nir: xr.DataArray = ds_b.sel(band=nir_id)
    
    ndvi = (nir-red)/(nir+red)
    ndvi.name = "ndvi"
    if rescale:
        ndvi = rescale_index(ndvi)
    return ndvi


def calc_vndvi(bandstack:xr.DataArray,red_id=1, green_id = 2, blue_id=3, rescale =True):
    """
    Normalized Difference Vegetation Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, appending the band ndvi.
    Assuming band 1 is RED, band 2 is NIR
    Rescale sets the min-max to 0-255
    """


    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=red_id)
    green: xr.DataArray = ds_b.sel(band=green_id)
    blue: xr.DataArray = ds_b.sel(band=blue_id)
    
    vndvi =  0.5268 * (red** -0.1294) * (green ** 0.3389) * (blue ** -0.3118)
    vndvi.name = "vndvi"
    if rescale:
        vndvi = rescale_index(vndvi)
    return vndvi

def calc_ndre(bandstack:xr.DataArray,rededge_id=1, nir_id=3, rescale =True):
    """
    Normalized Difference Red Edge Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, appending the band ndvi.
    Assuming band 1 is rededge, band 2 is NIR
    Rescale sets the min-max to 0-255
    """


    ds_b = bandstack.astype(float)
    rededge: xr.DataArray = ds_b.sel(band=rededge_id)
    nir: xr.DataArray = ds_b.sel(band=nir_id)
    
    ndre = (nir-rededge)/(nir+rededge)
    ndre.name = "ndre"
    if rescale:
        ndre = rescale_index(ndre)
    return ndre

def calc_gndvi(bandstack:xr.DataArray,greem_id=1, nir_id=3, rescale =True):
    """
    Green Normalized Difference Vegetation Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, appending the band hndvi.
    Assuming band 1 is green, band 2 is NIR
    Rescale sets the min-max to 0-255
    """


    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=red_id)
    nir: xr.DataArray = ds_b.sel(band=nir_id)
    
    gndvi = (nir-green)/(nir+green)
    gndvi.name = "gndvi"
    if rescale:
        gndvi = rescale_index(gndvi)
    return gndvi
    
def calc_bi(bandstack:xr.DataArray,red_id=1,green_id=2,blue_id=3, rescale =True):
    """
    Brightness Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in bi.
    Assuming band 1 is RED, band 2 is green, and blue is 3
    Rescale sets the min-max to 0-255, default = True
    """
    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=red_id)
    green: xr.DataArray = ds_b.sel(band=green_id)
    blue: xr.DataArray = ds_b.sel(band=blue_id)
    bi = math.sqrt((red**2 + green**2 + blue**2)/3)
    bi.name = "bi"
    if rescale:
        bi = rescale_index(bi)
    return bi
    
def calc_sci(bandstack:xr.DataArray,red_id=1,green_id=2, rescale = True):
    """
    Soil color Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in sci.
    Assuming band 1 is RED, band 2 is green
    Rescale sets the min-max to 0-255, default = True
    """
    

    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=red_id)
    green: xr.DataArray = ds_b.sel(band=green_id)
    sci = (red-green)/(red+green)
    sci.name = "sci"
    if rescale:
        sci = rescale_index(sci)
    return sci
    
def calc_gli(bandstack:xr.DataArray,red_id=1,green_id=2,blue_id=3, rescale = True):
    """
    Green Leaf index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in gli.
    Assuming band 1 is RED, band 2 is green, and blue is 3
    Rescale sets the min-max to 0-255, default = True
    """
    

    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=red_id)
    green: xr.DataArray = ds_b.sel(band=green_id)
    blue: xr.DataArray = ds_b.sel(band=blue_id)
    gli = ((2*green)-red-blue)/((2*green)+red+blue)
    gli.name = "gli"
    if rescale:
        gli = rescale_index(gli)
    return gli

def calc_hi(bandstack:xr.DataArray,red_id=1,green_id=2,blue_id=3):
    """
    Primary colours Hue Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in gli.
    Assuming band 1 is RED, band 2 is green, and blue is 3
    Rescale sets the min-max to 0-255, default = True
    """
    

    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=red_id)
    green: xr.DataArray = ds_b.sel(band=green_id)
    blue: xr.DataArray = ds_b.sel(band=blue_id)
    hi = (2*(red-green-blue)/(green-blue))
    hi.name = "hi"
    if rescale:
        hi = rescale_index(hi)
    return hi

def calc_ngrdi(bandstack:xr.DataArray,red_id=1,green_id=2,rescale=True):
    """
    Normalized green red difference index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in gli.
    Assuming band 1 is RED, band 2 is green
    Rescale sets the min-max to 0-255, default = True
    """
    
    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=red_id)
    green: xr.DataArray = ds_b.sel(band=green_id)

    ngrdi = (green-red)/(green+red)
    ngrdi.name = "ngrdi"
    if rescale:
        ngrdi = rescale_index(ngrdi)
    return ngrdi

def calc_si(bandstack:xr.DataArray,red_id=1,blue_id=2,rescale=True):
    """
    Spectral slope saturation index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in si.
    Assuming band 1 is RED, band 2 is blue
    Rescale sets the min-max to 0-255, default = True
    """
    
    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=red_id)
    blue: xr.DataArray = ds_b.sel(band=blue_id)

    si = (red-blue)/(blue+red)
    si.name = "si"
    if rescale:
        si = rescale_index(si)
    return ngrdi

def calc_vari(bandstack:xr.DataArray,red_id=1,green_id=2,blue_id=3,rescale=True):
    """
    Visible Atmospherically Resistant Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in vari.
    Assuming band 1 is RED, band 2 is green, and blue is 3
    Rescale sets the min-max to 0-255, default = True
    """
    
    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=red_id)
    green: xr.DataArray = ds_b.sel(band=green_id)
    blue: xr.DataArray = ds_b.sel(band=blue_id)
    vari = (green-red)/(green+red-blue)
    vari.name = "vari"
    if rescale:
        vari = rescale_index(vari)
    return vari

def calc_hue(bandstack:xr.DataArray,red_id=1,green_id=2,blue_id=3,rescale=True):
    """
    Overall Hue Index (adapted from FieldImageR)
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in hue.
    Assuming band 1 is RED, band 2 is green, and blue is 3
    Rescale sets the min-max to 0-255, default = True
    """
    
    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=red_id)
    green: xr.DataArray = ds_b.sel(band=green_id)
    blue: xr.DataArray = ds_b.sel(band=blue_id)
    hue = math.atan((2*(blue-green-red))/(30.5*(green-red)))
    hue.name = "hue"
    if rescale:
       hue = rescale_index(hue)
    return hue

def calc_bi(bandstack:xr.DataArray,green_id=2,blue_id=3,rescale=True):
    """
    Blue green Pigment Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in bi.
    Assuming band 2 is green, and blue is 3
    Rescale sets the min-max to 0-255, default = True
    """
    
    ds_b = bandstack.astype(float)
    green: xr.DataArray = ds_b.sel(band=green_id)
    blue: xr.DataArray = ds_b.sel(band=blue_id)
    bi = math.atan((2*(blue-green-red))/(30.5*(green-red)))
    bi.name = "bi"
    if rescale:
       bi = rescale_index(bi)
    return bi

def calc_psri(bandstack:xr.DataArray,red_id=1,green_id=2,rededge_id=3,rescale=True):
    """
    Plant Scenescence Reflectance Index 
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in psri.
    Assuming band 1 is RED, band 2 is green, and rededge is 3
    Rescale sets the min-max to 0-255, default = True
    """
    
    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=red_id)
    green: xr.DataArray = ds_b.sel(band=green_id)
    rededge: xr.DataArray = ds_b.sel(band=blue_id)
    psri = (red-green)/rededge
    psri.name = "psri"
    if rescale:
       psri= rescale_index(psri)
    return psri

def calc_rvi(bandstack:xr.DataArray,red_id=1,nir_id=3,rescale=True):
    """
    Ratio Vegetation Index 
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in rvi.
    Assuming band 1 is RED, band 3 is nir
    Rescale sets the min-max to 0-255, default = True
    """
    
    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=red_id)
    nir: xr.DataArray = ds_b.sel(band=nir_id)
    rvi = (red-green)/rededge
    rvi.name = "rvi"
    if rescale:
       rvi= rescale_index(rvi)
    return rvi

def calc_tvi(bandstack:xr.DataArray,red_id=1, green_id=2, nir_id=4, rescale =True):
    """
    Triangular Vegetation Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in tvi.
    Assuming band 1 is red, band 2 is green, band 4 is nir 
    Rescale sets the min-max to 0-255
    """


    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=rededge_id)
    nir: xr.DataArray = ds_b.sel(band=nir_id)
    green: xr.DataArray = ds_b.sel(band=green_id)

    tvi = 0.5 * (120*(nir-green)-200*(red-green))
    tvi.name = "tvi"
    if rescale:
        tvi = rescale_index(tvi)
    return tvi

def calc_cvi(bandstack:xr.DataArray,red_id=1, green_id=2, nir_id=4, rescale =True):
    """
    Chlorophyll Vegetation Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, appending the band ndvi.
    Assuming band 1 is red, band 2 is green, band 4 is nir 
    Rescale sets the min-max to 0-255
    """


    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=rededge_id)
    nir: xr.DataArray = ds_b.sel(band=nir_id)
    green: xr.DataArray = ds_b.sel(band=green_id)

    cvi = (nir*red)/(green**2)
    cvi.name = "cvi"
    if rescale:
        cvi = rescale_index(cvi)
    return cvi

def calc_evi(bandstack:xr.DataArray,red_id=1, blue_id=2, nir_id=4, rescale =True):
    """
    Enhanced Vegetation Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in evi.
    Assuming band 1 is red, band 2 is green, band 4 is nir 
    Rescale sets the min-max to 0-255
    """


    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=rededge_id)
    nir: xr.DataArray = ds_b.sel(band=nir_id)
    blue: xr.DataArray = ds_b.sel(band=blue_id)

    evi = 2.5*(nir-red)/(nir+6*red-7.5*blue+1)
    evi.name = "evi"
    if rescale:
        evi = rescale_index(evi)
    return evi

def calc_cig(bandstack:xr.DataArray,green_id=1, nir_id=2, rescale =True):
    """
    Chlorophyll Index Green
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in cig.
    Assuming band 1 is green, band 2 is nir 
    Rescale sets the min-max to 0-255
    """


    ds_b = bandstack.astype(float)
    green: xr.DataArray = ds_b.sel(band=green_id)
    nir: xr.DataArray = ds_b.sel(band=nir_id)

    cig = (nir/green)-1 
    cig.name = "cig"
    if rescale:
        cig = rescale_index(cig)
    return cig

def calc_cire(bandstack:xr.DataArray,rededge=1, nir_id=2, rescale =True):
    """
    Chlorophyll Index RedEdge
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in cire.
    Assuming band 1 is rededge, band 2 is nir 
    Rescale sets the min-max to 0-255
    """


    ds_b = bandstack.astype(float)
    rededge: xr.DataArray = ds_b.sel(band=rededge_id)
    nir: xr.DataArray = ds_b.sel(band=nir_id)

    cire = (nir/rededge)-1 
    cire.name = "cire"
    if rescale:
        cire = rescale_index(cire)
    return cire

def calc_dvi(bandstack:xr.DataArray,rededge=1, nir_id=2, rescale =True):
    """
    Difference Vegetation Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands, resulting in dvi.
    Assuming band 1 is rededge, band 2 is nir 
    Rescale sets the min-max to 0-255
    """


    ds_b = bandstack.astype(float)
    rededge: xr.DataArray = ds_b.sel(band=rededge_id)
    nir: xr.DataArray = ds_b.sel(band=nir_id)

    dvi = nir-rededge 
    dvi.name = "dvi"
    if rescale:
        dvi = rescale_index(dvi)
    return dvi

def calc_savi(bandstack:xr.DataArray,red_id=1, nir_id=3, l=0.5, rescale =True):
    """
    Normalized Difference Water Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands resutling in savi.
    Assuming band 1 is red, band 3 is NIR, l is set as 0.5 by default
    Rescale sets the min-max to 0-255
    """


    ds_b = bandstack.astype(float)
    nir: xr.DataArray = ds_b.sel(band=nir_id)
    red: xr.DataArray = ds_b.sel(band=red_id)
    
    savi = ((nir-red)/(nir+red+l))*(1+l) 
    savi.name = "savi"
    if rescale:
        savi = rescale_index(savi)
    return savi

def calc_ndwi(bandstack:xr.DataArray,green_id=1, nir_id=3, rescale =True):
    """
    Normalized Difference Water Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands resutling in ndwi.
    Assuming band 1 is green, band 3 is NIR
    Rescale sets the min-max to 0-255
    """


    ds_b = bandstack.astype(float)
    nir: xr.DataArray = ds_b.sel(band=nir_id)
    green: xr.DataArray = ds_b.sel(band=green_id)
    
    ndwi = (green-nir)/(nir+green)
    ndwi.name = "ndwi"
    if rescale:
        ndwi = rescale_index(ndwi)
    return ndwi

def calc_mndwi(bandstack:xr.DataArray,green_id=1, swir_id=3, rescale =True):
    """
    Modified Normalized Difference Water Index
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands resutling in mndwi.
    Assuming band 1 is green, band 3 is swir. Can be used on both SWIR bands of Landsat
    Rescale sets the min-max to 0-255
    """


    ds_b = bandstack.astype(float)
    red: xr.DataArray = ds_b.sel(band=red_id)
    green: xr.DataArray = ds_b.sel(band=green_id)
    
    mndwi = (green-swir)/(swir+green)
    mndwi.name = "mndwi"
    if rescale:
        mndwi = rescale_index(mndwi)
    return mndwi

def calc_aweish(bandstack:xr.DataArray,blue_id=1, green_id=2, nir_id=3, swir1_id=4, swir2_id=5, rescale =True):
    """
    Automated water extraction Index (sh)
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands resutling in awei-sh
    Assuming band 1 is green, band 2 is NIR
    Rescale sets the min-max to 0-255
    """


    ds_b = bandstack.astype(float)
    blue: xr.DataArray = ds_b.sel(band=blue_id)
    green: xr.DataArray = ds_b.sel(band=green_id)
    nir: xr.DataArray = ds_b.sel(band=nir_id)
    swir1: xr.DataArray = ds_b.sel(band=swir1_id)
    swir2: xr.DataArray = ds_b.sel(band=swir2_id)

    
    awei = blue+2.5*green - 1.5* (nir-swir1) - 0.25 * swir2
    awei.name = "awei-sh"
    if rescale:
        awei = rescale_index(awei)
    return awei

def calc_aweinsh(bandstack:xr.DataArray, green_id=2, nir_id=3, swir1_id=4,rescale =True):
    """
    Automated water extraction Index (nsh)
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands resutling in awei-nsh
    Assuming band 1 is green, band 3 is nir, band 4 is swir1, band 5 is swir2
    Rescale sets the min-max to 0-255
    """


    ds_b = bandstack.astype(float)

    green: xr.DataArray = ds_b.sel(band=green_id)
    nir: xr.DataArray = ds_b.sel(band=nir_id)
    swir1: xr.DataArray = ds_b.sel(band=swir1_id)

    awei = 4*(green-swir1) - (0.25*nir+2.75*swir1)
    awei.name = "awei-nsh"
    if rescale:
        awei = rescale_index(awei)
    return awei

def calc_custom(bandstack:xr.DataArray, func, rescale=True):
    
    ds_b = bandstack.astype(float)

    custom = func(ds_b)
    
    if rescale:
        custom = rescale_index(custom)
    return custom

def calc_rgbvi(bandstack:xr.DataArray, red_id = 1 ,green_id=2,blue_id=3,rescale =True):
    """
    Automated water extraction Index (nsh)
    Combine a xarray.DataArray (MS BANDS) inputs into an xarray.Dataset with 
    data variables named bands resutling in awei-nsh
    Assuming band 1 is green, band 3 is nir, band 4 is swir1, band 5 is swir2
    Rescale sets the min-max to 0-255
    """
    ds_b = bandstack.astype(np.float32)
    red: xr.DataArray = ds_b.sel(band=red_id)
    green: xr.DataArray = ds_b.sel(band=green_id)
    blue: xr.DataArray = ds_b.sel(band=blue_id)
    rgbvi = ((green**2)-(red*blue))/((green**2)+(red*blue))
    rgbvi.name = "rgbvi"
    if rescale:
        rgbvi = rescale_index(rgbvi)
    return rgbvi