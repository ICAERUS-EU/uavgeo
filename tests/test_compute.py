import unittest 
import uavgeo as ug
import xarray as xr
import numpy as np
import xarray as xr

# Define the values for each band
r_values = [
    [0.7, 0.8, 0.6, 0.4, 0.3, 0.7],
    [0.5, 0.6, 0.4, 0.2, 0.1, 0.5],
    [0.8, 0.9, 0.7, 0.5, 0.4, 0.8],
    [0.3, 0.4, 0.2, 0.1, 0.0, 0.3],
    [0.6, 0.7, 0.5, 0.3, 0.2, 0.6],
    [0.4, 0.5, 0.3, 0.1, 0.0, 0.4]
]

g_values = [
    [0.5, 0.6, 0.4, 0.2, 0.1, 0.5],
    [0.3, 0.4, 0.2, 0.1, 0.0, 0.3],
    [0.6, 0.7, 0.5, 0.3, 0.2, 0.6],
    [0.2, 0.3, 0.1, 0.0, 0.0, 0.2],
    [0.4, 0.5, 0.3, 0.1, 0.0, 0.4],
    [0.1, 0.2, 0.0, 0.0, 0.0, 0.1]
]

b_values = [
    [0.8, 0.9, 0.7, 0.5, 0.4, 0.8],
    [0.6, 0.7, 0.5, 0.3, 0.2, 0.6],
    [0.9, 1.0, 0.8, 0.6, 0.5, 0.9],
    [0.5, 0.6, 0.4, 0.2, 0.1, 0.5],
    [0.7, 0.8, 0.6, 0.4, 0.3, 0.7],
    [0.3, 0.4, 0.2, 0.1, 0.0, 0.3]
]

nir_values = [
    [0.2, 0.3, 0.1, 0.0, 0.0, 0.2],
    [0.1, 0.2, 0.0, 0.0, 0.0, 0.1],
    [0.4, 0.5, 0.3, 0.1, 0.0, 0.4],
    [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.2, 0.3, 0.1, 0.0, 0.0, 0.2],
    [0.0, 0.1, 0.0, 0.0, 0.0, 0.0]
]

# Create the DataArray
coords = {"y": range(6), "x": range(6)}
bands = ["R", "G", "B", "NIR"]
da = xr.DataArray(
    [r_values, g_values, b_values, nir_values],
    dims=["band", "y", "x"],
    coords=coords,
    name="RGBN",
    attrs={"bands": bands}
)


class TestCalcNDVI(unittest.TestCase):

    def test_calc_ndvi(self):
        # Expected NDVI values based on the provided formula
        expected_ndvi = np.array([[-0.55555556, -0.45454545, -0.71428571, -1.        , -1.        ,
        -0.55555556],
       [-0.66666667, -0.5       , -1.        , -1.        , -1.        ,
        -0.66666667],
       [-0.33333333, -0.28571429, -0.4       , -0.66666667, -1.        ,
        -0.33333333],
       [-1.        , -0.6       , -1.        , -1.        ,         np.nan,
        -1.        ],
       [-0.5       , -0.4       , -0.66666667, -1.        , -1.        ,
        -0.5       ],
       [-1.        , -0.66666667, -1.        , -1.        ,         np.nan,
        -1.        ]])

        # Call the function with the sample input
        result = ug.compute.calc_ndvi(da, red_id=0, nir_id=3, rescale=False)

        # Compare the result with the expected values
        np.testing.assert_array_almost_equal(result, expected_ndvi)

if __name__ == "__main__":
    unittest.main()