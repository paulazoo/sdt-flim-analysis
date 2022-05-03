dir = "L:/880_FLIM/paula_zhu/hela/dish2_CCCP/"
pc_dir = "C:/Users/paz279/Desktop/coding/"

# Scripts\activate.bat
# sklearn
# h5py
import dask.array as dr
# from dask.distributed import Client
import glob
import imageio as iio
from importlib import reload
from itertools import chain
from joblib import Parallel, delayed

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import napari
import numpy as np
import os
import pandas as pd
from pathlib import Path, PurePath
import seaborn as sns
from skimage.filters import gaussian
from skimage.measure import find_contours
import sys

import xarray as xr
xr.set_options(**{
    "display_expand_attrs": False,
    "display_expand_data": False,
})

import zarr
# pip install numcodecs==0.10.0a2 for python 3.10

sys.path.append(pc_dir + "pyTCSPC")
import pyTCSPC as pc


# IRF
# irf_loaded = pc.load_sdt(dir + "calibration/IRF.sdt", dims="CXM", dtype=np.uint32)
# [_, channel (M1, M2), 1 pixel, bins (ns)]
# irf = irf_loaded.sel(channel="M1").squeeze()
# [bins (ns)]
# dc_kwargs_M1 = {
#     "trunc": True,
#     "peak_start": 2.6,
#     "peak_end": 3.85,
#     "bgsub": True,
#     "bg_start": 8,
#     "bg_end": 10,
#     "fig": fig,
#     "ax": ax
# }

# load sdt image
sdt_loaded = pc.load_sdt(dir + 'img1.sdt')
# [_, channel (M1, M2), x, y, bins (ns)]

# time bins
time_bins = sdt_loaded['microtime_ns'].to_numpy()
np.save('time_bins', time_bins)

sdt_image = sdt_loaded.sel(channel="M1").squeeze()
# [x, y, bins (ns)]
np.save(dir+'sdt_image1', sdt_image)

int_image = sdt_image.sum(dim="microtime_ns")
# [x, y]
# save
np.save(dir+'int_image1', int_image)

