dir = "L:/880_FLIM/paula_zhu/hela/dish3_fasted/"
pc_dir = "C:/Users/paz279/Desktop/coding/"

# Scripts\activate.bat
# sklearn
# h5py
import dask.array as dr
# from dask.distributed import Client
import glob
import imageio as iio
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

from PIL import Image

import xarray as xr
xr.set_options(**{
    "display_expand_attrs": False,
    "display_expand_data": False,
})

import zarr
# pip install numcodecs==0.10.0a2 for python 3.10

sys.path.append(pc_dir + "pyTCSPC")
import pyTCSPC as pc

# load sdt image
print("loading sdt file...")
sdt_loaded = pc.load_sdt(dir + 'img1.sdt')
# [_, channel (M1, M2), x, y, bins (ns)]

# time bins
print("getting time bins...")
time_bins = sdt_loaded['microtime_ns'].to_numpy()
np.save('time_bins', time_bins)

sdt_image = sdt_loaded.sel(channel="M1").squeeze()
# [x, y, bins (ns)]
# np.save(dir+'sdt_image1', sdt_image)

print("calculating intensity image...")
int_image = sdt_image.sum(dim="microtime_ns")
# [x, y]
# save
print("saving intensity image...")
np.save(dir+'int_image1', int_image)

def as_strided2d(a, K):
    b = a
    view = np.lib.stride_tricks.as_strided(a,
        shape=(K, K, a.shape[0] - (K-1), a.shape[1] - (K-1)),
        strides=a.strides * 2
    )
    b[2:-2, 2:-2] = view.sum(axis=(0, 1))
    return b

# local binning size
K = 5

print('starting sr...')
# for all along time bin axis
sdt_image_sr = np.array([as_strided2d(slice, K) for slice in np.rollaxis(sdt_image.values, 2)])
print('finished, rolling axis...')
sdt_image_s = np.rollaxis(sdt_image_sr, 0, start=3)

# print('np saving sdt result...')
# np.save(dir+"sdt_image1_s", sdt_image_s)

print('calculating int image...')
int_image_s = sdt_image_s.sum(axis=2)
print('np saving int image of sdt result...')
np.save(dir+"int_image1_s", int_image_s)


def fm_analysis(sdt_image, int_image, time_bins_np, phot_cut=0):
    # int_image = sdt_image.sum(axis=2)
    # First Moment Quick Estimation
    # time_bins = sdt_image['microtime_ns'].to_numpy()
    flim_mult = sdt_image * time_bins_np
    flim_mult_sum = flim_mult.sum(axis=2)
    initial_zeros = dr.zeros_like(flim_mult_sum)
    return np.divide(flim_mult_sum, int_image, out=initial_zeros, where=int_image>phot_cut)

print("calculating fm analysis...")
fm_image1 = fm_analysis(sdt_image_s, int_image_s, time_bins, 4)

print('saving fm image...')
np.save(dir+'fm_image1', fm_image1)

def combine_hsv(flim_image, frange, int_image, imax):
    return np.stack(( np.clip( (flim_image - frange[1]) * 255.0/(frange[0]-frange[1]), 0, 255), \
    255*np.ones_like(flim_image), \
    np.clip(int_image * 255.0/imax, 0, 255) ))

print("combining fm and int image...")
x = combine_hsv(fm_image1, [2.8, 5], int_image_s, np.percentile(int_image_s, 99.9)) # 2.807 ns is the FM of the IRF
combined_image1 = np.rollaxis(x, 0, start=3)
print('saving combined image...')
np.save(dir+'combined_image1', combined_image1)
print('combined image saved')

# xpil = Image.fromarray(combined_image1.astype('uint8'), mode='HSV')
# # im = xpil.convert('RGB')
# xpil.show()
