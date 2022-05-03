
pc_dir = "C:/Users/paz279/Desktop/coding/"
calib_dir = "L:/880_FLIM/paula_zhu/hela/calibration/"

import dask.array as dr
# from dask.distributed import Client
import glob
import imageio as iio
from itertools import chain
from joblib import Parallel, delayed

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
import csv

import xarray as xr
xr.set_options(**{
    "display_expand_attrs": False,
    "display_expand_data": False,
})

import zarr
# pip install numcodecs==0.10.0a2 for python 3.10

sys.path.append(pc_dir + "pyTCSPC")
import pyTCSPC as pc

# IRF loading
print('Loading IRF...')
irf = pc.load_sdt(calib_dir + "IRF.sdt", dims="CXM", dtype=np.uint32)
fig, ax = plt.subplots(figsize=(12,4))
dc_kwargs_M1 = {
    "trunc": True,
    "peak_start": 2.6,
    "peak_end": 3.85,
    "bgsub": True,
    "bg_start": 8,
    "bg_end": 10,
    "fig": fig,
    "ax": ax
}
fig, ax, dc = pc.decay_curve(irf.sel(channel="M1"), plot=True, **dc_kwargs_M1)
dirs = ["L:/880_FLIM/paula_zhu/hela/dish1_cont/", "L:/880_FLIM/paula_zhu/hela/dish2_CCCP/", "L:/880_FLIM/paula_zhu/hela/dish3_fasted/"]

for dir in dirs:
    trials = ["img1", "img2", "img3"]
    # trial = trials[0]

    for trial in trials:
        print('loading new trial...')
        sdt_loaded = pc.load_sdt(dir + trial +'.sdt')
        print("calculating intensity image...")
        int_im = sdt_loaded.sel(channel="M1").squeeze().sum(axis=2).compute().data
        print("blurring and thresholding intensity image...")
        blur_im = gaussian(int_im, sigma=2)
        thresh_im = blur_im > np.percentile(blur_im, 90)

        # fig, ax = plt.subplots(figsize=(15,5), ncols=3)
        # ax[0].matshow(int_im)
        # ax[0].set_title("original intensity image")
        # ax[1].matshow(blur_im)
        # ax[1].set_title("smoothed intensity image");
        # ax[2].matshow(thresh_im)
        # ax[2].set_title("coarse thresholding of mitochondria")
        # plt.savefig(p.with_stem(p.stem + "_int_im").with_suffix(".png"), dpi=300, bbox_inches="tight")

        print("calculating decay curve...")
        dc = pc.decay_curve(sdt_loaded.sel(channel="M1").squeeze(), mask=thresh_im,).compute()
        dg = pc.decay_group(
            dc,
            irf.sel(channel="M1"),
            irf_kws=dc_kwargs_M1,
        )

        print("fitting 2 exp...")
        fitp, status = dg.fit(
            model="2exp",
            fixed_parameters=[],
            save_leastsq_params_array=True,
            verbose=False,
            plot=False,
        )

        fitp_value = fitp["value"].to_dict()
        fitp_err = fitp["err"].to_dict()

        result = {
            "filename": trial, 
            "photons_from": "mitochondria",
            "num_photons": dc.sum(),
            "intensity": dc.sum() / thresh_im.sum(),
            "fit_status": status,
            **fitp_value,
            **{item + "_err": fitp_err[item] for item in fitp_err},
        }

        print("appending result...")
        print(result)

        with open(dir+'calc_flim_' + trial + '.csv', 'w') as output:
            writer = csv.writer(output)
            for key, value in result.items():
                writer.writerow([key, value])
