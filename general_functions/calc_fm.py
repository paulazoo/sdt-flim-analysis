import numpy as np
import dask.array as dr
from PIL import Image 

dir = "L:/880_FLIM/paula_zhu/hela/dish1_cont/"

tpc_image = np.load(dir+'tpc_image1_s.npy')
int_image = np.load(dir+'int_image1_s.npy')
time_bins = np.load(dir+'time_bins.npy')

def fm_analysis(tpc_image, int_image, time_bins_np, phot_cut=0):
    # int_image = tpc_image.sum(axis=2)
    # First Moment Quick Estimation
    # time_bins = tpc_image['microtime_ns'].to_numpy()
    flim_mult = tpc_image * time_bins_np
    flim_mult_sum = flim_mult.sum(axis=2)
    initial_zeros = dr.zeros_like(flim_mult_sum)
    return np.divide(flim_mult_sum, int_image, out=initial_zeros, where=int_image>phot_cut)

fm_image1 = fm_analysis(tpc_image, int_image, time_bins, 4)
np.save(dir+'fm_image1', fm_image1)

