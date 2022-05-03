from readPTU_FLIM import PTUreader
import numpy as np
from matplotlib import pyplot as plt
# for choosing files
from tkinter import Tk
from tkinter.filedialog import askopenfilename
# directory manipulations
import os.path
# for saving objs
import pickle
import dask.array as dr

from PIL import Image

dir = "L:/880_FLIM/paula_zhu/convallaria/220408_convallaria/FLIM/220408_convallaria_run2/"
filename = "220408_convallaria_2.ptu"

# read in ptu data
ptu_file = PTUreader(dir+filename, print_header_data = True)
ptu_head = ptu_file.head

# save ptu_file headers
f = open(dir + 'ptu_head.pickle', 'ab')
pickle.dump(ptu_head, f) 

# Raw data can be accessed using ptu_file object
sync    = ptu_file.sync    # Macro photon arrival time
# tcspc   = ptu_file.tcspc   # Micro photon arrival time (tcspc time bin resolution)
# channel = ptu_file.channel # Detection channel of tcspc unit (<=8 for PQ hardware in 2019)
special = ptu_file.special # Special event markers, for e.g. Frame, LineStart, LineStop, etc.

#Get FLIM data stack
print('getting flim data stack...')
ptu_image = ptu_file.get_flim_data_stack()

# save FLIM data stack
# np.save(output_dir + '/' + 'ptu_image', ptu_image)
print("calculating data stack channels...")
ptu_image_0 = ptu_image[:, :, 0, :].squeeze()
ptu_image_1 = ptu_image[:, :, 1, :].squeeze()
ptu_image_total = ptu_image.sum(axis=2)
# print('saving flim data stacks...')
# np.save(dir + 'ptu_image_0', ptu_image_0)
# np.save(dir + 'ptu_image_1', ptu_image_1)

print('getting intensity images...')
# 2 channel image
if ptu_image.ndim == 4:
    int_image = np.sum(ptu_image, axis = 3) # sum across tcspc bin
    int_image_total = np.sum(int_image, axis  = 2) # sum across spectral channels

    int_image_0 = np.zeros([int(int_image.shape[0]), int(int_image.shape[1])])
    int_image_1 = np.zeros([int(int_image.shape[0]), int(int_image.shape[1])])
    # channel1 = 'red'
    int_image_0 = int_image[:, :, 0]
    # channel2 = 'green'
    int_image_1 = int_image[:, :, 1]

    # save color image
    print("saving intensity images...")
    np.save(dir + 'int_image', int_image_total)
    np.save(dir + 'int_image_0', int_image_0)
    np.save(dir + 'int_image_1', int_image_1)

# single channel image
elif ptu_image.ndim == 3:
    int_image_total  = np.sum(ptu_image, axis = 3) # sum across tcspc bin, only 1 detection channel
    np.save(dir + 'int_image', int_image_total)

print("getting time bins...")
# get x axis time intervals
# in nanoseconds (1e9)
time_bins = np.linspace(0,ptu_image_0.shape[2],ptu_image_0.shape[2], dtype = np.int)*(ptu_head["MeasDesc_Resolution"]*1e9)

# no sr
# def as_strided2d(a, K):
#     b = a
#     view = np.lib.stride_tricks.as_strided(a,
#         shape=(K, K, a.shape[0] - (K-1), a.shape[1] - (K-1)),
#         strides=a.strides * 2
#     )
#     b[2:-2, 2:-2] = view.sum(axis=(0, 1))
#     return b

# # local binning size
# K = 5

# print('starting sr...')
# # for all along time bin axis
# ptu_image_sr = np.array([as_strided2d(slice, K) for slice in np.rollaxis(ptu_image.values, 2)])
# print('finished, rolling axis...')
# ptu_image_s = np.rollaxis(ptu_image_sr, 0, start=3)


def fm_analysis(ptu_image, int_image, time_bins_np, phot_cut=0):
    # int_image = ptu_image.sum(axis=2)
    # First Moment Quick Estimation
    # time_bins = ptu_image['microtime_ns'].to_numpy()
    flim_mult = ptu_image * time_bins_np
    flim_mult_sum = flim_mult.sum(axis=2)
    initial_zeros = dr.zeros_like(flim_mult_sum)
    return np.divide(flim_mult_sum, int_image, out=initial_zeros, where=int_image>phot_cut)

print("calculating fm analysis...")
fm_image = fm_analysis(ptu_image_total, int_image_total, time_bins, 4)
fm_image_0 = fm_analysis(ptu_image_0, int_image_0, time_bins, 4)
fm_image_1 = fm_analysis(ptu_image_1, int_image_1, time_bins, 4)

print('saving fm images...')
np.save(dir+'fm_image', fm_image)
np.save(dir+'fm_image_0', fm_image_0)
np.save(dir+'fm_image_1', fm_image_1)

def combine_hsv(flim_image, frange, int_image, imax):
    return np.stack(( np.clip( (flim_image - frange[1]) * 255.0/(frange[0]-frange[1]), 0, 255), \
    255*np.ones_like(flim_image), \
    np.clip(int_image * 255.0/imax, 0, 255) ))

print("combining fm and int images...")
x = combine_hsv(fm_image, [1, 5], int_image_total, np.percentile(int_image_total, 99.9))
combined_image = np.rollaxis(x, 0, start=3)
x_0 = combine_hsv(fm_image_0, [1, 5], int_image_0, np.percentile(int_image_0, 99.9))
combined_image_0 = np.rollaxis(x_0, 0, start=3)
x_1 = combine_hsv(fm_image_1, [1, 5], int_image_1, np.percentile(int_image_1, 99.9))
combined_image_1 = np.rollaxis(x_1, 0, start=3)
print('saving combined images...')
np.save(dir+'combined_image', combined_image)
np.save(dir+'combined_image_0', combined_image_0)
np.save(dir+'combined_image_1', combined_image_1)
print('combined images saved')

# xpil = Image.fromarray(combined_image.astype('uint8'), mode='HSV')
# im = xpil.convert('RGB')
# xpil.show()
