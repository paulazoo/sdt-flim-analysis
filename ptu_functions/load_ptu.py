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

# Choose a file
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
# output directory
output_dir = os.path.dirname(filename) + "/"

# read in ptu data
ptu_file = PTUreader(filename, print_header_data = True)
ptu_head = ptu_file.head

# save ptu_file headers
f = open(output_dir + 'ptu_head.pickle', 'ab')
pickle.dump(ptu_head, f) 

# Raw data can be accessed using ptu_file object
sync    = ptu_file.sync    # Macro photon arrival time
# tcspc   = ptu_file.tcspc   # Micro photon arrival time (tcspc time bin resolution)
# channel = ptu_file.channel # Detection channel of tcspc unit (<=8 for PQ hardware in 2019)
special = ptu_file.special # Special event markers, for e.g. Frame, LineStart, LineStop, etc.

#Get FLIM data stack
print('getting flim data stack...')
flim_data_stack = ptu_file.get_flim_data_stack()

# save FLIM data stack
# np.save(output_dir + '/' + 'flim_data_stack', flim_data_stack)
print("calculating data stack channels...")
flim_data_stack_0 = flim_data_stack[:, :, 0, :].squeeze()
flim_data_stack_1 = flim_data_stack[:, :, 1, :].squeeze()
print('saving flim data stacks...')
np.save(output_dir + 'ptu_image_0', flim_data_stack_0)
np.save(output_dir + 'ptu_image_1', flim_data_stack_1)

print('getting intensity images...')
# 2 channel image
if flim_data_stack.ndim == 4:
    intensity_image = np.sum(flim_data_stack, axis = 3) # sum across tcspc bin
    total_intensity_image = np.sum(intensity_image, axis  = 2) # sum across spectral channels

    color_image_0 = np.zeros([int(intensity_image.shape[0]), int(intensity_image.shape[1])])
    color_image_1 = np.zeros([int(intensity_image.shape[0]), int(intensity_image.shape[1])])
    # channel1 = 'red'
    color_image_0 = intensity_image[:, :, 0]
    # channel2 = 'green'
    color_image_1 = intensity_image[:, :, 1]

    # save color image
    print("saving intensity images...")
    np.save(output_dir + 'int_image', total_intensity_image)
    np.save(output_dir + 'int_image_0', color_image_0)
    np.save(output_dir + 'int_image_1', color_image_1)

# single channel image
elif flim_data_stack.ndim == 3:
    total_intensity_image  = np.sum(flim_data_stack, axis = 3) # sum across tcspc bin, only 1 detection channel

print('Finished.')
