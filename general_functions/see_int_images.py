import matplotlib.pyplot as plt
import numpy as np

im1 = np.load('int_image1.npy')
im2 = np.load('int_image2.npy')
im3 = np.load('int_image3.npy')

def show_int_im(im1, im2, im3, climit=20):
    plt.subplot(1, 3, 1)
    plt.imshow(im1)
    plt.colorbar()
    plt.clim(0, climit)
    plt.subplot(1, 3, 2)
    plt.imshow(im2)
    plt.colorbar()
    plt.clim(0, climit)
    plt.subplot(1, 3, 3)
    plt.imshow(im3)
    plt.colorbar()
    plt.clim(0, climit)
    plt.show()
    return

# show_int_im(im1, im2, im3)

def show_im(im, climit=40, cfloor=0):
    plt.imshow(im)
    plt.colorbar()
    plt.set_cmap('jet')
    plt.clim(cfloor, climit)
    plt.show()
    return

# show_im(fm_image1_4, 8, 2)
