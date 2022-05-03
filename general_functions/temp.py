import numpy as np

dir = "L:/880_FLIM/paula_zhu/convallaria/220408_convallaria/FLIM/220408_convallaria_run2/"

fm_image = np.load("fm_image.npy")
fm_image_0 = np.load("fm_image_0.npy")
fm_image_1 = np.load("fm_image_1.npy")

int_image = np.load("int_image.npy")
int_image_0 = np.load("int_image_0.npy")
int_image_1 = np.load("int_image_1.npy")

def combine_hsv(flim_image, frange, int_image, imax):
    return np.stack(( np.clip( (flim_image - frange[1]) * 255.0/(frange[0]-frange[1]), 0, 255), \
    255*np.ones_like(flim_image), \
    np.clip(int_image * 255.0/imax, 0, 255) ))

print("combining fm and int images...")
x = combine_hsv(fm_image, [1, 5], int_image, np.percentile(int_image, 99.9))
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
