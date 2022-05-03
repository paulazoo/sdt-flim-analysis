import numpy as np
from PIL import Image

# dir = "L:/880_FLIM/paula_zhu/hela/dish3_fasted/"
dir = ""

def combine_hsv(flim_image, frange, int_image, imax):
    return np.stack(( np.clip( (flim_image - frange[1]) * 255.0/(frange[0]-frange[1]), 0, 255), \
    255*np.ones_like(flim_image), \
    np.clip(int_image * 255.0/imax, 0, 255) ))

def run_combine_hsv(dir, fm_color_max):
    print("loading fm and int images...")
    fm_image1 = np.load(dir+'fm_image1.npy')
    int_image_s = np.load(dir+'int_image1_s.npy')
    print("combining fm and int image...")
    x = combine_hsv(fm_image1, [2.8, fm_color_max], int_image_s, np.percentile(int_image_s, 99.9)) # 2.807 ns is the FM of the IRF
    combined_image1 = np.rollaxis(x, 0, start=3)
    # print('saving combined image...')
    # np.save(dir+'combined_image1', combined_image1)
    print('combined image saved')
    xpil = Image.fromarray(combined_image1.astype('uint8'), mode='HSV')
    # im = xpil.convert('RGB')
    xpil.show()
    return combined_image1