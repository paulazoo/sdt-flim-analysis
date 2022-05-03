import numpy as np
from PIL import Image

dirs = ["L:/880_FLIM/paula_zhu/convallaria/220408_convallaria/FLIM/220408_convallaria_run2/"]*3
fm_files = ['fm_image.npy', 'fm_image_0.npy', 'fm_image_1.npy']
int_files = ['int_image.npy', 'int_image_0.npy', 'int_image_1.npy']

def combine_hsv(flim_image, frange, int_image, imax):
    int_image[int_image < 5] = 0
    return np.stack(( np.clip( (flim_image - frange[1]) * 255.0/(frange[0]-frange[1]), 0, 255), \
    255*np.ones_like(flim_image), \
    np.clip(int_image * 255.0/imax, 0, 255) ))

def run_combine_hsv(dir, fm_file, int_file, out_file, \
    fm_start=2.8, fm_color_max=7, save_im=False, show_im=True):
    print("loading fm and int images...")
    fm_image1 = np.load(dir+fm_file)
    int_image_s = np.load(dir+int_file)
    print("combining fm and int image...")
    x = combine_hsv(fm_image1, [fm_start, fm_color_max], int_image_s, np.percentile(int_image_s, 95))
    combined_image1 = np.rollaxis(x, 0, start=3)
    if save_im == True:
        print('saving combined image...')
        np.save(dir+out_file, combined_image1)
        print('combined image saved')
    if show_im == True:
        pil_im = Image.fromarray(combined_image1.astype('uint8'), mode='HSV')
        # im = xpil.convert('RGB')
        pil_im.show()
    return combined_image1

def batch_view_hsv(dirs, fm_files, int_files, fm_start, fm_color_max):
    # [2.8 5.3] NADH
    # [1 7] convallaria 0
    # [4.25 6.25] convallaria 1
    # [1.5 6] convallaria total
    for i in range(len(dirs)):
        run_combine_hsv(dirs[i], fm_files[i], int_files[i], "combined_image"+str(i), \
            fm_start, fm_color_max)
    return
