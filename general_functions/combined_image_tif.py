from PIL import Image
import numpy as np

dirs = ["", \
    "L:/880_FLIM/paula_zhu/hela/dish2_CCCP/", \
    "L:/880_FLIM/paula_zhu/hela/dish3_fasted/"]
in_files = ["int_image1_s.npy"] * 3
out_files = ["int_rgb1_tif"] * 3

def np2tif(dir, in_file, out_file, convert_rgb = False):
    print("loading np file...")
    im_np = np.load(dir + in_file)
    if convert_rgb == True:
        print("creating PIL image...")
        im_pil = Image.fromarray(im_np.astype('uint8'), mode='HSV')
        im_final = im_pil.convert(mode="RGB")
    else:
        print('hi')
        print(im_np.shape)
        im_rgb = np.zeros([im_np.shape[0], im_np.shape[1], 3])
        im_norm = np.clip(im_np * 255 / np.percentile(im_np, 99.9), 0, 255)
        print(im_norm.max())
        print(im_norm.min())
        im_rgb[:, :, 1] = im_norm
        print("creating PIL image...")
        im_final = Image.fromarray(im_rgb.astype('uint8'), mode='RGB')
    print('saving image as tif...')
    im_final = im_final.save(dir + out_file+".tif")

def batch_np2tif(dirs, in_files, out_files):
    for i in range(len(dirs)):
        np2tif(dirs[i], in_files[i], out_files[i], False)
