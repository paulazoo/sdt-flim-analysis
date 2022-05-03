from PIL import Image
import numpy as np

dir = "L:/880_FLIM/paula_zhu/convallaria/220408_convallaria/FLIM/220408_convallaria_run2/"
red = np.load(dir + 'int_image_0.npy')
green = np.load(dir + 'int_image_1.npy')
im_rgb = np.zeros([red.shape[0], red.shape[1], 3])
im_rgb_red = np.zeros([red.shape[0], red.shape[1], 3])
red_norm = np.clip(red * 255 / np.percentile(red, 99.9), 0, 255)
green_norm = np.clip(green * 255 / np.percentile(green, 99.9), 0, 255)
im_rgb[:, :, 0] = red_norm
im_rgb[:, :, 1] = green_norm
im_rgb_red[:, :, 0] = red_norm
im_final_red = Image.fromarray(im_rgb_red.astype('uint8'), mode='RGB')
im_final_red = im_final_red.save(dir + "int_rgb0.tif")

im_rgb_green = np.zeros([red.shape[0], red.shape[1], 3])
im_rgb_green[:, :, 1] = green_norm
im_final_green = Image.fromarray(im_rgb_green.astype('uint8'), mode='RGB')
im_final_green = im_final_green.save(dir + "int_rgb1.tif")
