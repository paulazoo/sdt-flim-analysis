from PIL import Image
import numpy as np

# im_dirs = ["L:/880_FLIM/paula_zhu/hela/dish1_cont/", \
#     "L:/880_FLIM/paula_zhu/hela/dish2_CCCP/", \
#     "L:/880_FLIM/paula_zhu/hela/dish3_fasted/"]
im_dirs = ["L:/880_FLIM/paula_zhu/convallaria/220408_convallaria/FLIM/220408_convallaria_run2/combined_image.npy", \
    "L:/880_FLIM/paula_zhu/convallaria/220408_convallaria/FLIM/220408_convallaria_run2/combined_image_0.npy", \
    "L:/880_FLIM/paula_zhu/convallaria/220408_convallaria/FLIM/220408_convallaria_run2/combined_image_1.npy"]

for i, im_dir in enumerate(im_dirs):
    # combined_image = np.load(im_dir+"combined_image.npy")
    combined_image = np.load(im_dir)
    hsv_image = Image.fromarray(combined_image.astype('uint8'), mode='HSV')
    # hsv_image.show()
    rgb_image = hsv_image.convert(mode="RGB")
    rgb_image = rgb_image.save("fm_rgb_image"+str(i)+".tif")

