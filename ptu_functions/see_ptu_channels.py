import numpy as np
from matplotlib import pyplot as plt

def color_normalize(x):
    return ((x - x.min()) * (1/(x.max() - x.min()) * 255)).astype('uint8')

color_image = np.load('color_image.npy')

#plot color image
norm_color_image = color_normalize(color_image)
plt.imshow(norm_color_image)
plt.show()
