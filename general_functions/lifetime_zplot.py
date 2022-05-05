import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage.filters import gaussian
from skimage.measure import find_contours

dir = 'L:/880_FLIM/paula_zhu/hela/dish1_cont/'

fm_im = np.load("fm_image1_4.npy")[836:(836+500), 88:(88+500)] # 500x500
int_im = np.load("int_image1.npy")[836:(836+500), 88:(88+500)] # 500x500

# domains
x = np.linspace(0, 500, 500)
y = np.linspace(0, 500, 500)

# XY meshgrid
X, Y = np.meshgrid(x, y)    # 500x500

# fourth dimention - colormap
color_dimension = int_im # set color dimension to intensity 500x500
minn, maxx = color_dimension.min(), np.percentile(color_dimension, 90)
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
m.set_array([])
fcolors = m.to_rgba(color_dimension)

blur_im = gaussian(fm_im, sigma=2)

# plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,fm_im, facecolors=fcolors, vmin=minn, vmax=maxx)
ax.set_zlim(0, 25)
ax.set_xlim(0, 500)
ax.set_ylim(0, 500)
plt.show()


