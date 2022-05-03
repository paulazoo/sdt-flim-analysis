import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

dir = 'L:/880_FLIM/paula_zhu/hela/dish1_cont/'
hsv_im = np.load("combined_image1.npy")

fm_im = hsv_im[:, :, 1] # 2048x2048
int_im = hsv_im[:, :, 3] # 2048x2048

# domains
x = np.linspace(0, 2048, 2048)
y = np.linspace(0, 2048, 2048)

# XY meshgrid
X, Y = np.meshgrid(x, y)    # 2048x2048

# fourth dimention - colormap
color_dimension = fm_im # set color dimension to lifetimes 2048x2048
minn, maxx = color_dimension.min(), color_dimension.max()
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
m.set_array([])
fcolors = m.to_rgba(color_dimension)

# plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,int_im, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig.canvas.show()
