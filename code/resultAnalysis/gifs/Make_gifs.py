import numpy as np
import math
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from typing import List, Dict
from PIL import Image
import seaborn as sns
import json
from matplotlib import cm
from pprint import pformat
import json
import cv2
from pygifsicle import optimize
import imageio


from joblib import Parallel, delayed

from sklearn.metrics import confusion_matrix


plt.style.use("seaborn")
sns.color_palette("colorblind")

grey = 0.925

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def set_ax(ax):
    ax.patch.set_facecolor('white')
    ax.elev = 20
    ax.set_xlabel('left-right [mm]')
    ax.set_ylabel('anteroposterior [mm]')
    ax.set_zlabel('craniocaudal [mm]')
    set_axes_equal(ax)
    ax.w_xaxis.set_pane_color((grey, grey, grey, 1.0))
    ax.w_yaxis.set_pane_color((grey, grey, grey, 1.0))
    ax.w_zaxis.set_pane_color((grey, grey, grey, 1.0))

def plot_vol(ax, vol):
    visiblebox = vol != 0
    _ = ax.voxels(filled = visiblebox, facecolors = cm.gist_stern_r(vol*51), alpha=0.95)
    set_ax(ax)



colormap = cm.get_cmap('gist_stern_r')

os.system('rm *.png')
os.system('rm *.pdf')
os.system('rm *.gif')

base_path = r'/media/jan/DataStorage/ProjectData/temp/'
dataset = r'dataset_2_contrast_3'
source = r'xVertSeg'
image_nr = 2
mask_path = os.path.join(base_path, dataset,  f'{source}_masks', f'image{image_nr:03d}', 'mask_array.npy')
volume_path = os.path.join(base_path, dataset,  f'{source}_images', f'image{image_nr:03d}', 'volume.npy')

mask = np.rot90(np.load(mask_path), axes = (2,0))
volume = np.rot90(np.load(volume_path), axes = (2,0))

n_x, n_y, n_z = volume.shape
print(f'x : {n_x} - y: {n_y} - z: {n_z}')
min_val = volume.min()
max_val = volume.max()

max_dim = max([n_x, n_y, n_z])




def make_image(i):
    fig = plt.figure(figsize = (20, 20))
    x_s, y_s, z_s = int(n_x * i / max_dim), int(n_y * i / max_dim), int(n_z * i / max_dim)
    x_cut = volume[x_s,:,:]
    y_cut = volume[:,y_s,:]
    z_cut = volume[:,:,z_s]
    ax_x = fig.add_subplot(133, projection='3d')
    Y, Z = np.mgrid[0:n_y, 0:n_z]
    X_X = x_s * np.ones((n_y, n_z))
    ax_x.plot_surface(X_X, Y, Z, rstride=1, cstride=1, facecolors=cm.Greys_r(x_cut), shade=False)
    set_ax(ax_x)
    ax_x.set_xlim3d([0, max([n_x, n_y, n_z])])
    ax_x.set_ylim3d([0, max([n_x, n_y, n_z])])
    ax_x.set_zlim3d([0, max([n_x, n_y, n_z])])
    ax_x.set_title('Sagittal')
    
    ax_y = fig.add_subplot(132, projection='3d')
    X, Z = np.mgrid[0:n_x, 0:n_z]
    Y_Y = y_s * np.ones((n_x, n_z))
    ax_y.plot_surface(X, Y_Y, Z, rstride=1, cstride=1, facecolors=cm.Greys_r(y_cut), shade=False)
    set_ax(ax_y)
    ax_y.set_xlim3d([0, max([n_x, n_y, n_z])])
    ax_y.set_ylim3d([0, max([n_x, n_y, n_z])])
    ax_y.set_zlim3d([0, max([n_x, n_y, n_z])])
    ax_y.set_title('Coronal')
    
    ax_z = fig.add_subplot(131, projection='3d')
    X, Y = np.mgrid[0:n_x, 0:n_y]
    Z_Z = z_s * np.ones((n_x, n_y))
    ax_z.plot_surface(X, Y, Z_Z, rstride=1, cstride=1, facecolors=cm.Greys_r(z_cut), shade=False)
    set_ax(ax_z)
    ax_z.set_xlim3d([0, max([n_x, n_y, n_z])])
    ax_z.set_ylim3d([0, max([n_x, n_y, n_z])])
    ax_z.set_zlim3d([0, max([n_x, n_y, n_z])])
    ax_z.set_title('Transversal')
    plt.savefig(f'temp_{i}.png', bbox_inches = 'tight')
    plt.close()



slice_nr = [i for i in range(0, max_dim,5)]

Parallel(n_jobs=8)(delayed(make_image)(i) for i in slice_nr)

gif_path = "slicing.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.02) as writer:
    for i in slice_nr + slice_nr[::-1]:
        writer.append_data(imageio.imread(f'temp_{i}.png'))
os.system('rm temp_*.png')

optimize("slicing.gif")