# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits import mplot3d
from prepare_data import read_data, ground_truth, select_init_points
import matplotlib.image as mpimg

X_UG = scipy.io.loadmat('../data/ET_USER.mat')
X_UG = X_UG['ET'][:,:-1]
X_UGRID = scipy.io.loadmat('../data/ET_GRID.mat')
X_UGRID = X_UGRID['ET'][:,:-1]
X_UR = scipy.io.loadmat('../data/ET_RAND.mat')
X_UR = X_UR['ET'][:,:-1]


# create test data (true map)
GP, kernel = read_data('../data/ET_COMB.mat', 'ET')
data = scipy.io.loadmat('../data/ET_COMB.mat')
data = data['ET']
MeshRes = 0.05  # 0.5 mm
x1mesh = np.arange(np.floor(np.min(data[:, 0])), np.ceil(np.max(data[:, 0])), MeshRes)
x2mesh = np.arange(np.floor(np.min(data[:, 1])), np.ceil(np.max(data[:, 1])), MeshRes)
x1, x2 = np.meshgrid(x1mesh, x2mesh)
d1 = x1.shape[0]
d2 = x1.shape[1]
x_test = np.vstack((x1.flatten(), x2.flatten())).T
z_test = ground_truth(x_test, GP)
xlim = [np.min(x1), np.max(x1)]
ylim = [np.min(x2), np.max(x2)]
zlim = [0, np.max(z_test)]

###################

img1 = mpimg.imread('../figures/grid.png')
img2 = mpimg.imread('../figures/rand.png')
img3 = mpimg.imread('../figures/user.png')
# plt.imshow(img1)
# plt.show()

Points = [img1, X_UGRID, img2, X_UR, img3, X_UG]
fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, constrained_layout=False, figsize=(10,15))
# axes[0].remove()
# axes[0] = fig.add_subplot(1,4,1,projection='3d')
fig.add_subplot(111, frameon=False)
# Set the ticks and ticklabels for all axes
# plt.setp(axes, xticks=np.linspace(xlim[0], xlim[1], 3), yticks=np.linspace(ylim[0], ylim[1], 3))
label = ['$GRID$', '$GRID$', '$GRID$','$RAND$', '$USRG$', '$USRG$']
c = 0
for ax, i, j in zip(axes.flat, Points, label):
    # ax.imshow(img1)
    if c == 0:
        ax.imshow(img1)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    elif c == 2:
        ax.imshow(img2)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    elif c == 4:
        ax.imshow(img3)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    else:
        im = ax.pcolormesh(x1, x2, z_test.reshape(d1, d2), cmap='bwr')
        ax.scatter(i[:, 0], i[:, 1], s=4, facecolors='k', edgecolors='k', linewidths=1.3, label=j)
        ax.legend(loc="upper right", fontsize=8)
        ax.get_xaxis().set_ticks([-6, -3, 0])
        ax.get_yaxis().set_ticks([-8, -4, 0])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if c != 5:
            ax.get_xaxis().set_ticks([])

        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    c+=1



# plt.xlabel('$ANT-POST(cm)$', fontsize=15, fontweight='bold')
# plt.ylabel('$LAT-MED(cm)$', fontsize=15, fontweight='bold')
plt.subplots_adjust(hspace=0.1)
cbar = fig.colorbar(im, ax=axes.flat)
cbar.set_label(r'$MEP_{pp}(\mu v)$', fontsize=13, fontweight='bold')
plt.tick_params(labelcolor="none", bottom=False, left=False)
fig.text(0.45, 0.01, '$ANT-POST(cm)$', fontsize=15, fontweight='bold', ha='center')
fig.text(0.04, 0.5, '$LAT-MED(cm)$', fontsize=15, fontweight='bold', va='center', rotation='vertical')
plt.show()

