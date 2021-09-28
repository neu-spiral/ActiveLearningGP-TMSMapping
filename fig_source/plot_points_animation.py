import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from prepare_data import read_data, ground_truth, select_init_points


X_WGPE = (pd.read_csv('../Results/ET/X_WGPE.csv', header=None, index_col=False).values)

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



fig, ax = plt.subplots()
ax.set_xlim(xlim)
ax.set_ylim(ylim)

im = ax.pcolormesh(x1, x2, z_test.reshape(d1, d2), cmap='bwr')

def animation_frame(i):

	ax.scatter(X_WGPE[:i, 0], X_WGPE[:i, 1], s=7, facecolors='k', edgecolors='k', linewidths=1.3, label='WGPE')

	return ax
animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, len(X_WGPE), 1), interval=0.5)
cbar = fig.colorbar(im)
cbar.set_label(r'$MEP_{pp}(\mu v)$', fontsize=13, fontweight='bold')
fig.text(0.45, 0.0, '$ANT-POST(cm)$', fontsize=15, fontweight='bold', ha='center')
fig.text(0.04, 0.5, '$LAT-MED(cm)$', fontsize=15, fontweight='bold', va='center', rotation='vertical')
animation.save('../figures/animation.gif', writer='imagemagick', fps=30)
plt.show()