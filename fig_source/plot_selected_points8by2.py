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
from matplotlib import cm

"""
Z_UG_AS = (pd.read_csv('Results/AS/Z_UG.csv', header=None, index_col=False).values)[36:,:]
Z_UGRID_AS = (pd.read_csv('Results/AS_p/Z_UGRID.csv', header=None, index_col=False).values)[36:,:]
Z_UR_AS = (pd.read_csv('Results/AS_p/Z_UR.csv', header=None, index_col=False).values)[36:,:]
Z_GPE_AS = (pd.read_csv('Results/AS_p/Z_GPE.csv', header=None, index_col=False).values)[36:,:]
Z_GPEm_AS = (pd.read_csv('Results/AS_p/Z_GPEm.csv', header=None, index_col=False).values)[36:,:]
Z_GPRSm_AS = (pd.read_csv('Results/AS_p/Z_GPRSm.csv', header=None, index_col=False).values)[36:,:]
Z_WGPE_AS = (pd.read_csv('Results/AS_p/Z_WGPE.csv', header=None, index_col=False).values)[36:,:]

Z_UG_CCL = (pd.read_csv('Results/CCL/Z_UG.csv', header=None, index_col=False).values)[36:,:]
Z_UGRID_CCL = (pd.read_csv('Results/CCL_p/Z_UGRID.csv', header=None, index_col=False).values)[36:,:]
Z_UR_CCL = (pd.read_csv('Results/CCL_p/Z_UR.csv', header=None, index_col=False).values)[36:,:]
Z_GPE_CCL = (pd.read_csv('Results/CCL_p/Z_GPE.csv', header=None, index_col=False).values)[36:,:]
Z_GPEm_CCL = (pd.read_csv('Results/CCL_p/Z_GPEm.csv', header=None, index_col=False).values)[36:,:]
Z_GPRSm_CCL = (pd.read_csv('Results/CCL_p/Z_GPRSm.csv', header=None, index_col=False).values)[36:,:]
Z_WGPE_CCL = (pd.read_csv('Results/CCL_p/Z_WGPE.csv', header=None, index_col=False).values)[36:,:]

Z_UG_ET = (pd.read_csv('Results/ET/Z_UG.csv', header=None, index_col=False).values)[36:,:]
Z_UGRID_ET = (pd.read_csv('Results/ET_p/Z_UGRID.csv', header=None, index_col=False).values)[36:,:]
Z_UR_ET = (pd.read_csv('Results/ET_p/Z_UR.csv', header=None, index_col=False).values)[36:,:]
Z_GPE_ET = (pd.read_csv('Results/ET_p/Z_GPE.csv', header=None, index_col=False).values)[36:,:]
Z_GPEm_ET = (pd.read_csv('Results/ET_p/Z_GPEm.csv', header=None, index_col=False).values)[36:,:]
Z_GPRSm_ET = (pd.read_csv('Results/ET_p/Z_GPRSm.csv', header=None, index_col=False).values)[36:,:]
Z_WGPE_ET = (pd.read_csv('Results/ET_p/Z_WGPE.csv', header=None, index_col=False).values)[36:,:]

Z_UG_PAT = (pd.read_csv('Results/PAT/Z_UG.csv', header=None, index_col=False).values)[36:,:]
Z_UGRID_PAT = (pd.read_csv('Results/PAT_p/Z_UGRID.csv', header=None, index_col=False).values)[36:,:]
Z_UR_PAT = (pd.read_csv('Results/PAT_p/Z_UR.csv', header=None, index_col=False).values)[36:,:]
Z_GPE_PAT = (pd.read_csv('Results/PAT_p/Z_GPE.csv', header=None, index_col=False).values)[36:,:]
Z_GPEm_PAT = (pd.read_csv('Results/PAT_p/Z_GPEm.csv', header=None, index_col=False).values)[36:,:]
Z_GPRSm_PAT = (pd.read_csv('Results/PAT_p/Z_GPRSm.csv', header=None, index_col=False).values)[36:,:]
Z_WGPE_PAT = (pd.read_csv('Results/PAT_p/Z_WGPE.csv', header=None, index_col=False).values)[36:,:]

Z_UG_TM = (pd.read_csv('Results/TM/Z_UG.csv', header=None, index_col=False).values)[36:,:]
Z_UGRID_TM = (pd.read_csv('Results/TM_p/Z_UGRID.csv', header=None, index_col=False).values)[36:,:]
Z_UR_TM = (pd.read_csv('Results/TM_p/Z_UR.csv', header=None, index_col=False).values)[36:,:]
Z_GPE_TM = (pd.read_csv('Results/TM_p/Z_GPE.csv', header=None, index_col=False).values)[36:,:]
Z_GPEm_TM = (pd.read_csv('Results/TM_p/Z_GPEm.csv', header=None, index_col=False).values)[36:,:]
Z_GPRSm_TM = (pd.read_csv('Results/TM_p/Z_GPRSm.csv', header=None, index_col=False).values)[36:,:]
Z_WGPE_TM = (pd.read_csv('Results/TM_p/Z_WGPE.csv', header=None, index_col=False).values)[36:,:]

GPEm_mean = np.mean([np.sum(i > 50 for i in Z_GPEm_AS)/len(Z_GPEm_AS), np.sum(i > 50 for i in Z_GPEm_CCL)/len(Z_GPEm_CCL),
                     np.sum(i > 50 for i in Z_GPEm_ET)/len(Z_GPEm_ET), np.sum(i > 50 for i in Z_GPEm_PAT)/len(Z_GPEm_PAT),
                     np.sum(i > 50 for i in Z_GPEm_TM)/len(Z_GPEm_TM)])

GPEm_std = np.std([np.sum(i > 50 for i in Z_GPEm_AS)/len(Z_GPEm_AS), np.sum(i > 50 for i in Z_GPEm_CCL)/len(Z_GPEm_CCL),
                     np.sum(i > 50 for i in Z_GPEm_ET)/len(Z_GPEm_ET), np.sum(i > 50 for i in Z_GPEm_PAT)/len(Z_GPEm_PAT),
                     np.sum(i > 50 for i in Z_GPEm_TM)/len(Z_GPEm_TM)])

GPRSm_mean = np.mean([np.sum(i > 50 for i in Z_GPRSm_AS)/len(Z_GPRSm_AS), np.sum(i > 50 for i in Z_GPRSm_CCL)/len(Z_GPRSm_CCL),
                     np.sum(i > 50 for i in Z_GPRSm_ET)/len(Z_GPRSm_ET), np.sum(i > 50 for i in Z_GPRSm_PAT)/len(Z_GPRSm_PAT),
                     np.sum(i > 50 for i in Z_GPRSm_TM)/len(Z_GPRSm_TM)])

GPRSm_std = np.std([np.sum(i > 50 for i in Z_GPRSm_AS)/len(Z_GPRSm_AS), np.sum(i > 50 for i in Z_GPRSm_CCL)/len(Z_GPRSm_CCL),
                     np.sum(i > 50 for i in Z_GPRSm_ET)/len(Z_GPRSm_ET), np.sum(i > 50 for i in Z_GPRSm_PAT)/len(Z_GPRSm_PAT),
                     np.sum(i > 50 for i in Z_GPRSm_TM)/len(Z_GPRSm_TM)])

WGPE_mean = np.mean([np.sum(i > 50 for i in Z_WGPE_AS)/len(Z_WGPE_AS), np.sum(i > 50 for i in Z_WGPE_CCL)/len(Z_WGPE_CCL),
                     np.sum(i > 50 for i in Z_WGPE_ET)/len(Z_WGPE_ET), np.sum(i > 50 for i in Z_WGPE_PAT)/len(Z_WGPE_PAT),
                     np.sum(i > 50 for i in Z_WGPE_TM)/len(Z_WGPE_TM)])

WGPE_std = np.std([np.sum(i > 50 for i in Z_WGPE_AS)/len(Z_WGPE_AS), np.sum(i > 50 for i in Z_WGPE_CCL)/len(Z_WGPE_CCL),
                     np.sum(i > 50 for i in Z_WGPE_ET)/len(Z_WGPE_ET), np.sum(i > 50 for i in Z_WGPE_PAT)/len(Z_WGPE_PAT),
                     np.sum(i > 50 for i in Z_WGPE_TM)/len(Z_WGPE_TM)])

UGRID_mean = np.mean([np.sum(i > 50 for i in Z_UGRID_AS)/len(Z_UGRID_AS), np.sum(i > 50 for i in Z_UGRID_CCL)/len(Z_UGRID_CCL),
                     np.sum(i > 50 for i in Z_UGRID_ET)/len(Z_UGRID_ET), np.sum(i > 50 for i in Z_UGRID_PAT)/len(Z_UGRID_PAT),
                     np.sum(i > 50 for i in Z_UGRID_TM)/len(Z_UGRID_TM)])

UGRID_std = np.std([np.sum(i > 50 for i in Z_UGRID_AS)/len(Z_UGRID_AS), np.sum(i > 50 for i in Z_UGRID_CCL)/len(Z_UGRID_CCL),
                     np.sum(i > 50 for i in Z_UGRID_ET)/len(Z_UGRID_ET), np.sum(i > 50 for i in Z_UGRID_PAT)/len(Z_UGRID_PAT),
                     np.sum(i > 50 for i in Z_UGRID_TM)/len(Z_UGRID_TM)])

UR_mean = np.mean([np.sum(i > 50 for i in Z_UR_AS)/len(Z_UR_AS), np.sum(i > 50 for i in Z_UR_CCL)/len(Z_UR_CCL),
                     np.sum(i > 50 for i in Z_UR_ET)/len(Z_UR_ET), np.sum(i > 50 for i in Z_UR_PAT)/len(Z_UR_PAT),
                     np.sum(i > 50 for i in Z_UR_TM)/len(Z_UR_TM)])

UR_std = np.std([np.sum(i > 50 for i in Z_UR_AS)/len(Z_UR_AS), np.sum(i > 50 for i in Z_UR_CCL)/len(Z_UR_CCL),
                     np.sum(i > 50 for i in Z_UR_ET)/len(Z_UR_ET), np.sum(i > 50 for i in Z_UR_PAT)/len(Z_UR_PAT),
                     np.sum(i > 50 for i in Z_UR_TM)/len(Z_UR_TM)])

GPE_mean = np.mean([np.sum(i > 50 for i in Z_GPE_AS)/len(Z_GPE_AS), np.sum(i > 50 for i in Z_GPE_CCL)/len(Z_GPE_CCL),
                     np.sum(i > 50 for i in Z_GPE_ET)/len(Z_GPE_ET), np.sum(i > 50 for i in Z_GPE_PAT)/len(Z_GPE_PAT),
                     np.sum(i > 50 for i in Z_GPE_TM)/len(Z_GPE_TM)])

GPE_std = np.std([np.sum(i > 50 for i in Z_GPE_AS)/len(Z_GPE_AS), np.sum(i > 50 for i in Z_GPE_CCL)/len(Z_GPE_CCL),
                     np.sum(i > 50 for i in Z_GPE_ET)/len(Z_GPE_ET), np.sum(i > 50 for i in Z_GPE_PAT)/len(Z_GPE_PAT),
                     np.sum(i > 50 for i in Z_GPE_TM)/len(Z_GPE_TM)])

UG_mean = np.mean([np.sum(i > 50 for i in Z_UG_AS)/len(Z_UG_AS), np.sum(i > 50 for i in Z_UG_CCL)/len(Z_UG_CCL),
                     np.sum(i > 50 for i in Z_UG_ET)/len(Z_UG_ET), np.sum(i > 50 for i in Z_UG_PAT)/len(Z_UG_PAT),
                     np.sum(i > 50 for i in Z_UG_TM)/len(Z_UG_TM)])

UG_std = np.std([np.sum(i > 50 for i in Z_UG_AS)/len(Z_UG_AS), np.sum(i > 50 for i in Z_UG_CCL)/len(Z_UG_CCL),
                     np.sum(i > 50 for i in Z_UG_ET)/len(Z_UG_ET), np.sum(i > 50 for i in Z_UG_PAT)/len(Z_UG_PAT),
                     np.sum(i > 50 for i in Z_UG_TM)/len(Z_UG_TM)])
"""



X_UG = scipy.io.loadmat('../data/ET_USER.mat')
X_UG = X_UG['ET'][:,:-1]
X_UGRID = (pd.read_csv('../Results/ET/X_UGRID.csv', header=None, index_col=False).values)
X_UR = (pd.read_csv('../Results/ET/X_UR.csv', header=None, index_col=False).values)
X_GPE = (pd.read_csv('../Results/ET/X_GPE.csv', header=None, index_col=False).values)
X_GPEm = (pd.read_csv('../Results/ET/X_GPEm.csv', header=None, index_col=False).values)
X_GPRSm = (pd.read_csv('../Results/ET/X_GPRSm.csv', header=None, index_col=False).values)
X_WGPE = (pd.read_csv('../Results/ET/X_WGPE.csv', header=None, index_col=False).values)

X_UR[:36,:] = X_GPE[:36,:]

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

Points = [X_UG, X_UG, X_UGRID, X_UR, X_GPE, X_GPEm, X_GPRSm, X_WGPE]
fig, axes = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True, constrained_layout=False, figsize=(10,15))
axes[0,0].remove()
axes[0,0] = fig.add_subplot(4,2,1,projection='3d')
fig.add_subplot(111, frameon=False)
# Set the ticks and ticklabels for all axes
plt.setp(axes, xticks=np.linspace(xlim[0], xlim[1], 3), yticks=np.linspace(ylim[0], ylim[1], 3))
label = ['TargetMap', '$USRG$', '$UGRID$', '$URAND$', '$GPE$', '$GPE_\mu$', '$GPRS$', '$WGPE$']
for ax, i, j in zip(axes.flat, Points, label):

    if j == 'TargetMap':
        # ax = fig.add_subplot(4,2,2,projection='3d')
        ax.plot_surface(x1, x2, (z_test).reshape(d1, d2), cmap='bwr')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.view_init(30, 45)
        ax.set_proj_type('ortho')
        ax.set_zticks(np.linspace(0, zlim[1], 3))
        # ax.set_xlabel('$ANT-POST(cm)$', fontsize=8, rotation=150, fontweight='bold')
        # ax.set_ylabel('$LAT-MED(cm)$', fontsize=8, fontweight='bold')
        ax.set_zlabel(r'$MEP_{pp}(\mu v)$', fontsize=13, rotation=60, fontweight='bold')

    else:
        im = ax.pcolormesh(x1, x2, z_test.reshape(d1, d2), cmap='bwr')
        ax.scatter(i[:, 0], i[:, 1], s=4, facecolors='k', edgecolors='k', linewidths=1.3, label=j)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))



# plt.xlabel('$ANT-POST(cm)$', fontsize=15, fontweight='bold')
# plt.ylabel('$LAT-MED(cm)$', fontsize=15, fontweight='bold')
# plt.subplots_adjust(wspace=0.13, hspace=0.14)
cbar = fig.colorbar(im, ax=axes.flat)
cbar.set_label(r'$MEP_{pp}(\mu v)$', fontsize=13, fontweight='bold')
plt.tick_params(labelcolor="none", bottom=False, left=False)
fig.text(0.45, 0.04, '$ANT-POST(cm)$', fontsize=15, fontweight='bold', ha='center')
fig.text(0.04, 0.5, '$LAT-MED(cm)$', fontsize=15, fontweight='bold', va='center', rotation='vertical')
plt.show()



