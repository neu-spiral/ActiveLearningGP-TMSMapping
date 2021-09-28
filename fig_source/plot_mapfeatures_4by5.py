import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

plt.interactive(False)
import numpy as np
import pandas as pd
import scipy.io
from mpl_toolkits import mplot3d
from collections import OrderedDict

cmap = ['darkred', 'darkcyan', 'deepskyblue', 'k', 'navy', 'darkorange', 'darkgreen', 'r']
linestyles = OrderedDict(
    [('solid', (0, ())),
     ('loosely dotted', (0, (1, 10))),
     ('dotted', (0, (1, 5))),
     ('densely dotted', (0, (1, 1))),

     ('loosely dashed', (0, (5, 10))),
     ('dashed', (0, (5, 5))),
     ('densely dashed', (0, (5, 1))),

     ('loosely dashdotted', (0, (3, 10, 1, 10))),
     ('dashdotted', (0, (5, 2, 20, 2))),
     ('densely dashdotted', (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted', (0, (3, 2, 1, 2, 1, 2))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

path = ['../results/AS', '../results/CCL', '../results/ET', '../results/PAT', '../results/TM']

fig, axes = plt.subplots(nrows=4, ncols=5, sharex=True, sharey=False, constrained_layout=False, figsize=(30,30))
fig.add_subplot(111, frameon=False)
# Set the ticks and ticklabels for all axes
# plt.setp(axes, xticks=np.linspace(xlim[0], xlim[1], 3), yticks=np.linspace(ylim[0], ylim[1], 3))
n_train = 36
k = 0
num = 1
for j, subj in zip(range(5), path):
    mc_loop = 100
    Map_Feature_GT = pd.read_csv(subj + '/GT_features.csv', header=None, index_col=False).values
    Map_Feature_USRG = pd.read_csv(subj + '/Map_Feature_USRG.csv', header=None, index_col=False).values
    Map_Feature_UGRID = pd.read_csv(subj + '/Map_Feature_UGRID.csv', header=None, index_col=False).values
    Map_Feature_UR = pd.read_csv(subj + '/Map_Feature_UR.csv', header=None, index_col=False).values \
        .reshape(mc_loop, Map_Feature_UGRID.shape[0], Map_Feature_UGRID.shape[1])
    Map_Feature_GPE = pd.read_csv(subj + '/Map_Feature_GPE.csv', header=None, index_col=False).values
    Map_Feature_GPEm = pd.read_csv(subj + '/Map_Feature_GPEm.csv', header=None, index_col=False).values
    Map_Feature_GPRSm = pd.read_csv(subj + '/Map_Feature_GPRSm.csv', header=None, index_col=False).values \
        .reshape(mc_loop, Map_Feature_UGRID.shape[0], Map_Feature_UGRID.shape[1])
    Map_Feature_WGPE = pd.read_csv(subj + '/Map_Feature_WGPE.csv', header=None, index_col=False).values

    med_Map_Feature_UR = np.mean(Map_Feature_UR, axis=0)
    med_Map_Feature_GPRSm = np.mean(Map_Feature_GPRSm, axis=0)
    per5_Map_Feature_UR = 1.96 * np.std(Map_Feature_UR, axis=0)
    per5_Map_Feature_GPRSm = 1.96 * np.std(Map_Feature_GPRSm, axis=0)
    per95_Map_Feature_UR = 1.96 * np.std(Map_Feature_UR, axis=0)
    per95_Map_Feature_GPRSm = 1.96 * np.std(Map_Feature_GPRSm, axis=0)

    n = np.arange(n_train, n_train + len(Map_Feature_UGRID), 1)
    for ax, i, ylabel in zip(axes[:, j], [0,2,3,4], ['$MapVol$', '$MapArea$', '$CoGx1$', '$CoGx2$']):
        ax.plot(n, Map_Feature_USRG[n_train-1:n_train+len(n), i], color=cmap[0], linestyle=linestyles['dashdotdotted'], label='$USRG$', linewidth=1.3)

        ax.plot(n, Map_Feature_UGRID[:, i], color=cmap[1], linestyle=linestyles['densely dotted'], label='$UGRID$', linewidth=1.3)

        ax.plot(n, med_Map_Feature_UR[:, i], color=cmap[3], linestyle=linestyles['densely dashed'], label='$URAND$', linewidth=1.3)
        ax.fill_between(n, (med_Map_Feature_UR[:, i] - per5_Map_Feature_UR[:, i]),
                        (med_Map_Feature_UR[:, i] + per95_Map_Feature_UR[:, i]), color=cmap[3], alpha=.1, linewidth=1.3)

        ax.plot(n, Map_Feature_GPE[:, i], color=cmap[2], linestyle=linestyles['dashdotted'], label='$GPE$', linewidth=1.3)

        ax.plot(n, Map_Feature_GPEm[:, i], color=cmap[4], linestyle=linestyles['solid'], label='$GPE_\mu$', linewidth=1.3)

        ax.plot(n, med_Map_Feature_GPRSm[:, i], color=cmap[5], linestyle=linestyles['densely dashdotdotted'],
                label='$GPRS$', linewidth=1.3)
        ax.fill_between(n, (med_Map_Feature_GPRSm[:, i] - per5_Map_Feature_GPRSm[:, i]),
                        (med_Map_Feature_GPRSm[:, i] + per95_Map_Feature_GPRSm[:, i]), color=cmap[5], alpha=.1, linewidth=1.35)

        ax.plot(n, Map_Feature_WGPE[:, i], color=cmap[6], linestyle=linestyles['densely dashdotted'], label='$WGPE$', linewidth=1.3)

        ax.plot(n, Map_Feature_GT[i]*np.ones(len(n)), color=cmap[7], linestyle=linestyles['solid'],
                  label='$GT$', linewidth=1.3)

        if j == 0:
            ax.set_ylabel(ylabel, fontsize=15, fontweight='bold')

        if i ==0:
            ax.title.set_text('$Subject$' + "$% s$" % num)
            num += 1

        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.grid()
        if j == 0 and i == 0:
            print("hello")
            ax.legend(loc="lower right", fontsize=8, ncol=2)

    k += 5

ax.set_xlim(left=n_train, right=len(Map_Feature_UGRID))
ax.get_xaxis().set_ticks([36, 100, 150, 200, 250])
# ax.legend(loc="upper left", fontsize=8)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel(R'$Number\:\: of\:\: Stimuli$', fontsize=15, fontweight='bold')
plt.subplots_adjust(wspace=0.2, hspace=0.1)
# plt.tight_layout(8
# fig.legend(     # The line objects
#            labels= ['$USRG$', '$UGRID$', '$URAND$', '$GPE$', '$GPE_\mu$', '$GPRS$', '$WGPE$', '$GT$'],   # The labels for each line
#            loc='center right',   # Position of legend
#            borderaxespad=0.1,    # Small spacing around legend box
#            title="Methods",
#             ncol=1# Title for the legend
#            )
plt.show()
