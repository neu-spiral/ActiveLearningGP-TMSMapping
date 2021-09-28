# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

plt.interactive(False)
import numpy as np
import pandas as pd
import scipy.io
from mpl_toolkits import mplot3d
from collections import OrderedDict

cmap = ['darkred', 'darkcyan', 'deepskyblue', 'k', 'navy', 'darkorange', 'darkgreen']
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

path = ['../results/AS', '../results/CCL', '../results/ET', '../results/PAT', '../results/TM',
        '../results/AS', '../results/CCL', '../results/ET', '../results/PAT', '../results/TM']
fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, constrained_layout=False, figsize=(20,7))
fig.add_subplot(111, frameon=False)
# Set the ticks and ticklabels for all axes
# plt.setp(axes, xticks=np.linspace(xlim[0], xlim[1], 3), yticks=np.linspace(ylim[0], ylim[1], 3))
n_train = 36
num = 1
c = 0
position_WGPE = []
position_GPRS = []
NMSE_USRG_f = []
NMSE_USRG_f2 = []
for ax, i in zip(axes.flat, path):
    c+=1
    if c>=6:
        UG = pd.read_csv(i + '/NMSE_USRG.csv', header=None, index_col=False).values
        NMSE_UGRID = pd.read_csv(i + '/NMSE_UGRID.csv', header=None, index_col=False).values
        NMSE_UR = pd.read_csv(i + '/NMSE_UR.csv', header=None, index_col=False).values
        NMSE_GPE = pd.read_csv(i + '/NMSE_GPE.csv', header=None, index_col=False).values
        NMSE_GPEm = pd.read_csv(i + '/NMSE_GPEm.csv', header=None, index_col=False).values
        NMSE_GPRSm = pd.read_csv(i + '/NMSE_GPRSm.csv', header=None, index_col=False).values
        NMSE_WGPE = pd.read_csv(i + '/NMSE_WGPE.csv', header=None, index_col=False).values


        med_NMSE_UR = np.mean(NMSE_UR, axis=0)
        med_NMSE_GPRSm = np.mean(NMSE_GPRSm, axis=0)
        per5_NMSE_UR = 1.96 * np.std(NMSE_UR, axis=0)
        per5_NMSE_GPRSm = 1.96 * np.std(NMSE_GPRSm, axis=0)
        per95_NMSE_UR = 1.96 * np.std(NMSE_UR, axis=0)
        per95_NMSE_GPRSm = 1.96 * np.std(NMSE_GPRSm, axis=0)
        # NMSE_UR = np.sort(NMSE_UR, axis=0)
        # NMSE_GPRSm = np.sort(NMSE_GPRSm, axis=0)
        # med_NMSE_UR = np.percentile(NMSE_UR, 50, axis=0)
        # med_NMSE_GPRSm = np.percentile(NMSE_GPRSm, 50, axis=0)
        # per5_NMSE_UR = np.percentile(NMSE_UR, 5, axis=0)
        # per5_NMSE_GPRSm = np.percentile(NMSE_GPRSm, 5, axis=0)
        # per95_NMSE_UR = np.percentile(NMSE_UR, 95, axis=0)
        # per95_NMSE_GPRSm = np.percentile(NMSE_GPRSm, 95, axis=0)

        # position_WGPE.append([np.argmax(NMSE_WGPE < 0.1)])
        # position_GPRS.append([np.argmax(med_NMSE_GPRSm < 0.1)])
        NMSE_USRG_f.append(med_NMSE_GPRSm[150-35])

        n = np.arange(n_train, n_train + len(NMSE_GPE), 1)
        ax.plot(n, UG[n_train-1:n_train+len(n)], color=cmap[0], linestyle=linestyles['dashdotdotted'], label='$USRG$', linewidth=1.45)

        ax.plot(n, NMSE_UGRID, color=cmap[1], linestyle=linestyles['densely dotted'], label='$UGRID$', linewidth=1.45)
        #
        ax.plot(n, med_NMSE_UR, color=cmap[3], linestyle=linestyles['densely dashed'], label='$URAND$', linewidth=1.45)
        ax.fill_between(n, (med_NMSE_UR - per5_NMSE_UR),
                        (med_NMSE_UR + per95_NMSE_UR), color=cmap[3], alpha=.1)
        #
        ax.plot(n, NMSE_GPE, color=cmap[2], linestyle=linestyles['dashdotted'], label='$GPE$', linewidth=1.45)
        ax.set_xlim(left=n_train, right=len(NMSE_USRG))
        #
        ax.plot(n, NMSE_GPEm, color=cmap[4], linestyle=linestyles['solid'], label='$GPE_\mu$', linewidth=1.45)
        #
        ax.plot(n, med_NMSE_GPRSm, color=cmap[5], linestyle=linestyles['densely dashdotdotted'], label='$GPRS$', linewidth=1.45)
        ax.fill_between(n, (med_NMSE_GPRSm - per5_NMSE_GPRSm),
                        (med_NMSE_GPRSm + per95_NMSE_GPRSm), color=cmap[5], alpha=.1)

        # ax.title.set_text('$Subject$'+"$% s$" % num )
        num+=1
        ax.set_xticks([1, 36, 100, 150, 200, 250])

        ax.plot(n, NMSE_WGPE, color=cmap[6], linestyle=linestyles['densely dashdotted'], label='$WGPE$', linewidth=1.45)
        if i == 'results/AS':
            ax.legend(loc="upper right", fontsize=8, ncol=1)

        ax.grid()




    else:
        NMSE_USRG = pd.read_csv(i + '/NMSE_USRG.csv', header=None, index_col=False).values
        NMSE_GRID = pd.read_csv(i + '/NMSE_GRID.csv', header=None, index_col=False).values
        NMSE_RAND = pd.read_csv(i + '/NMSE_RAND.csv', header=None, index_col=False).values

        NMSE_USRG_f2.append(NMSE_USRG[150])
        # NMSE_UG_f.append([np.argmax(NMSE_UG < 0.1)])


        n = np.arange(1, 1 + np.minimum(len(NMSE_USRG), np.minimum(len(NMSE_GRID), len(NMSE_RAND))), 1)
        ax.plot(n, NMSE_USRG[:len(n)], color=cmap[0], linestyle=linestyles['dashdotdotted'], label='$USRG$',
                linewidth=1.45)

        ax.plot(n, NMSE_GRID[:len(n)], color=cmap[1], linestyle=linestyles['densely dotted'], label='$GRID$',
                linewidth=1.45)

        ax.plot(n, NMSE_RAND[:len(n)], color=cmap[3], linestyle=linestyles['densely dashed'], label='$RAND$',
                linewidth=1.45)
        ax.set_xlim(left=1, right=258)
        ax.title.set_text('$Subject$' + "$% s$" % num)
        num += 1

        ax.grid()
        if i == 'results/AS':
            ax.legend(loc="upper right", fontsize=8, ncol=1)

        ax.set_xticks([1, 36, 100, 150, 200, 250])





ax.set_ylim(0, 1.05)
# plt.xlim(1, n[-1])
# plt.subplots_adjust(wspace=1, hspace=0.01)
# ax.legend(loc="upper left", fontsize=8)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel(R'$Number\:\: of\:\: Stimuli$', fontsize=15, fontweight='bold')
plt.ylabel(R'$NMSE$', fontsize=15, fontweight='bold')
plt.subplots_adjust(wspace=0.06,hspace=0.01)
# plt.savefig("NMSE_.pdf")
plt.tight_layout()
plt.show()
