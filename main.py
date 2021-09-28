import numpy as np
import scipy.io
import pandas as pd

from prepare_data import read_data, ground_truth, select_init_points
from methods import experimental_methods, ugrid, urand, gpe, gprs, wgpe, map_features


''' configuration '''

subjects = ['AS', 'CCL', 'ET', 'PAT', 'TM']
for subj in subjects:
    save_path = 'results/'+subj
    save= 'True'
    mc_loop = 100
    z_max = np.array([1300.0]) # max for gp_rejection_sampling
    p = 1 #1.2 # regularization factor(power) for gp_rejection_sampling

    GP, kernel = read_data('data/'+subj+'_COMB.mat', subj)
    data = scipy.io.loadmat('data/'+subj+'_COMB.mat')
    data = data[subj]

    # read experimental data
    data_USRG = scipy.io.loadmat('data/'+subj+'_USER.mat')
    data_USRG = data_USRG[subj]
    X_USRG = np.vstack((data_USRG[:, 0], data_USRG[:, 1])).T
    Z_USRG = ground_truth(X_USRG, GP)
    n_USRG = X_USRG.shape[0]

    data_GRID = scipy.io.loadmat('data/'+subj+'_GRID.mat')
    data_GRID = data_GRID[subj]
    X_GRID = np.vstack((data_GRID[:, 0], data_GRID[:, 1])).T
    Z_GRID = ground_truth(X_GRID, GP)
    n_GRID = X_GRID.shape[0]

    data_RAND = scipy.io.loadmat('data/'+subj+'_RAND.mat')
    data_RAND = data_RAND[subj]
    X_RAND = np.vstack((data_RAND[:, 0], data_RAND[:, 1])).T
    Z_RAND = ground_truth(X_RAND, GP)
    n_RAND = X_RAND.shape[0]

    # create test data (true map)
    MeshRes = 0.05  # 0.5 mm
    x1mesh = np.arange(np.min(data[:, 0]), np.max(data[:, 0]), MeshRes)
    x2mesh = np.arange(np.min(data[:, 1]), np.max(data[:, 1]), MeshRes)
    x1, x2 = np.meshgrid(x1mesh, x2mesh)
    d1 = x1.shape[0]
    d2 = x1.shape[1]
    x_test = np.vstack((x1.flatten(), x2.flatten())).T
    z_test = ground_truth(x_test, GP)

    # boundaries of map
    min_bounds = [np.min(x_test[:, 0]), np.min(x_test[:, 1])]
    max_bounds = [np.max(x_test[:, 0]), np.max(x_test[:, 1])]

    # create train data (initial points)
    n_train = 36
    x_train = select_init_points(min_bounds, max_bounds, n_train, 'grid')  # grid or random
    z_train = ground_truth(x_train, GP)

    # create xgrid
    MeshRes_ = 0.2
    x1mesh_ = np.arange(np.min(data[:, 0]), np.max(data[:, 0]), MeshRes_)
    x2mesh_ = np.arange(np.min(data[:, 1]), np.max(data[:, 1]), MeshRes_)
    x1_, x2_ = np.meshgrid(x1mesh_, x2mesh_)
    x_grid = np.vstack((x1_.flatten(), x2_.flatten())).T

    # GT features
    GT_features = map_features(x1mesh, x2mesh, x1, x2, z_test.reshape(d1, d2))
    if save:
        df = pd.DataFrame(GT_features)
        df.to_csv(save_path + '/GT_features.csv', header=False, index=False)


    ''' methods '''
    # experimental
    X_USRG, Z_USRG, NMSE_USRG, z_est_USRG, Map_Feature_USRG = \
        experimental_methods(X_USRG, Z_USRG, x_test, z_test, 1, n_USRG, kernel, x1mesh, x2mesh, save_path, method = 'USRG',  save=save)

    X_GRID, Z_GRID, NMSE_GRID, z_est_GRID, Map_Feature_GRID = \
        experimental_methods(X_GRID, Z_GRID, x_test, z_test, 1, n_USRG, kernel, x1mesh, x2mesh, save_path, method = 'GRID',  save=save)

    X_RAND, Z_RAND, NMSE_RAND, z_est_RAND, Map_Feature_RAND = \
        experimental_methods(X_RAND, Z_RAND, x_test, z_test, 1, n_USRG, kernel, x1mesh, x2mesh, save_path, method = 'RAND',  save=save)

    #
    # # uniform_grid
    # X_UGRID, Z_UGRID, NMSE_UGRID, z_est_UGRID, Map_Feature_UGRID = \
    #     ugrid(x_test, z_test, min_bounds, max_bounds, n_train, n_USRG, kernel, ground_truth, GP, x1mesh,
    #                  x2mesh, save_path, save=save)
    #
    # # uniform_random
    # X_UR, Z_UR, NMSE_UR, z_est_UR, Map_Feature_UR = \
    #     urand(x_train, z_train,x_test, z_test, min_bounds, max_bounds, n_train, n_USRG, kernel, ground_truth, GP, x1mesh,
    #                    x2mesh, mc_loop, save_path, save=save)
    #
    # # gaussian process entropy
    # X_GPE, Z_GPE, NMSE_GPE, z_est_GPE, Map_Feature_GPE = \
    #     gpe(x_grid, x_train, z_train, x_test, z_test, kernel, n_train, n_USRG, 'entropy', ground_truth,
    #                GP, x1mesh, x2mesh, save_path, save=save)
    #
    # # gaussian process entropy on mean
    # X_GPEm, Z_GPEm, NMSE_GPEm, z_est_GPEm, Map_Feature_GPEm = \
    #     gpe(x_grid, x_train, z_train, x_test, z_test, kernel, n_train, n_USRG, 'entropy_mean', ground_truth,
    #                GP, x1mesh, x2mesh, save_path, save=save)
    #
    # # gaussian process rejection sampling on mean
    # X_GPRSm, Z_GPRSm, NMSE_GPRSm, z_est_GPRSm, Map_Feature_GPRSm = \
    #     gprs(x_grid, x_train, z_train, x_test, z_test, z_max, min_bounds, max_bounds,kernel, n_train,
    #                                n_USRG, p, ground_truth, GP,  x1mesh, x2mesh, 'grid', mc_loop, save_path, save=save)
    #
    # # wrapped gaussian process entropy = WGPE
    # X_WGPE, Z_WGPE, NMSE_WGPE, z_est_WGPE, Map_Feature_WGPE = \
    #     wgpe(x_grid, x_train, z_train, x_test, z_test, kernel, n_train, n_USRG, ground_truth, GP, x1mesh,
    #                       x2mesh, save_path, save=save)
