import random
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

from scipy.linalg import cholesky, cho_solve, solve_triangular


def normalized_mean_squared_error(x_true, x_est):
    N = len(x_true)
    mse = mean_squared_error(x_true, x_est)
    nmse = N * mse / (np.dot(x_true.T, x_true))
    return nmse


def map_features(x1mesh, x2mesh, x1, x2, z):
    z_ = z.copy()
    z_[z_ < 50] = 0

    MapVol = np.trapz(np.trapz(z_.T, x2mesh.T, 2), x1mesh.T, 1)

    MapMean = np.mean(z_[z_ != 0])
    MapArea = MapVol / MapMean
    CoGx1 = sum(sum(x1 * z_)) / sum(sum(z_))
    CoGx2 = sum(sum(x2 * z_)) / sum(sum(z_))
    map_feature = [MapVol, MapMean, MapArea, CoGx1, CoGx2]
    return map_feature


def inv_pd_mat(K, reg=1e-5):
    K = K.copy() + reg * np.eye(len(K))
    L = cholesky(K, lower=True)
    L_inv = solve_triangular(L.T, np.eye(L.shape[0]))

    return L_inv.dot(L_inv.T)


def z_fit(x_train, z_train, kernel):
    kernel__ = clone(kernel)
    gp = GaussianProcessRegressor(kernel=kernel__, n_restarts_optimizer=0)
    zw_train = np.sqrt(z_train)
    gp.fit(x_train, zw_train)
    return gp


def z_estimate(x, gp, return_std=False):
    zw_est, sw = gp.predict(x, return_std=True)
    z_est = zw_est ** 2

    if return_std:
        s = zw_est * sw.reshape(-1, 1) * zw_est
        return z_est, s
    else:
        return z_est


def experimental_methods(x_train, z_train, x_test, z_test, n_train, n, kernel, x1mesh, x2mesh, SaveName,
                       method = 'method', save='False'):
    X = x_train.copy()
    Z = z_train.copy().reshape(-1, 1)
    x1, x2 = np.meshgrid(x1mesh, x2mesh)
    d1 = x1.shape[0]
    d2 = x1.shape[1]
    NMSE = []
    Map_Feature = []

    for i in range(n_train, n + 1):
        gp = z_fit(X[:i, :], Z[:i], kernel)
        Z_est = z_estimate(x_test, gp)
        nmse = normalized_mean_squared_error(z_test, Z_est)
        NMSE.append(nmse)
        map_feature = map_features(x1mesh, x2mesh, x1, x2, Z_est.reshape(d1, d2))
        Map_Feature.append(map_feature)

    if save:
        df1 = pd.DataFrame(X)
        df1.to_csv(SaveName + '/X_' + method + '.csv', header=False, index=False)
        df2 = pd.DataFrame(Z)
        df2.to_csv(SaveName + '/Z_' + method + '.csv', header=False, index=False)
        df3 = pd.DataFrame(np.array(NMSE))
        df3.to_csv(SaveName + '/NMSE_' + method + '.csv', header=False, index=False)
        df4 = pd.DataFrame(np.array(Map_Feature))
        df4.to_csv(SaveName + '/Map_Feature_' + method + '.csv', header=False, index=False)

    return X, Z, np.array(NMSE), Z_est, np.array(Map_Feature)


def ugrid(x_test, z_test, x1_bounds, x2_bounds, n_train, n, kernel, ground_truth, GP, x1mesh, x2mesh, SaveName, save='False'):
    x1, x2 = np.meshgrid(x1mesh, x2mesh)
    d1 = x1.shape[0]
    d2 = x1.shape[1]
    NMSE = []
    Map_Feature = []

    for i in range(n_train, n + 1):
        x1M = np.linspace(x1_bounds[0], x2_bounds[0], int(np.sqrt(i)))
        x2M = np.linspace(x1_bounds[1], x2_bounds[1], int(np.sqrt(i)))
        x1g, x2g = np.meshgrid(x1M, x2M)
        X = np.vstack((x1g.flatten(), x2g.flatten())).T
        Z = ground_truth(X, GP)
        gp = z_fit(X, Z, kernel)
        Z_est = z_estimate(x_test, gp)
        nmse = normalized_mean_squared_error(z_test, Z_est)
        NMSE.append(nmse)
        map_feature = map_features(x1mesh, x2mesh, x1, x2, Z_est.reshape(d1, d2))
        Map_Feature.append(map_feature)

    if save:
        df1 = pd.DataFrame(X)
        df1.to_csv(SaveName + '/X_UGRID.csv', header=False, index=False)
        df2 = pd.DataFrame(Z)
        df2.to_csv(SaveName + '/Z_UGRID.csv', header=False, index=False)
        df3 = pd.DataFrame(np.array(NMSE))
        df3.to_csv(SaveName + '/NMSE_UGRID.csv', header=False, index=False)
        df4 = pd.DataFrame(np.array(Map_Feature))
        df4.to_csv(SaveName + '/Map_Feature_UGRID.csv', header=False, index=False)

    return X, Z, np.array(NMSE), Z_est, np.array(Map_Feature)


def urand(x_train, z_train, x_test, z_test, bounds_x1, bounds_x2, n_train, n, kernel, ground_truth, GP, x1mesh, x2mesh, mc_loop, SaveName, save='False'):

    NMSE_ = []
    Map_Feature_ = []
    for mc in range(mc_loop):
        X = x_train.copy()  # A
        Z = z_train.copy().reshape(-1, 1)
        x1, x2 = np.meshgrid(x1mesh, x2mesh)
        d1 = x1.shape[0]
        d2 = x1.shape[1]
        NMSE = []
        Map_Feature = []

        for i in range(n_train, n + 1):
            gp = z_fit(X[:i, :], Z[:i], kernel)
            Z_est = z_estimate(x_test, gp)
            nmse = normalized_mean_squared_error(z_test, Z_est)
            NMSE.append(nmse)
            map_feature = map_features(x1mesh, x2mesh, x1, x2, Z_est.reshape(d1, d2))
            Map_Feature.append(map_feature)

            x = np.random.uniform(bounds_x1, bounds_x2, [1, 2])
            X = np.concatenate([X, x.reshape(1, -1)])
            Z = np.concatenate([Z, [ground_truth(x.reshape(1, -1), GP)]])

        NMSE_.append(NMSE)
        Map_Feature_.append(Map_Feature)

    if save:
        df1 = pd.DataFrame(X)
        df1.to_csv(SaveName + '/X_UR.csv', header=False, index=False)
        df2 = pd.DataFrame(Z)
        df2.to_csv(SaveName + '/Z_UR.csv', header=False, index=False)
        df3 = pd.DataFrame(np.array(NMSE_))
        df3.to_csv(SaveName + '/NMSE_UR.csv', header=False, index=False)
        df4 = pd.DataFrame(np.array(Map_Feature_).flatten())
        df4.to_csv(SaveName + '/Map_Feature_UR.csv', header=False, index=False)

    return X, Z, np.array(NMSE_), Z_est, np.array(Map_Feature_)


def gpe(x_grid, x_train, z_train, x_test, z_test, kernel, n_train, n, method, ground_truth, GP, x1mesh, x2mesh, SaveName, save='False'):
    X = x_train.copy()  # A
    Z = z_train.copy().reshape(-1, 1)
    Xb = x_grid.copy()

    x1, x2 = np.meshgrid(x1mesh, x2mesh)
    d1 = x1.shape[0]
    d2 = x1.shape[1]

    NMSE = []
    Map_Feature = []

    for i in range(n_train, n + 1):
        # calculate error
        gp = z_fit(X, Z, kernel)
        Z_est = z_estimate(x_test, gp)
        nmse = normalized_mean_squared_error(z_test, Z_est)
        NMSE.append(nmse)
        map_feature = map_features(x1mesh, x2mesh, x1, x2, Z_est.reshape(d1, d2))
        Map_Feature.append(map_feature)

        kernel__ = clone(gp.kernel_)
        Zb = z_estimate(Xb, gp)

        if method == 'entropy':
            Sigmay2 = np.diagonal(kernel__(Xb))
            SigmayA = kernel__(Xb, X)
            invSigmaAA = inv_pd_mat(kernel__(X))
            SigmaAy = kernel__(X, Xb)
            elem1 = np.dot(SigmayA, invSigmaAA)
            elem2 = np.dot(elem1, SigmaAy)
            # elem2 = (elem1 * SigmaAy.T).sum(-1)
            # dy = Sigmay2 - elem2
            dy = Sigmay2 - np.diagonal(elem2)

        if method == 'entropy_mean':
            Sigmay2 = np.diagonal(kernel__(Zb))
            SigmayA = kernel__(Zb, Z)
            invSigmaAA = inv_pd_mat(kernel__(Z))
            SigmaAy = kernel__(Z, Zb)
            elem1 = np.dot(SigmayA, invSigmaAA)
            elem2 = np.dot(elem1, SigmaAy)
            # elem2 = (elem1 * SigmaAy.T).sum(-1)
            dy = Sigmay2 - np.diagonal(elem2)

        arg_max = np.argmax(dy)
        X = np.concatenate([X, Xb[arg_max].reshape(1, -1)])
        Z = np.concatenate([Z, [ground_truth(Xb[arg_max].reshape(1, -1), GP)]])
        Xb = np.delete(Xb, arg_max, axis=0)

    if save and method == 'entropy':
        df1 = pd.DataFrame(X)
        df1.to_csv(SaveName + '/X_GPE.csv', header=False, index=False)
        df2 = pd.DataFrame(Z)
        df2.to_csv(SaveName + '/Z_GPE.csv', header=False, index=False)
        df3 = pd.DataFrame(np.array(NMSE))
        df3.to_csv(SaveName + '/NMSE_GPE.csv', header=False, index=False)
        df4 = pd.DataFrame(np.array(Map_Feature))
        df4.to_csv(SaveName + '/Map_Feature_GPE.csv', header=False, index=False)

    if save and method == 'entropy_mean':
        df1 = pd.DataFrame(X)
        df1.to_csv(SaveName + '/X_GPEm.csv', header=False, index=False)
        df2 = pd.DataFrame(Z)
        df2.to_csv(SaveName + '/Z_GPEm.csv', header=False, index=False)
        df3 = pd.DataFrame(np.array(NMSE))
        df3.to_csv(SaveName + '/NMSE_GPEm.csv', header=False, index=False)
        df4 = pd.DataFrame(np.array(Map_Feature))
        df4.to_csv(SaveName + '/Map_Feature_GPEm.csv', header=False, index=False)

    return X, Z, np.array(NMSE), Z_est, np.array(Map_Feature)


def gprs(x_grid, x_train, z_train, x_test, z_test, z_max, bounds_x1, bounds_x2,
         kernel, n_train, n, p, ground_truth, GP, x1mesh, x2mesh, method, mc_loop, SaveName, save='False'):
    NMSE_ = []
    Map_Feature_ = []
    for mc in range(mc_loop):
        z_min = np.array([0.0])
        # z_max = np.array([1300.0 ** (1 / p)])
        z_max = z_max ** (1 / p)
        X = x_train.copy()
        Z = z_train.copy().reshape(-1, 1)
        Xb = x_grid.copy()
        gp = z_fit(X, Z, kernel)

        x1, x2 = np.meshgrid(x1mesh, x2mesh)
        d1 = x1.shape[0]
        d2 = x1.shape[1]

        NMSE = []
        Map_Feature = []

        iter = n_train
        while iter != n + 1:

            if method == 'continuous':
                x = np.random.uniform(bounds_x1, bounds_x2).reshape(1, -1)
            elif method == 'grid':
                idx = random.randrange(len(Xb) - 1)
                x = Xb[idx].reshape(1, -1)
            else:
                raise Exception('method is not correct')

            z = z_estimate(x, gp).reshape(-1, 1)

            if np.random.uniform(z_min, z_max) < z ** (1 / p):
                # calculate error
                gp = z_fit(X, Z, kernel)
                Z_est = z_estimate(x_test, gp)
                nmse = normalized_mean_squared_error(z_test, Z_est)
                NMSE.append(nmse)
                map_feature = map_features(x1mesh, x2mesh, x1, x2, Z_est.reshape(d1, d2))
                Map_Feature.append(map_feature)

                iter += 1
                X = np.concatenate([X, x])
                Z = np.concatenate([Z, [ground_truth(x, GP)]])
                if method == 'grid':
                    Xb = np.delete(Xb, idx, axis=0)

        NMSE_.append(NMSE)
        Map_Feature_.append(Map_Feature)

    if save:
        df1 = pd.DataFrame(X)
        df1.to_csv(SaveName + '/X_GPRSm.csv', header=False, index=False)
        df2 = pd.DataFrame(Z)
        df2.to_csv(SaveName + '/Z_GPRSm.csv', header=False, index=False)
        df3 = pd.DataFrame(np.array(NMSE_))
        df3.to_csv(SaveName + '/NMSE_GPRSm.csv', header=False, index=False)
        df4 = pd.DataFrame(np.array(Map_Feature_).flatten())
        df4.to_csv(SaveName + '/Map_Feature_GPRSm.csv', header=False, index=False)

    return X, Z, np.array(NMSE_), Z_est, np.array(Map_Feature_)


def wgpe(x_grid, x_train, z_train, x_test, z_test, kernel, n_train, n, ground_truth, GP, x1mesh, x2mesh, SaveName, save='False'):
    X = x_train.copy()
    Z = z_train.copy().reshape(-1, 1)
    Xb = x_grid.copy()

    x1, x2 = np.meshgrid(x1mesh, x2mesh)
    d1 = x1.shape[0]
    d2 = x1.shape[1]

    NMSE = []
    Map_Feature = []

    for i in range(n_train, n + 1):
        # calculate error
        gp = z_fit(X, Z, kernel)
        Z_est = z_estimate(x_test, gp)
        nmse = normalized_mean_squared_error(z_test, Z_est)
        NMSE.append(nmse)
        map_feature = map_features(x1mesh, x2mesh, x1, x2, Z_est.reshape(d1, d2))
        Map_Feature.append(map_feature)

        _, s = z_estimate(Xb, gp, return_std=True)
        arg_max = np.argmax(s)

        X = np.concatenate([X, Xb[arg_max].reshape(1, -1)])
        Z = np.concatenate([Z, [ground_truth(Xb[arg_max].reshape(1, -1), GP)]])
        Xb = np.delete(Xb, arg_max, axis=0)

    if save:
        df1 = pd.DataFrame(X)
        df1.to_csv(SaveName + '/X_WGPE.csv', header=False, index=False)
        df2 = pd.DataFrame(Z)
        df2.to_csv(SaveName + '/Z_WGPE.csv', header=False, index=False)
        df3 = pd.DataFrame(np.array(NMSE))
        df3.to_csv(SaveName + '/NMSE_WGPE.csv', header=False, index=False)
        df4 = pd.DataFrame(np.array(Map_Feature))
        df4.to_csv(SaveName + '/Map_Feature_WGPE.csv', header=False, index=False)

    return X, Z, np.array(NMSE), Z_est, np.array(Map_Feature)