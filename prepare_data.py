import scipy.io
import numpy as np

from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel as C


def read_data(data_name, header):
    data = scipy.io.loadmat(data_name)
    data = data[header]

    x = np.vstack((data[:, 0], data[:, 1])).T
    z = data[:, 2]
    zw = np.sqrt(z)

    # kernel = C(1.0) * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
    #          + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-10, 0.05))
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3))

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
    gp.fit(x, zw)
    kernel = clone(gp.kernel_)
    return gp, kernel


def ground_truth(x, gp):
    zw_est, sw = gp.predict(x, return_std=True)
    z_est = zw_est ** 2
    return z_est


def select_init_points(x1_bounds, x2_bounds, n, method):
    if method == 'grid':
        x1 = np.linspace(x1_bounds[0] + 1, x2_bounds[0] - 1, int(np.sqrt(n)))
        x2 = np.linspace(x1_bounds[1] + 1, x2_bounds[1] - 1, int(np.sqrt(n)))
        xv1, xv2 = np.meshgrid(x1, x2)
        x = np.vstack((xv1.flatten(), xv2.flatten())).T
        return x

    if method == 'random':
        x = np.random.uniform(x1_bounds, x2_bounds, [n, 2])
        return x

    else:
        raise Exception('method is not correct')