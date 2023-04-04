import numpy as np


def lorenz(q, sigma=10., beta=8./3, rho=28.):
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    dxdt = sigma*(y - x)
    dydt = x*(rho - z) - y
    dzdt = x*y - beta*z
    return np.asarray([dxdt, dydt, dzdt])


def dlorenz(q, sigma=10., beta=8./3, rho=28.):
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]

    dxdx = -sigma
    dxdy = sigma
    dxdz = 0

    dydx = rho - z
    dydy = -1
    dydz = -x

    dzdx = y
    dzdy = x
    dzdz = -beta

    return np.asarray([[dxdx, dxdy, dxdz],
                       [dydx, dydy, dydz],
                       [dzdx, dzdy, dzdz]])
