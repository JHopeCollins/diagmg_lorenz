import numpy as np
from scipy.optimize import newton_krylov


def theta_method(f, q1, q0, dt, theta):
    return q1 - q0 - dt*(theta*f(q1) + (1 - theta)*f(q0))


def forward_euler(f, q1, q0, dt):
    return theta_method(f, q1, q0, dt, 0)


def backward_euler(f, q1, q0, dt):
    return theta_method(f, q1, q0, dt, 1)


def trapezium_rule(f, q1, q0, dt):
    return theta_method(f, q1, q0, dt, 0.5)


def serial_steps(f, q0, nt, dt, stepper, timeseries=False):

    if timeseries:
        nx = len(q0)
        series = np.zeros((nt+1, nx))
        series[0, :] = q0

    for i in range(nt):

        def func(q):
            return stepper(f, q, q0, dt)

        q1 = newton_krylov(func, q0)
        q0 = q1

        if timeseries:
            series[i+1, :] = q0

    if timeseries:
        return series
    else:
        return q1
