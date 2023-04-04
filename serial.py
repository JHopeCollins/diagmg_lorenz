
from math import log
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton_krylov

from lorenz import lorenz
from timestepping import theta_method, serial_steps
from functools import partial

# ## === --- --- === ###
#
# Solve the Lorenz '63 model using implicit theta method
# and serial timestepping
#
# dx/dt = sigma*(y - x)
# dy/dt = x*(rho - z) - y
# dz/dt = x*y - beta*z
#
# ## === --- --- === ###

# parameters

ntimescales = 1
ntperscale = 1024

Tlambda = log(10)/0.9
T = ntimescales*Tlambda
nt = ntperscale*ntimescales

dt = T/nt
theta = 0.0

# initial conditions

x0 = 9
y0 = 13
z0 = 20

qinit = np.asarray([x0, y0, z0])

# timestepping loop

stepper = partial(theta_method, theta=theta)

xyz = serial_steps(lorenz, qinit,
                   nt, dt, stepper,
                   timeseries=True)

# plot

plt.plot(xyz[:, 0], xyz[:, 2])
plt.show()
