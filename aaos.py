
from math import log
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import scipy.sparse.linalg as spla
from scipy.fft import fft, ifft
from scipy.optimize import newton_krylov

from lorenz import lorenz, dlorenz
from timestepping import theta_method, serial_steps

# ## === --- --- === ###
#
# Solve the Lorenz '63 model using implicit theta method
# solving the all-at-once system with ParaDiag
#
# dx/dt = sigma*(y - x)
# dy/dt = x*(rho - z) - y
# dz/dt = x*y - beta*z
#
# ## === --- --- === ###

# parameters

verbose = True
noise_level = 0.2
alpha = 1e-2
init_its = 1

inner_maxiter = 200

init_weight = 0.0

ntimescales = 1
ntperscale = 1024

Tlambda = log(10)/0.9
T = ntimescales*Tlambda
nt = ntperscale*ntimescales
dt = T/nt

theta = 0.5

x0 = 9
y0 = 13
z0 = 20

qinit = np.asarray([x0, y0, z0])

# setup timeseries

xyz = np.zeros((nt, 3))

xyz[:, 0] = x0
xyz[:, 1] = y0
xyz[:, 2] = z0

# ## timestepping toeplitz matrices

# # mass toeplitz

# first column
b1 = np.zeros(nt)
b1[0] = 1/dt
b1[1] = -1/dt

# # function toeplitz

# first column
b2 = np.zeros(nt)
b2[0] = -theta
b2[1] = -(1-theta)

# timestepping functions

#xyz[-1,:] = serial_steps(qinit, nt)

qNk = qinit
qNk1 = qinit
def aaosfunc(q, a=alpha):
    r = np.zeros_like(q)

    q0 = qinit
    q1 = q[0,:]
    r[0, :] = theta_method(lorenz, q1, q0, dt, theta)

    for i in range(nt-1):
        q0 = q[i, :]
        q1 = q[i+1, :]
        r[i+1, :] = theta_method(lorenz, q1, q0, dt, theta)
    return r


class BlockCirculantLinearOperator(spla.LinearOperator):
    def __init__(self, b1, b2, block_op, nx, alpha=1):
        self.nt = len(b1)
        self.nx = nx
        self.dim = self.nt*self.nx
        self.shape = tuple((self.dim, self.dim))
        self.dtype = b1.dtype

        self.gamma = alpha**(np.arange(self.nt, dtype=complex)/self.nt)

        self.eigvals1 = fft(b1*self.gamma, norm='backward')
        self.eigvals2 = fft(b2*self.gamma, norm='backward')

    def _make_blocks(self, qav):
        return tuple((block_op(l1, l2, qav)
                      for l1, l2 in zip(self.eigvals1, self.eigvals2)))

    def _to_eigvecs(self, v):
        y = np.matmul(np.diag(self.gamma), v)
        return fft(y, axis=0)

    def _from_eigvecs(self, v):
        y = ifft(v, axis=0)
        return np.matmul(np.diag(1/self.gamma), y)

    def _block_solve(self, v):
        for i in range(self.nt):
            v[i] = self.blocks[i].matvec(v[i])
        return v

    def _matvec(self, v):
        y = v.reshape((self.nt, self.nx))
        y = self._to_eigvecs(y)
        y = self._block_solve(y)
        y = self._from_eigvecs(y)
        return y.reshape(self.dim).real

    def update(self, q, f):
        q = q.reshape((self.nt, self.nx))
        self.qav = np.sum(q, axis=0)/q.shape[0]
        self.blocks = self._make_blocks(self.qav)


class InverseOperator(spla.LinearOperator):
    def __init__(self, mat):
        self.shape = mat.shape
        self.dtype = mat.dtype
        self.mat = mat

    def _matvec(self, v):
        return np.linalg.solve(self.mat, v)


def block_op(l1, l2, q):
    mat = l1*np.identity(3) + l2*dlorenz(q)
    return InverseOperator(mat)


P = BlockCirculantLinearOperator(b1, b2, block_op, 3, alpha=alpha)
P.update(xyz, aaosfunc(xyz))

stepper = partial(theta_method, theta=theta)
xyz[0] = serial_steps(lorenz, qinit, 1, dt, stepper)
for i in range(1, nt):
    xyz[i] = serial_steps(lorenz, xyz[i-1], 1, dt, stepper)

xyz += noise_level*np.random.normal(0, 1.0, xyz.shape)
qNk = xyz[-1,:]
qNk1 = qinit + init_weight*(qNk - qinit)


# ## solve all-at-once system

krylov_its = 0
newton_its = 0

# solver callbacks for nicer output


def gmres_callback(pr_norm):
    global krylov_its
    krylov_its += 1
    #print(f"krylov_its: {str(krylov_its).rjust(5,' ')} | residual: {pr_norm}")
    return

plt.plot(xyz[:, 0], xyz[:, 2])
plt.show()

def newton_callback(x, f):
    global krylov_its, newton_its

    newton_its += 1

    if verbose:
        #print("\n")
        print(f"newton_its: {str(newton_its).rjust(5,' ')} | krylov its: {str(krylov_its).rjust(5,' ')} | residual: {np.linalg.norm(f)}")
        #print("\n")
        q = x.reshape(nt, 3)
        plt.cla()
        plt.plot(q[:, 0], q[:, 2])
        plt.plot(q[-2:,0], q[-2:,2])
        plt.pause(0.1)
    krylov_its = 0


writs = 0
print(f"Waveform iteration: {writs} | Residual: {np.linalg.norm(aaosfunc(xyz, a=0))} | Tail norm: {alpha*np.linalg.norm(qNk-qNk1)}")
xyz = newton_krylov(aaosfunc, xyz, method='gmres',
                    #verbose=True,
                    maxiter=100,
                    f_rtol = 1e-5,
                    callback=newton_callback,
                    inner_M=P,
                    inner_maxiter=inner_maxiter,
                    inner_tol=1e-5,
                    inner_callback=gmres_callback,
                    inner_callback_type='pr_norm')
qNk1 = qNk
qNk = xyz[-1,:]
writs += 1

print(f"Waveform iteration: {writs} | Residual: {np.linalg.norm(aaosfunc(xyz, a=0))} | Tail norm: {alpha*np.linalg.norm(qNk-qNk1)}")

plt.plot(xyz[:, 0], xyz[:, 2])
plt.show()
