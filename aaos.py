
import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse.linalg as spla
from scipy.fft import fft, ifft
from scipy.optimize import newton_krylov

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

serial_init = True
verbose = True
noise_level = 0.2
tail_term = True
alpha = 1e-0
init_its = 1
nwrs = 15
wr_tol = 1e-5

inner_maxiter = 200

init_weight = 0.0

nscales = 5.0
T = nscales*2.5
nt = int(nscales*256)
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
    dydy = 1
    dydz = -x

    dzdx = y
    dzdy = x
    dzdz = -beta

    return np.asarray([[dxdx, dxdy, dxdz],
                       [dydx, dydy, dydz],
                       [dzdx, dzdy, dzdz]])

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

def theta_method(q1, q0):
    return q1 - q0 - dt*(theta*lorenz(q1) + (1 - theta)*lorenz(q0))


def serial_steps(q0, n):
    q1 = q0
    for i in range(n):
        def f(q):
            return theta_method(q, q0)
        q1 = newton_krylov(f, q0)
        q0 = q1
    return q1

#xyz[-1,:] = serial_steps(qinit, nt)

qNk = qinit
qNk1 = qinit
def aaosfunc(q, a=alpha):
    r = np.zeros_like(q)

    q0 = qinit
    if tail_term:
        q0 = q0 +  a*(qNk - qNk1)

    q1 = q[0,:]
    r[0, :] = theta_method(q1, q0)

    for i in range(nt-1):
        q0 = q[i, :]
        q1 = q[i+1, :]
        r[i+1, :] = theta_method(q1, q0)
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
        eigvals = zip(self.eigvals1, self.eigvals2)
        return tuple((block_op(l1, l2, qav)
                      for l1, l2 in eigvals))

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

if serial_init:
    xyz[0] = serial_steps(qinit, 1)
    for i in range(1, nt):
        xyz[i] = serial_steps(xyz[i-1], 1)
else:
    for i in range(init_its):
        xyz -= P.matvec(aaosfunc(xyz, a=0).reshape(nt*3)).reshape(nt,3)

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


for i in range(nwrs):
    print(f"Waveform iteration: {i} | Residual: {np.linalg.norm(aaosfunc(xyz, a=0))} | Tail norm: {alpha*np.linalg.norm(qNk-qNk1)}")
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
    if np.linalg.norm(aaosfunc(xyz)) < wr_tol: break

print(f"Waveform iteration: {nwrs} | Residual: {np.linalg.norm(aaosfunc(xyz, a=0))} | Tail norm: {alpha*np.linalg.norm(qNk-qNk1)}")

plt.plot(xyz[:, 0], xyz[:, 2])
plt.show()
