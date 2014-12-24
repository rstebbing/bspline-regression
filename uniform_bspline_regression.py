# uniform_bspline_regression.py

# Imports
import numpy as np

from scipy.linalg import block_diag
from scipy.optimize import fmin_bfgs

from uniform_bspline import Contour
from util import raise_if_not_shape


# Solver
class Solver(object):
    def __init__(self, contour):
        self._c = contour

        i = np.arange(self._c.num_control_points if self._c.is_closed else
                      self._c.num_control_points - 1)
        j = (i + 1) % self._c.num_control_points
        self._r_ij = i, j

    def minimise(self, Y, w, lambda_, u, X):
        w = np.atleast_1d(w)
        (N,) = w.shape

        Y = np.atleast_2d(Y)
        raise_if_not_shape('Y', Y, (N, self._c.dim))

        if lambda_ <= 0.0:
            raise ValueError('lambda_ <= 0.0 (= {})'.format(lambda_))

        u = np.atleast_1d(u)
        raise_if_not_shape('u', u, (N,))

        X = np.atleast_2d(X)
        raise_if_not_shape('X', X, (self._c.num_control_points, self._c.dim))

        self._Y = Y
        self._w = np.sqrt(w)
        self._lambda = np.sqrt(lambda_)

        def x_to_u_X(x):
            x = x.copy()
            u = x[:N]
            u[u < 0] = 0.0
            i = u >= self._c.num_segments
            u[i] = self._c.num_segments - 1e-9
            return u, x[N:].reshape(-1, 2)

        def f(x):
            u, X = x_to_u_X(x)
            r = self._r(u, X)
            return 0.5 * np.dot(r, r)

        def fprime(x):
            u, X = x_to_u_X(x)
            J = self._J(u, X)
            r = self._r(u, X)
            return np.dot(J.T, r)

        x = np.r_[u, X.ravel()]
        x1 = fmin_bfgs(f, x, fprime)
        return x_to_u_X(x1)

    def _r(self, u, X):
        R = self._w[:, np.newaxis] * (self._Y - self._c.M(u, X))

        i, j = self._r_ij
        Q = self._lambda * (X[j] - X[i])

        return np.r_[R.ravel(), Q.ravel()]

    def _J(self, u, X):
        d = self._c.dim
        Mu = -self._w[:, np.newaxis] * self._c.Mu(u, X)
        E = block_diag(*Mu.reshape(-1, d, 1))

        F = -np.repeat(self._w, d)[:,np.newaxis] * self._c.MX(u)

        i, j = self._r_ij
        n = i.shape[0]
        G = np.zeros((n * d, self._c.num_control_points * d), dtype=float)
        r = np.arange(n)
        for k in range(d):
            G[d * r + k, d * i + k] = -self._lambda
            G[d * r + k, d * j + k] =  self._lambda

        Z = np.zeros((n * d, E.shape[1]))
        J = np.r_['0,2', np.c_[E, F],
                         np.c_[Z, G]]
        return J


# Main
import matplotlib.pyplot as plt

# Example contour `c` and control points `X`.
c = Contour(2, 5, 2, is_closed=False)
X = np.r_[0.0, 0,
          1, 0,
          2, 1,
          3, 0,
          4, 0].reshape(-1, 2)

u = c.uniform_parameterisation(128)
x, y = c.M(u, X).T
dx, dy = c.Mu(u, X).T

f, axs = plt.subplots(3, 1)
axs[0].plot(x, y)
axs[1].plot(u, dx)
axs[2].plot(u, dy)

# Example `Y`.
x = np.linspace(0.0, np.pi, 64)
y = np.sin(x)
np.random.seed(0)
x += 0.01 * np.random.randn(y.size)
y += 0.02 * np.random.randn(y.size)
Y = np.c_[x, y]
Y *= 0.5
Y[:, 0] += 1.0

# Example parameters.
w = 100.0 * np.ones(Y.shape[0])
lambda_ = 1e-3

# Initialise `u`.
import scipy.spatial
u0 = c.uniform_parameterisation(16)
D = scipy.spatial.distance.cdist(Y, c.M(u0, X))
u = u0[np.argmin(D, axis=1)]

# Minimise.
s = Solver(c)
u1, X1 = s.minimise(Y, w, lambda_, u, X)

f, ax = plt.subplots()
ax.set_aspect('equal')

x, y = Y.T
ax.plot(x, y, 'ro')

for y, y1 in zip(Y, c.M(u1, X1)):
    x, y = np.r_['0,2', y, y1].T
    ax.plot(x, y, 'r.-')

x, y = X1.T
ax.plot(x, y, 'bo--')

x, y = c.M(c.uniform_parameterisation(256), X1).T
ax.plot(x, y, 'b-', lw=3)

x, y = X.T
ax.plot(x, y, 'mo--')

x, y = c.M(c.uniform_parameterisation(256), X).T
ax.plot(x, y, 'm-', lw=3)

plt.show()
