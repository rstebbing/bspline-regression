# uniform_bspline_regression.py

# Imports
import numpy as np
import scipy.linalg

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

    def minimise(self, Y, w, lambda_, u, X, max_num_iterations):
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

        # TODO Settings
        self._max_radius = 1e12
        self._min_radius = 1e-9

        self._decrease_factor = 2.0
        self._radius = 1e-4

        n, d = u.shape[0], self._c.dim

        reuse = False

        for i in range(max_num_iterations):
            # Compute Levenberg-Marquardt Step.
            if self._radius <= self._min_radius:
                break

            # E is a block-diagonal matrix of `N` blocks, each of shape
            # `(dim, 1)`.
            # `E[i]` is a vector of shape `(dim,)` of the ith block.
            if not reuse:
                E = self._E(u, X)
                F, G = self._F(u), self._G()

                EtF = np.empty((n, F.shape[1]))
                for i in range(n):
                    EtF[i] = np.dot(E[i], F[d * i: d * (i + 1)])
                FtE = EtF.T

                H0 = np.dot(F.T, F) + np.dot(G.T, G)

                ra, rb = self._r(u, X)
                r = np.r_[ra, rb]
                e = 0.5 * (np.dot(r, r))

                # a = Et * ra
                a = (E * ra.reshape(-1, d)).sum(axis=1)
                b = np.dot(F.T, ra) + np.dot(G.T, rb)

            diag_EtEi = 1.0 / ((E * E).sum(axis=1) + 1.0 / self._radius)

            H = (H0 + np.diag([1.0 / self._radius] * H0.shape[0])
                    - np.dot(FtE, diag_EtEi[:, np.newaxis] * EtF))
            try:
                c_and_lower = scipy.linalg.cho_factor(H)
            except scipy.linalg.LinAlgError:
                # Step is invalid.
                self._reject_step()
                reuse = True

            t = b - np.dot(FtE, diag_EtEi * a)
            v1 = scipy.linalg.cho_solve(c_and_lower, t)
            v0 = diag_EtEi * (a - np.dot(EtF, v1))
            delta_u = -v0
            delta_X = -v1.reshape(-1, d)

            # delta = np.r_[v0, v1]
            # E_ = scipy.linalg.block_diag(*E[..., np.newaxis])
            # Z = np.zeros((G.shape[0], E_.shape[1]))
            # J = np.r_['0,2', np.c_[E_, F],
            #                  np.c_[Z, G]]
            # b_ = np.dot(J.T, np.r_[ra, rb])
            # A_ = np.dot(J.T, J) + np.diag([1.0 / self._radius] * J.shape[1])
            # assert np.allclose(delta, np.dot(np.linalg.inv(A_), b_), atol=1e-4)

            Jdelta = np.r_[(E * delta_u[:, np.newaxis]).ravel() +
                            np.dot(F, delta_X.ravel()),
                           np.dot(G, delta_X.ravel())]
            # Jdelta_ = np.dot(J, delta)
            # assert np.allclose(Jdelta, Jdelta, atol=1e-4)

            model_cost_change = -np.dot(Jdelta, r + Jdelta / 2.0)
            assert model_cost_change >= 0.0

            u1 = u + delta_u
            u1[u1 < 0] = 0.0
            u1[u1 >= self._c.num_segments] = (self._c.num_segments +
                                              np.finfo(u1.dtype).epsneg)
            X1 = X + delta_X

            e1 = self._e(u1, X1)
            step_quality = (e - e1) / model_cost_change
            if step_quality > 0:
                self._accept_step(step_quality)
                u, X = u1, X1
                reuse = False
            else:
                self._reject_step()
                reuse = True

        return u, X

    def _accept_step(self, step_quality):
        assert step_quality > 0.0
        self._radius /= max(1.0 / 3.0,
                            1.0 - (2.0 * step_quality - 1.0)**3)
        self._radius = min(self._max_radius, self._radius)
        self._decrease_factor = 2

    def _reject_step(self):
        self._radius /= self._decrease_factor
        self._decrease_factor *= 2

    def _r(self, u, X):
        R = self._w[:, np.newaxis] * (self._Y - self._c.M(u, X))

        i, j = self._r_ij
        Q = self._lambda * (X[j] - X[i])

        return R.ravel(), Q.ravel()

    def _e(self, u, X):
        ra, rb = self._r(u, X)
        return 0.5 * (np.dot(ra, ra) + np.dot(rb, rb))

    def _E(self, u, X):
        Mu = -self._w[:, np.newaxis] * self._c.Mu(u, X)
        return Mu

    def _F(self, u):
        return -np.repeat(self._w, self._c.dim)[:,np.newaxis] * self._c.MX(u)

    def _G(self):
        i, j = self._r_ij
        n, d = i.shape[0], self._c.dim
        G = np.zeros((n * d, self._c.num_control_points * d), dtype=float)
        r = np.arange(n)
        for k in range(d):
            G[d * r + k, d * i + k] = -self._lambda
            G[d * r + k, d * j + k] =  self._lambda
        return G


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
w = np.ones(Y.shape[0])
lambda_ = 1e-2

# Initialise `u`.
import scipy.spatial
u0 = c.uniform_parameterisation(16)
D = scipy.spatial.distance.cdist(Y, c.M(u0, X))
u = u0[np.argmin(D, axis=1)]

# Minimise.
s = Solver(c)
u1, X1 = s.minimise(Y, w, lambda_, u, X, 200)

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
