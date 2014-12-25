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
        self._ij = i, j

    def minimise(self, Y, w, lambda_, u, X, max_num_iterations=100,
                 min_radius=1e-9, max_radius=1e12, initial_radius=1e4):
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

        self._max_radius = max_radius
        self._min_radius = min_radius

        self._decrease_factor = 2.0
        self._radius = initial_radius

        N, d = u.shape[0], self._c.dim

        update_schur_components = True

        for i in range(max_num_iterations):
            if self._radius <= self._min_radius:
                # Terminate if the trust region radius is too small.
                break

            # Compute Levenberg-Marquardt step.
            if update_schur_components:
                # The actual E is a block-diagonal matrix of `N` blocks, each
                # of shape `(dim, 1)`.
                # Here, `E[i]` is a vector of shape `(dim,)` of the `i`th
                # block.
                E, F, G = self._E(u, X), self._F(u), self._G()

                EtF = np.empty((N, F.shape[1]))
                for i in range(N):
                    EtF[i] = np.dot(E[i], F[d * i: d * (i + 1)])
                FtE = EtF.T

                H0 = np.dot(F.T, F) + np.dot(G.T, G)

                e, (ra, rb, r) = self._e(u, X, return_all=True)

                # a = Et * ra
                a = (E * ra.reshape(-1, d)).sum(axis=1)
                b = np.dot(F.T, ra) + np.dot(G.T, rb)

            # `diag_EtEi` is the vector so that `np.diag(diag_EtEi)` is equal
            # to (Et * E + diag)^-1, where `diag` is the diagonal matrix with
            # entries equal to `1.0 / self._radius`.
            diag_EtEi = 1.0 / ((E * E).sum(axis=1) + 1.0 / self._radius)

            # Solve the Schur reduced system for `delta_u` and `delta_X`, the
            # updates for `u` and `X` respectively.
            H = (H0 + np.diag([1.0 / self._radius] * H0.shape[0])
                    - np.dot(FtE, diag_EtEi[:, np.newaxis] * EtF))
            try:
                c_and_lower = scipy.linalg.cho_factor(H)
            except scipy.linalg.LinAlgError:
                # Step is invalid.
                self._reject_step()
                update_schur_components = False
                continue

            t = b - np.dot(FtE, diag_EtEi * a)
            v1 = scipy.linalg.cho_solve(c_and_lower, t)
            v0 = diag_EtEi * (a - np.dot(EtF, v1))
            delta_u = -v0
            delta_X = -v1.reshape(-1, d)

            # Equivalent:
            #   E_ = scipy.linalg.block_diag(*E[..., np.newaxis])
            #   Z = np.zeros((G.shape[0], E_.shape[1]))
            #   J = np.r_['0,2', np.c_[E_, F],
            #                    np.c_[Z, G]]
            #   b_ = np.dot(J.T, np.r_[ra, rb])
            #   A_ = np.dot(J.T, J) + np.diag(
            #       [1.0 / self._radius] * J.shape[1])
            #   assert np.allclose(np.r_[v0, v1],
            #                      np.dot(np.linalg.inv(A_), b_), atol=1e-4)

            # Evaluate the change in energy as expected by the quadratic
            # approximation.
            Jdelta = np.r_[(E * delta_u[:, np.newaxis]).ravel() +
                            np.dot(F, delta_X.ravel()),
                           np.dot(G, delta_X.ravel())]

            # Equivalent:
            #   Jdelta_ = np.dot(J, -np.r_[v0, v1])
            #   assert np.allclose(Jdelta, Jdelta_, atol=1e-4)

            model_cost_change = -np.dot(Jdelta, r + Jdelta / 2.0)
            assert model_cost_change >= 0.0

            # Evaluate the updated coordinates `u1` and control points `X1`.
            u1 = self._c.clip(u + delta_u)
            X1 = X + delta_X

            # Accept the updates if the energy has decreased, and reject it
            # otherwise. Also update the trust region radius depending on how
            # well the quadratic approximation modelled the change in energy.
            e1 = self._e(u1, X1)
            step_quality = (e - e1) / model_cost_change
            if step_quality > 0:
                self._accept_step(step_quality)
                u, X = u1, X1
                update_schur_components = True
            else:
                self._reject_step()
                update_schur_components = False

        return u, X

    def _accept_step(self, step_quality):
        # Refer to Ceres and "Methods for Non-Linear Least Squares Problems"
        # by Madsen.
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

        i, j = self._ij
        Q = self._lambda * (X[j] - X[i])

        return R.ravel(), Q.ravel()

    def _e(self, u, X, return_all=False):
        ra, rb = self._r(u, X)
        r = np.r_[ra, rb]
        e = 0.5 * np.dot(r, r)
        return (e if not return_all else
                (e, (ra, rb, r)))

    def _E(self, u, X):
        Mu = -self._w[:, np.newaxis] * self._c.Mu(u, X)
        return Mu

    def _F(self, u):
        return -np.repeat(self._w, self._c.dim)[:,np.newaxis] * self._c.MX(u)

    def _G(self):
        i, j = self._ij
        N, d = i.shape[0], self._c.dim
        G = np.zeros((N * d, self._c.num_control_points * d), dtype=float)
        r = np.arange(N)
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
