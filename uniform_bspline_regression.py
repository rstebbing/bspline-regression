# uniform_bspline_regression.py

# Imports
import argparse
import json
import numpy as np
import os
import scipy.linalg
from time import time

from uniform_bspline import Contour
from util import raise_if_not_shape


# Solver
class Solver(object):
    DEBUG = False

    def __init__(self, contour):
        self._c = contour

        i = np.arange(self._c.num_control_points if self._c.is_closed else
                      self._c.num_control_points - 1)
        j = (i + 1) % self._c.num_control_points
        self._ij = i, j

    def minimise(self, Y, w, lambda_, u, X, max_num_iterations=100,
                 min_radius=1e-9, max_radius=1e12, initial_radius=1e4,
                 return_all=False):
        w = np.atleast_2d(w)
        N = w.shape[0]
        raise_if_not_shape('w', w, (N, self._c.dim))
        if np.any(w <= 0.0):
            raise ValueError('w <= 0.0')

        Y = np.atleast_2d(Y)
        raise_if_not_shape('Y', Y, (N, self._c.dim))

        if lambda_ <= 0.0:
            raise ValueError('lambda_ <= 0.0 (= {})'.format(lambda_))

        u = np.atleast_1d(u)
        raise_if_not_shape('u', u, (N,))
        u = self._c.clip(u)

        X = np.atleast_2d(X)
        raise_if_not_shape('X', X, (self._c.num_control_points, self._c.dim))

        self._Y = Y
        self._w = np.sqrt(w)
        self._lambda = np.sqrt(lambda_)

        self._min_radius = min_radius
        self._max_radius = max_radius

        self._decrease_factor = 2.0
        self._radius = initial_radius

        N, d = u.shape[0], self._c.dim

        if return_all:
            states = []
            def save_state(u, X, *args):
                states.append((u.copy(), X.copy()) + args)
        else:
            def save_state(*args):
                pass

        save_state(u, X, self._e(u, X), self._radius)

        t0 = time()
        update_schur_components, has_converged = True, False
        for i in range(max_num_iterations):
            if self._radius <= self._min_radius:
                # Terminate if the trust region radius is too small.
                has_converged = True
                break

            # Compute Levenberg-Marquardt step.
            if update_schur_components:
                # The actual E is a block-diagonal matrix of `N` blocks, each
                # of shape `(dim, 1)`.
                # Here, `E[i]` is a vector for the `i`th block and is of shape
                # `(dim,)`.
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

            # Equivalent.
            if self.DEBUG:
                J = self._J(u, X)
                b_ = np.dot(J.T, np.r_[ra, rb])
                A_ = np.dot(J.T, J) + np.diag(
                    [1.0 / self._radius] * J.shape[1])
                assert np.allclose(np.r_[v0, v1],
                                   np.dot(np.linalg.inv(A_), b_), atol=1e-4)

            # Evaluate the change in energy as expected by the quadratic
            # approximation.
            Jdelta = np.r_[(E * delta_u[:, np.newaxis]).ravel() +
                            np.dot(F, delta_X.ravel()),
                           np.dot(G, delta_X.ravel())]

            # Equivalent.
            if self.DEBUG:
                Jdelta_ = np.dot(J, -np.r_[v0, v1])
                assert np.allclose(Jdelta, Jdelta_, atol=1e-4)

            model_e_decrease = -np.dot(Jdelta, r + Jdelta / 2.0)
            assert model_e_decrease >= 0.0

            # Evaluate the updated coordinates `u1` and control points `X1`.
            u1 = self._c.clip(u + delta_u)
            X1 = X + delta_X

            # Accept the updates if the energy has decreased, and reject it
            # otherwise. Also update the trust region radius depending on how
            # well the quadratic approximation modelled the change in energy.
            e1 = self._e(u1, X1)
            step_quality = (e - e1) / model_e_decrease
            if step_quality > 0:
                save_state(u1, X1, e1, self._radius)

                self._accept_step(step_quality)
                e, u, X = e1, u1, X1
                update_schur_components = True
            else:
                self._reject_step()
                update_schur_components = False

        t1 = time()

        return (((u, X), has_converged, states, i, t1 - t0) if return_all else
                (u, X))

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
        R = self._w * (self._Y - self._c.M(u, X))

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
        Mu = -self._w * self._c.Mu(u, X)
        return Mu

    def _F(self, u):
        return -self._w.reshape(-1, 1) * self._c.MX(u)

    def _G(self):
        i, j = self._ij
        N, d = i.shape[0], self._c.dim
        G = np.zeros((N * d, self._c.num_control_points * d), dtype=float)
        r = np.arange(N)
        for k in range(d):
            G[d * r + k, d * i + k] = -self._lambda
            G[d * r + k, d * j + k] =  self._lambda
        return G

    def _J(self, u, X):
        """Calculate dense Jacobian. For debugging use only."""
        E, F, G = self._E(u, X), self._F(u), self._G()
        E_ = scipy.linalg.block_diag(*E[..., np.newaxis])
        Z = np.zeros((G.shape[0], E_.shape[1]))
        return np.r_['0,2', np.c_[E_, F],
                            np.c_[Z, G]]


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('--output-all', default=False, action='store_true')
    args = parser.parse_args()

    print 'Input:', args.input_path
    with open(args.input_path, 'rb') as fp:
        z = json.load(fp)

    degree, num_control_points, dim, is_closed = (
        z['degree'], z['num_control_points'], z['dim'], z['is_closed'])

    print '  degree:', degree
    print '  num_control_points:', num_control_points
    print '  dim:', dim
    print '  is_closed:', is_closed
    c = Contour(degree, num_control_points, dim, is_closed=is_closed)

    Y, w, u, X = map(lambda k: np.array(z[k]), 'YwuX')
    lambda_ = z['lambda_']
    print '  num_data_points:', Y.shape[0]
    print '  lambda_:', lambda_

    print 'Solving:'
    ((u1, X1),
     has_converged, states, num_iterations, time_taken) = Solver(c).minimise(
        Y, w, lambda_, u, X, return_all=True)
    print '  has_converged:', has_converged
    print '  num_iterations:', num_iterations
    print '  num_successful_iterations:', len(states) - 1
    print '  initial_energy: {:.3e}'.format(states[0][2])
    print '  final_energy: {:.3e}'.format(states[-1][2])
    print '  time_taken: {:.3e}s'.format(time_taken)
    print '  per_iteration: {:.3e}s'.format(time_taken / num_iterations)

    print 'Output:', args.output_path
    if args.output_all:
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        for i, (u, X, e, radius) in enumerate(states):
            z['u'], z['X'] = u.tolist(), X.tolist()
            z['e'], z['radius'] = e, radius
            output_path = os.path.join(args.output_path, '{}.json'.format(i))
            print '  ', output_path
            with open(output_path, 'wb') as fp:
                fp.write(json.dumps(z, indent=4))

    else:
        z['u'], z['X'] = u1.tolist(), X1.tolist()
        with open(args.output_path, 'wb') as fp:
            fp.write(json.dumps(z, indent=4))


if __name__ == '__main__':
    main()
