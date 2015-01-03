##########################################
# File: uniform_bspline_regression.py    #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

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
        # Ensure input dimensions and values are valid.
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

        # Set `_Y`, `_w`, and `_lambda` for internal evaluation methods.
        self._Y = Y
        self._w = np.sqrt(w)
        self._lambda = np.sqrt(lambda_)

        # Set internal variables for `_accept_step` and `_reject_step`.
        self._min_radius = min_radius
        self._max_radius = max_radius

        self._decrease_factor = 2.0
        self._radius = initial_radius

        if return_all:
            states = []
            def save_state(u, X, *args):
                states.append((u.copy(), X.copy()) + args)
        else:
            def save_state(*args):
                pass

        d = self._c.dim

        save_state(u, X, self._e(u, X), self._radius)

        t0 = time()
        update_schur_components, has_converged = True, False
        for i in range(max_num_iterations):
            if self._radius <= self._min_radius:
                # Terminate if the trust region radius is too small.
                has_converged = True
                break

            # Compute damped Newton step.
            if update_schur_components:
                # Error and residual components.
                e, (ra, rb, r) = self._e(u, X, return_all=True)

                # First derivatives.
                # The actual E is a block-diagonal matrix of `N` blocks, each
                # of shape `(dim, 1)`.
                # Here, `E[i]` is a vector for the `i`th block and is of shape
                # `(dim,)`.
                E, F, G = self._E(u, X), self._F(u), self._G()

                # Second derivatives.
                # `P` is the same dimensions as `E`.
                P, Q = self._P(u, X), self._Q(u)

                # (Partial) Schur diagonal.
                D_EtE_rP = ((E * E).sum(axis=1) +
                            (P * ra.reshape(-1, d)).sum(axis=1))

                # Schur upper right and lower left blocks.
                EtF_rQ = np.empty((N, F.shape[1]))
                for i in range(N):
                    EtF_rQ[i] = (np.dot(E[i], F[d * i: d * (i + 1)]) +
                                  np.dot(ra[d * i: d * (i + 1)],
                                         Q[d * i: d * (i + 1)]))
                FtE_rQ = EtF_rQ.T

                # (Partial) Schur lower right block.
                H0 = np.dot(F.T, F) + np.dot(G.T, G)

                # Schur right-hand side (a = Et * ra).
                a = (E * ra.reshape(-1, d)).sum(axis=1)
                b = np.dot(F.T, ra) + np.dot(G.T, rb)

            # `D` is the vector of the inverse of the complete Schur diagonal.
            D = 1.0 / (D_EtE_rP + 1.0 / self._radius)

            # Solve the Schur reduced system for `delta_u` and `delta_X`.
            H = (H0 + np.diag([1.0 / self._radius] * H0.shape[0])
                    - np.dot(FtE_rQ, D[:, np.newaxis] * EtF_rQ))
            try:
                c_and_lower = scipy.linalg.cho_factor(H)
            except scipy.linalg.LinAlgError:
                # Step is invalid.
                self._reject_step()
                update_schur_components = False
                continue

            t = b - np.dot(FtE_rQ, D * a)
            v1 = scipy.linalg.cho_solve(c_and_lower, t)
            v0 = D * (a - np.dot(EtF_rQ, v1))
            delta_u = -v0
            delta_X = -v1.reshape(-1, d)

            # Equivalent.
            if self.DEBUG:
                J, S = self._J(u, X), self._S(u, X)

                b_ = np.dot(J.T, np.r_[ra, rb])
                A_ = (np.dot(J.T, J) + S +
                      np.diag([1.0 / self._radius] * J.shape[1]))
                assert np.allclose(np.r_[v0, v1],
                                   np.dot(np.linalg.inv(A_), b_), atol=1e-4)

            # Evaluate the change in energy as expected by the quadratic
            # approximation.
            Jdelta = np.r_[(E * delta_u[:, np.newaxis]).ravel() +
                            np.dot(F, delta_X.ravel()),
                           np.dot(G, delta_X.ravel())]

            Sdelta = np.r_[D_EtE_rP * delta_u +
                            np.dot(EtF_rQ, delta_X.ravel()),
                           np.dot(EtF_rQ.T, delta_u) +
                            np.dot(H0, delta_X.ravel())]
            model_e_decrease = -(np.dot(r, Jdelta) +
                                 0.5 * np.dot(np.r_[delta_u, delta_X.ravel()],
                                              Sdelta))
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
        return -self._w * self._c.Mu(u, X)

    def _P(self, u, X):
        return -self._w * self._c.Muu(u, X)

    def _F(self, u):
        return -self._w.reshape(-1, 1) * self._c.MX(u)

    def _Q(self, u):
        return -self._w.reshape(-1, 1) * self._c.MuX(u)

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

    def _S(self, u, X):
        """Calculate symmetric dense matrix of second and mixed derivatives.
        For debugging use only."""
        P, Q = self._P(u, X), self._Q(u)
        ra, rb = self._r(u, X)

        N, d = u.shape[0], self._c.dim

        n = N + X.size
        Su = np.zeros((n, n), dtype=float)
        Su[np.diag_indices(N)] = 0.5 * (P * ra.reshape(-1, d)).sum(axis=1)
        for i in range(N):
            for j in range(d):
                Su[i, N:] += ra[d * i + j] * Q[d * i + j]
        return Su + Su.T

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('--output-all', default=False, action='store_true')
    parser.add_argument('--max-num-iterations', type=int, default=100)
    parser.add_argument('--min-radius', type=float, default=1e-9)
    parser.add_argument('--max-radius', type=float, default=1e12)
    parser.add_argument('--initial-radius', type=float, default=1e4)
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

    print 'Solver:'
    print '  max_num_iterations:', args.max_num_iterations
    print '  min_radius: {:g}'.format(args.min_radius)
    print '  max_radius: {:g}'.format(args.max_radius)
    print '  initial_radius: {:g}'.format(args.initial_radius)

    print 'Solver Output:'
    ((u1, X1),
     has_converged, states, num_iterations, time_taken) = Solver(c).minimise(
        Y, w, lambda_, u, X,
        max_num_iterations=args.max_num_iterations,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        initial_radius=args.initial_radius,
        return_all=True)
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
        z['e'], z['radius'] = states[-1][2:]
        with open(args.output_path, 'wb') as fp:
            fp.write(json.dumps(z, indent=4))


if __name__ == '__main__':
    main()
