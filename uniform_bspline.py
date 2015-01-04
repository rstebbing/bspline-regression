##########################################
# File: uniform_bspline.py               #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import numpy as np
import sympy as sp

from itertools import groupby
from util import previous_float, raise_if_not_shape

# __all__
__all__ = ['B',
           'basis_functions',
           'uniform_bspline_basis',
           'UniformBSpline']


# B
@sp.cacheit
def B(i, k, x):
    """Symbolically evaluate the B-spline basis with uniform knot spacing
    (http://en.wikipedia.org/wiki/B-spline).

    Parameters
    ----------
    i : int
        Translation (delay) of the basis on the x-axis.

    k: int
        Order of the B-spline.

    x : sp.Symbol
        Variable.

    Returns
    -------
    b : list of sp.Expr, len = k + i
        `b[i]` is the expression (in `x`) for the basis over the half-open
        interval [i, i + 1).
    """
    if k < 1:
        raise ValueError('k < 1 (= {})'.format(k))
    if i < 0:
        raise ValueError('i < 0 (= {})'.format(i))
    if not isinstance(x, sp.Symbol):
        raise ValueError('x is not sympy.Symbol (type(x) = "{}")'.format(
            type(x).__name__))

    if k == 1:
        f = [sp.S.One]
    else:
        b0, b1 = B(0, k - 1, x), B(1, k - 1, x)

        h = sp.Rational(1, k - 1)
        p0 = [h* x * e for e in b0]
        p0.append(sp.S.Zero)
        p1 = [h * (k - x) * e for e in b1]
        assert len(p0) == len(p1)
        f = map(lambda e0, e1: (e0 + e1).expand(), p0, p1)

    return [sp.S.Zero] * i + [e.subs({x : x - i}) for e in f]


# basis_functions
@sp.cacheit
def basis_functions(d, x):
    """Symbolically evaluate the uniform B-spline basis functions.

    Parameters
    ----------
    d : int
        The degree of the uniform B-spline.

    x : sp.Symbol
        Interpolation parameter.

    Returns
    -------
    b : list of sp.Expr, len = d + 1
        List of B-spline basis functions, where `b[i]` is the interpolating
        expression over the unit interval.
    """
    if d < 0:
        raise ValueError('d < 0 (= {})'.format(d))

    return [ib[1].subs({x : x + ib[0]}).expand()
            for ib in enumerate(B(0, d + 1, x))][::-1]


# uniform_bspline_basis
UNIFORM_BSPLINE_TEMPLATE = """def {func_name}(t):
    t = np.atleast_1d(t)
    if np.any((t < 0) | (t > 1)):
        raise ValueError('t < 0 or t > 1')

    (N,) = t.shape
    W = np.empty((N, {num_control_points}), dtype=float)
{W}
    return W
"""
@sp.cacheit
def uniform_bspline_basis(d, p=0):
    """Generate a "Numpy friendly" function to facilitate fast evaluation of
    uniform B-spline basis functions.

    Parameters
    ----------
    d : int
        The degree of the uniform B-spline.

    p : optional, int
        The order of the derivative with respect to the interpolation
        parameter.

    Returns
    -------
    uniform_bspline_basis_d : function
        "Numpy friendly" function to evaluate the uniform B-spline
        interpolation components.
    """
    t = sp.Symbol('t')
    b = basis_functions(d, t)
    for i in range(p):
        b = [sp.diff(e, t) for e in b]

    func_name = 'uniform_bspline_basis_{}_{}'.format(d, p)
    W = ['    W[:, {}] = {}'.format(ie[0], ie[1].evalf())
         for ie in enumerate(b)]
    code = UNIFORM_BSPLINE_TEMPLATE.format(func_name=func_name,
                                           num_control_points=len(W),
                                           W='\n'.join(W))
    globals_ = {'np' : np}
    exec(code, globals_)
    return globals_[func_name]


# UniformBSpline
class UniformBSpline(object):
    """UniformBSpline

    Class to facilitate evaluation of points on, and derivatives of, a uniform
    B-spline.

    Parameters
    ----------
    degree : int
        The degree of the uniform B-spline.

    num_control_points : int
        The number of control points for the B-spline.

    dim : int
        Number of dimensions of each control point (typically 2 or 3).

    is_closed optional, bool
        True if the contour is closed, False otherwise.
    """

    def __init__(self, degree, num_control_points, dim, is_closed=False):
        if degree < 1:
            raise ValueError('degree < 1 (= {})'.format(degree))

        self._degree = degree
        self.num_control_points = num_control_points
        self.dim = dim
        self.is_closed = is_closed

        self.num_segments = (num_control_points if is_closed else
                             num_control_points - degree)
        if self.num_segments <= 0:
            raise ValueError('num_segments <= 0 (= {})'.format(
                self.num_segments))

        self._max_u = previous_float(self.num_segments)

        self._W = uniform_bspline_basis(degree, 0)
        self._Wt = uniform_bspline_basis(degree, 1)
        self._Wtt = uniform_bspline_basis(degree, 2)

    def uniform_parameterisation(self, N):
        """Generate a vector of coordinates `u` which parameterise linearly
        distributed points on the contour.

        Parameters
        ----------
        N : int
            The number of points.

        Returns
        -------
        u : float, np.ndarray of shape = (N,)
            The vector of contour coordinates.
        """
        return np.linspace(0.0, self._max_u, N, endpoint=True)

    def clip(self, u):
        """Clip a vector of coordinates `u` to the domain of the contour.

        Parameters
        ----------
        u : array_like
            The vector of contour coordinates.

        Returns
        -------
        clipped_u : float, np.ndarray
            The vector of clipped contour coordinates.
        """
        return ((np.asarray(u) % self.num_segments) if self.is_closed else
                 np.clip(u, 0.0, self._max_u))

    def M(self, u, X):
        """Evaluate points on the contour.

        Parameters
        ----------
        u : float, array_like of shape = (N,)
            The vector of contour coordinates.

        X : float, array_like of shape = (num_control_points, dim)
            The matrix of control point positions.

        Returns
        -------
        M : float, np.ndarray of shape = (N, dim)
            The matrix of evaluated positions.
        """
        return self._f(self._W, u, X)

    def Mu(self, u, X):
        """Evaluate first derivatives with respect to `u` on the contour.

        Parameters
        ----------
        u : float, array_like of shape = (N,)
            The vector of contour coordinates.

        X : float, array_like of shape = (num_control_points, dim)
            The matrix of control point positions.

        Returns
        -------
        Mu : float, np.ndarray of shape = (N, dim)
            The matrix of evaluated first derivatives.
        """
        return self._f(self._Wt, u, X)

    def Muu(self, u, X):
        """Evaluate second derivatives with respect to `u` on the contour.

        Parameters
        ----------
        u : float, array_like of shape = (N,)
            The vector of contour coordinates.

        X : float, array_like of shape = (num_control_points, dim)
            The matrix of control point positions.

        Returns
        -------
        Mu : float, np.ndarray of shape = (N, dim)
            The matrix of evaluated second derivatives.
        """
        return self._f(self._Wtt, u, X)

    def MX(self, u):
        """Evaluate first derivatives with respect to `X` on the contour.

        Parameters
        ----------
        u : float, array_like of shape = (N,)
            The vector of contour coordinates.

        Returns
        -------
        J : float, np.ndarray of shape = (N * dim, num_control_points * dim)
            The full Jacobian of first derivatives.
            `J[dim * i + k, dim * j + l]` is the derivative of the `k`th
            component of the `i`th point with respect to component `l` of the
            `j`th control point.
        """
        return self._fX(self._W, u)

    def MuX(self, u):
        """Evaluate the mixed derivatives with respect to `u` and `X` on the
        contour.

        Parameters
        ----------
        u : float, array_like of shape = (N,)
            The vector of contour coordinates.

        Returns
        -------
        M : float, np.ndarray of shape = (N * dim, num_control_points * dim)
            The full matrix of mixed derivatives.
            `M[dim * i + k, dim * j + l]` is the derivative of the `k`th
            component of the `i`th point with respect to `u[i]` and component
            `l` of the `j`th control point.
        """
        return self._fX(self._Wt, u)

    def _f(self, f, u, X):
        u, s, t = self._u_to_s_t(u)
        (N,) = u.shape

        X = np.atleast_2d(X)
        raise_if_not_shape('X', X, (self.num_control_points, self.dim))

        R = np.empty((N, self.dim), dtype=float)
        for s_, i in groupby(np.argsort(s), key=lambda i: s[i]):
            i = list(i)
            R[i] = np.dot(f(t[i]), X[self._i(s_)])

        return R

    def _fX(self, f, u):
        u, s, t = self._u_to_s_t(u)
        (N,) = u.shape

        d = self.dim
        R = np.zeros((N * d, self.num_control_points * d), dtype=float)
        for s_, i in groupby(np.argsort(s), key=lambda i: s[i]):
            i = np.array(list(i))
            for j, w in zip(self._i(s_), f(t[i]).T):
                for k in range(self.dim):
                    R[d * i + k, d * j + k] = w
        return R

    def _u_to_s_t(self, u):
        """Ensure the contour coordinate `u` is valid and translate it to a
        segment index `s` and segment coordinate `t`."""
        u = np.atleast_1d(u)
        (N,) = u.shape

        s = np.floor(u).astype(int)
        if np.any((s < 0) | (s >= self.num_segments)):
            raise ValueError('s < 0 or s >= {}'.format(
                self.num_segments))
        t = u - s
        assert np.all((0.0 <= t) & (t <= 1.0))

        return u, s, t

    def _i(self, s):
        """Provide the control point indices for segment `s`."""
        return [i % self.num_control_points
                for i in range(s, s + self._degree + 1)]
