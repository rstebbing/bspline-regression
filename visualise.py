##########################################
# File: visualise.py                     #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
from __future__ import print_function
import argparse
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from uniform_bspline import UniformBSpline


# Colours
C = dict(b='#377EB8', r='#E41A1C', g='#4DAF4A', o='#FF7F00')


# generate_figure
def generate_figure(z, num_samples, empty=False, disable={}, verbose=True):
    degree, num_control_points, dim, is_closed = (
        z['degree'], z['num_control_points'], z['dim'], z['is_closed'])

    if verbose:
        print('  degree:', degree)
        print('  num_control_points:', num_control_points)
        print('  dim:', dim)
        print('  is_closed:', is_closed)
    c = UniformBSpline(degree, num_control_points, dim, is_closed=is_closed)

    Y, w, u, X = [np.array(z[k]) for k in 'YwuX']
    if verbose:
        print('  num_data_points:', Y.shape[0])

    kw = {}
    if Y.shape[1] == 3:
        kw['projection'] = '3d'
    f = plt.figure()
    if empty:
        ax = f.add_axes((0, 0, 1, 1), **kw)
        ax.set_xticks([])
        ax.set_yticks([])
        if Y.shape[1] == 3:
            ax.set_zticks([])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax = f.add_subplot(111, **kw)
    ax.set_aspect('equal')
    def plot(X, *args, **kwargs):
        ax.plot(*(tuple(X.T) + args), **kwargs)

    if 'Y' not in disable:
        plot(Y, '.', c=C['r'])

    if 'u' not in disable:
        for m, y in zip(c.M(u, X), Y):
            plot(np.r_['0,2', m, y], '-', c=C['o'])

    if 'X' not in disable:
        plot(X, 'o--', ms=6.0, c='k', mec='k')
    if 'M' not in disable:
        plot(c.M(c.uniform_parameterisation(num_samples), X), '-',
             c=C['b'], lw=3.0)

    if not empty:
        e = z.get('e')
        if e is not None:
            ax.set_title('Energy: {:.7e}'.format(e))

    return f


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_path', nargs='?')
    parser.add_argument('--num-samples', type=int, default=1024)
    parser.add_argument('--width', type=float, default=6.0)
    parser.add_argument('--height', type=float, default=4.0)
    parser.add_argument('--dpi', type=int, default=100)
    parser.add_argument('--empty', default=False, action='store_true')
    parser.add_argument('-d', '--disable', action='append', default=[],
                        choices={'Y', 'u', 'M', 'X'})
    args = parser.parse_args()

    if not os.path.isdir(args.input_path):
        print('Input:', args.input_path)
        with open(args.input_path, 'r') as fp:
            z = json.load(fp)
        f = generate_figure(z, args.num_samples,
                            empty=args.empty, disable=args.disable)
        if args.output_path is None:
            plt.show()
        else:
            print('Output:', args.output_path)
            f.set_size_inches((args.width, args.height))
            f.savefig(args.output_path, dpi=args.dpi,
                      bbox_inches=0.0, pad_inches='tight')
    else:
        if args.output_path is None:
            raise ValueError('`output_path` required')
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        input_files = sorted(os.listdir(args.input_path),
                             key=lambda f: int(os.path.splitext(f)[0]))
        input_paths = [os.path.join(args.input_path, f) for f in input_files]
        print('Input:')
        states = []
        for input_path in input_paths:
            print('  ', input_path)
            with open(input_path, 'r') as fp:
                states.append(json.load(fp))

        bounds = sum([[(np.min(z[k], axis=0),
                                                  np.max(z[k], axis=0)) for z in states] for k in 'XY'],
                     [])
        min_, max_ = list(zip(*bounds))
        min_, max_ = np.min(min_, axis=0), np.max(max_, axis=0)
        d = 0.025 * (max_ - min_)

        ndim = d.size
        if ndim == 2:
            xlim, ylim = np.c_[min_ - d, max_ + d]
        elif ndim == 3:
            xlim, ylim, zlim = np.c_[min_ - d, max_ + d]
        else:
            raise ValueError('unable to handle ndim (= {})'.format(ndim))

        print('Output:')
        for input_file, z in zip(input_files, states):
            f = generate_figure(z, args.num_samples,
                                empty=args.empty, disable=args.disable,
                                verbose=False)

            (ax,) = f.axes
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if ndim == 3:
                ax.set_zlim(*zlim)

            input_stem, _ = os.path.splitext(input_file)
            output_path = os.path.join(args.output_path,
                                       '{}.png'.format(input_stem))
            print('  ', output_path)

            f.set_size_inches((args.width, args.height))
            f.savefig(output_path, dpi=args.dpi,
                      bbox_inches=0.0, pad_inches='tight')
            plt.close(f)

        f, axs = plt.subplots(2, 1)

        axs[0].plot([z['e'] for z in states], '.-', c=C['b'])
        axs[0].set_xlim(0, len(states) - 1)
        axs[0].set_yscale('log', basey=2)
        axs[0].set_title('Energy')

        axs[1].plot([z['radius'] for z in states], '.-', c=C['b'])
        axs[1].set_xlim(0, len(states) - 1)
        axs[1].set_title('Radius')
        axs[1].set_yscale('log')

        output_path = os.path.join(args.output_path, 'Optimisation.png')
        print('  ', output_path)
        f.savefig(output_path, dpi=args.dpi,
                  bbox_inches=0.0, pad_inches='tight')
        plt.close(f)


if __name__ == '__main__':
    main()
