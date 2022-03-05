# Script to randomly generate loc/var of a Gaussian.
# Yibo Yang 2021

import numpy as np
import argparse
import os


def gen_params(dim, loc='rand', var='rand', dtype='float32'):
    assert loc in ('zeros', 'rand')
    assert var in ('ones', 'rand')
    if loc == 'zeros':
        loc = np.zeros(dim, dtype=dtype)
    else:
        loc = np.random.uniform(-0.5, 0.5, dim).astype(dtype)

    if var == 'ones':
        var = np.ones(dim, dtype=dtype)
    else:
        var = np.random.uniform(0, 2, dim).astype(dtype)

    return loc, var


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", default="./data", help="Directory to save in.")
    parser.add_argument("--dim", type=int, help="Data dimensionality.")
    parser.add_argument("--loc", type=str, default="rand", help="loc of Gaussian source to use.")
    parser.add_argument("--var", type=str, default="rand", help="variance of Gaussian source to use.")

    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)

    loc, var = gen_params(dim=args.dim, loc=args.loc, var=args.var)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    prefix = 'gaussian_params'
    save_path = os.path.join(save_dir, f'{prefix}-dim={args.dim}.npz')

    np.savez(save_path, loc=loc, var=var, scale=var ** 0.5)
    print(f'Saved to {save_path}')
