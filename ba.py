# Blahut-Arimoto algorithm for a continuous data source under MSE distortion.
# Yibo Yang, 2021
import numpy as np


def blahut_arimoto(Rho, p_x, lamb, steps=1000, tol=1e-6, verbose=False):
    """
    Blahut-Arimoto (BA) algorithm for computing the R-D function of a discrete memoryless source.
    Call the source alphabet X and reproduction Y. The BA algorithm creates two variational
    distributions, Q and q, where Q is a conditional probability distribution of y given x, and
    q is a marginal distribution on Y, then minimizes a variational objective by coordinate descent
    w.r.t. q and Q. See Blahut 1972, "Computation of channel capacity and rate-distortion functions".
    The coordinate descent steps are done in log domain for numerical stability.

    :param Rho: a |X| by |Y| matrix (2d numpy array) of distortion values b/w every pair of values in
    X and Y, s.t. Rho[i, j] = rho(X[i], Y[j]).
    :param p_x: a length |X| 1d array of source probabilities.
    :param lamb: positive Lagrange multiplier in front of the distortion term of the Lagrangian.
    :param steps: max num coordinate descent steps.
    :param tol: terminate early if improvement dropped below this
    :return: final variational distributions and estimated R-D (rate given in nats per sample).
    """
    import numpy as np
    from scipy.special import logsumexp

    # Q is a |X| by |Y| matrix of conditional probabilities, with Q[i, j] = Q(Y = j | X = i), so
    # the ith row gives the conditional prob of Q(y | x = i).
    Q = np.ones_like(Rho)
    Q /= np.sum(Q, axis=1, keepdims=True)  # initialize each row to uniform distributions
    log_Q = np.log(Q)

    p_x = p_x / np.sum(p_x)  # normalize just to be sure
    log_p_x = np.log(p_x)

    scaled_Rho = -lamb * Rho  # cached for convenience
    records = []
    ub_prev = np.inf
    for step in range(steps):
        # Compute the marginal q(y) from the joint
        # q_y = np.matmul(p_x, Q)
        log_q_y = logsumexp(log_p_x + log_Q.T, axis=1)

        # Update the conditional distribution
        # Q = np.exp(-lamb * dist_mat) * q_y
        # Q /= np.expand_dims(np.sum(Q, 1), 1)
        log_Q = scaled_Rho + log_q_y
        log_Q -= logsumexp(log_Q, axis=1, keepdims=True)  # normalize

        # Compute objective
        Q = np.exp(log_Q)
        # rate = np.matmul(p_x, Q * np.log(Q / np.expand_dims(q_y, 0))).sum()
        rate = np.matmul(p_x, Q * (log_Q - log_q_y)).sum()
        distortion = np.matmul(p_x, (Q * Rho)).sum()

        ub = rate + lamb * distortion  # UB on the R-axis intercept of the slope-lamb tangent line, also the Lagrangian

        rcd = dict(step=step, ub=ub, D=distortion, R=rate)
        records.append(rcd)
        if verbose and (10 * step) % steps == 0:
            print(rcd)
        if (ub_prev - ub) / ub < tol:
            if verbose:
                print(f'Tolerance reached, terminating after {step} steps')
            break
        else:
            ub_prev = ub

    print(rcd)
    return records, log_Q, log_q_y


def bin_edges_to_grid_pts(edges):
    # Given an array of histogram bin edges as returned by numpy histogram,
    # create an array of grid points located at the center of the bins
    left_edges = edges[:-1]
    right_edges = edges[1:]
    grid_pts = (left_edges + right_edges) / 2
    return grid_pts


def vectorized_mse(xs, ys):
    xs_shape = xs.shape  # [N, d1, d2, ...]
    ys_shape = ys.shape  # [M, d1, d2, ...]
    assert np.all(xs_shape[1:] == ys_shape[1:])
    xs = np.expand_dims(xs, axis=1)  # [N, 1, d1, d2, ...]
    num_data_dims = len(ys_shape) - 1
    axes_except_first_two = tuple(range(2, 2 + num_data_dims))
    mse = np.mean((xs - ys) ** 2, axis=axes_except_first_two)  # [N, M, d1, d2, ...] -> [N, M]
    return mse


if __name__ == "__main__":
    import argparse
    import sys, os, json

    parser = argparse.ArgumentParser()

    # High-level options.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=str, help='Path to the .npy file containing data samples.')
    parser.add_argument("--lamb", type=float, default=1.0)
    parser.add_argument("--save_dir", default="./results/")
    parser.add_argument("--bins", type=int, default=50, help='Number of discretization bins per dimension.')
    parser.add_argument("--tol", type=float, default=1e-6, help='Convergence tolerance.')
    parser.add_argument("--steps", type=int, default=1000, help='Max number of iterations.')
    parser.add_argument("-V", '--verbose', default=False, action='store_true')

    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)

    # Estimate (a discretized approximation of) the source by binning samples
    bins = args.bins
    samples = np.load(args.samples)  # num samples by n-dim
    n = samples.shape[1]
    assert issubclass(samples.dtype.type, np.floating), 'Assuming data is continuous for now.'
    hist_res = np.histogramdd(samples, bins=bins)  # , range=(xlim, ylim))  # here using the default range of samples
    counts = hist_res[0]  # joint counts
    bin_edges = hist_res[1]  # length-n list of arrays, each array has length bins+1
    if args.verbose:
        print('bin ranges:')
        for edges in bin_edges:
            print(edges[0], edges[-1])

    grid_axes = [bin_edges_to_grid_pts(edges) for edges in bin_edges]

    # Enumerate grid points corresponding to the histogram (using the center of each bin).
    meshgrid = np.meshgrid(*grid_axes, indexing='ij')  # length-n list, one 'mesh' for each data dimension
    grid_pts = np.dstack(meshgrid)  # each grid point (n-tuple) resides in the inner-most dimension
    grid_pts_flat = np.reshape(grid_pts, [-1, n])  # preserve the inner-most dim while flatten the rest
    counts_flat = counts.ravel()

    if args.verbose:
        print('prop of zero bins', np.mean(counts_flat == 0))
    good_pts_ind = (counts_flat != 0)
    src_alphabet = grid_pts_flat[good_pts_ind]
    src_dist = counts_flat[good_pts_ind]
    src_dist /= src_dist.sum()

    rep_alphabet = grid_pts_flat  # use all grid points from the sample histogram
    Rho = vectorized_mse(src_alphabet, rep_alphabet)

    if args.verbose:
        print(f'(source, reproduction) alphabets have size {Rho.shape}')

    #### BEGIN: boilerplate for training logistics ####
    from utils import config_dict_to_str, MyJSONEncoder

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    runname = config_dict_to_str(vars(args), record_keys=('lamb', 'bins'), prefix=script_name,
                                 use_abbr=False)
    save_dir = os.path.join(args.save_dir, runname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    records, log_Q, log_q_y = blahut_arimoto(Rho=Rho, p_x=src_dist, steps=args.steps, lamb=args.lamb,
                                             verbose=args.verbose, tol=args.tol)

    final_rcd = records[-1]
    R, D, obj = final_rcd['R'], final_rcd['D'], final_rcd['ub']
    steps = final_rcd['step'] + 1  # total num coord descent steps taken
    save_name = f'steps={steps}-R={R:.3g}-D={D:.3g}-obj={obj:.3g}'
    save_path = os.path.join(save_dir, f'{save_name}.jsonl')
    log_file_path = save_path

    import json

    with open(log_file_path, 'a') as f:  # write one line to the json log
        for rcd in records:
            json.dump(rcd, f, cls=MyJSONEncoder)
            f.write('\n')

    print(f'Saved to {save_path}')
