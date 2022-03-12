# Estimate a lower bound on the R-D function of a data source, by solving an maximization problem based on Csiszár's dual
# characterization of R(D).
#
# See details in the paper,
# Towards Empirical Sandwich Bounds on the Rate-Distortion Function, ICLR 2022, Yibo Yang, Stephan Mandt
# https://arxiv.org/abs/2111.12166
# Yibo Yang, 2021

import argparse
import os
import sys
import math

import numpy as np
import tensorflow as tf
from absl import app
from absl.flags import argparse_flags
from utils import MyJSONEncoder, preprocess_float_dict
from tensorflow_compression.python.ops import math_ops

upper_bound = math_ops.upper_bound
lower_bound = math_ops.lower_bound
import configs


@tf.function
def compute_Ckobj(log_u_fun, x, y, lamb, return_log=True):
    """
    Computes the sample average approximation of E[exp(-lamb rho(X,y) - log u(X)]; this is called \gamma_k in the paper
    appendix.
    :param y: point in Y space where to evaluate the function
    :param x: a batch of data
    :param lamb: Lagrange multiplier that scales the distortion
    :param log_u_fun: a callable implementing the "log u(x)" function
    :return:
    """
    x_shape = tf.shape(x)
    mse = tf.reduce_mean(tf.math.squared_difference(x, y),
                         axis=list(range(1, len(x_shape))))  # squared error avged over all dims except batch, i.e., MSE
    log_u = log_u_fun(x)  # [B, 1]
    log_u = tf.squeeze(log_u)  # drop the vacuous "channel" dimension
    batchsize = tf.cast(x_shape[0], 'float32')
    Ckobj = tf.reduce_logsumexp(- lamb * mse - log_u) - tf.math.log(batchsize)
    if not return_log:  # usually a bad idea and get overflow/underlow from exp
        Ckobj = tf.exp(Ckobj)
    return Ckobj, log_u


def batch_mse(x, y, chunksize=None):
    """
    Given two batches of ndarrays, compute pairwise MSE.
    :param x: [B, d1, d2, ...]
    :param y: [P, d1, d2, ...]
    :param chunksize: int, should be able to evenly divide len(x). If provided, will divide the x batch into 'chunks'
    (minibatches) with the specified chunksize, to avoid running out of memory when computing MSE on large data tensors.
    :return:  [B, P] tensor of pairwise MSE.
    """
    x_shape = tf.shape(x)  # [B, d1, d2, ...]
    num_data_dims = len(x_shape) - 1
    # assert np.all(x_shape[-num_data_dims:] == y_shape[-num_data_dims:])

    xs = tf.expand_dims(x, axis=1)  # [B, 1, d1, d2, ...]
    axes_except_first_two = list(range(2, 2 + num_data_dims))
    if not chunksize:
        mse = tf.reduce_mean(tf.math.squared_difference(xs, y),
                             axis=axes_except_first_two)  # [B, P, d1, d2, ...] -> [B, P]
    else:
        batchsize = x_shape[0]
        mse_chunks = []
        num_chunks = tf.cast(tf.math.ceil(batchsize / chunksize), dtype='int32')
        for i in tf.range(num_chunks):
            beg = i * chunksize
            end = beg + chunksize
            x_chunk = xs[beg: end]
            mse_chunk = tf.reduce_mean(tf.math.squared_difference(x_chunk, y),
                                       axis=axes_except_first_two)  # [C, P, d1, d2, ...] -> [C, P]
            mse_chunks.append(mse_chunk)
        mse = tf.concat(mse_chunks, axis=0)

    return mse


def optimize_y(log_u_fun, lamb, x, num_steps=500, lr=1e-2, tol=1e-6, init='quick',
               quick_topn=10, verbose=False, chunksize=None):
    """
    Find the y that (approximately) achieves the sup in the definition of the sup-partition function.
    :param log_u_fun:
    :param lamb:
    :param x: a tf tensor of k data points (batchsize in the first dimension).
    :param num_steps: max num steps
    :param lr: lr for each of the k grad ascent runs
    :param tol: convergence tolerance for each of the k grad ascent runs
    :param init: either 'exhaustive' or 'quick'. 'exhaustive' mode implements the full procedure of Carreira-Perpinan
    (2007) and starts hill-climbing from each of the k centroids; 'quick' mode is our heuristic approximation of only
    starting hill-climbing from the t best centroids to speed up training.
    :param quick_topn: in quick mode, start ascent from this many best starting points
    :param verbose:
    :return: a result dict
    """
    if num_steps <= 0:  # i.e., skip the optimization entirely
        opt_y = tf.reduce_mean(x, axis=0)
        return dict(opt_y=opt_y)
    assert num_steps > 0
    assert init in ('exhaustive', 'quick')

    y = tf.Variable(tf.zeros(x.shape[1:], dtype=x.dtype), trainable=True, name='y')

    # do some setup/precomputation for the inner optimization
    x_shape = tf.shape(x)
    batchsize = tf.cast(x_shape[0], 'float32')  # 'k' in the paper
    log_batchsize = tf.math.log(batchsize)
    mse_reduction_axes = tf.range(1, len(x_shape))
    log_u = log_u_fun(x)  # [B, 1]
    log_u = tf.squeeze(log_u)  # drop the vacuous "channel" dimension

    if init == 'exhaustive':
        init_ys = x  # an array of values to initialize y to
    elif init == 'quick':  # only run t gradient ascent attempts, starting from the top t most promising x data points
        # log_supobj_at_x_vals = - log_u  # using 1 / g(x_k) (actually just the log) for x_k in batch as a cheap approximation
        mse = batch_mse(x, x, chunksize)
        log_supobj_at_x_vals = tf.reduce_logsumexp(- lamb * mse - log_u, axis=1) - log_batchsize
        # get index of the top_n x vals with the highest objective vals
        top_n_indices = np.argsort(log_supobj_at_x_vals)[-quick_topn:]
        init_ys = tf.gather(x, top_n_indices)
    else:
        raise NotImplementedError
    y_vals = []  # list of local maxima as tf tensors
    log_supobj_vals = []  # values of the log supremum objective valuated at each local mode found by gradient ascent
    for i, init_y in enumerate(init_ys):
        if verbose:
            print(f'\tGrad ascent from init #{i}')
        y.assign(init_y)
        prev_loss = np.inf
        y_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        for step in range(num_steps):
            with tf.GradientTape() as tape:
                mse = tf.reduce_mean(tf.math.squared_difference(x, y),
                                     axis=mse_reduction_axes)  # squared error avged over all dims except batch, i.e., MSE
                log_supobj = tf.reduce_logsumexp(- lamb * mse - log_u) - log_batchsize  # optimize in log domain
                loss = - log_supobj

            loss_val = float(loss)
            if np.isinf(loss_val):
                print(f"\tloss = {loss_val} saturated, can't optimize")
                break
            elif abs((prev_loss - loss_val)) < tol:
                if verbose:
                    print(f'\tOptimization w.r.t. y converged after {step} steps')
                break
            else:
                prev_loss = loss_val

            # backward pass, grad update
            grads = tape.gradient(loss, [y])
            y_optimizer.apply_gradients(zip(grads, [y]))

        y_vals.append(tf.convert_to_tensor(y))
        log_supobj_vals.append(log_supobj.numpy())

    opt_idx = np.argmax(log_supobj_vals)
    opt_log_supobj = log_supobj_vals[opt_idx]
    opt_y = y_vals[opt_idx]
    res = dict(opt_y=opt_y, opt_log_supobj=opt_log_supobj)

    return res


def get_model(**kwargs):
    from nn_models import make_mlp, get_activation, get_convnet
    n = kwargs['data_dim']
    if kwargs['model'] == 'mlp':
        model = make_mlp(units=kwargs['units'] + [1], activation=get_activation(kwargs['activation']),
                         name='mlp', input_shape=[n])
    elif kwargs['model'] == 'cnn':
        model = get_convnet(num_units=kwargs['units'], kernel_dims=kwargs['kernel_dims'],
                            activation='selu', preoutput_activation=None,
                            output_scale=None, output_bias=None,
                            raw_pixel_input=False)  # assumes img data is always in [0, 1]
    else:
        raise NotImplementedError

    log_u_model = model
    return log_u_model


def get_runname(args):
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    from utils import config_dict_to_str
    runname = config_dict_to_str(vars(args),
                                 record_keys=('data_dim', 'model', 'units', 'lamb', 'batchsize', 'seed'),
                                 prefix=script_name, use_abbr=True)
    return runname


def lr_scheduler(epoch, total_epochs, base_lr, decay_factor=0.2, warmup_epochs=0):
    if epoch < warmup_epochs:
        return base_lr
    if epoch < 1 / 2 * total_epochs:
        return base_lr
    if epoch < 3 / 4 * total_epochs:
        return base_lr * decay_factor ** 1
    if epoch < 7 / 8 * total_epochs:
        return base_lr * decay_factor ** 2
    return base_lr * decay_factor ** 3


def linear_schedule(epoch, total_epochs, base_val, final_val, warmup_epochs=0, cooldown_epochs=0):
    if epoch < warmup_epochs:
        val = base_val
    elif epoch + cooldown_epochs > total_epochs:
        val = final_val
    else:
        T = total_epochs - warmup_epochs - cooldown_epochs  # the number of epochs where linear interpolation happens
        val = base_val + (epoch - warmup_epochs) / T * (final_val - base_val)
    return val


def get_dataset(dataset_spec: str, data_dim: int, batchsize: int, preprocess_threads=8, dtype='float32', **kwargs):
    """
    This returns an 'infinite' batched dataset representing the data source.
    :param dataset_spec: a string specifying the dataset
    :param data_dim:
    :param batchsize:
    :return:
    """
    if dataset_spec in ('gaussian', 'banana'):
        if dataset_spec == 'gaussian':
            if kwargs.get('gparams_path', None):
                gparams = np.load(kwargs['gparams_path'])
                loc = gparams['loc'].astype(dtype)
                scale = gparams['scale'].astype(dtype)
            else:
                loc = np.zeros(data_dim, dtype=dtype)
                scale = np.ones(data_dim, dtype=dtype)
            import tensorflow_probability
            source = tensorflow_probability.distributions.Normal(loc=loc, scale=scale)
            map_sample_fun = lambda _: source.sample(batchsize)
        elif dataset_spec == 'banana':
            import ntc_sources
            source = ntc_sources.get_banana()  # a tfp.distributions.TransformedDistribution object
            if data_dim == 2:
                map_sample_fun = lambda _: source.sample(batchsize)
            else:
                from ntc_sources import get_nd_banana
                map_sample_fun, _ = get_nd_banana(data_dim, batchsize, kwargs.get('seed', 0))
        else:
            raise NotImplementedError
        dataset = tf.data.Dataset.from_tensors([])
        dataset = dataset.repeat()
        dataset = dataset.map(map_sample_fun)
    elif dataset_spec == 'mnist':
        patchsize = data_dim
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        arr = np.vstack([x_train, x_test])
        arr = arr[..., None]  # model expects channel dim
        arr = arr.astype(dtype) / 255.
        dataset = tf.data.Dataset.from_tensor_slices(arr)
        dataset = dataset.shuffle(len(arr), reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        if patchsize is not None and patchsize < 28:
            dataset = dataset.map(lambda x: tf.image.random_crop(x, (patchsize, patchsize, 1)),
                                  num_parallel_calls=preprocess_threads)
        dataset = dataset.batch(batchsize)
    else:
        raise NotImplementedError

    return dataset


def get_dataset_iter(args):
    """
    Get an iterator to an infinite batched dataset, representing the i.i.d. source whose R-D we want to estimate.
    :param args:
    :return:
    """
    gan_src = args.dataset in configs.biggan_class_names_to_ids
    if gan_src:
        # with tf.device("/gpu:0"):
        import biggan
        sampler = biggan.get_sampler(args.dataset, args.data_dim,
                                     post_process_fun=lambda x: (x + 1.) * 0.5)  # map from [-1, 1] to [0, 1]

        if not args.chunksize:
            def dataset():
                while True:
                    yield sampler(args.batchsize)
        else:  # sample each batch in small chunks then combine (otherwise can get OOM on GPU)
            def dataset():
                biggan_chunksize = 64  # this seems to work well on our Titan RTX GPU
                while True:
                    chunks = []
                    for i in range(math.ceil(args.batchsize / biggan_chunksize)):
                        chunk = sampler(biggan_chunksize)
                        chunks.append(chunk)
                    x = tf.concat(chunks, axis=0)
                    x = x[:args.batchsize]
                    yield x

        dataset_iter = dataset()
    else:
        if args.dataset.endswith('.npy') or args.dataset.endswith('.npz'):
            from utils import get_np_datasets
            dataset = get_np_datasets(args.dataset, args.batchsize, get_validation_data=False)
        else:
            dataset = get_dataset(dataset_spec=args.dataset, data_dim=args.data_dim, batchsize=args.batchsize,
                                  gparams_path=args.gparams_path)
        dataset_iter = iter(dataset)
    return dataset_iter


def train(args):
    """Trains the model."""

    # Get input data pipeline.
    train_dataset_iter = get_dataset_iter(args)
    # Create nn model
    log_u_model = get_model(**vars(args))
    nn_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, global_clipnorm=args.global_clipnorm)

    #### BEGIN: boilerplate for training logistics ####
    runname = get_runname(args)
    save_dir = os.path.join(args.checkpoint_dir, runname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    import json
    from utils import get_time_str
    time_str = get_time_str()
    log_file_path = os.path.join(save_dir, f'record-{time_str}.jsonl')
    print(f'Logging to {log_file_path}')

    # For continued training.
    if args.cont is not None:
        if args.cont == '':  # if flag specified but no value given, use the latest ckpt in save_dir
            restore_ckpt_path = tf.train.latest_checkpoint(save_dir)
            if not restore_ckpt_path:
                print(f'No checkpoints found in {save_dir}; training from scratch!')
        else:  # then the supplied ckpt path had better be valid
            if os.path.isdir(args.cont):
                ckpt_dir = args.cont
                restore_ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
                assert restore_ckpt_path is not None, f'No checkpoints found in {ckpt_dir}'
            else:  # assuming this is a checkpoint name
                restore_ckpt_path = args.cont
        if restore_ckpt_path:
            load_status = log_u_model.load_weights(restore_ckpt_path).expect_partial()
            # load_status.assert_consumed()
            print('Loaded model weights from', restore_ckpt_path)

    # optional tensorboard logging
    step_counter = nn_optimizer.iterations
    tf.summary.experimental.set_step(step_counter)
    if args.logdir != '':
        tf_log_dir = os.path.join(args.logdir, runname)
        writer = tf.summary.create_file_writer(tf_log_dir)
    #### END: boilerplate for training logistics ####

    ### BEGIN: Set up lambda annealing scheduler
    if args.anneal_lamb:
        assert len(args.target_lambs) >= 2
        target_lambs = list(sorted(args.target_lambs))  # this will be mutated throughout and contains the lambs to save
        warmup_epochs = cooldown_epochs = args.ready_steps
        # linear schedule
        base_lamb = tf.constant(target_lambs[0], dtype='float32')
        final_lamb = tf.constant(target_lambs[-1], dtype='float32')

        def lamb_schedule(epoch):
            lamb = linear_schedule(epoch, args.last_step, base_lamb, final_lamb, warmup_epochs, cooldown_epochs)
            return lamb

        target_lambs = target_lambs[1:]  # no need to save when training with the base lamb
        next_target_lamb = target_lambs.pop(0)

    else:
        lamb = tf.constant(args.lamb, dtype='float32')
    ### END: Set up lamb scheduler

    if not args.num_Ck_samples:
        args.num_Ck_samples = 1
    M = args.num_Ck_samples
    log_M = np.log(float(M))
    for step in range(-1, args.last_step):
        if args.anneal_lamb:
            lamb = lamb_schedule(step)

        # Run M many global optimization runs to draw M samples of C_k, in order to estimate the training objective
        opt_ys = []
        log_Ck_samples = []
        x_M_batch = []  # a batch containing M minibatches of data samples, each minibatch consisting of k samples
        for j in range(M):  # embarrassingly parallelizable
            x = next(train_dataset_iter)
            x_M_batch.append(x)

            res = optimize_y(log_u_fun=log_u_model, x=x, lamb=lamb, num_steps=args.y_steps,
                             lr=args.y_lr, tol=args.y_tol, init=args.y_init,
                             quick_topn=args.y_quick_topn, verbose=args.verbose, chunksize=args.chunksize)
            opt_y = res['opt_y']
            opt_ys.append(opt_y)
            log_Ck_samples.append(res['opt_log_supobj'])

        if step == -1:  # the initial run is just to set the log expansion point alpha
            # 'log_avg_Ck' here is just the avg of the M samples of Ck in log domain, for numerical reasons
            # (and as a R.V., its expected value underestimates the true log E[C_k], just like in IWAE)
            log_avg_Ck = tf.reduce_logsumexp(log_Ck_samples) - log_M
            log_alpha = prev_log_avg_Ck = log_avg_Ck  # for next iter
            continue
        # Update log_alpha for next iter
        beta = args.beta  # should be in [0, 1)
        if abs(prev_log_avg_Ck - log_alpha) <= 10.0:  # heuristic, skips update if the new value would be too extreme
            if beta == 0:
                log_alpha = prev_log_avg_Ck
            else:  # retain beta fraction of its current value, and update alpha with (1-beta) fraction of prev_log_Ck
                # alpha = beta * alpha + (1-beta) * prev E[C_k]
                log_alpha = tf.reduce_logsumexp(
                    [log_alpha + tf.math.log(beta), prev_log_avg_Ck + tf.math.log(1 - beta)])

        # Estimate the RD LB objective and do gradient update
        with tf.GradientTape() as tape:
            log_Ck_samples = []
            log_us = []
            for j in tf.range(M):
                x = x_M_batch[j]
                opt_y = opt_ys[j]
                log_Ck, log_u = compute_Ckobj(log_u_fun=log_u_model, x=x, y=opt_y, lamb=lamb)
                log_Ck_samples.append(log_Ck)
                log_us.append(log_u)  # each is a length B tensor

            log_avg_Ck = tf.reduce_logsumexp(log_Ck_samples) - log_M
            log_avg_Ck = upper_bound(log_avg_Ck, prev_log_avg_Ck + args.log_E_Ck_max_delta)
            log_avg_Ck = lower_bound(log_avg_Ck, prev_log_avg_Ck - args.log_E_Ck_max_delta)
            log_us = tf.concat(log_us, axis=0)
            E_log_u = tf.reduce_mean(log_us)
            log_E_Ck_est = tf.math.exp(
                log_avg_Ck - log_alpha) + log_alpha - 1  # overestimator of log(E[C_k]) by linearization
            loss = E_log_u + log_E_Ck_est

        prev_log_avg_Ck = log_avg_Ck  # for next iter
        trainable_vars = log_u_model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        nn_optimizer.apply_gradients(zip(grads, trainable_vars))
        step_rcd = dict(log_alpha=log_alpha, log_avg_Ck=log_avg_Ck, E_log_u=E_log_u,
                        log_E_Ck_est=log_E_Ck_est, loss=loss)

        # Log to console/file
        print_to_console = args.verbose
        if print_to_console:
            str_to_print = f"step {step}:\t\tloss = {loss:.4g}, log_alpha = {log_alpha:.4g}, log_avg_Ck = {log_avg_Ck:.4g}, log_E_Ck_est = {log_E_Ck_est:.4g}, "
            str_to_print += f"E_log_u = {E_log_u:.4}"
            if args.anneal_lamb:
                str_to_print += f', lamb = {lamb:.5g}'
            print(str_to_print)
        step_rcd['step'] = step
        if args.anneal_lamb:
            step_rcd['lamb'] = lamb
        step_rcd = preprocess_float_dict(step_rcd)  # to mollify json.dump
        with open(log_file_path, 'a') as f:  # write one line to the json log
            json.dump(step_rcd, f, cls=MyJSONEncoder)
            f.write('\n')

        # tensorboard logging
        if args.logdir != '':
            with writer.as_default():
                for (k, v) in step_rcd.items():
                    tf.summary.scalar(k, v, step=step_counter)
            writer.flush()

        # Update lr
        if args.lr_schedule:
            old_lr = float(nn_optimizer.lr)
            lr = lr_scheduler(epoch=step, total_epochs=args.last_step, base_lr=args.lr, decay_factor=0.2,
                              warmup_epochs=0)
            if abs(old_lr - lr) > 1e-6:  # if should update
                from keras import backend
                backend.set_value(nn_optimizer.lr, lr)
                print(f'Reset lr from {old_lr:.4g} to {float(nn_optimizer.lr):.4g}')

        # Manually save model
        # At this point we've actually already taken (step + 1) gradient steps
        save_model_flag = (step + 1) % args.checkpoint_interval == 0 or step + 1 == args.last_step
        pop_lamb_flag = args.anneal_lamb and lamb_schedule(step + 1) >= next_target_lamb
        save_model_flag = save_model_flag or pop_lamb_flag
        if save_model_flag:
            save_path = os.path.join(save_dir, f'step={step + 1}-lamb={lamb:.5g}-loss={float(loss):.3f}-kerasckpt')
            log_u_model.save_weights(save_path)
            if pop_lamb_flag:
                print(f'Right before reaching target lamb {next_target_lamb}, saved to {save_path}')
                if len(target_lambs) > 0:
                    next_target_lamb = target_lambs.pop(0)
                else:
                    next_target_lamb = np.Inf  # done, no more lambda values to target/save

        finished = (step + 1 >= args.last_step)
        if finished:
            break

    print(f'Logged to {log_file_path}')


def est_R_(log_u_fun, lamb, dataset_iter, args):
    """
    Estimate the R-axis intercept of a linear underestimator to R(D) with slope -lambda; call this quantity R_, then
    the resulting linear underestimator has the equation R_L(D) = - lamb D + R_.
    This computes \tilde \ell_k (Eq (8)) in the paper. This uses 2M samples of C_k; the first half are used to set the
    log expansion point, the second half to estimate the E[C_k] term. Also see Algorithm 1.
    :param log_u_fun:
    :param lamb:
    :param dataset_iter:
    :param args:
    :return:
    """
    log_Ck_samples = []
    E_log_us = []
    M = args.num_Ck_samples
    assert M >= 1
    for n in range(2 * M):
        x = next(dataset_iter)
        res = optimize_y(log_u_fun=log_u_fun, x=x, lamb=lamb, num_steps=args.y_steps,
                         lr=args.y_lr, tol=args.y_tol,
                         init=args.y_init, quick_topn=args.y_quick_topn,
                         verbose=args.verbose, chunksize=args.chunksize)
        log_Ck_samples.append(res['opt_log_supobj'])
        if args.verbose:
            print(f"GOP run {n} opt_log_supobj = {res['opt_log_supobj']}")

        log_u = log_u_fun(x)  # B-tensor
        E_log_us.append(tf.reduce_mean(log_u))

    E_log_us = np.array(E_log_us)
    # Use the first half of Ck samples to set log expansion point
    log_alpha = tf.reduce_logsumexp(log_Ck_samples[:M]) - np.log(float(M))  # a = E[Ck]
    # Use the second half of Ck samples and E_log_us to estimate the LB objective. We define the estimator
    # $\xi := - \frac{1}{k} \sum_i^k \log u (X_i) - C_k/alpha - \log \alpha + 1$; each \xi sample is computed
    # from a separate k-minibatch of data (and C_k), but we use the same \alpha. The LB objective is then the population
    # mean (expectation) of \xi.
    xi_samples = -E_log_us[M:] - np.exp(log_Ck_samples[M:] - log_alpha) - log_alpha + 1
    R_ = tf.reduce_mean(xi_samples)

    res = dict(E_log_us=E_log_us, log_Ck_samples=np.array(log_Ck_samples), xi_samples=xi_samples, log_alpha=log_alpha,
               R_=R_)

    return res


def eval(args):
    """
    Load a trained model and evaluate R(D) lower bound. Will run the full Gaussian mixture global optimization procedure
    (i.e., in 'exhaustive' mode) to be correct.
    :param args:
    :return:
    """
    if not args.ckpt:  # use the latest checkpoint in run dir, based on args
        runname = get_runname(args)
        ckpt_dir = os.path.join(args.checkpoint_dir, runname)  # run dir
        save_dir = ckpt_dir
        restore_ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
        assert restore_ckpt_path is not None, f'No checkpoints found in {ckpt_dir}'
    else:
        if os.path.isdir(args.ckpt):
            ckpt_dir = args.ckpt
            save_dir = ckpt_dir
            restore_ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
            assert restore_ckpt_path is not None, f'No checkpoints found in {ckpt_dir}'
        else:
            restore_ckpt_path = args.ckpt
            if restore_ckpt_path.endswith('.index'):
                restore_ckpt_path = os.path.splitext(restore_ckpt_path)[0]  # remove extension for tf
            save_dir = os.path.dirname(restore_ckpt_path)

    log_u_model = model = get_model(**vars(args))
    load_status = model.load_weights(restore_ckpt_path).expect_partial()
    print('Loaded model weights from', restore_ckpt_path)

    dataset_iter = get_dataset_iter(args)

    if not args.lamb:
        print('Non-effective lamb provided in args; will try to use lamb in ckpt name')
        try:
            import re
            ckpt_file_name = os.path.basename(restore_ckpt_path)
            # search for a numeric string (possibly in scientific notation) for the lamb value
            lamb_str = re.search('lamb=(\d*\.?\d+(?:e[+-]?\d+)?)', ckpt_file_name).group(1)
            lamb = float(lamb_str)
            lamb = tf.constant(lamb, dtype='float32')
            print(f'Defaulting lamb to ckpt value = {lamb} instead')
        except Exception as e:
            print('Failed to get valid lamb!')
            raise e
    else:
        lamb = args.lamb
    assert lamb > 0

    assert args.y_init == 'exhaustive', 'Should run full global optimization procedure to be correct.'
    res = est_R_(log_u_model, lamb, dataset_iter, args)
    R_ = res['R_']
    save_path = os.path.join(save_dir,
                             f'rd-seed={args.seed}-k={args.batchsize}-M={args.num_Ck_samples}-lamb={lamb:.5g}-R_={R_:.3f}.npz')
    np.savez(save_path, **res)
    print(f'Saved final R_ est to {save_path}')

    return None


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="Report bitrate and distortion when training or compressing.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint_dir", default="./checkpoints",
        help="Parent directory for model checkpoints; each run creates a subdirectory based on runname and saves checkpoints there.")
    parser.add_argument("--chunksize", type=int, default=None,
                        help="Do things in chunks to avoid running out of memory.")

    # Dataset.
    parser.add_argument("--data_dim", type=int, help="Data dimension. For Gaussian toy source, this is simply the"
                                                     "dimensionality; for images, this is the crop dim/height/width.")
    parser.add_argument(
        "--dataset",
        help="Dataset specifier. 'gaussian' or a glob pattern identifying training data. This pattern must expand "
             "to a list of RGB images in PNG format.")
    parser.add_argument("--gparams_path", type=str, default=None, help="Path to Gaussian loc/scale params. ")

    # Model settings
    parser.add_argument(
        "--lamb", type=float,
        help="lambda; should be positive float.")
    parser.add_argument("--model", type=str, default='mlp', help='Type of log_u model to use.')
    # For NN model
    parser.add_argument(
        "--units", type=lambda s: [int(i) for i in s.split(',')], default=[],
        help="A comma delimited list, specifying the number of hidden units per NN layer (if using NN model).")
    parser.add_argument(
        "--activation", type=str, default='selu',
        help="Activation for hidden layers of the NN model.")
    # For CNN specifically
    parser.add_argument(
        "--kernel_dims", type=lambda s: tuple(int(i) for i in s.split(',')),
        default=tuple(),
        help="A comma delimited list, specifying the kernel dim of each conv layer; no effect if not using a conv net")

    subparsers = parser.add_subparsers(
        title="commands", dest="command",
        help="What to do: 'train' loads training data and trains (or continues "
             "to train) a new model. 'compress' reads an image file (lossless "
             "PNG format) and writes a compressed binary file. 'decompress' "
             "reads a binary file and reconstructs the image (in PNG format). "
             "input and output filenames need to be provided for the latter "
             "two options. Invoke '<command> -h' for more information.")

    # 'train' subcommand.
    train_cmd = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trains (or continues to train) a new model.")
    train_cmd.add_argument(
        "--lr", type=float, default=1e-4,
        help="Main optimizer lr.")
    train_cmd.add_argument(
        "--lr_schedule", default=False, action='store_true',
        help="Whether to use lr schedule")
    train_cmd.add_argument(
        "--global_clipnorm", type=float, default=100.,
        help="Implement global_clipnorm to this value for the main optimizer.")
    train_cmd.add_argument(
        "--last_step", type=int, default=10000,
        help="Train up to this number of steps.")
    train_cmd.add_argument(
        "--preprocess_threads", type=int, default=16,
        help="Number of CPU threads to use for parallel decoding of training "
             "images.")
    train_cmd.add_argument(
        "--checkpoint_interval", type=int, default=500,
        help="Write a checkpoint every `checkpoint_interval` training steps.")
    train_cmd.add_argument(
        "--logdir", default="",  # '--log_dir' seems to conflict with absl.flags's existing
        help="Directory for storing Tensorboard logging files; set to empty string '' to disable Tensorboard logging.")

    train_cmd.add_argument(
        "--cont", nargs='?', default=None, const='',  # see https://docs.python.org/3/library/argparse.html#nargs
        help="Path to the checkpoint (either the directory containing the checkpoint (will use the latest), or"
             "full checkpoint name (should not have the .index extension)) to continue training from;"
             "if no path is given, will try to use the latest ckpt in the run dir.")
    train_cmd.add_argument(
        "--beta", type=float, default=0.2,
        help="Fraction of historic value to use in the moving average estimation of alpha; set to 0 to disable history.")
    train_cmd.add_argument(
        "--log_E_Ck_max_delta", type=float, default=1.0,
        help="Allow log_Ck to differ by at most this number across steps to prevent drastic updates in log_u resulting in NaN.")

    train_cmd.add_argument(
        "--anneal_lamb", default=False, action='store_true',
        help="Whether to use lambda annealing; if specified, the --lamb argument sets the final lambda value, and"
             "a linear schedule is used to interpolate .")
    train_cmd.add_argument(
        "--ready_steps", type=int, default=50,
        help="Number of steps to get ready (warmup or cooldown) before/after lambda annealing.")
    train_cmd.add_argument(
        "--target_lambs",
        type=lambda s: tuple(float(i) for i in s.split(',')),
        default=tuple(),
        help="A comma delimited list of floats, specifiying the lambda values at which to save model checkpoints."
             "The first and last values will be used to set a linear annealing schedule, and a new checkpoint"
             "will be saved every time the current value of lambda reaches one of the values on the list.")

    # 'eval' subcommand.
    eval_cmd = subparsers.add_parser(
        "eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluates model saves results in .npz.")
    eval_cmd.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to the checkpoint (either the directory containing the checkpoint (will use the latest), or"
             "full checkpoint name (with optional .index extension)) to load;"
             "by default (None) uses the latest ckpt in the auto-generated run dir in checkpoint_dir/runname")

    for cmd in (train_cmd, eval_cmd):
        cmd.add_argument(
            "--batchsize", "-k", type=int, default=16,
            help="Batch size of x samples.")
        cmd.add_argument(
            "--num_Ck_samples", "-M", type=int, default=None,
            help="Number of samples of Ck used to estimate E[C_k] or the log expansion point. Drawing each C_k sample"
                 "requires a global optimization run.")
        # Settings for global optimization of Gaussian mixture density.
        cmd.add_argument(
            "--y_init", type=str, default='quick',
            help="How to initialize inner optimization w.r.t. y, exhaustive|quick.")
        cmd.add_argument(
            "--y_quick_topn", type=int, default=10,
            help="When using 'quick' for y init, only run hill climbing from this many 'best' centroids.")
        cmd.add_argument(
            "--y_steps", type=int, default=100,
            help="Max number of steps to run inner gradient ascent w.r.t. y to compute C_k sample.")
        cmd.add_argument(
            "--y_tol", type=float, default=1e-3,
            help="Convergence threshold for inner gradient ascent w.r.t. y.")
        cmd.add_argument(
            "--y_lr", type=float, default=1e-2,
            help="Optimizer lr for the inner gradient ascent w.r.t. y.")

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    # Invoke subcommand.
    seed = args.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if args.command == "train":
        train(args)
    elif args.command == "eval":
        eval(args)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
