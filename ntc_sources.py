# Based on https://github.com/tensorflow/compression/blob/66228f0faf9f500ffba9a99d5f3ad97689595ef8/models/toy_sources/toy_sources.ipynb
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfm = tf.math
tfkl = tf.keras.layers
tfpb = tfp.bijectors
tfpd = tfp.distributions


def _rotation_2d(degrees):
    phi = tf.convert_to_tensor(degrees / 180 * np.pi, dtype=tf.float32)
    rotation = [[tfm.cos(phi), -tfm.sin(phi)], [tfm.sin(phi), tfm.cos(phi)]]
    rotation = tf.linalg.LinearOperatorFullMatrix(
        rotation, is_non_singular=True, is_square=True)
    return rotation


def get_laplace(loc=0, scale=1):
    return tfpd.Independent(
        tfpd.Laplace(loc=[loc], scale=[scale]),
        reinterpreted_batch_ndims=1,
    )


def get_banana():
    return tfpd.TransformedDistribution(
        tfpd.Independent(tfpd.Normal(loc=[0, 0], scale=[3, .5]), 1),
        tfpb.Invert(tfpb.Chain([
            tfpb.RealNVP(
                num_masked=1,
                shift_and_log_scale_fn=lambda x, _: (.1 * x ** 2, None)),
            tfpb.ScaleMatvecLinearOperator(_rotation_2d(240)),
            tfpb.Shift([1, 1]),
        ])),
    )


def get_nd_banana(n, batchsize, seed=0):
    """
    Returns a callable that generates samples of n-d banana
    :param n: desired dimensionality
    :param seed:
    :return:
    """
    source = get_banana()
    from nn_models import make_mlp
    tf.random.set_seed(seed)
    embedder = make_mlp([n], activation=tf.nn.softplus, name='banana_embedder',
                        input_shape=[2], dtype=source.dtype)
    return lambda _: embedder(source.sample(batchsize)), embedder
