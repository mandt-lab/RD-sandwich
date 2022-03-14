# Based on https://www.tensorflow.org/hub/lib_overview
import tensorflow as tf
import tensorflow_hub as hub

url = 'https://tfhub.dev/deepmind/biggan-deep-128/1'  # use 128x128 imgs for now (no lower-res model available)
src_model = hub.KerasLayer(url)

from configs import biggan_class_names_to_ids as class_names_to_ids

BIGGAN_LATENT_DIM = 128  # fixed to this number in bigGAN
NUM_CLASSES = 1000
default_truncation = 0.8  # for this project; want more diversity, hence a value closer to 1


def gen_samples(img_class, intrinsic_dim: int, batch_size: int, truncation: float = default_truncation):
    """

    :param img_class: either a string for the class name, or an integer class id for the IMAGENET class (see, e.g.,
    list of class names and ids https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)
    :param intrinsic_dim:
    :param batch_size:
    :param truncation:
    :return:
    """
    # Sample random noise (z) and ImageNet label (y) inputs.
    if type(img_class) is str:
        class_name = img_class
        assert class_name in class_names_to_ids
        class_id = class_names_to_ids[class_name]
    else:
        assert type(img_class) is int
        class_id = img_class
    assert intrinsic_dim <= BIGGAN_LATENT_DIM
    assert 0.0 <= truncation <= 1.0
    # batch_size = 8
    # truncation = 0.5  # scalar truncation value in [0.0, 1.0]
    # intrinsic_dim = 8

    # Control (upper bound on) the intrinsic dimension of samples by fixing all except
    # intrinsic_dim many coordinates of z to zeros.
    # z = truncation * tf.random.truncated_normal([batch_size, LATENT_DIM])  # noise sample
    z = truncation * tf.random.truncated_normal([batch_size, intrinsic_dim])  # noise sample
    zeros = tf.zeros([batch_size, BIGGAN_LATENT_DIM - intrinsic_dim], dtype=z.dtype)
    z = tf.concat([z, zeros], axis=-1)

    # y_index = tf.random.uniform([batch_size], maxval=1000, dtype=tf.int32)
    y_index = tf.zeros([batch_size], dtype=tf.int32) + class_id
    y = tf.one_hot(y_index, NUM_CLASSES)  # one-hot ImageNet label

    # Call BigGAN on a dict of the inputs to generate a batch of images with shape
    # [8, 128, 128, 3] and range [-1, 1].
    samples = src_model(dict(y=y, z=z, truncation=truncation))
    return samples


def get_sampler(img_class, intrinsic_dim: int, truncation: float = default_truncation, post_process_fun=None):
    # Sample random noise (z) and ImageNet label (y) inputs.
    if type(img_class) is str:
        class_name = img_class
        assert class_name in class_names_to_ids
        class_id = class_names_to_ids[class_name]
    else:
        assert type(img_class) is int
        class_id = img_class
    assert intrinsic_dim <= BIGGAN_LATENT_DIM
    assert 0.0 <= truncation <= 1.0

    # batch_size = 8
    # truncation = 0.5  # scalar truncation value in [0.0, 1.0]
    # intrinsic_dim = 8

    def sampler(batch_size):
        # Control (upper bound on) the intrinsic dimension of samples by fixing all except
        # intrinsic_dim many coordinates of z to zeros.
        # z = truncation * tf.random.truncated_normal([batch_size, LATENT_DIM])  # noise sample
        z = truncation * tf.random.truncated_normal([batch_size, intrinsic_dim])
        zeros = tf.zeros([batch_size, BIGGAN_LATENT_DIM - intrinsic_dim], dtype=z.dtype)
        z = tf.concat([z, zeros], axis=-1)  # noise sample

        y_index = tf.zeros([batch_size], dtype=tf.int32) + class_id
        y = tf.one_hot(y_index, NUM_CLASSES)  # one-hot ImageNet label

        # Call BigGAN on a dict of the inputs to generate a batch of images with shape
        # [8, 128, 128, 3] and range [-1, 1].
        samples = src_model(dict(y=y, z=z, truncation=truncation))
        if post_process_fun:
            samples = post_process_fun(samples)
        return samples

    return sampler


if __name__ == '__main__':

    # parser = argparse_flags.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    import argparse

    parser = argparse.ArgumentParser()

    # High-level options.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", default="gan_imgs", help="Directory where to save.")

    parser.add_argument("-d", type=int, help="Intrinsic Data dimensionality.")
    parser.add_argument("-n", type=int, default=8, help="num samples to generate.")
    parser.add_argument("-c", help="Class name or id.")
    parser.add_argument("--truncation", type=float, default=default_truncation, help="For BigGAN.")

    args = parser.parse_args()
    seed = args.seed

    import numpy as np
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)

    import os

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    samples = gen_samples(args.c, args.d, args.n, args.truncation)

    import matplotlib.pyplot as plt

    samples = samples.numpy() * 0.5 + 0.5  # [-1, 1] -> [0, 1]
    for i, s in enumerate(samples):
        plt.imsave(os.path.join(save_dir, f'class={args.c}-d={args.d}-{i}.png'), s)

    print(f'Saved {args.n} imgs to {save_dir}')
