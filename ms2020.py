# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Nonlinear transform coder with hyperprior for RGB images.

This is the image compression model published in:
D. Minnen and S. Singh:
"Channel-wise autoregressive entropy models for learned image compression"
Int. Conf. on Image Compression (ICIP), 2020
https://arxiv.org/abs/2007.08739

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.

This script requires TFC v2 (`pip install tensorflow-compression==2.*`).

# Adapted from https://github.com/tensorflow/compression/blob/f9edc949fa58381ffafa5aa8cb37dc8c3ce50e7f/models/ms2020.py,
# with minor refactoring to simplify training and evaluation.
# Yibo Yang, 2021
"""

import argparse
import functools
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
from absl import app
from absl.flags import argparse_flags


class AnalysisTransform(tf.keras.Sequential):
    """The analysis transform."""

    def __init__(self, latent_depth):
        super().__init__()
        conv = functools.partial(tfc.SignalConv2D, corr=True, strides_down=2,
                                 padding="same_zeros", use_bias=True)
        layers = [
            tf.keras.layers.Lambda(lambda x: x / 255.),
            conv(192, (5, 5), name="layer_0", activation=tfc.GDN(name="gdn_0")),
            conv(192, (5, 5), name="layer_1", activation=tfc.GDN(name="gdn_1")),
            conv(192, (5, 5), name="layer_2", activation=tfc.GDN(name="gdn_2")),
            conv(latent_depth, (5, 5), name="layer_3", activation=None),
        ]
        for layer in layers:
            self.add(layer)


class SynthesisTransform(tf.keras.Sequential):
    """The synthesis transform."""

    def __init__(self):
        super().__init__()
        conv = functools.partial(tfc.SignalConv2D, corr=False, strides_up=2,
                                 padding="same_zeros", use_bias=True)
        layers = [
            conv(192, (5, 5), name="layer_0",
                 activation=tfc.GDN(name="igdn_0", inverse=True)),
            conv(192, (5, 5), name="layer_1",
                 activation=tfc.GDN(name="igdn_1", inverse=True)),
            conv(192, (5, 5), name="layer_2",
                 activation=tfc.GDN(name="igdn_2", inverse=True)),
            conv(3, (5, 5), name="layer_3",
                 activation=None),
            tf.keras.layers.Lambda(lambda x: x * 255.),
        ]
        for layer in layers:
            self.add(layer)


class HyperAnalysisTransform(tf.keras.Sequential):
    """The analysis transform for the entropy model parameters."""

    def __init__(self, hyperprior_depth):
        super().__init__()
        conv = functools.partial(tfc.SignalConv2D, corr=True, padding="same_zeros")

        # See Appendix C.2 for more information on using a small hyperprior.
        layers = [
            conv(320, (3, 3), name="layer_0", strides_down=1, use_bias=True,
                 activation=tf.nn.relu),
            conv(256, (5, 5), name="layer_1", strides_down=2, use_bias=True,
                 activation=tf.nn.relu),
            conv(hyperprior_depth, (5, 5), name="layer_2", strides_down=2,
                 use_bias=False, activation=None),
        ]
        for layer in layers:
            self.add(layer)


class HyperSynthesisTransform(tf.keras.Sequential):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self):
        super().__init__()
        conv = functools.partial(
            tfc.SignalConv2D, corr=False, padding="same_zeros", use_bias=True,
            kernel_parameter="variable", activation=tf.nn.relu)

        # Note that the output tensor is still latent (it represents means and
        # scales but it does NOT hold mean or scale values explicitly). Therefore,
        # the final activation is ReLU rather than None or Exp). For the same
        # reason, it is not a requirement that the final depth of this transform
        # matches the depth of `y`.
        layers = [
            conv(192, (5, 5), name="layer_0", strides_up=2),
            conv(256, (5, 5), name="layer_1", strides_up=2),
            conv(320, (3, 3), name="layer_2", strides_up=1),
        ]
        for layer in layers:
            self.add(layer)


class SliceTransform(tf.keras.layers.Layer):
    """Transform for channel-conditional params and latent residual prediction."""

    def __init__(self, latent_depth, num_slices):
        super().__init__()
        conv = functools.partial(
            tfc.SignalConv2D, corr=False, strides_up=1, padding="same_zeros",
            use_bias=True, kernel_parameter="variable")

        # Note that the number of channels in the output tensor must match the
        # size of the corresponding slice. If we have 10 slices and a bottleneck
        # with 320 channels, the output is 320 / 10 = 32 channels.
        slice_depth = latent_depth // num_slices
        if slice_depth * num_slices != latent_depth:
            raise ValueError("Slices do not evenly divide latent depth (%d / %d)" % (
                latent_depth, num_slices))

        self.transform = tf.keras.Sequential([
            conv(224, (5, 5), name="layer_0", activation=tf.nn.relu),
            conv(128, (5, 5), name="layer_1", activation=tf.nn.relu),
            conv(slice_depth, (3, 3), name="layer_2", activation=None),
        ])

    def call(self, tensor):
        return self.transform(tensor)


class MS2020Model(tf.keras.Model):
    """Main model class."""

    def __init__(self, lmbda,
                 num_filters, latent_depth, hyperprior_depth,
                 num_slices, max_support_slices,
                 num_scales, scale_min, scale_max):
        super().__init__()
        self.lmbda = lmbda
        self.num_scales = num_scales
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        offset = tf.math.log(scale_min)
        factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
                num_scales - 1.)
        self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
        self.analysis_transform = AnalysisTransform(latent_depth)
        self.synthesis_transform = SynthesisTransform()
        self.hyper_analysis_transform = HyperAnalysisTransform(hyperprior_depth)
        self.hyper_synthesis_mean_transform = HyperSynthesisTransform()
        self.hyper_synthesis_scale_transform = HyperSynthesisTransform()
        self.cc_mean_transforms = [
            SliceTransform(latent_depth, num_slices) for _ in range(num_slices)]
        self.cc_scale_transforms = [
            SliceTransform(latent_depth, num_slices) for _ in range(num_slices)]
        self.lrp_transforms = [
            SliceTransform(latent_depth, num_slices) for _ in range(num_slices)]
        self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=[hyperprior_depth])
        self.build((None, None, None, 3))
        # The call signature of decompress() depends on the number of slices, so we
        # need to compile the function dynamically.
        self.decompress = tf.function(
            input_signature=3 * [tf.TensorSpec(shape=(2,), dtype=tf.int32)] +
                            (num_slices + 1) * [tf.TensorSpec(shape=(1,), dtype=tf.string)]
        )(self.decompress)

    @classmethod
    def create_model(cls, args):
        return cls(args.lmbda, None, args.latent_depth, args.hyperprior_depth,
                   args.num_slices, args.max_support_slices, args.num_scales, args.scale_min, args.scale_max)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--lambda", type=float, default=0.01, dest="lmbda",
            help="Lambda for rate-distortion tradeoff.")

        parser.add_argument(
            "--latent_depth", type=int, default=320,
            help="Number of filters of the last layer of the analysis transform.")
        parser.add_argument(
            "--hyperprior_depth", type=int, default=192,
            help="Number of filters of the last layer of the hyper-analysis "
                 "transform.")
        parser.add_argument(
            "--num_slices", type=int, default=10,
            help="Number of channel slices for conditional entropy modeling.")
        parser.add_argument(
            "--max_support_slices", type=int, default=5,
            help="Maximum number of preceding slices to condition the current slice "
                 "on. See Appendix C.1 of the paper for details.")

        parser.add_argument(
            "--num_scales", type=int, default=64,
            help="Number of Gaussian scales to prepare range coding tables for.")
        parser.add_argument(
            "--scale_min", type=float, default=.11,
            help="Minimum value of standard deviation of Gaussians.")
        parser.add_argument(
            "--scale_max", type=float, default=256.,
            help="Maximum value of standard deviation of Gaussians.")

    def call(self, x, training):
        """Computes rate and distortion losses."""
        # Build the encoder (analysis) half of the hierarchical autoencoder.
        y = self.analysis_transform(x)
        y_shape = tf.shape(y)[1:-1]

        z = self.hyper_analysis_transform(y)

        # Build the entropy model for the hyperprior (z).
        em_z = tfc.ContinuousBatchedEntropyModel(
            self.hyperprior, coding_rank=3, compression=False)

        # When training, z_bpp is based on the noisy version of z (z_tilde).
        _, z_bits = em_z(z, training=training)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_hat = em_z.quantize(z)

        # Build the decoder (synthesis) half of the hierarchical autoencoder.
        latent_scales = self.hyper_synthesis_scale_transform(z_hat)
        latent_means = self.hyper_synthesis_mean_transform(z_hat)

        # Build a conditional entropy model for the slices.
        em_y = tfc.LocationScaleIndexedEntropyModel(
            tfc.NoisyNormal, num_scales=self.num_scales, scale_fn=self.scale_fn,
            coding_rank=3, compression=False)

        # En/Decode each slice conditioned on hyperprior and previous slices.
        y_slices = tf.split(y, self.num_slices, axis=-1)
        y_hat_slices = []
        y_bits = []
        for slice_index, y_slice in enumerate(y_slices):
            # Model may condition on only a subset of previous slices.
            support_slices = (y_hat_slices if self.max_support_slices < 0 else
                              y_hat_slices[:self.max_support_slices])

            # Predict mu and sigma for the current slice.
            mean_support = tf.concat([latent_means] + support_slices, axis=-1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :y_shape[0], :y_shape[1], :]

            # Note that in this implementation, `sigma` represents scale indices,
            # not actual scale values.
            scale_support = tf.concat([latent_scales] + support_slices, axis=-1)
            sigma = self.cc_scale_transforms[slice_index](scale_support)
            sigma = sigma[:, :y_shape[0], :y_shape[1], :]

            _, slice_bits = em_y(y_slice, sigma, loc=mu, training=training)
            y_bits.append(slice_bits)

            # For the synthesis transform, use rounding. Note that quantize()
            # overrides the gradient to create a straight-through estimator.
            y_hat_slice = em_y.quantize(y_slice, sigma, loc=mu)

            # Add latent residual prediction (LRP).
            lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * tf.math.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        # Merge slices and generate the image reconstruction.
        y_hat = tf.concat(y_hat_slices, axis=-1)
        x_hat = self.synthesis_transform(y_hat)

        y_bits = tf.convert_to_tensor(y_bits)  # [num_slices, batchsize]
        y_bits = tf.reduce_sum(y_bits, axis=0)
        bits = y_bits + z_bits

        # Total number of bits divided by total number of pixels.
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), y_bits.dtype)
        y_bpp = tf.reduce_sum(y_bits) / num_pixels
        z_bpp = tf.reduce_sum(z_bits) / num_pixels
        bpp = y_bpp + z_bpp  # scalars
        # bpp = tf.reduce_sum(bits) / num_pixels

        # Mean squared error across pixels.
        # Don't clip or round pixel values while training.
        # mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))

        axes_except_batch = list(range(1, len(x.shape)))  # should be [1,2,3]
        mses = tf.reduce_mean(tf.math.squared_difference(x, x_hat), axis=axes_except_batch)  # per img
        mse = tf.reduce_mean(mses)
        psnrs = 20 * np.log10(255) - 10 * tf.math.log(mses) / np.log(10)  # PSNR for each img in batch
        psnr = tf.reduce_mean(psnrs)
        # The rate-distortion Lagrangian.
        loss = bpp + self.lmbda * mse

        return dict(loss=loss, bpp=bpp, bits=bits, y_bits=y_bits, z_bits=z_bits, mse=mse, mses=mses, x_hat=x_hat,
                    psnr=psnr)

    def train_step(self, x):
        with tf.GradientTape() as tape:
            res = self(x, training=True)
        variables = self.trainable_variables
        loss = res['loss']
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        for m in self.my_metrics:
            m.update_state(res[m.name])
        return {m.name: m.result() for m in self.my_metrics}

    def test_step(self, x):
        res = self(x, training=False)
        for m in self.my_metrics:
            m.update_state(res[m.name])
        return {m.name: m.result() for m in self.my_metrics}

    def predict_step(self, x):
        raise NotImplementedError("Prediction API is not supported.")

    def compile(self, **kwargs):
        super().compile(
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            **kwargs,
        )
        self.metric_names = ('loss', 'bpp', 'psnr')  # mse)
        self.my_metrics = [tf.keras.metrics.Mean(name=name) for name in self.metric_names]  # can't use self.metrics

    def fit(self, *args, **kwargs):
        retval = super().fit(*args, **kwargs)
        # After training, fix range coding tables.
        self.set_entropy_model()  # the resulting self.entropy_model won't actually be saved if using model.save_weights
        return retval

    def set_entropy_model(self):
        self.em_z = tfc.ContinuousBatchedEntropyModel(
            self.hyperprior, coding_rank=3, compression=True)
        self.em_y = tfc.LocationScaleIndexedEntropyModel(
            tfc.NoisyNormal, num_scales=self.num_scales, scale_fn=self.scale_fn,
            coding_rank=3, compression=True)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
    ])
    def compress(self, x):
        """Compresses an image."""
        # Add batch dimension and cast to float.
        x = tf.expand_dims(x, 0)
        x = tf.cast(x, dtype=tf.float32)

        y_strings = []
        x_shape = tf.shape(x)[1:-1]

        # Build the encoder (analysis) half of the hierarchical autoencoder.
        y = self.analysis_transform(x)
        y_shape = tf.shape(y)[1:-1]

        z = self.hyper_analysis_transform(y)
        z_shape = tf.shape(z)[1:-1]

        z_string = self.em_z.compress(z)
        z_hat = self.em_z.decompress(z_string, z_shape)

        # Build the decoder (synthesis) half of the hierarchical autoencoder.
        latent_scales = self.hyper_synthesis_scale_transform(z_hat)
        latent_means = self.hyper_synthesis_mean_transform(z_hat)

        # En/Decode each slice conditioned on hyperprior and previous slices.
        y_slices = tf.split(y, self.num_slices, axis=-1)
        y_hat_slices = []
        for slice_index, y_slice in enumerate(y_slices):
            # Model may condition on only a subset of previous slices.
            support_slices = (y_hat_slices if self.max_support_slices < 0 else
                              y_hat_slices[:self.max_support_slices])

            # Predict mu and sigma for the current slice.
            mean_support = tf.concat([latent_means] + support_slices, axis=-1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :y_shape[0], :y_shape[1], :]

            # Note that in this implementation, `sigma` represents scale indices,
            # not actual scale values.
            scale_support = tf.concat([latent_scales] + support_slices, axis=-1)
            sigma = self.cc_scale_transforms[slice_index](scale_support)
            sigma = sigma[:, :y_shape[0], :y_shape[1], :]

            slice_string = self.em_y.compress(y_slice, sigma, mu)
            y_strings.append(slice_string)
            y_hat_slice = self.em_y.decompress(slice_string, sigma, mu)

            # Add latent residual prediction (LRP).
            lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * tf.math.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        return (x_shape, y_shape, z_shape, z_string) + tuple(y_strings)

    def decompress(self, x_shape, y_shape, z_shape, z_string, *y_strings):
        """Decompresses an image."""
        assert len(y_strings) == self.num_slices

        z_hat = self.em_z.decompress(z_string, z_shape)

        # Build the decoder (synthesis) half of the hierarchical autoencoder.
        latent_scales = self.hyper_synthesis_scale_transform(z_hat)
        latent_means = self.hyper_synthesis_mean_transform(z_hat)

        # En/Decode each slice conditioned on hyperprior and previous slices.
        y_hat_slices = []
        for slice_index, y_string in enumerate(y_strings):
            # Model may condition on only a subset of previous slices.
            support_slices = (y_hat_slices if self.max_support_slices < 0 else
                              y_hat_slices[:self.max_support_slices])

            # Predict mu and sigma for the current slice.
            mean_support = tf.concat([latent_means] + support_slices, axis=-1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :y_shape[0], :y_shape[1], :]

            # Note that in this implementation, `sigma` represents scale indices,
            # not actual scale values.
            scale_support = tf.concat([latent_scales] + support_slices, axis=-1)
            sigma = self.cc_scale_transforms[slice_index](scale_support)
            sigma = sigma[:, :y_shape[0], :y_shape[1], :]

            y_hat_slice = self.em_y.decompress(y_string, sigma, loc=mu)

            # Add latent residual prediction (LRP).
            lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * tf.math.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        # Merge slices and generate the image reconstruction.
        y_hat = tf.concat(y_hat_slices, axis=-1)
        x_hat = self.synthesis_transform(y_hat)
        # Remove batch dimension, and crop away any extraneous padding.
        x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
        # Then cast back to 8-bit integer.
        return tf.saturate_cast(tf.round(x_hat), tf.uint8)


def get_runname(args):
    from utils import config_dict_to_str
    model_name = os.path.splitext(os.path.basename(__file__))[0]
    runname = config_dict_to_str(vars(args), record_keys=('latent_depth', 'hyperprior_depth', 'lmbda'),
                                 prefix=model_name)
    return runname


# Unavoidable boilerplate below.
import boilerplate
from functools import partial

# Note: needed to specify as kwargs in partial; o/w would be incorrect ('argv' would receive the value for create_model)
main = partial(boilerplate.main, create_model=MS2020Model.create_model, get_runname=get_runname)
parse_args = partial(boilerplate.parse_args, add_model_specific_args=MS2020Model.add_model_specific_args)

if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
