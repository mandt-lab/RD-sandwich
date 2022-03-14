# A ResNet-VAE like beta-VAE model for image data, consisting of a series of deterministic EncoderBlocks, DecoderBlocks,
# and LatentBlocks. Each pair of (EncoderBlock, DecoderBlock) is followed by a LatentBlock, which is in charge of a
# latent tensor. A LatentBlock implements bi-directional inference for its latent tensor using bottom-up features
# computed by the downstream neighbor EncoderBlock, as well as top-down features from the upstream DecoderBlock;
# then it samples a latent tensor from the inferred posterior and passes it along to the downstream DecoderBlock.
# See the appendix of the IAF paper for more details on bidrectional inference.
# I also implement a channelwise autoregressive prior in each LatentBlock for added expressiveness.
#
# Towards Empirical Sandwich Bounds on the Rate-Distortion Function, ICLR 2022, Yibo Yang, Stephan Mandt
# https://arxiv.org/abs/2111.12166
# Yibo Yang, 2021

import os

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
from absl import app

import tensorflow_probability as tfp

tfd = tfp.distributions

from utils import reshape_spatially_as, transform_scale_indexes, diag_normal_from_features


class ChannelwiseARTransform(tf.keras.layers.Layer):
    """Given channel slices [x0,x1,...,xn], returns [0, f(x0), f(x0,x1), ...]
     Useful as the mean/scale autoregressive function for the shift/scale in an IAF.
     A similar idea is proposed by Minnen et al., in "Channel-wise autoregressive entropy models for learned image compression"
     See https://github.com/tensorflow/compression/blob/f9edc949fa58381ffafa5aa8cb37dc8c3ce50e7f/models/ms2020.py
     """

    def __init__(self, input_num_channels, num_slices, max_support_ratio=None, activation=tf.nn.leaky_relu,
                 output_activation=None, cond_first_slices=True):
        super().__init__()
        import functools
        conv = functools.partial(
            tfc.SignalConv2D, corr=False, strides_up=1, padding="same_zeros",
            use_bias=True, kernel_parameter="variable")
        self.cond_first_slices = cond_first_slices  # whether to always condition on the first few slices, as in ms2020

        if num_slices > input_num_channels:
            print(f'Warning: num_slices ({num_slices}) > input_num_channels ({input_num_channels}), '
                  f'setting num_slices to equal input_num_channels (thus will have slice_depth=1)')
            num_slices = input_num_channels
        slice_depth = input_num_channels // num_slices
        if slice_depth * num_slices != input_num_channels:
            raise ValueError("Slices do not evenly divide latent depth (%d / %d)" % (
                input_num_channels, num_slices))

        self.num_slices = num_slices
        if max_support_ratio is None:
            max_support_slices = num_slices - 1  # condition all previous slices
        else:
            max_support_slices = round(num_slices * max_support_ratio)
        self.max_support_slices = max_support_slices

        def interpolate(alpha, x=max_support_slices, y=1):
            return alpha * x + (1 - alpha) * y

        self.transforms = [tf.keras.Sequential([
            conv(round(interpolate(0.6)) * slice_depth, (3, 3), name="layer_0", activation=activation),
            conv(round(interpolate(0.3)) * slice_depth, (3, 3), name="layer_1", activation=activation),
            conv(1 * slice_depth, (5, 5), name="layer_2",
                 activation=output_activation)]) for _ in range(num_slices - 1)]
        self.transforms = [None] + self.transforms  # use a dummy zeroth transform

    def call(self, tensor, return_slices=False):
        input_slices = tf.split(tensor, self.num_slices, axis=-1)
        output_slices = []
        for i in range(self.num_slices):
            if i == 0:
                output_slice = tf.zeros_like(input_slices[0])
            else:
                if self.cond_first_slices:
                    support_slices = input_slices[:min(self.max_support_slices, i)]
                else:  # Condition on all previous slices up to max_support_slices
                    support_slices = input_slices[max(0, i - self.max_support_slices): i]
                output_slice = self.transforms[i](tf.concat(support_slices, axis=-1))  # has slice_depth num channels
            output_slices.append(output_slice)

        if return_slices:
            return output_slices
        else:
            return tf.concat(output_slices, axis=-1)


# The implementation of Encoder/DecoderBlocks are inspired by Cheng et al., 2020,
# "Learned image compression with discretized gaussian mixture likelihoods and attention modules."
def enc_conv(filters, kernel_support, stride, name=None, activation=tf.nn.leaky_relu,
             kernel_parameter='rdft'):
    return tfc.SignalConv2D(filters, kernel_support, name=name, corr=True,
                            strides_down=stride, kernel_parameter=kernel_parameter,
                            padding="same_zeros", use_bias=True,
                            activation=activation)


def dec_conv(filters, kernel_support, stride, name=None, activation=tf.nn.leaky_relu,
             kernel_parameter='rdft'):
    return tfc.SignalConv2D(filters, kernel_support, name=name, corr=False,
                            strides_up=stride, kernel_parameter=kernel_parameter,
                            padding="same_zeros", use_bias=True,
                            activation=activation)


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_support=(3, 3), stride=1, name=None,
                 output_activation='gdn',
                 kernel_parameter='rdft', scale_img=False):
        """

        :param filters: num filters in the conv, as well as the output
        :param kernel_support:
        :param stride:
        :param name:
        :param output_activation:
        :param kernel_parameter:
        :param scale_img:
        """
        super().__init__(name=name)
        self.filters = filters
        self.kernel_support = kernel_support
        self.kernel_parameter = kernel_parameter
        self.stride = stride
        self.scale_img = scale_img
        if output_activation == 'gdn':
            output_activation = tfc.GDN()
        self.conv2a = output_activation

    def build(self, input_shape):
        input_channels = input_shape[-1]

        self.conv1 = enc_conv(self.filters, self.kernel_support, self.stride,
                              kernel_parameter=self.kernel_parameter, name='conv1')
        self.conv2 = enc_conv(self.filters, self.kernel_support, 1, activation=self.conv2a,
                              kernel_parameter=self.kernel_parameter, name='conv2')

        if input_channels == self.filters and self.stride == 1:
            self.shortcut = lambda x: x
        else:
            self.shortcut = enc_conv(self.filters, (1, 1), self.stride, activation=None, name='shortcut',
                                     kernel_parameter='variable')
        super().build(input_shape)

    def call(self, x):
        if self.scale_img:
            x = x / 255. - 0.5
        return self.shortcut(x) + self.conv2(self.conv1(x))


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_support=(3, 3), stride=1, name=None,
                 kernel_parameter='rdft',
                 img_output=False):
        """

        :param filters:
        :param kernel_support:
        :param stride:
        :param name:
        :param kernel_parameter:
        :param img_output: whether this decoder block should output an img; currently uses a single dec_conv for this
        """
        super().__init__(name=name)
        self.filters = filters
        self.kernel_support = kernel_support
        self.stride = stride
        self.img_output = img_output

        if not img_output:
            self.conv1 = dec_conv(filters, kernel_support, stride, kernel_parameter=kernel_parameter, name='conv1')
            self.conv2 = dec_conv(filters, kernel_support, 1, kernel_parameter=kernel_parameter, name='conv2')
        else:
            assert filters == 3
            self.conv = dec_conv(filters, kernel_support, stride, activation=None, kernel_parameter=kernel_parameter,
                                 name='conv')

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if input_channels == self.filters and self.stride == 1:
            self.shortcut = lambda x: x
        else:
            self.shortcut = dec_conv(self.filters, (1, 1), self.stride, activation=None, name='shortcut',
                                     kernel_parameter='variable')
        super().build(input_shape)

    def call(self, x):
        if self.img_output:
            out = self.conv(x)
            out = (out + 0.5) * 255.
        else:
            out = self.shortcut(x) + self.conv2(self.conv1(x))
        return out


logit_offset = 1.0


class LatentBlock(tf.keras.Model):
    def __init__(self, feature_channels, latent_channels, ar_prior,
                 kernel_support=(3, 3),
                 res_q_param=True,
                 name='',
                 kernel_parameter='variable',
                 **kwargs):
        """

        :param feature_channels: number of channels in the deterministic bottom-up/top-down feature tensors
        :param latent_channels: number of channels in the latent tensor at this level
        :param ar_prior: whether to use channel-wise MAF prior
        :param kernel_support:
        :param res_q_param: whether to parameterize the variational distribution relative to the prior
        :param name:
        :param kernel_parameter:
        :param kwargs:
        """
        super().__init__(name=name)
        self.__dict__.update(kwargs)
        self.feature_channels = feature_channels
        self.latent_channels = latent_channels
        self.kernel_parameter = kernel_parameter
        self.res_q_param = res_q_param
        if ar_prior and res_q_param:
            print('Warning: using both ar_prior and residual q param, might not make sense.')

        # BU inference net (computes part 1 of posterior stats)
        self.inf_net = enc_conv(2 * latent_channels, kernel_support, stride=1, activation=None,
                                kernel_parameter=kernel_parameter, name=f"{name}-inf_net")
        # TD inference net (computes part 2 of posterior stats)
        self.td_inf_net = dec_conv(2 * latent_channels, kernel_support, stride=1, name=f"{name}-td_inf_net",
                                   kernel_parameter=kernel_parameter)

        # first chunk will be top-down deterministic feature, second/third chunks are prior mean/scale parameters
        self.gen_net = dec_conv(latent_channels * 3, kernel_support, stride=1, activation=None,
                                kernel_parameter=kernel_parameter, name=f"{name}-gen_net")

        # Convert residual top-down features to the same channels as the input t, in order to combine with it;
        # 'new_td_feats_adaptor' would probably have been a better name than 't_adaptor'
        self.t_adaptor = dec_conv(feature_channels, (3, 3), stride=1, name=f"{name}-t_adaptor",
                                  kernel_parameter=kernel_parameter)
        self.ar_prior = ar_prior
        if ar_prior:
            ar_num_slices = kwargs['ar_num_slices']
            max_support_ratio = kwargs.get('max_support_ratio', 0.5)
            self.ar_prior_shift_transform = ChannelwiseARTransform(input_num_channels=latent_channels,
                                                                   num_slices=ar_num_slices,
                                                                   max_support_ratio=max_support_ratio,
                                                                   cond_first_slices=True)
            self.ar_prior_scale_transform = ChannelwiseARTransform(input_num_channels=latent_channels,
                                                                   num_slices=ar_num_slices,
                                                                   max_support_ratio=max_support_ratio,
                                                                   cond_first_slices=True)

    def call(self, b, t, training):
        """

        :param b: bottom up feature tensor from the latent hierarchy below
        :param t: top down feature tensor from the latent hierarchy above
        :param training:
        :return: updated t, with the same shape as before; z sample, and bits at this level of latent hierarchy
        """
        det_feats, p_loc, p_scale = tf.split(self.gen_net(t), num_or_size_splits=3, axis=-1)
        p_scale = transform_scale_indexes(p_scale)
        q_feats = b

        bu_q_loc, bu_q_scale = tf.split(self.inf_net(q_feats), num_or_size_splits=2, axis=-1)  # BU q(z|..) statistics
        bu_q_scale = transform_scale_indexes(bu_q_scale)
        td_q_mean, td_q_scale = tf.split(self.td_inf_net(t), num_or_size_splits=2, axis=-1)  # TD q(z|..) statistics
        td_q_scale = transform_scale_indexes(td_q_scale)
        q_loc = bu_q_loc + td_q_mean
        q_scale = bu_q_scale + td_q_scale

        if self.res_q_param:  # parameterize q relative to p
            q_loc += p_loc
            q_scale += p_scale

        p = tfd.Normal(loc=p_loc, scale=p_scale)
        q = tfd.Normal(loc=q_loc, scale=q_scale)
        z = q.sample()

        reduce_axes = tuple(range(-self.coding_rank, 0))
        if self.ar_prior:
            # To evaluate the density of z~q(z|..) under MAF prior, need to perform inverse AR transform on z to get
            # underlying noise epsilon, then evaluate the noise density under the noise distribution, which is now p.
            log_q = tf.reduce_sum(q.log_prob(z), axis=reduce_axes)

            shift = self.ar_prior_shift_transform(z, return_slices=False)  # m_t in Eq (13) of IAF
            scale = tf.nn.sigmoid(
                logit_offset + self.ar_prior_scale_transform(z, return_slices=False))  # sigma_t in Eq (13) of IAF
            epsilon = scale * z + (1 - scale) * shift  # everything done elementwise
            log_det_J = tf.reduce_sum(tf.math.log(scale),
                                      axis=reduce_axes)  # [batch_size]; this is log det of the 'inverse AR transform': z->\epsilon
            log_p = tf.reduce_sum(p.log_prob(epsilon), axis=reduce_axes) + log_det_J

            bits = (log_q - log_p) / tf.math.log(tf.constant(2, dtype=log_p.dtype))  # [batch_size]
        else:  # good old Gaussian prior/posterior
            bits = tfd.kl_divergence(q, p)
            bits = tf.reduce_sum(bits, axis=reduce_axes) \
                   / tf.math.log(tf.constant(2, dtype=p_loc.dtype))  # sum along all axes except batch

        new_t_feats = self.t_adaptor(tf.concat([z, det_feats], axis=-1))  # 2 * latent_channels -> t_channels
        t += new_t_feats  # residual update
        return dict(z=z, bits=bits, t=t)


class Model(tf.keras.Model):
    """Main model class for the ResNet-VAE used in image experiments in the paper."""

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.z0_channels = z0_channels = self.latent_channels[-1]

        # Build the autoencoder from bottom-up
        stride = self.downsample_factor
        for bu_idx in range(self.num_levels):
            td_idx = self.num_levels - bu_idx - 1
            num_filters = self.num_filters  # Enc block's output num channels
            j = 0  # legacy index into the latent along possibly many latents at a spatial hierarchy
            latent_block_name = f'latent_block_{td_idx}_{j}'
            if td_idx == 0:  # topmost hierarchy
                latent_block = None  # dud; z0 is taken care of manually
            else:
                use_ar_prior = (bu_idx < self.ar_prior_levels)
                latent_block = LatentBlock(num_filters, self.latent_channels[bu_idx],
                                           ar_prior=use_ar_prior,
                                           res_q_param=(not use_ar_prior),
                                           name=latent_block_name,
                                           coding_rank=self.coding_rank,
                                           ar_num_slices=self.ar_slices
                                           )
            self.set_latent_block(td_idx, j, latent_block)

            # assume working with images in [0, 255], hence need to scale model input/output at the bottom layer
            scale_img = bu_idx == 0

            enc_block_name = f'enc_block_{bu_idx}'
            if td_idx == 0:  # topmost encoder computes parameters of q(z0)
                if num_filters != 2 * z0_channels:
                    print(f'Overriding the last encoder num_filters to 2 times z0 channels = {z0_channels}'
                          f'for q(z0) mean,scale')
                enc_block = EncoderBlock(filters=2 * z0_channels, stride=stride,
                                         name=enc_block_name, output_activation=None,
                                         scale_img=scale_img)
            else:
                enc_block = EncoderBlock(filters=num_filters, stride=stride,
                                         name=enc_block_name,
                                         scale_img=scale_img)
            self.set_enc_block(bu_idx, enc_block)

            dec_block_name = f'dec_block_{td_idx}'
            if bu_idx == 0:  # the bottom-most decoder
                dec_block = DecoderBlock(filters=3, stride=stride, name=dec_block_name,
                                         img_output=scale_img)
            else:
                # the b and t inputs at each L block should have the same num channels for symmetry reasons
                prev_enc_num_filters = num_filters
                dec_block = DecoderBlock(filters=prev_enc_num_filters, stride=stride,
                                         name=dec_block_name, img_output=scale_img)
            self.set_dec_block(td_idx, dec_block)

        if not self.flat_z0:  # fully convolutional
            self.pz0 = tfc.DeepFactorized(batch_shape=(z0_channels,))  # no conv with UNoise
        else:
            self.z0_spatial_dim = z0_spatial_dim = int(self.img_dim / self.cum_downsample_factor)
            assert z0_spatial_dim * self.cum_downsample_factor == self.img_dim, 'Must divide evenly.'
            self.z0_flat_dim = int(z0_spatial_dim ** 2 * z0_channels)
            print('Using MAF prior on z0 with flat dim', self.z0_flat_dim)
            self.pz0_base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([self.z0_flat_dim], dtype=self.dtype),
                                                            scale_diag=tf.ones([self.z0_flat_dim], dtype=self.dtype))
            tfb = tfp.bijectors
            from rdub_mlp import af_transform
            self._maf_mades = [
                tfb.AutoregressiveNetwork(params=2, activation=tf.nn.leaky_relu, hidden_units=self.maf_units) for _
                in range(self.maf_stacks)]
            self.pz0 = af_transform(self.pz0_base_dist, self._maf_mades, permute=True, iaf=False)

        self.build((None, self.img_dim, self.img_dim, 3))

    @classmethod
    def create_model(cls, args):
        return cls(**vars(args))

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--lambda", type=float, default=0.01, dest="lmbda",
            help="Lambda for rate-distortion tradeoff.")
        parser.add_argument(
            "--img_dim", type=int, default=None,
            help="Input img dim for non-fully convolutional arch.")

        # Architecture.
        parser.add_argument(
            "--num_filters", type=int, default=256,
            help="Number of filters/channels for the deterministic features; this is"
                 "just the number of output channels of each Enc block.")
        parser.add_argument(
            "--downsample_factor", type=int, default=2,
            help="The downsampling factor ('stride') of every EncoderBlock")
        parser.add_argument(
            "--latent_channels",
            type=lambda s: tuple(int(i) for i in s.split(',')),
            default=(4, 8, 16, 32, 64, 128),
            help="A comma delimited list of integers, specifying the number of channels "
                 "each latent tensor should have (in a LatentBlock) after each EncoderBlock (i.e., in bottom-up"
                 "/inference order). This determines the topology of the beta-VAE.")
        parser.add_argument(
            "--ar_prior_levels", type=int, default=0,
            help="Use channelwise MAF prior at the this many bottom-most levels of generative (decoding) hierarchy.")
        parser.add_argument(
            "--ar_slices", type=int, default=0,
            help="Number of channel-wise slices for the channel-wise MAF prior.")
        parser.add_argument(
            "--ar_max_support_ratio", type=float, default=0.5,
            help="The fraction of maximum number of channel slices to be conditioned on in the channel-wise MAF prior.")

        # Option to use dense / flat top-level latents with MAF prior.
        parser.add_argument(
            "--flat_z0", default=0, action='store_const', const=1,
            help="If provided, will use flat z0 with (flat) MAF prior (no longer fully convolution arch).")
        parser.add_argument(
            "--maf_units", type=lambda s: [int(i) for i in s.split(',')], default=[32, 32],
            help="A comma delimited list, specifying the number of hidden units per MLP layer in the AutoregressiveNetworks"
                 "for normalizing flow.")
        parser.add_argument(
            "--maf_stacks", type=int, default=3,
            help="Num of MAF transforms.")

    def call(self, x, training=True):
        """Given a batch of input imgs, perform a full inference -> generative pass through the model, and computes
        rate and distortion losses."""

        coding_rank = self.coding_rank
        reduce_axes = tuple(range(-coding_rank, 0))

        bu_features = []  # ordered by generative hierarchy (top to bottom)
        fx = x
        for i in range(self.num_levels):
            enc = self.get_enc_block(i)
            fx = enc(fx)
            bu_features.insert(0, fx)

        # go down the generative hierarchy
        bits_per_latent = []  # contains vector of shape [batchsize]
        for i in range(self.num_levels):
            b = bu_features[i]
            dec_block = self.get_dec_block(i)
            j = 0  # legacy index into the latent along possibly many latents at a spatial hierarchy
            if i == 0:  # topmost latent
                qz = diag_normal_from_features(b, name='qz0', scale_lb_reparam=True)
                z = qz.sample()
                log_q = qz.log_prob(z)
                log_q = tf.reduce_sum(log_q, axis=reduce_axes)  # [batch_size]

                if self.flat_z0:  # flatten z0 to evaluate under flat prior
                    # orig_z0_shape = z.shape  # keras compilation crashes due to unknown dims
                    orig_z0_shape = tf.constant([-1, self.z0_spatial_dim, self.z0_spatial_dim, self.z0_channels],
                                                dtype='int32')
                    flat_z0_shape = tf.constant([-1, self.z0_flat_dim])
                    z = tf.reshape(z, flat_z0_shape)
                    log_p = self.pz0.log_prob(z)  # just [batch_size], one number per each elem in the batch
                    z = tf.reshape(z, orig_z0_shape)  # back to spatial tensor
                else:
                    log_p = self.pz0.log_prob(z)
                    log_p = tf.reduce_sum(log_p, axis=reduce_axes)  # [batch_size]

                z_bits = (log_q - log_p) / tf.math.log(tf.constant(2, dtype=log_p.dtype))  # [batch_size]
                t = z
            else:
                latent_block = self.get_latent_block(i, j)
                out = latent_block(b, t, training=training)
                z, z_bits, t = out['z'], out['bits'], out['t']
            bits_per_latent.append(z_bits)

            t = dec_block(t)  # compute top-down feature for the next hierarchy

        x_hat = t
        if not training:
            x_hat = reshape_spatially_as(x_hat, x)

        bits_per_latent = tf.convert_to_tensor(bits_per_latent)  # [latent_hierarchy_depth, batchsize]
        bits = tf.reduce_sum(bits_per_latent, axis=0)  # [batchsize]

        # Mean squared error distortion.
        axes_except_batch = list(range(1, len(x.shape)))  # should be [1,2,3]
        mses = tf.reduce_mean(tf.math.squared_difference(x, x_hat), axis=axes_except_batch)  # per img
        mse = tf.reduce_mean(mses)

        # Total number of bits divided by total number of pixels.
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
        bpp = tf.reduce_sum(bits) / num_pixels
        psnrs = 20 * np.log10(255) - 10 * tf.math.log(mses) / np.log(10)  # PSNR for each img in batch
        psnr = tf.reduce_mean(psnrs)
        # The rate-distortion Lagrangian.
        loss = bpp + self.lmbda * mse

        return dict(loss=loss, bpp=bpp, mse=mse, mses=mses, bits=bits, x_hat=x_hat, psnr=psnr,
                    bits_per_latent=bits_per_latent)

    @property
    def num_levels(self):
        return len(self.latent_channels)

    @property
    def cum_downsample_factor(self):
        return self.downsample_factor ** len(self.latent_channels)

    @property
    def coding_rank(self):
        coding_rank = 3  # this many innermost channels under the prior; see tfc.ContinuousBatchedEntropyModel
        return coding_rank

    def set_enc_block(self, bu_idx, obj):
        setattr(self, f'enc_block_{bu_idx}', obj)

    def get_enc_block(self, bu_idx):
        return getattr(self, f'enc_block_{bu_idx}')

    def set_dec_block(self, td_idx, obj):
        setattr(self, f'dec_block_{td_idx}', obj)

    def get_dec_block(self, td_idx):
        return getattr(self, f'dec_block_{td_idx}')

    def set_latent_block(self, td_idx, j, obj):
        setattr(self, f'latent_block_{td_idx}_{j}', obj)

    def get_latent_block(self, td_idx, j):
        return getattr(self, f'latent_block_{td_idx}_{j}')

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
        self.metric_names = ('loss', 'bpp', 'psnr')
        self.my_metrics = [tf.keras.metrics.Mean(name=name) for name in self.metric_names]  # can't use self.metrics


def get_runname(args):
    from utils import config_dict_to_str
    model_name = os.path.splitext(os.path.basename(__file__))[0]
    runname = config_dict_to_str(vars(args), record_keys=('lmbda', 'num_filters', 'latent_channels',
                                                          'ar_prior_levels', 'ar_slices'
                                                          ),
                                 prefix=model_name, use_abbr=True)
    return runname


# Unavoidable boilerplate below.
import boilerplate
from functools import partial

# Note: needed to specify as kwargs in partial; o/w would be incorrect ('argv' would receive the value for create_model)
main = partial(boilerplate.main, create_model=Model.create_model, get_runname=get_runname)
parse_args = partial(boilerplate.parse_args, add_model_specific_args=Model.add_model_specific_args)

if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
