# Train/eval a beta-VAE with a MSE distortion (Gaussian likelihood) to get a variational bound on the source rate-distortion
# function. This script assume data consists of i.i.d. flat vectors, and uses MLP encoder/decoder; see other scripts
# that handle img-like data with convolutional encoder/decoder.
# Inspired by https://github.com/tensorflow/compression/blob/66228f0faf9f500ffba9a99d5f3ad97689595ef8/models/toy_sources/ntc.py
#
# See details in the paper,
# Towards Empirical Sandwich Bounds on the Rate-Distortion Function, ICLR 2022, Yibo Yang, Stephan Mandt
# https://arxiv.org/abs/2111.12166
# Yibo Yang, 2021

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
from absl import app
from absl.flags import argparse_flags

import tensorflow_probability as tfp

from utils import softplus_inv_1

tfd = tfp.distributions
tfb = tfp.bijectors

from nn_models import get_activation, make_mlp


def af_transform(base_distribution, mades, permute=True, iaf=False):
    """
    Apply a cascade of autoregressive transforms to a base distribution. Default is MAF.
    """
    if permute:
        dims = np.arange(base_distribution.event_shape[0])
    dist = base_distribution
    for i, made in enumerate(mades):
        af = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made)
        if iaf:
            af = tfb.Invert(af)
        if permute:
            permutation_order = np.roll(dims, i)  # circular shift
            permutation = tfb.Permute(permutation_order)
            bij = tfb.Chain([permutation, af])
        else:
            bij = af
        dist = tfd.TransformedDistribution(distribution=dist, bijector=bij)
    return dist


def check_no_decoder(decoder_units):
    # When decoder_units = [] (default), the code still uses a decoder network mapping from latent_dim to
    # data_dim. In order to specify "no decoder network at all", we follow the convention of setting decoder_units=[0]
    return len(decoder_units) == 1 and decoder_units[0] <= 0


class Model(tf.keras.Model):
    """R-D VAE"""

    # def __init__(self, lmbda, data_dim, latent_dim, encoder_units, decoder_units,
    #              encoder_activation, decoder_activation, prior_type='deep', posterior_type='gaussian',
    #              dtype='float32', ar_hidden_units=[10, 10], ar_activation='relu', maf_stacks=3, iaf_stacks=3,
    #              rpd=False):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        dtype = self.dtype
        posterior_type = self.posterior_type
        latent_dim = self.latent_dim
        ar_activation = get_activation(self.ar_activation, dtype)

        if posterior_type in ('gaussian', 'iaf'):
            encoder_output_dim = latent_dim * 2  # currently IAF uses a base Gaussian distribution conditioned on x
            if posterior_type == 'iaf':
                self._iaf_mades = [
                    tfb.AutoregressiveNetwork(params=2, activation=ar_activation, hidden_units=self.ar_hidden_units) for
                    _ in range(self.iaf_stacks)]
        else:
            encoder_output_dim = latent_dim

        # We always require an encoder network in order to produce the variational distribution Q(Y|X).
        # encoder_units = [] gives the minimal network.
        encoder = make_mlp(
            units=self.encoder_units + [encoder_output_dim],
            activation=get_activation(self.encoder_activation, dtype),
            name="encoder",
            input_shape=[self.data_dim],
            dtype=dtype,
        )

        # However, a decoder network is optional when dim(Y) == dim(X).
        # When decoder_units = [] (default), the code still uses a decoder network mapping from latent_dim to
        # data_dim. In order to specify "no decoder network at all", we follow the convention of setting decoder_units=[0]
        if check_no_decoder(self.decoder_units):
            decoder = None  # no decoder
            assert self.data_dim == latent_dim
            print('Not using decoder')
        else:  # decoder_units = [] allowed
            decoder = make_mlp(
                self.decoder_units + [self.data_dim],
                get_activation(self.decoder_activation, dtype),
                "decoder",
                [latent_dim],
                dtype,
            )

        self.encoder = encoder
        self.decoder = decoder

        self._prior = None
        if self.prior_type == "deep":  # this implements the deep factorized CDF entropy model
            self._prior = tfc.DeepFactorized(
                batch_shape=[self.latent_dim], dtype=self.dtype)
        elif self.prior_type == 'std_gaussian':  # use 'gmm_1' for gaussian prior with learned mean/scale
            self._prior = tfd.MultivariateNormalDiag(loc=tf.zeros([self.latent_dim], dtype=self.dtype),
                                                     scale_diag=tf.ones([self.latent_dim], dtype=self.dtype))
        elif self.prior_type == 'maf':
            self._maf_mades = [
                tfb.AutoregressiveNetwork(params=2, activation=ar_activation, hidden_units=self.ar_hidden_units) for _
                in range(self.maf_stacks)]
            base_distribution = tfd.MultivariateNormalDiag(loc=tf.zeros([self.latent_dim], dtype=self.dtype),
                                                           scale_diag=tf.ones([self.latent_dim], dtype=self.dtype))
            self._prior = af_transform(base_distribution, self._maf_mades, permute=True, iaf=False)
        elif self.prior_type[:4] in ("gsm_", "gmm_", "lsm_", "lmm_"):  # mixture prior; specified like 'gmm_2'
            # This only implements a scalar mixture for each dimension, and the dimensions themselves are factorized
            components = int(self.prior_type[4:])
            shape = (self.latent_dim, components)
            self.logits = tf.Variable(tf.random.normal(shape, dtype=self.dtype))
            self.log_scale = tf.Variable(
                tf.random.normal(shape, mean=2., dtype=self.dtype))
            if "s" in self.prior_type:  # scale mixture
                self.loc = 0.
            else:
                self.loc = tf.Variable(tf.random.normal(shape, dtype=self.dtype))
        else:
            raise ValueError(f"Unknown prior_type: '{self.prior_type}'.")

        self.build([None, self.data_dim])

    def prior(self, conv_unoise=False):
        if self._prior is not None:
            prior = self._prior
        elif self.prior_type[:4] in ("gsm_", "gmm_", "lsm_", "lmm_"):
            cls = tfd.Normal if self.prior_type.startswith("g") else tfd.Logistic
            prior = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=self.logits),
                components_distribution=cls(
                    loc=self.loc, scale=tf.math.softplus(self.log_scale)),
            )
        if conv_unoise:  # convolve with uniform noise for NTC compression model
            prior = tfc.UniformNoiseAdapter(prior)
        return prior

    def encode_decode(self, x):
        """Given a batch of inputs, perform a full inference -> generative pass through the model."""
        if self.posterior_type in ('gaussian', 'iaf'):
            encoder_res = self.encoder(x)
            qy_loc = encoder_res[..., :self.latent_dim]
            qy_scale = tf.nn.softplus(encoder_res[..., self.latent_dim:] + softplus_inv_1)
            y_dist = tfd.MultivariateNormalDiag(loc=qy_loc, scale_diag=qy_scale, name="q_y")
            if self.posterior_type == 'iaf':
                y_dist = af_transform(y_dist, self._iaf_mades, permute=True, iaf=True)

            y_tilde = y_dist.sample()  # Y ~ Q(Y|X); batch_size by latent_dim
            log_q_tilde = y_dist.log_prob(y_tilde)  # [batch_size]; should be 0 on avg for uniform distribution
            prior = self.prior(conv_unoise=False)
        elif self.posterior_type == 'uniform':
            encoder_res = self.encoder(x)
            y_dist = tfd.Uniform(low=encoder_res - 0.5, high=encoder_res + 0.5, name="q_y")
            y_tilde = y_dist.sample()  # Y ~ Q(Y|X); batch_size by latent_dim
            log_q_tilde = 0.  # [batch_size]; should be 0 on avg for uniform distribution
            prior = self.prior(conv_unoise=True)  # NTC
        else:
            raise NotImplementedError(f'unknown posterior_type={self.posterior_type}')

        if self.prior_type == 'maf':
            log_prior = prior.log_prob(y_tilde)  # just [batch_size], one number per each x in the batch
        else:
            log_prior = tf.reduce_sum(prior.log_prob(y_tilde),
                                      axis=-1)  # sum across latent_dim (since the prior is fully factorized)
        kls = log_q_tilde - log_prior

        if self.decoder:
            y_tilde = self.decoder(y_tilde)

        return y_dist, y_tilde, kls

    def get_losses(self, x):
        _, y_tilde, kls = self.encode_decode(x)
        mse = tf.reduce_mean(tf.math.squared_difference(x, y_tilde))
        if self.nats:
            rates = kls
        else:
            rates = (kls / tf.cast(tf.math.log(2.), self.dtype))  # convert to bits
        if self.rpd:  # normalize by number of data dimension
            rate = tf.reduce_mean(rates) / float(self.data_dim)
        else:
            rate = tf.reduce_mean(rates)
        loss = rate + self.lmbda * mse
        return loss, rate, mse

    def call(self, x):
        return self.get_losses(x)

    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss, rate, mse = self.get_losses(x)
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.loss.update_state(loss)
        self.rate.update_state(rate)
        self.mse.update_state(mse)
        return {m.name: m.result() for m in [self.loss, self.rate, self.mse]}

    def test_step(self, x):
        loss, rate, mse = self(x, training=False)
        self.loss.update_state(loss)
        self.rate.update_state(rate)
        self.mse.update_state(mse)
        return {m.name: m.result() for m in [self.loss, self.rate, self.mse]}

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
        self.loss = tf.keras.metrics.Mean(name="loss")
        self.rate = tf.keras.metrics.Mean(name="rate")
        self.mse = tf.keras.metrics.Mean(name="mse")

    @classmethod
    def create_model(cls, args):
        return cls(**vars(args))

    def fit(self, *args, **kwargs):
        retval = super().fit(*args, **kwargs)
        return retval


def get_runname(args):
    from utils import config_dict_to_str
    model_name = os.path.splitext(os.path.basename(__file__))[0]
    runname = config_dict_to_str(vars(args),
                                 record_keys=('data_dim', 'latent_dim', 'lmbda', 'encoder_units', 'decoder_units',
                                              'prior_type', 'posterior_type',
                                              'maf_stacks', 'iaf_stacks'), prefix=model_name)
    return runname


def gen_dataset(dataset_spec: str, data_dim: int, batchsize: int, dtype='float32', **kwargs):
    """
    This returns an 'infinite' batched dataset for training.
    If only one batch of data is desired, you can create a Python iterator like 'batched_iterator = iter(dataset)'
     and then call 'batch = next(batch_iterator)'; see https://github.com/tensorflow/tensorflow/issues/40285
    :param dataset: a string specifying the dataset
    :param data_dim:
    :param batchsize:
    :return:
    """
    if dataset_spec == 'gaussian':
        if kwargs.get('gparams_path', None):
            gparams = np.load(kwargs['gparams_path'])
            loc = gparams['loc'].astype(dtype)
            scale = gparams['scale'].astype(dtype)
        else:
            loc = np.zeros(data_dim, dtype=dtype)
            scale = np.ones(data_dim, dtype=dtype)
        source = tfd.Normal(loc=loc, scale=scale)
        map_sample_fun = lambda _: source.sample(batchsize)
    elif dataset_spec == 'banana':
        if kwargs.get('gparams_path', None):
            print('Did you mean to run with --dataset gaussian instead?')
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
    return dataset


def get_lr_scheduler(learning_rate, epochs, decay_factor=0.1, warmup_epochs=0):
    """Returns a learning rate scheduler function for the given configuration."""

    def scheduler(epoch, lr):
        del lr  # unused
        if epoch < warmup_epochs:
            return learning_rate * 10. ** (epoch - warmup_epochs)
        if epoch < 1 / 2 * epochs:
            return learning_rate
        if epoch < 3 / 4 * epochs:
            return learning_rate * decay_factor ** 1
        if epoch < 7 / 8 * epochs:
            return learning_rate * decay_factor ** 2
        return learning_rate * decay_factor ** 3

    return scheduler


def train(args):
    """Instantiates and trains the model."""

    model = Model.create_model(args)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
    )

    if args.dataset.endswith('.npy') or args.dataset.endswith('.npz'):
        from utils import get_np_datasets
        train_dataset, validation_dataset = get_np_datasets(args.dataset, args.batchsize)
    else:
        train_dataset = gen_dataset(dataset_spec=args.dataset, data_dim=args.data_dim, batchsize=args.batchsize,
                                    gparams_path=args.gparams_path)
        validation_dataset = gen_dataset(dataset_spec=args.dataset, data_dim=args.data_dim, batchsize=args.batchsize,
                                         gparams_path=args.gparams_path)

    validation_dataset = validation_dataset.take(
        args.max_validation_steps)  # keras crashes without this (would be using an infinite validation set)

    ##################### BEGIN: Good old bookkeeping #########################
    runname = get_runname(args)
    save_dir = os.path.join(args.checkpoint_dir, runname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    from utils import get_time_str
    time_str = get_time_str()
    # log to file during training
    log_file_path = os.path.join(save_dir, f'record-{time_str}.jsonl')
    from utils import get_json_logging_callback
    file_log_callback = get_json_logging_callback(log_file_path)
    print(f'Logging to {log_file_path}')
    ##################### END: Good old bookkeeping #########################

    tmp_save_dir = os.path.join('/tmp/rdvae', save_dir)
    lr_scheduler = get_lr_scheduler(args.lr, args.epochs, decay_factor=0.2)
    hist = model.fit(
        train_dataset.prefetch(tf.data.AUTOTUNE),
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=validation_dataset.cache(),
        validation_freq=1,
        verbose=int(args.verbose),
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            # tf.keras.callbacks.TensorBoard(
            #     log_dir=tmp_save_dir,
            #     histogram_freq=1, update_freq="epoch"),
            tf.keras.callbacks.experimental.BackupAndRestore(tmp_save_dir),
            file_log_callback,
            tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
        ],
    )
    records = hist.history
    ckpt_path = os.path.join(save_dir, f"ckpt-lmbda={args.lmbda}-epoch={args.epochs}-loss={records['loss'][-1]:.3f}")
    model.save_weights(ckpt_path)
    print('Saved checkpoint to', ckpt_path)
    return hist


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="Report progress and metrics when training or compressing.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint_dir", default="./checkpoints",
        help="Directory where to save/load model checkpoints.")

    # Specifying dataset
    parser.add_argument("--data_dim", type=int, default=None, help="Data dimensionality.")
    parser.add_argument("--dataset", type=str, default="banana",
                        help="Dataset specifier. This can be known dataset names ('gaussian'|'banana')"
                             "handled by gen_dataset, or a path to a numpy array of data vectors.")
    parser.add_argument("--gparams_path", type=str, default=None, help="Path to custom Gaussian loc/scale params. ")

    # Model specific args
    parser.add_argument(
        "--encoder_units", type=lambda s: [int(i) for i in s.split(',')], default=[],
        help="A comma delimited list, specifying the number of units per hidden layer in the encoder.")
    parser.add_argument(
        "--decoder_units", type=lambda s: [int(i) for i in s.split(',')], default=[],
        help="A comma delimited list, specifying the number of units per hidden layer in the decoder;"
             "set to 0 to not use decoder (for quantization experiments).")
    parser.add_argument("--encoder_activation", type=str, default="softplus", help="Activation in encoder MLP")
    parser.add_argument("--decoder_activation", type=str, default="softplus", help="Activation in decoder MLP")

    parser.add_argument("--latent_dim", type=int, help="Latent space dimensionality."
                                                       "Will be automatically set to be the same as data_dim if decoder_units=0.")
    parser.add_argument(
        "--posterior_type", type=str, default='gaussian', help="Posterior type. One of 'gaussian|iaf|uniform'.")
    parser.add_argument(
        "--prior_type", type=str, default='deep', help="Prior type. Can be 'deep|maf|std_gaussian' or a factorized"
                                                       "mixture model specified like 'gmm_2'. Use prior_type='deep' with"
                                                       "posterior_type='uniform' for a compressive autoencoder (NTC).")
    parser.add_argument(
        "--ar_hidden_units", type=lambda s: [int(i) for i in s.split(',')], default=[10, 10],
        help="A comma delimited list, specifying the number of hidden units per MLP layer in the AutoregressiveNetworks"
             "for normalizing flow.")
    parser.add_argument(
        "--ar_activation", type=str, default=None,
        help="Activation function to use in the AutoregressiveNetworks"
             "for normalizing flow. No need to worry about output activation as tfb.MaskedAutoregressiveFlow operates on"
             "log_scale outputted by the AutoregressiveNetworks.")
    parser.add_argument(
        "--maf_stacks", type=int, default=0, help="Number of stacks of transforms to use for MAF prior.")
    parser.add_argument(
        "--iaf_stacks", type=int, default=0, help="Number of stacks of transforms to use for IAF posterior.")
    parser.add_argument(
        "--lambda", type=float, default=0.01, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    parser.add_argument(
        "--rpd", default=False, action='store_true',
        help="Whether to normalize the rate (per sample) by the number of data dimensions; default is False, i.e., bits/nats per sample.")
    parser.add_argument(
        "--nats", default=False, action='store_true',
        help="Whether to compute rate in terms of nats (instead of bits)")

    subparsers = parser.add_subparsers(
        title="commands", dest="command",
        help="What to do: 'train' loads training data and trains (or continues "
             "to train) a new model. Invoke '<command> -h' for more information.")

    # 'train' subcommand.
    train_cmd = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trains (or continues to train) a new model."
                    "When training on an infinite data stream produced by"
                    "a generator (see gen_dataset method), the validation set"
                    "is created from a number of random batches (max_validation_steps)"
                    "from the generator and kept fixed throughout training."
                    "When training on a numpy array dataset, the code looks for a corresponding"
                    "val/test dataset as the validation data; if not found, a random subset of"
                    "training data is used as validation data.")

    train_cmd.add_argument(
        "--batchsize", type=int, default=1024,
        help="Batch size for training and validation.")
    train_cmd.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate.")
    train_cmd.add_argument(
        "--epochs", type=int, default=100,
        help="Train up to this number of epochs. (One epoch is here defined as "
             "the number of steps given by --steps_per_epoch, not iterations "
             "over the full training dataset.)")
    train_cmd.add_argument(
        "--steps_per_epoch", type=int, default=1000,
        help="Perform validation and produce logs after this many batches.")
    train_cmd.add_argument(
        "--max_validation_steps", type=int, default=10,
        help="Maximum number of batches to use for validation.")

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    # Invoke subcommand.
    if check_no_decoder(args.decoder_units):
        print(f'Using Z=Y; resetting latent_dim={args.latent_dim} to data_dim={args.data_dim}')
        args.latent_dim = args.data_dim

    seed = args.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if args.command == "train":
        train(args)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
