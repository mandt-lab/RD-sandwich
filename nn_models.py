import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, Flatten


def get_activation(activation: str, dtype=None):
    if not activation or activation.lower() == 'none':
        return None
    if activation in ("gdn", "igdn"):
        import tensorflow_compression as tfc
        if activation == "gdn":
            return tfc.GDN(dtype=dtype)
        elif activation == "igdn":
            return tfc.GDN(inverse=True, dtype=dtype)
    else:
        return getattr(tf.nn, activation)


def make_mlp(units, activation, name='mlp', input_shape=None, dtype=None, no_last_activation=True):
    kwargs = [dict(units=u, use_bias=True, activation=activation,
                   name=f"{name}_{i}", dtype=dtype,
                   ) for i, u in enumerate(units)]
    if input_shape is not None:
        kwargs[0].update(input_shape=input_shape)
    if no_last_activation:
        kwargs[-1].update(activation=None)
    layers = [tf.keras.layers.Dense(**k) for k in kwargs]
    return tf.keras.Sequential(layers, name=name)


# Simple convnet used for R-D LB on imgs.
def get_convnet(num_units, kernel_dims=(5, 3), activation='selu', preoutput_activation=None, output_scale=1,
                output_bias=0,
                raw_pixel_input=False):
    """
    Simple convnet.
    :param num_units: iterable, which sets the num of filters of dense units per layer; the first len(kernel_dims)
    entries correspond to num filters of conv layers, while the rest correspond to num of dense units in MLP.
    :param kernel_dims: iterable
    :param activation: hidden activation
    :return:
    """
    if raw_pixel_input:
        layers = [tf.keras.layers.Lambda(lambda x: x / 255.)]
    else:
        layers = []

    i = 0
    for kernel_dim in kernel_dims:
        layers += [
            Conv2D(num_units[i], (kernel_dim, kernel_dim), strides=2, padding='same', activation=activation)]
        i += 1

    if i > 0:  # there was at least one conv layer
        layers += [Flatten()]

    for h in num_units[i:]:
        layers += [Dense(h, activation=activation)]

        # Dense(1, activation='sigmoid')  # 0 <= g <= 1 # seems to saturate around 1, not good
        # Dense(1, activation='softplus')  # 0 <= g <= 1
    layers += [Dense(1, activation=preoutput_activation)]

    if output_bias or output_scale:
        if not output_bias:
            output_bias = 0
        if not output_scale:
            output_scale = 1
        layers += [
            tf.keras.layers.Lambda(lambda x: x * output_scale + output_bias)
        ]
    print('Constructing CNN with layers', layers)
    return tf.keras.Sequential(layers)
