import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers

import tensorflow as tf

import tsgm

latent_dim = 64
output_dim = 2
feature_dim = 1
seq_len = 100
batch_size = 128

generator_in_channels = latent_dim + output_dim
discriminator_in_channels = feature_dim + output_dim

X, y_i = tsgm.utils.gen_sine_vs_const_dataset(5000, seq_len, 1, max_value=20, const=10)

scaler = tsgm.utils.TSFeatureWiseScaler((-1, 1))
X_train = scaler.fit_transform(X)
y = keras.utils.to_categorical(y_i, 2)

X_train = X_train.astype(np.float32)
y = y.astype(np.float32)

dataset = tf.data.Dataset.from_tensor_slices((X_train, y))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

tsgm.utils.visualize_ts_lineplot(X_train, y_i)

architecture = tsgm.models.architectures.zoo["cgan_base_c4_l1"](
    seq_len=seq_len, feat_dim=feature_dim,
    latent_dim=latent_dim, output_dim=output_dim)
discriminator, generator = architecture.discriminator, architecture.generator

cond_gan = tsgm.models.cgan.ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

cbk = tsgm.models.monitors.GANMonitor(num_samples=3, latent_dim=latent_dim, save=False, labels=y, save_path="/tmp")
cond_gan.fit(dataset, epochs=1000, callbacks=[cbk])

limit = 500
X_gen = cond_gan.generate(y[:limit])
X_gen = X_gen.numpy()
y_gen = y[:limit]

tsgm.utils.visualize_tsne(X_train[:limit], y[:limit], X_gen, y_gen)
