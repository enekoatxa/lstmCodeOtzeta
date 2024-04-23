# example of a wgan for generating handwritten digits
from numpy import expand_dims
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras import backend
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.constraints import Constraint
from matplotlib import pyplot
import tensorflow as tf
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader
from lstmDataset import lstmDataset
from pickle import dump, load
import numpy as np

################################################
# global variables for training and dimensions #
################################################
n_features = 192
seq_length = 30

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value

	# clip model weights to hypercube
	def __call__(self, weights):
		return tf.clip_by_value(weights, -self.clip_value, self.clip_value)

	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}

# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return tf.math.reduce_mean(y_true * y_pred)

# define the standalone critic model
def define_critic(in_shape=(n_features, seq_length,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# weight constraint
	const = ClipConstraint(0.01)
	# define model
	model = Sequential()
	model.add(Input(shape=(seq_length, n_features)))
	# input layer
	model.add(Dense(n_features, activation=LeakyReLU(alpha=0.2), kernel_constraint=const, kernel_initializer=init))
	model.add(Dropout(0.4))
	# layer 1
	model.add(Dense(256, activation=LeakyReLU(alpha=0.2), kernel_constraint=const, kernel_initializer=init))
	model.add(Dropout(0.4))
	# layer 2
	model.add(Dense(128, activation=LeakyReLU(alpha=0.2), kernel_constraint=const, kernel_initializer=init))
	model.add(Dropout(0.4))
	# layer 3
	model.add(Dense(64, activation=LeakyReLU(alpha=0.2), kernel_constraint=const, kernel_initializer=init))
	model.add(Dropout(0.4))
	model.add(Flatten())
	# classifier
	model.add(Dense(1))
	# compile model
	opt = RMSprop(learning_rate=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model

# define the standalone generator model
def define_generator(latent_dim):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
	model = Sequential()
	model.add(Input(shape=(seq_length, latent_dim)))
	# layer 1
	model.add(Dense(latent_dim, activation=LeakyReLU(alpha=0.2), kernel_initializer=init))
	model.add(Dropout(0.4))
	# layer 2
	model.add(Dense(256, activation=LeakyReLU(alpha=0.2), kernel_initializer=init))
	model.add(Dropout(0.4))
	# layer 3
	model.add(Dense(128, activation=LeakyReLU(alpha=0.2), kernel_initializer=init))
	model.add(Dropout(0.4))
    # layer 4
	model.add(Dense(256, activation=LeakyReLU(alpha=0.2), kernel_initializer=init))
	model.add(Dropout(0.4))
	#  output layer
	model.add(Dense(n_features, activation=LeakyReLU(alpha=0.2), kernel_initializer=init))
	model.add(Dropout(0.4))
	return model

# define the combined generator and critic model, for updating the generator
def define_gan(generator, critic):
	# make weights in the critic not trainable
	for layer in critic.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the critic
	model.add(critic)
	# compile model
	opt = RMSprop(learning_rate=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model

# load images
def load_real_samples():
	# TODO: oraindik firstpersonframes ez dut erabili
	scaler = load(open("scalerDifferences.pkl", "rb"))
	x, firstPersonFrames, ids = bvhLoader.loadDataset(datasetName="silenceDataset3sec", partition="Train", trim=False,
                                                verbose=True, onlyPositions=False,
                                                onlyRotations=False, removeHandsAndFace=True, 
                                                scaler=scaler, loadDifferences = True, jump = 0, specificSize = 100)
	return x

# select real samples
def generate_real_samples(datamodule, n_samples):
    X = []
    for n in range(0, n_samples):
        # choose random instance
        id = randint(0, len(datamodule))
        # select images
        person = datamodule[id]
        frame = randint(0, len(person)-seq_length)
        X.append(datamodule[id][frame:frame+seq_length])
	# generate class labels, -1 for 'real'
    y = -ones((n_samples, 1))
    return np.stack(X), np.stack(y)

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples * seq_length)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, seq_length, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels with 1.0 for 'fake'
	y = ones((n_samples, 1))
	return np.stack(X), np.stack(y)

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
	# plot history
	pyplot.plot(d1_hist, label='crit_real')
	pyplot.plot(d2_hist, label='crit_fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	pyplot.savefig('plot_line_plot_loss.png')
	pyplot.close()

# train the generator and critic
def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=50, n_batch=512, n_critic=5):
	datasetSize = 0
	for person in dataset:
		datasetSize = datasetSize + len(person) - seq_length
	print("dataset size: " + str(datasetSize) + " sequences")
	# calculate the number of batches per training epoch
	bat_per_epo = int(datasetSize / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# lists for keeping track of loss
	c1_hist, c2_hist, g_hist = list(), list(), list()
	# manually enumerate epochs
	for i in range(n_steps):
		# update the critic more than the generator
		c1_tmp, c2_tmp = list(), list()
		for _ in range(n_critic):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# update critic model weights
			c_loss1 = c_model.train_on_batch(X_real, y_real)
			print(str(c_loss1) + "closs1")
			c1_tmp.append(c_loss1)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update critic model weights
			c_loss2 = c_model.train_on_batch(X_fake, y_fake)
			print(str(c_loss1) + "closs2")
			c2_tmp.append(c_loss2)
		# store critic loss
		c1_hist.append(mean(c1_tmp))
		c2_hist.append(mean(c2_tmp))
		# prepare points in latent space as input for the generator
		X_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = -ones((n_batch, 1))
		# update the generator via the critic's error
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		g_hist.append(g_loss)
		# summarize loss on this batch
		print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))
	# line plots of loss
	plot_history(c1_hist, c2_hist, g_hist)

# size of the latent space
latent_dim = 50
# create the critic
critic = define_critic()
critic.summary()
# create the generator
generator = define_generator(latent_dim)
generator.summary()
# create the gan
gan_model = define_gan(generator, critic)
# load image data
dataset = load_real_samples()
print(len(dataset))
# train model
train(generator, critic, gan_model, dataset, latent_dim)
generator.save("models/generator.keras")
critic.save("models/critic.keras")