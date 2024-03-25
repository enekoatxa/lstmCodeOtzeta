import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
from keras import layers

from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.constraints import Constraint

import tensorflow as tf
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/code/util')
import bvhLoader
from lstmDataset import lstmDataset
from pickle import dump, load
import numpy as np

# scaler = load(open("scalerDifferences.pkl", "rb"))

# datamodule = lstmDataset(root="/home/bee/Desktop/idle animation generator", specificSize = 1000, batchSize = 128, partition="Train", datasetName = "dataset", 
#                              sequenceSize = n_steps, trim=False, verbose=True, outSequenceSize=n_steps_out, removeHandsAndFace = True, scaler = scaler, loadDifferences = True, jump = 5)
# datamoduleVal = lstmDataset(root="/home/bee/Desktop/idle animation generator", specificSize = 200, batchSize = 128, partition="Validation", datasetName = "dataset", 
#                             sequenceSize = n_steps, trim=False, verbose=True, outSequenceSize=n_steps_out, removeHandsAndFace = True, scaler = scaler, loadDifferences = True, jump = 5)

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
def get_discriminator_model():
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# weight constraint
	const = ClipConstraint(0.01)
	# define model
	model = Sequential()
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
	# classifier
	model.add(Dense(1))
	# compile model
	opt = RMSprop(learning_rate=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model

# define the standalone generator model
def get_generator_model(latent_dim):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
	model = Sequential()
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

n_steps = 20
n_steps_out = 1
n_features = 192
learning_rate = 0.1

d_model = get_discriminator_model()
d_model.build()
d_model.summary()