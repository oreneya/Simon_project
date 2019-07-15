import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

class Property(object):
	"""Each property should have its own model.
	Each part has several properties.
	Each property links to several serial numbers."""
	def __init__(self, name, lower_limit, upper_limit):
		self.name = name
		self.lower_limit = lower_limit
		self.upper_limit = upper_limit
		self.serial_numbers = []
	
	def train(self, sn_train_idx, sn_valid_idx):
		"""Train method takes several time-serieses to train on, and others to validate on."""
		self.data_to_train = np.array([])
		self.data_to_validate = np.array([])

		# fill data structure to train with
		for i in sn_train_idx:
			self.data_to_train = np.append(self.data_to_train, self.serial_numbers[i])

		# fill data structure to validate with
		for i in sn_valid_idx:
			self.data_to_validate = np.append(self.data_to_validate, self.serial_numbers[i])

		# prepare data for supervised training
		# model
		num_units = 4 # number of neurons to the hidden layer
		len_ts_in = 10 # length of time-series as input
		len_out = 8 # length of output vector
		model = Sequential()
		model.add(LSTM(num_units, input_shape=(len(sn_train_idx), len_ts_in)))
		model.add(Dense(len_out))
		model.compile(loss='mean_squared_error', optimizer='adam')
		model.fit()
		return self, model

	def predict(self, data_to_predict):
		"""Predict method takes one time-series to predict its future."""
		self.data_to_predict = data_to_predict

class SerialNumber(object):
	"""Class SerialNumber is for the historic data of specific physical part w.r.t some property."""
	def __init__(self, name, x, y):
		self.name = name
		self.x = x
		self.y = y
		