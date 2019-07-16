import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

class PartAttribute(object):
	"""Each part attribute should have its own model.
	Each part has several attributes.
	Each part attribute links to several serial numbers."""
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
		self.data_to_validate = self.serial_numbers[sn_valid_idx]

		# prepare data for supervised training
		lin = 10 # length of time-series as input
		lout = 8 # length of output vector
		
		x_train = np.array([])
		y_train = np.array([])
		
		for sn in self.data_to_train:
			for i in range(1+len(sn.measurement.values[:-(lin+lout)])):
				x_train = np.append(x_train, sn.measurement.values[i:i+lin])
				y_train = np.append(y_train, sn.measurement.values[i+lin:i+lin+lout])
		
		x_train = x_train.reshape(len(x_train) / lin, lin)
		print(x_train.shape)
		y_train = y_train.reshape(len(y_train) / lout, lout)
		print(y_train.shape)
				
		x_val = np.array([])
		y_val = np.array([])
		
		sn = self.data_to_validate
		for i in range(1+len(sn.measurement.values[:-(lin+lout)])):
			x_val = np.append(x_val, sn.measurement.values[i:i+lin])
			y_val = np.append(y_val, sn.measurement.values[i+lin:i+lin+lout])
		
		x_val = x_val.reshape(len(x_val) / lin, lin)
		y_val = y_val.reshape(len(y_val) / lout, lout)
		
		# model
		num_units = 4 # number of neurons to the hidden layer

		model = Sequential()
		model.add(LSTM(num_units, input_shape=(x_train.shape[0], lin)))
		model.add(Dense(lout))
		model.compile(loss='mean_squared_error', optimizer='adam')
		model.fit(x_train, y_train, epochs=10)#, validation_data=(x_val, y_val))
		return self, model

	def predict(self, data_to_predict):
		"""Predict method takes one time-series to predict its future."""
		self.data_to_predict = data_to_predict

class SerialNumber(object):
	"""Class SerialNumber is for the historic data of specific physical part w.r.t some part attribute."""
	def __init__(self, name, cycle, measurement):
		self.name = name
		self.cycle = cycle
		self.measurement = measurement
		
