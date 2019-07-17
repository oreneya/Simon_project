import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

class PartAttribute(object):
	"""Each part attribute should have its own model.
	Each part has several attributes.
	Each part attribute links to several serial numbers."""
	def __init__(self, name, lower_limit, upper_limit):
		self.name = name
		self.lower_limit = lower_limit
		self.upper_limit = upper_limit
		self.serial_numbers = []
	
	def standardize(self, x, y, flag=None):
		"""Just a standardization of data beacause... activation functions."""
		
		if flag == 'train':
			self.scaler = MinMaxScaler(feature_range=(-1, 1)) # range due to tanh in the LSTM (default)
			data = np.concatenate([np.concatenate(x), np.concatenate(y)])
			self.scaler.fit(data.reshape(-1,1))
		
		else:
			data = np.concatenate([np.concatenate(x), np.concatenate(y)])
		
		scaled = self.scaler.transform(data.reshape(-1,1))
		x_scaled = scaled[:len(x)*len(x[0])]
		y_scaled = scaled[len(x)*len(x[0]):]
		x = np.reshape(x_scaled, (len(x), len(x[0])))
		y = np.reshape(y_scaled, (len(y), len(y[0])))
			
	def train(self, sn_train_idx, sn_valid_idx):
		"""Train method takes several time-serieses to train on, and others to validate on."""
		self.data_to_train = []
		self.data_to_validate = []

		# fill data structure to train with
		for i in sn_train_idx:
			self.data_to_train.append(list(self.serial_numbers[i].measurement.values))
		
		# fill data structure to validate with
		self.data_to_validate = list(self.serial_numbers[sn_valid_idx].measurement.values)

		# prepare data for supervised training
		lin = 10 # length of time-series as input
		lout = 8 # length of output vector
		
		x_train = []
		y_train = []
		
		for sn in self.data_to_train:
			for i in range(len(sn[:-(lin+lout)])):
				x_train.append(sn[i:i+lin])
				y_train.append(sn[i+lin:i+lin+lout])
		
		self.standardize(x_train, y_train, 'train')

		x_train = np.reshape(x_train, (len(x_train), lin, 1))
		y_train = np.reshape(y_train, (len(y_train), lout))
		
		x_val = []
		y_val = []
		
		sn = self.data_to_validate
		for i in range(len(sn[:-(lin+lout)])):
			x_val.append(sn[i:i+lin])
			y_val.append(sn[i+lin:i+lin+lout])
		
		self.standardize(x_val, y_val)
		
		x_val = np.reshape(x_val, (len(x_val), lin, 1))
		y_val = np.reshape(y_val, (len(y_val), lout))
		
		# model
		num_units = 4 # number of neurons to the (currently only) hidden layer

		model = Sequential()
		model.add(LSTM(num_units, input_shape=(x_train.shape[1], 1)))
		model.add(Dense(lout))
		model.compile(loss='mean_squared_error', optimizer='adam')
		model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
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
		
