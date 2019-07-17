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
		
		return x, y
			
	def prepare_data(self, sn_idx, lin, lout, flag=None):
		"""Prepare data for training."""
		
		data = []
		x = []
		y = []
				
		if flag == 'train':
		
			# fill in data structure
			for i in sn_idx:
				data.append(list(self.serial_numbers[i].measurement.values))

			# transform data into supervised learning problem
			for sn in data:
				for i in range(len(sn[:-(lin+lout)])):
					x.append(sn[i:i+lin])
					y.append(sn[i+lin:i+lin+lout])

		else:
			
			# fill in data structure
			data = list(self.serial_numbers[sn_idx].measurement.values)
			
			# transform data into supervised learning problem
			sn = data
			for i in range(len(sn[:-(lin+lout)])):
				x.append(sn[i:i+lin])
				y.append(sn[i+lin:i+lin+lout])
			
		x, y = self.standardize(x, y, flag)
		
		# reshape data - necessary for NN input shape
		x = np.reshape(x, (len(x), lin, 1))
		y = np.reshape(y, (len(y), lout))
		
		return x, y

	def train(self, lin, lout, sn_train_idx, sn_val_idx):
		"""Train method takes several time-series to train on, and others to validate on."""

		# ------------------------------ #
		#    prepare data for training   #
		# ------------------------------ #

		x_train, y_train = self.prepare_data(sn_train_idx, lin, lout, 'train')
		x_val, y_val = self.prepare_data(sn_val_idx, lin, lout)
		
		# ----------- #
		#    model    #
		# ----------- #
		
		num_units = 4 # number of neurons to the (currently only) hidden layer
		
		# configure NN architecture
		model = Sequential()
		model.add(LSTM(num_units, input_shape=(x_train.shape[1], 1)))
		model.add(Dense(lout))
		
		model.compile(loss='mean_squared_error', optimizer='adam')
		
		# finally begin training
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




