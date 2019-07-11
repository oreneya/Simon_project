import numpy as np

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
		self.data_to_train = np.empty(len(sn_train_idx))
		self.data_to_validate = np.empty(len(sn_valid_idx))

		# fill data structure to train with
		for i in sn_train_idx:
			self.data_to_train = np.append(self.data_to_train, self.serial_numbers[i])

		# fill data structure to validate with
		for i in sn_valid_idx:
			self.data_to_validate = np.append(self.data_to_validate, self.serial_numbers[i])

		# construct model
		# compile model
		# train
		# save model

	def predict(self, data_to_predict):
		"""Predict method takes one time-series to predict its future."""
		self.data_to_predict = data_to_predict

class SerialNumber(object):
	"""Class SerialNumber is for the historic data of specific physical part w.r.t some property."""
	def __init__(self, name, x, y):
		self.name = name
		self.x = x
		self.y = y
		