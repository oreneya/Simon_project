class Property(object):
	"""Each property should have its own model.
	Each part has several properties.
	Each property links to several serial numbers."""
	def __init__(self, name, lower_limit, upper_limit):
		self.name = name
		self.lower_limit = lower_limit
		self.upper_limit = upper_limit
		self.serial_numbers = []
	
	def train(self, data_to_train):
		"""Train method takes several time-serieses to train on."""
		self.data_to_train = data_to_train

	def validate(self, data_to_validate):
		"""Validate method takes several time-serieses to validate with."""
		self.data_to_validate = data_to_validate

	def predict(self, data_to_predict):
		"""Predict method takes one time-series to predict its future."""
		self.data_to_predict = data_to_predict

class SerialNumber(object):
	"""Class SerialNumber is for the historic data of specific physical part w.r.t some property."""
	def __init__(self, name, x, y):
		self.name = name
		self.x = x
		self.y = y
		