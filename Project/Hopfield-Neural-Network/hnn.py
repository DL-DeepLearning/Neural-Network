import numpy as np

class HNN:
	def __init__(self, input_data):
		mat_len = input_data.shape[1]
		self.X = input_data
		self.weights = np.matrix(np.zeros((mat_len, mat_len)))
		self.threashold = np.matrix(np.zeros((mat_len)))
		self.train()
		
	def train(self):
		X = np.matrix(self.X)
		mat_len = X.shape[1]
		W = self.weights
		
		for x in X:
			W += x.transpose() * x

		for i, j in zip(xrange(mat_len), xrange(mat_len)):
			W[i, j] = 0
		
		self.threashold = -0.5 * W.sum(axis = 1)
		self.weights = W

		
	def inference(self, input_data):
		W = self.weights
		theata = self.threashold
		prev_Y = np.matrix(input_data).transpose()
		vec_len = input_data.shape[0]
		positive = np.matrix(np.zeros((vec_len, 1)) + 1)
		negative = np.matrix(np.zeros((vec_len, 1)) - 1)
		
		while True:
			R =  W * prev_Y + theata
			Y = np.multiply((R > 0), positive) + np.multiply((R == 0) , prev_Y) + np.multiply((R < 0), negative)
			if not (Y != prev_Y).any():
				break
			else:
				prev_Y = Y

		return Y.reshape(-1)
		
if __name__ == "__main__":
	input_data = np.random.randint(2, size=(2, 4)) * 2 - 1
	test_data = np.random.randint(2, size=(2, 4)) * 2 - 1
	Hopfield = HNN(input_data)
	print input_data[0]
	print Hopfield.inference(input_data[0])

	
	
				