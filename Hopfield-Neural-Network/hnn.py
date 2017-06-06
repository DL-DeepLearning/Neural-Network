import numpy as np

class HNN:
	def __init__(self, train_data):
		mat_len = train_data.shape[1]
		self.weights = np.matrix(np.zeros((mat_len, mat_len)))
		self.threashold = np.matrix(np.zeros((mat_len)))
		self.train(train_data)
		
	def train(self, train_data):
		X = np.matrix(train_data)
		mat_len = X.shape[1]
		W = self.weights
		
		# Train weight
		for x in X:
			W += x.transpose() * x
			
		# Set the diagonal of weights to zero
		for i, j in zip(xrange(mat_len), xrange(mat_len)):
			W[i, j] = 0
		
		self.threashold = -0.5 * W.sum(axis = 1)
		self.weights = W
		print W

		
	def inference(self, test_data):
		W = self.weights
		theata = self.threashold
		prev_Y = np.matrix(test_data).transpose()
		# binary mask
		vec_len = test_data.shape[0]
		positive = np.matrix(np.zeros((vec_len, 1)) + 1)
		negative = np.matrix(np.zeros((vec_len, 1)) - 1)
		# interations
		while True:
			R =  W * prev_Y
			Y = np.multiply((R > 0), positive) + np.multiply((R == 0) , prev_Y) + np.multiply((R < 0), negative)
			if not (Y != prev_Y).any():
				break
			else:
				prev_Y = Y

		return Y.reshape(-1)
		
if __name__ == "__main__":
	input_data = np.array([	[ 1, -1,  1, -1,  1, -1],
							[-1,  1, -1,  1, -1,  1],
							[ 1,  1,  1,  1,  1,  1],
							[-1, -1, -1, -1, -1, -1]
							])
	test_data = np.random.randint(2, size=(2, 4)) * 2 - 1
	Hopfield = HNN(input_data)
	print np.array([1, 1, 1, -1, 1, -1])
	print Hopfield.inference(np.array([1, 1, 1, -1, 1, -1]))
	print Hopfield.inference(np.array([ 1, -1,  1, -1,  1, -1]))
	print Hopfield.inference(np.array([-1,  1, -1,  1, -1,  1]))

	
	
				