# your code goes here
import numpy as np
from math import exp, sqrt

class SOM:
	# init
	def __init__(self, height, width, input_datas, R = None, lr = None, shrink = None):
		self.height = height
		self.width = width
		self.dim = input_datas.shape[1]
		self.input_datas = input_datas
		self.neurons = np.random.rand(height * width, self.dim)
		# approximate radius
		if R == None or float(R) > min(height, width):
			self.R = min(height, width)
		else:
			self.R = float(R)
		# learning rate
		if lr == None:
			self.lr = 0.1
		else:
			self.lr = float(lr)
		# Shrinkage
		if shrink == None:
			self.shrink = 0.8
		else:	
			self.shrink = float(shrink)

	# train
	def train(self, iterations = 100):
		# setup parameter
		input_datas = self.input_datas
		neurons = self.neurons
		winner_neuron = 0
		# training iterations
		for i in range(iterations):
			for idx, input_data in enumerate(input_datas):
				w_n = np.argmin(np.sqrt(((neurons - input_data) ** 2).sum(axis = 1)), axis = 0)
				winner_neuron = w_n
				# update weight
				self.update_weight(winner_neuron, input_data)
			# update parameter	
			self.R = self.shrink * self.R
			self.lr = self.shrink * self.lr
	
	# inference
	def inference(self, inference_data, value):
		# setup parameter
		input_data = inference_data
		neurons = self.neurons
		graph = np.zeros((self.height, self.width), dtype = np.float32)
		winner_neuron = 0
		# mapping to neurons
		w_n = np.argmin(np.sqrt(((neurons - input_data) ** 2).sum(axis = 1)), axis = 0)
		winner_neuron = w_n
		w_row = winner_neuron / self.width
		w_col = winner_neuron % self.width
		graph[w_row][w_col] = value
		return graph
	
	# update weight		
	def update_weight(self, winner_neuron, input_data):
		# setup parameter
		width = self.width
		neurons = self.neurons
		X = input_data
		# convert 1D to 2D coordinate
		w_row = winner_neuron / width
		w_col = winner_neuron % width
		for n_idx, weight in  enumerate(X - neurons):
			# convert 1D to 2D coordinate
			c_row = n_idx / width
			c_col = n_idx % width
			# approximate formulate
			dis = sqrt((c_row - w_row) ** 2 + (c_col - w_col) ** 2)
			k = exp(-(dis / self.R) ** 2)
			self.neurons[n_idx] += self.lr * weight * k
			
if __name__ == "__main__":
	height = 10
	width = 20
	dim = 50
	num_of_input = 100
	R = 8.0
	lr = 0.1
	shrink = 0.8
	input_datas = np.random.rand(num_of_input, dim)
	my_som = SOM(height, width, input_datas, R, lr, shrink)
	my_som.train(1000)
	graph = np.zeros((height, width), dtype = np.float32)
	for input_data in input_datas:
		graph += my_som.inference(input_data, 1)
	print graph	

