from __future__ import print_function
import numpy as np
from math import exp, sqrt


class SOM:
	# init
	def __init__(self, height, width, input_datas, R = None, lr = None, shrink = None):
		self.height = height
		self.width = width
		self.dim = input_datas.shape[1]
		self.train_datas = input_datas
		self.neurons = np.random.rand(height * width, self.dim)
		# approximate radius
		if R == None or float(R) > min(height, width):
			self.R = min(height, width)
		else:
			self.R = float(R)
		# learning rate
		if lr == None:
			self.init_lr = 0.1
			self.lr = 0.1
		else:
			self.init_lr = float(lr)
			self.lr = float(lr)
			
		# Shrinkage
		if shrink == None:
			self.shrink = 0.8
		else:	
			self.shrink = float(shrink)

	# train
	def train(self, iterations = 100):
		# setup parameter
		train_datas = self.train_datas
		neurons = self.neurons
		winner_neuron = 0
		total_data = train_datas.shape[0] * 1.0
		# training iterations
		for i in xrange(iterations):
			print(str(i + 1) + ' / '+ str(iterations) + ' iterations', end = ' | ')
			for idx, trn_dt in enumerate(train_datas):
				if (idx / total_data * 100) % 10 == 0: print(u"\u2588", end = '')
				w_n = np.argmin(np.sqrt(((neurons - trn_dt) ** 2).sum(axis = 1)), axis = 0)
				winner_neuron = w_n
				# update weight
				self.update_weight(winner_neuron, trn_dt)
			# update parameter	
			self.R = self.shrink * self.R
			self.lr = self.init_lr * exp(float(i) / iterations)
			print("")	
	# inference
	def inference(self, inference_data, value):
		# setup parameter
		neurons = self.neurons
		graph = np.zeros((self.height, self.width), dtype = np.float32)
		winner_neuron = 0
		# mapping to neurons
		w_n = np.argmin(np.sqrt(((neurons - inference_data) ** 2).sum(axis = 1)), axis = 0)
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
		w_row, w_col = winner_neuron / width, winner_neuron % width
		for n_idx, weight in  enumerate(X - neurons):
			# convert 1D to 2D coordinate
			c_row, c_col = n_idx / width, n_idx % width
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
	train_datas = np.random.rand(num_of_input, dim)
	my_som = SOM(height, width, train_datas, R, lr, shrink)
	my_som.train(1000)
	graph = np.zeros((height, width), dtype = np.float32)
	for input_data in train_datas:
		graph += my_som.inference(input_data, 1)
	print (graph)

