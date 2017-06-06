import numpy as np
import matplotlib.pyplot as plt
import hnn

execfile("load_data.py")
train_datas = train_datas[::2]
test_datas = test_datas
height = height
width = width
hopfield = hnn.HNN(train_datas)

for t_d in train_datas:
	print "test_data"
	td_reshap = 255 * (t_d.reshape(height, width) == -1)
	print td_reshap
	result = hopfield.inference(t_d)
	plt.imshow(td_reshap)
	plt.show()
	print "answer"
	result_reshap = 255 * (result.reshape(height, width) == -1)
	print result_reshap
	plt.imshow(result_reshap)
	plt.show()