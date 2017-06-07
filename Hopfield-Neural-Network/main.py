import numpy as np
import matplotlib.pyplot as plt
import hnn

execfile("load_data.py")
start_num = 0
train_datas = train_datas[start_num: start_num + 2]
test_datas = test_datas[start_num: 6 * (start_num + 2)]
height = height
width = width
hopfield = hnn.HNN(train_datas)

for t_d in test_datas:
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