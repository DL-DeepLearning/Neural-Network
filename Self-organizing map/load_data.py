import numpy as np
import re

non_repeat_answer = []
classify = []
features = []
''' load data'''
with open("data/iris.data.txt") as file:
	for line in file.readlines():
		new_str = re.split('\n', line)[0]
		if len(new_str) > 0:
			features.append(new_str.split(",")[:-1])
			classify.append(new_str.split(",")[-1])
			
			if not classify[-1] in non_repeat_answer:
				non_repeat_answer.append(classify[-1])

features = np.array(features, dtype = np.float32)


