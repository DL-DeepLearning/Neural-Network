import som
import matplotlib.pyplot as plt
import matplotlib.cm as cm

execfile("load_data.py")

''' som '''
height = 100
width = 150
my_som = som.SOM(height, width, features)

print "Train..."
my_som.train(10)

print "Inference..."
Y = []
for f in features:
	graph = my_som.inference(f, 1)
	[[x], [y]] = np.where(graph == 1)
	Y.append([x, y])
Y = np.array(Y, dtype = np.float32)

print "visualize..."	
plt.figure(8)
colors = cm.rainbow(np.linspace(0, 1, len(non_repeat_answer)))
for idx, coor in enumerate(Y):
	c = colors[non_repeat_answer.index(classify[idx])]
	x, y = coor[0], coor[1]
	plt.scatter(x, y, color = c)
'''
# text annotate	
for label, x, y in zip(classify, Y[:, 0], Y[:, 1]):
	plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
'''	
plt.show()