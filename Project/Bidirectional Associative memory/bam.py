import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
from PIL import Image

class bam():
	pass




def main():
	from scipy import misc
	f = misc.face()
	misc.imsave('face.png', f) # uses the Image module (PIL)

	import matplotlib.pyplot as plt
	plt.imshow(f)
	plt.show()
	
	
if __name__ == "__main__":
	main()
