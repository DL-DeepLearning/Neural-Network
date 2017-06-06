import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def read_image(filepath, width, height, scale):
	imgs = []
	rez = [width / scale, height / scale]
	rez_half = [rez[0] / 2, rez[1] / 2]
	img_center = [width / 2 , height / 2]
	for dir_item in os.listdir(filepath):
		# join dir path and file name
		dir_item_path = os.path.join(filepath, dir_item)
		# check whether a file exists before read
		if os.path.isfile(dir_item_path):
			with Image.open(dir_item_path, 'r') as im:
				n_im_bg = Image.new('RGB', (width, height), (255, 255, 255))
				n_im = im.resize((rez[0], rez[1]), Image.BILINEAR)
				n_im_bg.paste(n_im, ( img_center[0] - rez_half[0] , img_center[1] - rez_half[1]))
				# n_im_bg.save(dir_item_path)
				pixels = list(n_im_bg.getdata())
				binary_pixels = [1 * (((r + g + b) / 3.0) > 220) for r, g, b in pixels]
				bipolar_pixels = np.array(binary_pixels) * 2 -1
				imgs.append(bipolar_pixels)
	return np.array(imgs)
	
train_datas = []
test_datas = []
width = 32
height = 32

train_datas = read_image("train/", width, height, 1)
test_datas = read_image("test/", width, height, 1)
#print train_data.shape
