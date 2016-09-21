import lmdb
import caffe
from PIL import Image
from StringIO import StringIO
import numpy as np
import matplotlib.pyplot as plt
import cv2

def read(file_name):
	lmdb_env = lmdb.open(file_name, readonly=True)
	print lmdb_env.stat()

	lmdb_txn = lmdb_env.begin()
	lmdb_cursor = lmdb_txn.cursor()
	datum = caffe.proto.caffe_pb2.Datum()

	for key, value in lmdb_cursor:
		datum.ParseFromString(value)

		data = caffe.io.datum_to_array(datum)
		im = np.array(bytearray(datum.data)).reshape(datum.channels, datum.height, datum.width)
	
		print key, im.size, im.shape, im.transpose((1, 2, 0)).shape
		if int(key) % 200 == 0:
			#plt.imshow(im.transpose((1, 2, 0)))
			#plt.show()
    
			# cv2 
			image = np.transpose(data, (1, 2, 0))
			cv2.imshow('cv2', image)
			cv2.waitKey(0)

file = 'val_label_lmdb'
#file = 'train_img_lmdb'
read(file)
