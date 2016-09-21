import os
import numpy as np
from scipy import io
import lmdb
import caffe
from PIL import Image

NUM_IDX_DIGITS = 10
IDX_FMT = '{:0>%d' % NUM_IDX_DIGITS + 'd}'

# src
def img_to_lmdb(paths_src_file, path_dst):
	"""
	paths_src: src img paths 
	path_dst: src img lmdb file
	"""
	print("Creating Training Data LMDB File ..... ")

	paths_src = []
	with open(paths_src_file) as f: 
		for line in f.readlines():
			#print line 
			line = line.strip('\n')
			paths_src.append(line)
	
	#path_dst = 'train-lmdb'
	in_db = lmdb.open(path_dst, map_size=int(1e12))
	with in_db.begin(write=True) as in_txn:
		for in_idx, in_ in enumerate(paths_src):
			print in_idx, in_
			# load image:
			# - as np.uint8 {0, ..., 255}
			# - in BGR (switch from RGB)
			# - in Channel x Height x Width order (switch from H x W x C)
			im = np.array(Image.open(in_)) # or load whatever ndarray you need
			im = im[:,:,::-1]
			im = im.transpose((2,0,1)) # convert to CxHxW
			im_dat = caffe.io.array_to_datum(im)
			in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
	in_db.close()
	print 'end.'

# label 
def label_to_lmdb(paths_label_file, path_dst):
	"""
	paths_src: label img paths 
	path_dst: label lmdb file
	"""
	print("Creating Training Label LMDB File ..... ")
	paths_src = []
	with open(paths_label_file) as f: 
		for line in f.readlines():
			line = line.strip('\n')
			paths_src.append(line)

	in_db = lmdb.open(path_dst, map_size=int(1e12))
	with in_db.begin(write=True) as in_txn:
		for in_idx, in_ in enumerate(paths_src):
			print in_idx, in_
			# load image:
			# - as np.uint8 {0, ..., 255}
			# - in BGR (switch from RGB)
			# - in Channel x Height x Width order (switch from H x W x C)
			im = np.array(Image.open(in_)) # or load whatever ndarray you need

			# 
			im = im.reshape(im.shape[0], im.shape[1], 1)
			im = im.transpose((2,0,1))
			##

			#im = np.expand_dims(im, axis=0)	

			# create the dataset
			im_dat = caffe.io.array_to_datum(im)
			in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
	in_db.close()
	print 'end.'

# train
train_img_paths = 'train_img.txt'
train_img_lmdb = 'train_img_lmdb'

train_label_paths = 'train_label.txt'
train_label_lmdb = 'train_label_lmdb'

img_to_lmdb(train_img_paths, train_img_lmdb)
label_to_lmdb(train_label_paths, train_label_lmdb)


# val
val_img_paths = 'val_img.txt'
val_img_lmdb = 'val_img_lmdb'

val_label_paths = 'val_label.txt'
val_label_lmdb = 'val_label_lmdb1'

img_to_lmdb(val_img_paths, val_img_lmdb)
label_to_lmdb(val_label_paths, val_label_lmdb)
