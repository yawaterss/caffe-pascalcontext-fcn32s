import numpy as np
from PIL import Image
import sys

import caffe
caffe.set_device(0) 
caffe.set_mode_gpu()

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe

#img_name = '/home/ss/Desktop/outdoor-30-10-b.jpg' 
img_name = sys.argv[1] + '.png'
im = Image.open(img_name)

# check im's mode is "RGB"
if im.mode == 'RGBA':
	print "change img mode 'RGBA' to 'RGB'"
	im.load()
	bg = Image.new("RGB", im.size, (255, 255, 255))
	bg.paste(im, mask=im.split()[3])
	bg.save(img_name)
	im = Image.open(img_name)
#im.show()

in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
#in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ -= np.array((110,112,109))
in_ = in_.transpose((2,0,1))


# load net
#net = caffe.Net('deploy.prototxt', 'snapshot/train_iter_640000.caffemodel', caffe.TEST)
net = caffe.Net('deploy.prototxt', 'snapshot_bak1/train_iter_500000.caffemodel', caffe.TEST)
#net = caffe.Net('deploy.prototxt', 'snapshot_bak1/train_iter_400000.caffemodel', caffe.TEST)

# shape for input (data blob is N x C x H x W), set data
# put img to net
net.blobs['data'].reshape(1, *in_.shape)  # 1: batch size, *in_.shape 3 channel ?
net.blobs['data'].data[...] = in_

# run net and take argmax for prediction
output = net.forward()

# print
def print_param(output):
	# the blobs
	print '--------------------------'
	print 'the blobs'
	for k, v in net.blobs.items():
		print k, v.data.shape

	# the parameters
	print '--------------------------'
	print 'the paramsters'
	for k, v in net.params.items():
		print k, v[0].data.shape

	# the conv layer weights
	print '--------------------------'
	print 'the conv layer weights'
	print net.params['conv1_1'][0].data

	# the data blob	
	print '--------------------------'
	print 'the data blob'
	print net.blobs['data'].data

	# the conv1_1 blob
	print '--------------------------'
	print 'the conv1_1 blob'
	print net.blobs['conv1_1'].data

	# the pool1 blob
	print '--------------------------'
	print 'the pool1 blob'
	print net.blobs['pool1'].data
	
	weights = net.blobs['fc6'].data[0]
	print 'blobs fc6'
	print np.unique(weights)
	weights = net.blobs['fc7'].data[0]
	print 'blobs fc7'
	print np.unique(weights)
	weights = net.blobs['score_fr_sign'].data[0]
	print 'blobs score_fr_sign'
	print np.unique(weights)
	weights = net.blobs['upscore_sign'].data[0]
	print 'blobs upscore_sign'
	print np.unique(weights)
	weights = net.blobs['score'].data[0]
	print 'blobs score'
	print np.unique(weights)

print_param(output)

out = net.blobs['score'].data[0].argmax(axis=0)

#np.savetxt("vote", out, fmt="%02d")
np.savetxt("vote", out, fmt="%d")

print im.height
print im.width
print out.shape, len(out.shape)

def array2img(out):
	out1 = np.array(out, np.unit8)
	img = Image.fromarray(out1,'L')
	for x in range(img.size[0]):
		for y in range(img.size[1]):
			if not img.getpixel((x, y)) == 0:
				print 'PLz', str(img.getpixel((x, y)))

	img.show()


def show_pred_img(file_name):
	file = open(file_name, 'r')
	lines = file.read().split('\n')

	#img_name = str(sys.argv[1])
	im = Image.open(img_name)
	im_pixel = im.load()

	img = Image.new('RGB', im.size, "black")
	pixels = img.load()

	w, h = 0, 0
	for l in lines:
		w = 0
		if len(l) > 0:
			word = l.split(' ')
			for x in word:
				if int(x) == 1:
					pixels[w, h] = im_pixel[w, h]
				w += 1
			h += 1
	print im.size
	#img.show()
	img.save(img_name+'_result.png')
show_pred_img('vote')
