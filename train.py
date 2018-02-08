import tensorflow as tf
import pandas as pd
import numpy as np
from dan import DAN
import time
import numpy as np
from remtime import *
import random
import glob
import Image
import warnings

warnings.filterwarnings("ignore")


LEARNING_RATE = 0.001
BATCH_SIZE = 25
N_EPOCHS = 1
PER=5.0
NUM_VID = 100
NUM_IMAGES = NUM_VID*80
NUM_TEST_IMAGES = NUM_VID*20



imgs = tf.placeholder('float', [None, 224, 224, 3])
val = tf.placeholder('float', [None, 5])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
with tf.Session(config=config) as sess:
	
	model = DAN(imgs)

	cost = tf.reduce_mean(tf.squared_difference(model.output, val))
	optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
	
	images, labels = dataBatch('train500.tfrecords', BATCH_SIZE=BATCH_SIZE, N_EPOCHS=N_EPOCHS)
	images2, labels2 = dataBatch('test500.tfrecords', BATCH_SIZE=BATCH_SIZE, N_EPOCHS=1)
	
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess.run(init_op)
	
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	
	model.load_weights('vgg16_weights.npz', sess)
	
	for epoch in range(N_EPOCHS):
		sess.run(tf.local_variables_initializer())
		mean_acc = 0
		i=0
		stime = time.time()
		while i<NUM_IMAGES:
			epoch_x, epoch_y = sess.run([images, labels])
			_, c, output = sess.run([optimizer, cost, model.output], feed_dict = {imgs: epoch_x, val: epoch_y})
			mean_acc += np.mean(1-np.absolute(output-epoch_y))
			i+=BATCH_SIZE
			x=100/PER
			if i%(NUM_IMAGES/x)==0:
				per = float(i)/NUM_IMAGES*100
				print("Epoch:"+str(round(per,2))+"% Of "+str(epoch+1)+"/"+str(N_EPOCHS)+", Batch loss:"+str(round(c,4)))
				ftime = time.time()
				remtime = (ftime-stime)*((NUM_IMAGES-i)/(NUM_IMAGES/x)) + (ftime-stime)*x*(N_EPOCHS-epoch-1)
				stime=ftime
				printTime(remtime)
		mean_acc = mean_acc*BATCH_SIZE/NUM_IMAGES
		print("Epoch"+ str(epoch+1)+" completed out of "+str(N_EPOCHS)+", Mean Acc:"+str(round(mean_acc,4)))



	sess.run(tf.local_variables_initializer())
	mean_acc = 0
	i=0
	stime = time.time()
	while i<NUM_TEST_IMAGES:
		epoch_x, epoch_y = sess.run([images2, labels2])
		output = sess.run([model.output], feed_dict = {imgs: epoch_x, val: epoch_y})
		mean_acc += np.mean(1-np.absolute(output-epoch_y))
		i+=BATCH_SIZE
		x=100/PER
		if i%(NUM_TEST_IMAGES/x)==0:
			ftime = time.time()
			remtime = (ftime-stime)*((NUM_TEST_IMAGES-i)/(NUM_TEST_IMAGES/x))
			stime=ftime
			printTime(remtime)
	mean_acc = mean_acc*BATCH_SIZE/NUM_TEST_IMAGES
	print("Validation Mean Acc:"+str(round(mean_acc,4)))
	coord.request_stop()
	# Wait for threads to stop
	coord.join(threads)


	