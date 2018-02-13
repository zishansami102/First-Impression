import tensorflow as tf
import numpy as np
from dan import DAN
import time
from remtime import *
import warnings
import pickle

warnings.filterwarnings("ignore")


LEARNING_RATE = 0.0005
BATCH_SIZE = 25
N_EPOCHS = 2
REG_PENALTY = 0.5
PER=5.0
NUM_VID = 500
NUM_IMAGES = NUM_VID*80
NUM_TEST_IMAGES = NUM_VID*20



imgs = tf.placeholder('float', [None, 224, 224, 3])
val = tf.placeholder('float', [None, 5])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
with tf.Session(config=config) as sess:
	
	model = DAN(imgs, REG_PENALTY=REG_PENALTY, pretrained='vggface')

	cost = tf.reduce_mean(tf.squared_difference(model.output, val))+ model.cost_reg
	optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
	

	
	tr_reader = tf.TFRecordReader()
	tr_filename_queue = tf.train.string_input_producer(['train500_new.tfrecords'], num_epochs=2*N_EPOCHS)
	_, tr_serialized_example = tr_reader.read(tr_filename_queue)
	# Decode the record read by the reader
	tr_feature = {'train/image': tf.FixedLenFeature([], tf.string), 'train/label': tf.FixedLenFeature([], tf.string)}
	tr_features = tf.parse_single_example(tr_serialized_example, features=tr_feature)
	# Convert the image data from string back to the numbers
	tr_image = tf.decode_raw(tr_features['train/image'], tf.float32)
	tr_label = tf.decode_raw(tr_features['train/label'], tf.float32)
	# Reshape image data into the original shape
	tr_image = tf.reshape(tr_image, [224, 224, 3])
	tr_label = tf.reshape(tr_label, [5])
	tr_images, tr_labels = tf.train.shuffle_batch([tr_image, tr_label], batch_size=BATCH_SIZE, capacity=100, min_after_dequeue=BATCH_SIZE, allow_smaller_final_batch=True)

	

	val_reader = tf.TFRecordReader()
	val_filename_queue = tf.train.string_input_producer(['val500_new.tfrecords'], num_epochs=N_EPOCHS)
	_, val_serialized_example = val_reader.read(val_filename_queue)
	# Decode the record read by the reader
	val_feature = {'val/image': tf.FixedLenFeature([], tf.string), 'val/label': tf.FixedLenFeature([], tf.string)}
	val_features = tf.parse_single_example(val_serialized_example, features=val_feature)
	# Convert the image data from string back to the numbers
	val_image = tf.decode_raw(val_features['val/image'], tf.float32)
	val_label = tf.decode_raw(val_features['val/label'], tf.float32)
	# Reshape image data into the original shape
	val_image = tf.reshape(val_image, [224, 224, 3])
	val_label = tf.reshape(val_label, [5])
	val_images, val_labels = tf.train.shuffle_batch([val_image, val_label], batch_size=BATCH_SIZE, capacity=100, min_after_dequeue=BATCH_SIZE, allow_smaller_final_batch=True)


	
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess.run(init_op)
	
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	
	model.initialize_with_vggface('vgg-face.mat', sess)
	loss_list=[]
	for epoch in range(N_EPOCHS):
		tr_acc_list = []
		val_acc_list=[]
		sess.run(tf.local_variables_initializer())
		i=0
		stime = time.time()
		while i<NUM_IMAGES:
			epoch_x, epoch_y = sess.run([tr_images, tr_labels])
			_, c = sess.run([optimizer, cost], feed_dict = {imgs: epoch_x, val: epoch_y})
			loss_list.append(np.power(c,0.5))
			i+=BATCH_SIZE
			x=100/PER
			if i%(NUM_IMAGES/x)==0:
				per = float(i)/NUM_IMAGES*100
				print("Epoch:"+str(round(per,2))+"% Of "+str(epoch+1)+"/"+str(N_EPOCHS)+", Batch loss:"+str(round(c,4)))
				ftime = time.time()
				remtime = (ftime-stime)*((NUM_IMAGES-i)/(NUM_IMAGES/x)) + (ftime-stime)*x*(N_EPOCHS-epoch-1)
				stime=ftime
				printTime(remtime)
		sess.run(tf.local_variables_initializer())
		print("Computing Training Accuracy..")
		i=0
		while i<NUM_IMAGES:
			epoch_x, epoch_y = sess.run([tr_images, tr_labels])
			output = sess.run([model.output], feed_dict = {imgs: epoch_x})
			tr_mean_acc = np.mean(1-np.absolute(output-epoch_y))
			tr_acc_list.append(tr_mean_acc)
			i+=BATCH_SIZE
		tr_mean_acc = np.mean(tr_acc_list)

		print("Computing Validation Accuracy..")
		i=0
		while i<NUM_TEST_IMAGES:
			epoch_x, epoch_y = sess.run([val_images, val_labels])
			output = sess.run([model.output], feed_dict = {imgs: epoch_x})
			val_mean_acc = np.mean(1-np.absolute(output-epoch_y))
			val_acc_list.append(val_mean_acc)
			i+=BATCH_SIZE
		val_mean_acc = np.mean(val_acc_list)
		print("Epoch"+ str(epoch+1)+" completed out of "+str(N_EPOCHS))
		print("Tr. Mean Acc:"+str(round(tr_mean_acc,4)))
		print("Val. Mean Acc:"+str(round(val_mean_acc,4)))


	coord.request_stop()
	# Wait for threads to stop
	coord.join(threads)

	# saver = tf.train.Saver()
	# saver.save(sess, 'model500', [model])

	with open('loss.pkl', 'wb') as pfile:
	    pickle.dump(loss_list, pfile, pickle.HIGHEST_PROTOCOL)
	print "Loss List Saved!!"



	