import tensorflow as tf

def printTime(remtime):
	hrs = int(remtime)/3600
	mins = int((remtime/60-hrs*60))
	secs = int(remtime-mins*60-hrs*3600)
	print(str(hrs)+"Hrs "+str(mins)+"Mins "+str(secs)+"Secs remaining..")

def dataBatch(data_path, BATCH_SIZE, N_EPOCHS=1):
	reader = tf.TFRecordReader()
	filename_queue = tf.train.string_input_producer([data_path], num_epochs=N_EPOCHS)
	_, serialized_example = reader.read(filename_queue)
	# Decode the record read by the reader
	feature = {'train/image': tf.FixedLenFeature([], tf.string), 'train/label': tf.FixedLenFeature([], tf.string)}
	features = tf.parse_single_example(serialized_example, features=feature)
	# Convert the image data from string back to the numbers
	image = tf.decode_raw(features['train/image'], tf.float32)
	label = tf.decode_raw(features['train/label'], tf.float32)
	# Reshape image data into the original shape
	image = tf.reshape(image, [224, 224, 3])
	label = tf.reshape(label, [5])

	images, labels = tf.train.shuffle_batch([image, label], batch_size=BATCH_SIZE, capacity=100, min_after_dequeue=BATCH_SIZE, allow_smaller_final_batch=True)
	return images, labels



# VID_BATCH = 100


# df = pd.read_csv('small_train_sample.csv')
# NUM_VID = len(df)
# images=np.empty((NUM_VID*100, 224, 224, 3))
# traits=np.zeros((NUM_VID*100, 5))

# for i in range(NUM_VID):
# 	filelist=glob.glob('ImageData/trainingData/'+(df['VideoName'].iloc[i]).split('.mp4')[0]+'/*.jpg')
# 	print(str(i)+" "+ str(len(filelist))+" "+df['VideoName'].iloc[i])
# 	images[i*100:(i+1)*100] = np.array([np.array(Image.open(fname).resize((224,224), Image.ANTIALIAS)) for fname in filelist])
# 	traits[i*100:(i+1)*100]=np.array(df.drop(['VideoName'], 1, inplace=False).iloc[i])	


# ind = [i for i in range(NUM_IMAGES)]

# print(images.shape)
# print(traits.shape)