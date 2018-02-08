from random import shuffle
import glob
import pandas as pd
import tensorflow as tf
import sys
import numpy as np
import Image
import cv2



df = pd.read_csv('small_train_sample.csv')
NUM_VID = len(df)
addrs = []
labels = []
for i in range(NUM_VID):
	filelist=glob.glob('ImageData/trainingData/'+(df['VideoName'].iloc[i]).split('.mp4')[0]+'/*.jpg')
	addrs+=filelist
	labels+=[np.array(df.drop(['VideoName'], 1, inplace=False).iloc[i]).astype(np.float32)]*100
# print labels[101]
# print len(labels[0]), len(addrs)

c = list(zip(addrs, labels))
shuffle(c)
addrs, labels = zip(*c)

# Divide the hata into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.8*len(addrs))]
train_labels = labels[0:int(0.8*len(labels))]

val_addrs = addrs[int(0.8*len(addrs)):]
val_labels = labels[int(0.8*len(addrs)):]

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# train_filename = 'train500.tfrecords'  # address to save the TFRecords file
# # open the TFRecords file
# writer = tf.python_io.TFRecordWriter(train_filename)
# for i in range(len(train_addrs)):
#     # print how many images are saved every 1000 images
#     if not i % 1000:
#         print 'Train data: {}/{}'.format(i, len(train_addrs))
#         sys.stdout.flush()
#     # Load the image
#     img = load_image(train_addrs[i])
#     # print img.shape
#     label = train_labels[i].astype(np.float32)
#     # print label.shape
#     # print label
#     # Create a feature
#     feature = {'train/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
#                'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
#     # Create an example protocol buffer
#     example = tf.train.Example(features=tf.train.Features(feature=feature))
    
#     # Serialize to string and write on the file
#     writer.write(example.SerializeToString())



test_filename = 'test500.tfrecords'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)

for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print 'Val data: {}/{}'.format(i, len(val_addrs))
        sys.stdout.flush()
    # Load the image
    img = load_image(val_addrs[i])
    # print img.shape
    label = val_labels[i].astype(np.float32)
    # print label.shape
    # print label
    # Create a feature
    feature = {'train/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()