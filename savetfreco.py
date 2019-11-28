from random import shuffle
import glob
import pandas as pd
import tensorflow as tf
import sys
import numpy as np
from PIL import Image 
import cv2



def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = np.array(Image.open(addr).resize((224,224), Image.ANTIALIAS))
    # img = cv2.imread(addr)
    # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)
    return img

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


df = pd.read_csv('training_gt.csv')
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
train_addrs, train_labels = zip(*c)
# train_addrs = train_addrs[0:10000]
# train_labels = train_labels[0:10000]
train_filename = 'train_full.tfrecords'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print ('Train data: {}/{}'.format(i, len(train_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(train_addrs[i])
    # print img[0,0:5]
    # break
    # print img.shape
    label = train_labels[i]
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

print (len(train_addrs), "training images saved.. ")


df = pd.read_csv('validation_gt.csv')
NUM_VID = len(df)
addrs = []
labels = []
for i in range(NUM_VID):
	filelist=glob.glob('ImageData/validationData/'+(df['VideoName'].iloc[i]).split('.mp4')[0]+'/*.jpg')
	addrs+=filelist
	labels+=[np.array(df.drop(['VideoName'], 1, inplace=False).iloc[i]).astype(np.float32)]*100

c = list(zip(addrs, labels))
shuffle(c)
val_addrs, val_labels = zip(*c)

val_filename = 'val_full.tfrecords'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(val_filename)

for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print ('Val data: {}/{}'.format(i, len(val_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(val_addrs[i])
    # print img.shape
    label = val_labels[i].astype(np.float32)

    feature = {'val/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
               'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())


writer.close()
sys.stdout.flush()

print (len(train_addrs), "training images saved.. ")
print (len(val_addrs), "validation images saved.. ")


# test_filename = 'val_full.tfrecords'  # address to save the TFRecords file
# # open the TFRecords file
# writer = tf.python_io.TFRecordWriter(test_filename)

# for i in range(len(val_addrs)):
#     # print how many images are saved every 1000 images
#     if not i % 1000:
#         print ('Val data: {}/{}'.format(i, len(val_addrs)))
#         sys.stdout.flush()
#     # Load the image
#     img = load_image(val_addrs[i])
#     # print (img.shape)
#     label = val_labels[i].astype(np.float32)
#     # print (label.shape)
#     # print (label)
#     # Create a feature
#     feature = {'val/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
#                'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
#     # Create an example protocol buffer
#     example = tf.train.Example(features=tf.train.Features(feature=feature))
    
#     # Serialize to string and write on the file
#     writer.write(example.SerializeToString())
