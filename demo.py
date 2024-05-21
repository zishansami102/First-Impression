import glob
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

import cv2
from dan import DAN

warnings.filterwarnings("ignore")


def predict(file_name):

    num = []

    cap = cv2.VideoCapture(file_name)

    file_name = (file_name.split(".mp4"))[0]
    ## Creating folder to save all the 100 frames from the video
    try:
        os.makedirs("ImageData/testingData/" + file_name)
    except OSError:
        print("Error: Creating directory of data")

    ## Setting the frame limit to 100
    cap.set(cv2.CAP_PROP_FRAME_COUNT, 101)
    length = 101
    count = 0
    ## Running a loop to each frame and saving it in the created folder
    while cap.isOpened():
        count += 1
        if length == count:
            break
        _, frame = cap.read()
        if frame is None:
            continue

        ## Resizing it to 256*256 to save the disk space and fit into the model
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
        # Saves image of the current frame in jpg file
        name = (
            "ImageData/testingData/" + str(file_name) + "/frame" + str(count) + ".jpg"
        )
        cv2.imwrite(name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    addrs = []

    def load_image(addr):
        img = np.array(Image.open(addr).resize((224, 224), Image.ANTIALIAS))
        img = img.astype(np.uint8)
        return img

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    addrs = []

    filelist = glob.glob("ImageData/testingData/" + str(file_name) + "/*.jpg")
    addrs += filelist

    train_addrs = addrs
    train_filename = "test.tfrecords"  # address to save the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)
    for i in range(len(train_addrs)):
        # Load the image
        img = load_image(train_addrs[i])
        feature = {"test/image": _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

    BATCH_SIZE = 20
    REG_PENALTY = 0
    NUM_IMAGES = 100
    N_EPOCHS = 1

    imgs = tf.placeholder("float", [None, 224, 224, 3], name="image_placeholder")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.Session(config=config) as sess:

        model = DAN(imgs, REG_PENALTY=REG_PENALTY, preprocess="vggface")
        tr_reader = tf.TFRecordReader()
        tr_filename_queue = tf.train.string_input_producer(
            ["test.tfrecords"], num_epochs=N_EPOCHS
        )
        _, tr_serialized_example = tr_reader.read(tr_filename_queue)
        tr_feature = {"test/image": tf.FixedLenFeature([], tf.string)}
        tr_features = tf.parse_single_example(
            tr_serialized_example, features=tr_feature
        )

        tr_image = tf.decode_raw(tr_features["test/image"], tf.uint8)
        tr_image = tf.reshape(tr_image, [224, 224, 3])
        tr_images = tf.train.shuffle_batch(
            [tr_image],
            batch_size=BATCH_SIZE,
            capacity=100,
            min_after_dequeue=BATCH_SIZE,
            allow_smaller_final_batch=True,
        )
        init_op = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer()
        )
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        file_list = ["param1.pkl", "param2.pkl"]
        epoch = 0
        for pickle_file in file_list:
            error = 0
            model.load_trained_model(pickle_file, sess)
            i = 0
            while i < NUM_IMAGES:
                i += BATCH_SIZE
                try:
                    epoch_x = sess.run(tr_images)
                except:
                    if error >= 5:
                        break
                    error += 1
                    continue
                output = sess.run(
                    [model.output], feed_dict={imgs: epoch_x.astype(np.float32)}
                )
                num.append(output[0])
            epoch += 1
        coord.request_stop()
        # Wait for threads to stop
        coord.join(threads)
    a = np.round(np.mean(np.concatenate(num), axis=0), 3)
    a_json = {
        "Extraversion": a[0],
        "Neuroticism": a[1],
        "Agreeableness": a[2],
        "Conscientiousness": a[3],
        "Openness": a[4],
    }
    return a_json


output = predict("_uNup91ZYw0.002.mp4")
print(output)
