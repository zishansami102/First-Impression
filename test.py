import pickle
import time
import warnings

import numpy as np
import tensorflow as tf

from dan import DAN
from remtime import *

warnings.filterwarnings("ignore")


BATCH_SIZE = 50
REG_PENALTY = 0
NUM_IMAGES = 599900
NUM_TEST_IMAGES = 199900
N_EPOCHS = 1


imgs = tf.placeholder("float", [None, 224, 224, 3], name="image_placeholder")
values = tf.placeholder("float", [None, 5], name="value_placeholder")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
with tf.Session(config=config) as sess:

    model = DAN(imgs, REG_PENALTY=REG_PENALTY, preprocess="vggface")
    # output = model.output

    tr_reader = tf.TFRecordReader()
    tr_filename_queue = tf.train.string_input_producer(
        ["train_full.tfrecords"], num_epochs=N_EPOCHS
    )
    _, tr_serialized_example = tr_reader.read(tr_filename_queue)
    # Decode the record read by the reader
    tr_feature = {
        "train/image": tf.FixedLenFeature([], tf.string),
        "train/label": tf.FixedLenFeature([], tf.string),
    }
    tr_features = tf.parse_single_example(tr_serialized_example, features=tr_feature)
    # Convert the image data from string back to the numbers
    tr_image = tf.decode_raw(tr_features["train/image"], tf.uint8)
    tr_label = tf.decode_raw(tr_features["train/label"], tf.float32)
    # Reshape image data into the original shape
    tr_image = tf.reshape(tr_image, [224, 224, 3])
    tr_label = tf.reshape(tr_label, [5])
    tr_images, tr_labels = tf.train.shuffle_batch(
        [tr_image, tr_label],
        batch_size=BATCH_SIZE,
        capacity=100,
        min_after_dequeue=BATCH_SIZE,
        allow_smaller_final_batch=True,
    )

    val_reader = tf.TFRecordReader()
    val_filename_queue = tf.train.string_input_producer(
        ["val_full.tfrecords"], num_epochs=N_EPOCHS
    )
    _, val_serialized_example = val_reader.read(val_filename_queue)
    # Decode the record read by the reader
    val_feature = {
        "val/image": tf.FixedLenFeature([], tf.string),
        "val/label": tf.FixedLenFeature([], tf.string),
    }
    val_features = tf.parse_single_example(val_serialized_example, features=val_feature)
    # Convert the image data from string back to the numbers
    val_image = tf.decode_raw(val_features["val/image"], tf.uint8)
    val_label = tf.decode_raw(val_features["val/label"], tf.float32)
    # Reshape image data into the original shape
    val_image = tf.reshape(val_image, [224, 224, 3])
    val_label = tf.reshape(val_label, [5])
    val_images, val_labels = tf.train.shuffle_batch(
        [val_image, val_label],
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

    file_list = [
        "param" + str((60 / N_EPOCHS) * (x + 1)) + ".pkl" for x in range(0, N_EPOCHS)
    ]
    file_list = ["param25.pkl"]
    training_accuracy = []
    validation_accuracy = []
    epoch = 0
    stime = time.time()
    print("Testing Started")
    for pickle_file in file_list:
        error = 0
        model.load_trained_model(pickle_file, sess)
        tr_acc_list = []
        val_acc_list = []

        i = 0
        while i < NUM_IMAGES:
            i += BATCH_SIZE
            try:
                epoch_x, epoch_y = sess.run([tr_images, tr_labels])
            except:
                print("Error in reading this batch")
                if error >= 5:
                    break
                error += 1
                continue
            output = sess.run(
                [model.output], feed_dict={imgs: epoch_x.astype(np.float32)}
            )
            tr_mean_acc = np.mean(1 - np.absolute(output - epoch_y))
            tr_acc_list.append(tr_mean_acc)
            if not i % 20000:
                print(i, "images completed in training")

        tr_mean_acc = np.mean(tr_acc_list)
        training_accuracy.append(tr_mean_acc)

        i = 0
        while i < NUM_TEST_IMAGES:
            i += BATCH_SIZE
            try:
                epoch_x, epoch_y = sess.run([val_images, val_labels])
            except:
                print("Error in reading this batch")
                if error >= 5:
                    break
                error += 1
                continue
            output = sess.run(
                [model.output], feed_dict={imgs: epoch_x.astype(np.float32)}
            )
            val_mean_acc = np.mean(1 - np.absolute(output - epoch_y))
            val_acc_list.append(val_mean_acc)
            if not i % 20000:
                print(i, "images completed in validation")
        sess.run(tf.local_variables_initializer())

        val_mean_acc = np.mean(val_acc_list)
        validation_accuracy.append(val_mean_acc)

        print("Epoch" + str(epoch + 1) + " completed out of " + str(N_EPOCHS))
        print(
            "Tr. Mean Acc:"
            + str(round(tr_mean_acc, 4))
            + ", Val. Mean Acc:"
            + str(round(val_mean_acc, 4))
        )

        ftime = time.time()
        remtime = (ftime - stime) * (N_EPOCHS - epoch - 1)
        stime = ftime
        printTime(remtime)
        epoch += 1
        if not epoch % (N_EPOCHS / 2):
            with open("acc_plot25.pkl", "wb") as nfile:
                pickle.dump([training_accuracy, validation_accuracy], nfile)
            print("Half testing saved")

    coord.request_stop()
    # Wait for threads to stop
    coord.join(threads)

    print("Testing done... Values saved successfully")

