import pickle
import warnings

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

warnings.filterwarnings("ignore")


class DAN_PLUS:
    def __init__(self, imgs, REG_PENALTY=0, preprocess=None):
        self.imgs = imgs
        if preprocess == "vggface":
            self.mean = [129.1862793, 104.76238251, 93.59396362]
        self.convlayers()
        self.dan_part()
        self.output = tf.nn.sigmoid(self.reg_head, name="output")
        self.cost_reg = REG_PENALTY * tf.reduce_mean(tf.square(self.parameters[-2])) / 2

    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope("preprocess") as scope:
            mean = tf.constant(
                self.mean, dtype=tf.float32, shape=[1, 1, 1, 3], name="img_mean"
            )
            images = self.imgs - mean

        # conv1_1
        with tf.name_scope("conv1_1") as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.Variable(
                tf.constant(0.0, shape=[64], dtype=tf.float32),
                trainable=True,
                name="biases",
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope("conv1_2") as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.Variable(
                tf.constant(0.0, shape=[64], dtype=tf.float32),
                trainable=True,
                name="biases",
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(
            self.conv1_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool1",
        )

        # conv2_1
        with tf.name_scope("conv2_1") as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.Variable(
                tf.constant(0.0, shape=[128], dtype=tf.float32),
                trainable=True,
                name="biases",
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope("conv2_2") as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.Variable(
                tf.constant(0.0, shape=[128], dtype=tf.float32),
                trainable=True,
                name="biases",
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(
            self.conv2_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool2",
        )

        # conv3_1
        with tf.name_scope("conv3_1") as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.Variable(
                tf.constant(0.0, shape=[256], dtype=tf.float32),
                trainable=True,
                name="biases",
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope("conv3_2") as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.Variable(
                tf.constant(0.0, shape=[256], dtype=tf.float32),
                trainable=True,
                name="biases",
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope("conv3_3") as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.Variable(
                tf.constant(0.0, shape=[256], dtype=tf.float32),
                trainable=True,
                name="biases",
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(
            self.conv3_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool3",
        )

        # conv4_1
        with tf.name_scope("conv4_1") as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name="biases",
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope("conv4_2") as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name="biases",
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope("conv4_3") as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name="biases",
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(
            self.conv4_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool4",
        )

        # conv5_1
        with tf.name_scope("conv5_1") as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name="biases",
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope("conv5_2") as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name="biases",
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # MaxPool5_2
        self.maxpool5_2 = tf.nn.max_pool(
            self.conv5_2,
            ksize=[1, 14, 14, 1],
            strides=[1, 1, 1, 1],
            padding="SAME",
            name="maxpool5_2",
        )

        # AvgPool5_2
        self.avgpool5_2 = tf.nn.avg_pool(
            self.conv5_2,
            ksize=[1, 14, 14, 1],
            strides=[1, 1, 1, 1],
            padding="SAME",
            name="avgpool5_2",
        )
        # conv5_3
        with tf.name_scope("conv5_3") as scope:
            kernel = tf.Variable(
                tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                name="weights",
            )
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name="biases",
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(
            self.conv5_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool5",
        )
        # MaxPool5_3
        self.maxpool5_3 = tf.nn.max_pool(
            self.pool5,
            ksize=[1, 7, 7, 1],
            strides=[1, 1, 1, 1],
            padding="SAME",
            name="maxpool5_3",
        )

        # AvgPool5_3
        self.avgpool5_3 = tf.nn.avg_pool(
            self.pool5,
            ksize=[1, 7, 7, 1],
            strides=[1, 1, 1, 1],
            padding="SAME",
            name="avgpool5_3",
        )



    def dan_part(self):

        # fc1
        with tf.name_scope("reg_head") as scope:
            shape = 2 * int(np.prod(self.maxpool5_3.get_shape()[1:]))
            
            fc1w = tf.Variable(
                tf.truncated_normal([shape, 5], dtype=tf.float32, stddev=1e-1),
                name="weights",
            )
            fc1b = tf.Variable(
                tf.constant(1.0, shape=[5], dtype=tf.float32),
                trainable=True,
                name="biases",
            )
            maxpool5_2_flat = tf.nn.l2_normalize(
                tf.reshape(self.maxpool5_2, [-1, int(shape / 2)]), 1
            )
            avgpool5_2_flat = tf.nn.l2_normalize(
                tf.reshape(self.avgpool5_2, [-1, int(shape / 2)]), 1
            )

            maxpool5_3_flat = tf.nn.l2_normalize(
                tf.reshape(self.maxpool5_3, [-1, int(shape / 2)]), 1
            )
            avgpool5_3_flat = tf.nn.l2_normalize(
                tf.reshape(self.avgpool5_3, [-1, int(shape / 2)]), 1
            )

            self.concat = tf.concat([maxpool5_3_flat, avgpool5_3_flat,maxpool5_2_flat,avgpool5_2_flat], 1)
            self.reg_head = tf.nn.bias_add(
                tf.matmul(self.concat, fc1w), fc1b, name=scope
            )
            self.parameters += [fc1w, fc1b]

    def initialize_with_vggface(self, weight_file, sess):
        data = loadmat(weight_file)
        layers = data["layers"][0]
        i = 0
        for layer in layers:
            name = layer[0]["name"][0][0]
            layer_type = layer[0]["type"][0][0]
            if layer_type == "conv" and name[0:2] != "fc":
                kernel, bias = layer[0]["weights"][0][0]
                sess.run(self.parameters[i].assign(kernel))
                sess.run(self.parameters[i + 1].assign(bias.reshape(bias.shape[0])))
                i += 2

    def load_trained_model(self, pickle_file, sess):
        with open(pickle_file, "rb") as pfile:
            param = pickle.load(pfile)
        for i in range(len(param)):
            sess.run(self.parameters[i].assign(param[i]))
