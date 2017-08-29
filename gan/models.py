# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
import utils

import tensorflow.contrib.slim as slim


def lrelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


class BaseModel(object):
    """Contains the base class for models."""
    """Inherit from this class when implementing new models."""

    def create_model(self, **unused_params):
        """Define variables of the model."""
        raise NotImplementedError()

    def run_model(self, unused_model_input, **unused_params):
        """Run model with given input."""
        raise NotImplementedError()

    def get_variables(self):
        """Return all variables used by the model for training."""
        raise NotImplementedError()


class SampleGenerator(BaseModel):
    def __init__(self):
        self.noise_input_size = 100

    def create_model(self, output_size, **unused_params):
        self.reuse = False

    def run_model(self, model_input, is_training=True, **unused_params):
        with tf.variable_scope('generator', reuse=self.reuse):
            net = tf.contrib.layers.fully_connected(model_input,
                                                    4*4*1024,
                                                    activation_fn=tf.nn.sigmoid,
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(1e-8))
            net = tf.reshape(net, [-1, 4, 4, 1024])
            net = tf.layers.batch_normalization(net, training=True)

            # consider using tf.contrib.layers.conv2d_transpose on both side
            # (generator and discriminator), because of initializer and weight
            # regularizer
            net = tf.layers.conv2d_transpose(net, 512, [5, 5], (2, 2), 'SAME',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-8))
            net = tf.layers.batch_normalization(net, training=True)
            net = tf.nn.relu(net)

            net = tf.layers.conv2d_transpose(net, 256, [5, 5], (2, 2), 'SAME',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-8))
            net = tf.layers.batch_normalization(net, training=True)
            net = tf.nn.relu(net)

            net = tf.layers.conv2d_transpose(net, 128, [5, 5], (2, 2), 'SAME',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-8))
            net = tf.layers.batch_normalization(net, training=True)
            net = tf.nn.relu(net)

            net = tf.layers.conv2d_transpose(net, 1, [5, 5], (2, 2), 'SAME',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-8))

            net = tf.image.resize_images(net, [50, 50])
            net = tf.contrib.layers.flatten(net)
            output = tf.nn.tanh(net)
            self.reuse = True
            return {"output": output}

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='tower/generator')


class SampleDiscriminator(BaseModel):
    def create_model(self, input_size, **unused_params):
        self.reuse = False

    def run_model(self, model_input, is_training=True, **unused_params):
        with tf.variable_scope('discriminator', reuse=self.reuse):
            net = tf.reshape(model_input, [-1, 50, 50, 1])
            # net = tf.image.resize_images(net, [64, 64])

            net = tf.layers.conv2d(net, 64, [5, 5], (2, 2), padding='SAME',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-8))
            net = tf.layers.batch_normalization(net, training=True)
            net = lrelu(net)

            net = tf.layers.conv2d(net, 128, [5, 5], (2, 2), padding='SAME',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-8))
            net = tf.layers.batch_normalization(net, training=True)
            net = lrelu(net)

            net = tf.layers.conv2d(net, 256, [5, 5], (2, 2), padding='SAME',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-8))
            net = tf.layers.batch_normalization(net, training=True)
            net = lrelu(net)

            net = tf.layers.conv2d(net, 512, [5, 5], (2, 2), padding='SAME',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-8))
            net = tf.layers.batch_normalization(net, training=True)
            net = lrelu(net)

            net = tf.contrib.layers.flatten(net)
            logits = tf.contrib.layers.fully_connected(net,
                                                       1,
                                                       activation_fn=None,
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(1e-8))
            predictions = lrelu(logits)
            self.reuse = True
            return {"logits": logits, "predictions": predictions}

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='tower/discriminator')
