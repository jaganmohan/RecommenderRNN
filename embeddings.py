from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

class Embeddings(object):

    def __init__(self, emb_size, num_classes, num_samples, batch_size,
      learning_rate=0.1, epoch=20):
	self._emb_size = emb_size
	self._lr = learning_rate
	self._epoch = epoch
	self._batch_size = batch_size
	self._num_classes = num_classes+1
	self._emb = tf.Variable(tf.random_normal([self._num_classes, emb_size]), name="embeddings")

	w_out = tf.Variable(tf.random_normal([self._num_classes, emb_size]))
	biases = tf.Variable(tf.zeros([self._num_classes]))

	self._inputs = tf.placeholder(dtype=tf.int64, shape=[batch_size])
	self._targets = tf.placeholder(dtype=tf.int64, shape=[batch_size, 1])

	# shape(hidden_states): [batch_size x emb_size]
	hidden_states = tf.nn.embedding_lookup(self._emb, self._inputs)
	# Calculate loss 
	self._loss = tf.nn.sampled_softmax_loss(weights=w_out, biases=biases, labels=self._targets,
	    inputs=hidden_states, num_sampled=num_samples, num_classes=self._num_classes)
	# Define optimizer
	optimizer = tf.train.GradientDescentOptimizer(self._lr)
	# Train by minimizeing the calculated loss
	train_op = optimizer.minimize(self._loss)
	self._train = train_op

    @property
    def inputs(self):
	return self._inputs

    @property
    def targets(self):
	return self._targets

    @property
    def embeddings(self):
	return self._emb.eval()

    @property
    def num_classes(self):
	return self._num_classes

    @property
    def lr(self):
	return self._lr

    @property
    def batch_size(self):
	return self._batch_size

    @property
    def epoch(self):
	return self._epoch

    @property
    def cost(self):
	return self._loss

    @property
    def train(self):
	return self._train

    def run_batch(self, sess, inputs):
	if len(inputs) < self.batch_size:
	    inputs = list(inputs)
	    inputs.extend(np.zeros(self.batch_size-len(inputs)))
	fetches = [self.cost, self.train]
	feed_dict = {}
	feed_dict[self.inputs] = inputs
	feed_dict[self.targets] = np.reshape(inputs,[-1, 1])
	errors, _ = sess.run(fetches, feed_dict)
	return errors

    def run_epoch(self, sess, permutation):
	total_cost = 0
	for j in range(0, len(permutation), self.batch_size):
	    total_cost += self.run_batch(sess, permutation[j:(j+self.batch_size)])
	return total_cost
	
    def create_embeddings(self, sess):
	classes = range(self._num_classes)
	for i in range(1,self._epoch+1):
            permutation = np.random.permutation(classes)
            cost = self.run_epoch(sess, permutation)
	    cost = np.nansum(cost)
            print("[Epoch {}] Total cost : {}".format(i, cost))
	print("Completed training")
