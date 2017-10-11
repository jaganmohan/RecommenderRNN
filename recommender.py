from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf

from cell import RecommenderCell
from reader import Dataset
from embeddings import Embeddings

class RecommenderRNN(object):
    
    def __init__(self, num_users, num_items, max_seq_len, is_training, u_emb, i_emb,
		hidden_size=20, batch_size=1, chunk_size=10, learning_rate=0.1, inp_D=20):
        
	self._batch_size = batch_size
        self._chunk_size = chunk_size
	self._hidden_unit_size = hidden_size
	# sort inputs for user_lstm and item_lstm
        # Placeholders for input data
	self._inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2, chunk_size, 3], name="inputs")
	self._targets = tf.placeholder(dtype=tf.float32, shape=[batch_size, chunk_size], name="targets")
        self._seq_lengths = tf.placeholder(tf.int32, shape=[batch_size], name="seq_lengths")
	self._u_emb = u_emb 
	self._i_emb = i_emb 

	# getting rating sequence
        if is_training:
          u_ratings = self._inputs[:,0,:,1]
          i_ratings = self._inputs[:,1,:,1]
        else:
          u_ratings = tf.zeros((batch_size, chunk_size, 1))
          i_ratings = tf.zeros((batch_size, chunk_size, 1))
        # concatenating input embedding, time component and rating
	u_emb_seq = tf.concat(
		[tf.nn.embedding_lookup(self._i_emb, tf.to_int32(self._inputs[:,0,:,0])),  # embedding
                tf.reshape(self._inputs[:,0,:,2], [batch_size,chunk_size,1]),  # time-component
		tf.reshape(u_ratings, [batch_size,chunk_size,1])], 2)  # rating
	i_emb_seq = tf.concat(
		[tf.nn.embedding_lookup(self._u_emb, tf.to_int32(self._inputs[:,1,:,0])), # embedding
                tf.reshape(self._inputs[:,1,:,2], [batch_size,chunk_size,1]),  #time-component
		tf.reshape(i_ratings, [batch_size,chunk_size,1])], 2)  #rating

	inputs = tf.concat([u_emb_seq, i_emb_seq], 2)
        
        # Recommender LSTM cell
        cell = RecommenderCell(hidden_size, batch_size, inp_D, is_training)
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        
	outputs, self._final_state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.seq_lengths, 
                               initial_state=self._initial_state)
	#shape(hid_state): [batch_size x seq_len x hidden_size]

	_out_u, _out_i = tf.split(value=outputs, num_or_size_splits=2, axis=2)
	_out_u = tf.reshape(_out_u, [batch_size, chunk_size, hidden_size, 1])
	_out_i = tf.reshape(_out_i, [batch_size, chunk_size, hidden_size, 1])
        self._outputs = tf.squeeze(tf.matmul(_out_u, _out_i, transpose_a=True, name="_outputs"))

        outputs = tf.one_hot(self._seq_lengths-1, chunk_size)
        self._outputs = tf.multiply(self._outputs, outputs)

	# Back-propagating errors only for the last time step
	# outputs[:len(outputs)-2] = 0
	# _outputs = tf.zeros(tf.shape(outputs))
	# _outputs[len(_outputs)-1] = outputs[len(outputs)-1] 

	# Calculating loss
        # Need to check with reduction type None and change outputs to softmax instead of dotproduct
	self._loss = tf.losses.mean_squared_error(self._targets, self._outputs)

	if not is_training:
	    self._train_op = tf.no_op()
	    return

	# Optimization for training
	optimizer = tf.train.AdamOptimizer(learning_rate)
	self._train_op = optimizer.minimize(self.loss)

	# Initializing embedding variables
	#self._embeddings_reset = list()
	#op = user_emb.assign()
	#self._optimizer_reset.append(op)

    # Need to define necessary properties
    @property
    def inputs(self):
	return self._inputs

    @property
    def targets(self):
	return self._targets

    @property
    def seq_lengths(self):
	return self._seq_lengths

    @property
    def initial_state(self):
	return self._initial_state

    @property
    def batch_size(self):
	return self._batch_size

    @property
    def hidden_unit_size(self):
	return self._hidden_unit_size

    @property
    def u_emb(self):
	return self._u_emb

    @property
    def i_emb(self):
	return self._i_emb

    @property
    def loss(self):
	return self._loss

    @property
    def final_state(self):
	return self._final_state

    @property
    def outputs(self):
	return self._outputs

    @property
    def train_op(self):
	return self._train_op

def run_batch(sess, model, iterator, init_state):
    """
    """
    costs = 0#np.zeros(model.batch_size)
    state = init_state
    # shape(inputs): [seq_len x D x 2]
    # shape(targets): [seq_len]
    # where D = shape[embedding,ts] or D = shape[embedding,1/0,ts]
    try:
      for (inputs,targets,seq_lens) in iterator:
        fetches = [model.loss, model.final_state, model.outputs, model.train_op]
        feed_dict = {}
        feed_dict[model.inputs] = inputs
        feed_dict[model.targets] = targets
        feed_dict[model.seq_lengths] = seq_lens
        feed_dict[model.initial_state] = state
        errors, state, outputs, _ = sess.run(fetches, feed_dict)
        costs += errors
#        print("targets {}".format(outputs))
#        print("outputs {}".format(outputs))
      return costs, state
    except Exception as e:
        print(e)
        #print("targets:",targets)
        #print("feed-inputs: ",inputs)
        #print("feed-seq_lens: ", seq_lens)

def run_epoch(sess, train_model, valid_model, train_iter, valid_iter):
    """
    """
    train_errors = list()
    valid_errors = list()
    print("Training model on train data")
    for train in train_iter:
	state = sess.run(train_model.initial_state)
	errors, state = run_batch(sess, train_model, train, state)
	train_errors.append(errors)
    print("Validating on probe data")
    for valid in valid_iter:
        state = sess.run(train_model.initial_state)
	errors, state = run_batch(sess, valid_model, valid, state)
	valid_errors.append(errors)

    return (np.nansum(train_errors), np.nansum(valid_errors))

def main(args):
    """
    """

    print("Forming training data...")
    train_data = Dataset.from_path(args.train_path, args.batch_size,
        args.chunk_size, args.hidden_size-1, mode="train")
    print("Forming validation data...")
    valid_data = Dataset.from_path(args.valid_path, args.batch_size,
        args.chunk_size, args.hidden_size-1, mode="valid")
    num_users = train_data.num_users
    num_items = train_data.num_items
    max_seq_len = train_data.max_seq_len
    valid_data.max_seq_len = max_seq_len
    print("num of users: ", num_users)
    print("num of items: ", num_items)
    print("train_max_seq_len ",train_data.max_seq_len)
    print("valid_max_seq_len ",valid_data.max_seq_len)
    print("Doing batch preprocessing for training data...")
    train_data.batch_preprocessing()
    print("Doing batch preprocessing for validation data...")
    valid_data.batch_preprocessing()

    emb_settings = {
	"emb_size": args.input_emb_size-1, # decreasing to add time-component later
	"num_samples": 100,
	"batch_size": 20,
	"learning_rate": 0.1,
	"epoch": 20,
    }

    with tf.Graph().as_default(), tf.Session() as sess:
      with tf.variable_scope("Embeddings_model"):
        with tf.variable_scope("User_Embeddings"):
	    u_emb_model = Embeddings(num_classes=num_users, **emb_settings)
        with tf.variable_scope("Item_Embeddings"):
	    i_emb_model = Embeddings(num_classes=num_items, **emb_settings)
        tf.global_variables_initializer().run()
        # Create embeddings for user and item sequences and prepare batches
        print("Creating embeddings for users...")
        u_emb_model.create_embeddings(sess)
        u_emb = u_emb_model.embeddings
        print("Creating embeddings for items...")
        i_emb_model.create_embeddings(sess)
        i_emb = i_emb_model.embeddings

    settings = {
        "inp_D": args.input_emb_size,
        "hidden_size": args.hidden_size,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "learning_rate": args.learning_rate,
	"u_emb":u_emb,
	"i_emb":i_emb,
        "max_seq_len":max_seq_len,
    }

#    config = tf.ConfigProto(device_count={'GPU':0})
    with tf.Graph().as_default(), tf.Session() as sess:
#      with tf.variable_scope("Main_model"):
#      with tf.device("/cpu:0"):
        with tf.variable_scope("model"):
            train_model = RecommenderRNN(num_users, num_items,
                 is_training=True, **settings)
        with tf.variable_scope("model", reuse=True):
            valid_model = RecommenderRNN(num_users, num_items,
                is_training=False, **settings)

	tf.global_variables_initializer().run()

	for i in range(1, args.num_epochs+1):
	    # Train on random batches of data
	    train_iter = train_data.prepare_and_iter_batches()
	    valid_iter = valid_data.prepare_and_iter_batches()

	    train_error, valid_error = run_epoch(sess, train_model, valid_model,
			train_iter, valid_iter)

	    print("[Epoch {}] Train RMSE: {:.3f}".format(i, train_error))
	    print("[Epoch {}] Valid RMSE: {:.3f}".format(i, valid_error))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", help="path to training data")
    parser.add_argument("valid_path", help="path to validation data")
    parser.add_argument("--batch-size", type=int, default=10,
	help="number of sequences processed in parallel")
    parser.add_argument("--chunk-size", type=int, default=30,
        help="sequence window to be processed in each computation")
    parser.add_argument("--input-emb-size", type=int, default=20,
	help="dimension of input features")
    parser.add_argument("--hidden-size", type=int, default=20,
	help="number of hidden units in the RNN cell")
    parser.add_argument("--learning-rate", type=float, default=0.01,
	help="model learning rate")
    parser.add_argument("--num-epochs", type=int, default=10,
	help="number of epochs to learn")
    parser.add_argument("--verbose", action="store_true", default=False,
	help="enable display of debugging messages")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
#    if args.verbose:
    main(args)

