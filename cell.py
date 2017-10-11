from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear

class RecommenderCell(tf.contrib.rnn.BasicLSTMCell):

    def __init__(self, num_units, batch_size, inp_D, is_training,
		forget_bias=1):
        # Number of units = H, Number of users = U, Number of items = I
        self._num_units = num_units
        self._batch_size = batch_size
        self._inp_D = inp_D
        self._is_train = is_training
	self._forget_bias = forget_bias

    @property
    def state_size(self):
        return 6 * self._num_units

    @property
    def output_size(self):
        return 2 * self._num_units

    def __call__(self, inputs, state, scope=None):
	""" Long Short-Term Memory cell wrapper with one LSTM cell for users
            and one LSTM cell for items
           
            Might include dropout and batch/layer normalization
            https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
            https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
        """
	# Parameters of gates are concatenated into one multiply for efficiency.
        # shape(inputs): [batch_size x inp_D*2] 
        # shape(state): [batch_size x num_units*4]
	print(inputs.shape)
        inputs_u, inputs_i = tf.split(value=inputs, num_or_size_splits=2, axis=1)
        _inputs_u, r_u = inputs_u[:,:inputs_u.shape[1]-1],inputs_u[:,inputs_u.shape[1]-1:]
        _inputs_i, r_i = inputs_i[:,:inputs_i.shape[1]-1],inputs_i[:,inputs_i.shape[1]-1:]
        print(inputs_u[:,:inputs_u.shape[1]-1], r_u)
        # shape(c):
        # shape(i):


        # Define two variables one for user and one for item and do a scalar multiply with rating
        # and pass it to linear method as an additional parameter
        # throw _r_u and _r_i 
 
        c_u, h_u, c_i, h_i, _r_u, _r_i = tf.split(value=state, num_or_size_splits=6, axis=1)
 
        with tf.variable_scope(scope or type(self).__name__):
#	  with tf.device("/cpu:0"):
            with tf.variable_scope("Gates"):
		with tf.variable_scope("Users"):
                    # linear will take care of affine multiplication with weights (W * x + b)
                    # shape(concat):
                    concat_u = _linear([_inputs_u, h_u, _r_u], 4 * self._num_units, True, 1.0)
                    # f = forget gate, i = input gate, o = output gate, j = new gate
                    # shape(f,i,o,j) :
                    f_u, i_u, o_u, j_u = tf.split(value=concat_u, num_or_size_splits=4, axis=1)
		    # Calculating new input and new hidden states for users
                    new_c_u = (c_u * tf.sigmoid(f_u + self._forget_bias) + tf.sigmoid(i_u) * tf.tanh(j_u))
		    new_h_u = tf.tanh(new_c_u) * tf.sigmoid(o_u)

		with tf.variable_scope("Items"):
                    concat_i = _linear([_inputs_i, h_i, _r_i], 4 * self._num_units, True, 1.0)
                    f_i, i_i, o_i, j_i = tf.split(axis=1, num_or_size_splits=4, value=concat_i)
                    # Calculating new input and hidden states for items
                    new_c_i = (c_i * tf.sigmoid(f_i + self._forget_bias) + tf.sigmoid(i_i) * tf.tanh(j_i))
                    new_h_i = tf.tanh(new_c_i) * tf.sigmoid(o_i)

	    with tf.variable_scope("Outputs"):
 		# Calculating outputs for users
                with tf.variable_scope("outputs_user"):
                    w_out_u = tf.get_variable("w_out_u",
			[self._batch_size, self._num_units, self._num_units],dtype=tf.float32)
                    b_out_u = tf.get_variable("b_out_u",
			[self._batch_size, self._num_units, 1],dtype=tf.float32)
		    # shape of _h: [batch_size x num_units x 1]
                    _h = tf.reshape(new_h_u, new_h_u.shape.concatenate(1))
                    _out_users = tf.squeeze(tf.add(tf.matmul(w_out_u, _h), b_out_u))

                # Calculating outputs for items
                with tf.variable_scope("outputs_items"):
                    w_out_i = tf.get_variable("w_out_i",
			[self._batch_size, self._num_units, self._num_units],dtype=tf.float32)
                    b_out_i = tf.get_variable("b_out_i",
			[self._batch_size, self._num_units, 1],dtype=tf.float32)
                    _h = tf.reshape(new_h_i, new_h_i.shape.concatenate(1))
                    _out_items = tf.squeeze(tf.add(tf.matmul(w_out_i, _h), b_out_i))

		# shape of outputs:[batch_size]
	        #outputs = tf.squeeze(tf.matmul(_out_users, _out_items, transpose_a=True, name="_outputs"))

                # Incorporating user item ratings of previous step
                #_inp_u_r_u = tf.transpose(tf.multiply(tf.transpose(_input_u,_r_u))
                #_inp_i_r_i = tf.transpose(tf.multiply(tf.transpose(_input_i,_r_i))
                _inp_u_r_u = tf.multiply(_inputs_u, r_u)
                _inp_i_r_i = tf.multiply(_inputs_i, r_i)

        return tf.concat([_out_users,_out_items],1),tf.concat([new_c_u, new_h_u, new_c_i, new_h_i, _inp_u_r_u, _inp_i_r_i],1)
