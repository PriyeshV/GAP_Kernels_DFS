import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
import numpy as np
from utils import *


class propogation_Cell(RNNCell):
    def __init__(self, config):
        self.config = config
        self.attention = self.config.mRNN.attention
        self.num_units = self.config.mRNN._hidden_size
        self.n_labels = self.config.data_sets._len_labels
        self.x_size = self.config.data_sets._len_features_curr
        self.attn_size = 16#self.num_units

        self.W_xh = tf.get_variable('W_xh', [self.x_size, self.num_units])
        self.node_bias = tf.get_variable('bias', [self.num_units])

        self.W_hh = tf.get_variable('W_hh', [self.num_units, self.num_units])
        self.c_bias = tf.get_variable('c_bias', [self.num_units])

        self.W_L = tf.get_variable('W_L', [self.num_units, self.n_labels], initializer=tanh_init)
        self.b_L = tf.get_variable('b_L', [self.n_labels], initializer=zeros)

        #self.W_g = tf.get_variable('g_w', [self.num_units + self.n_labels, 1])
        #self.b_g = tf.get_variable('g_b', [1], initializer=zeros)

    @property
    def state_size(self):
        return tuple([self.num_units, self.num_units])

    @property
    def output_size(self):
        return self.num_units

    def get_node_weights(self):
        return self.W_xh, self.node_bias

    def get_prediction_weights(self):
        return self.W_L, self.b_L

    def attentive_combine2(self, node, states, A):

        with tf.variable_scope('Attentive_Ensemble'):
            state_size = self.num_units
            context_size = node.get_shape().as_list()[1]
            batch_size = tf.shape(node)[0]
            context = node

            if self.attention == 0:
                A = A/(tf.reduce_sum(A, axis=1, keep_dims=True)+1e-15)
                c = tf.matmul(A, states)
                return c, tf.ones(tf.shape(A))

            else:
                # Attention Mechanism
                score_weights = tf.get_variable("ScoreW", [self.attn_size, 1])  # [A, 1]

                k = tf.get_variable("AttnW", [state_size, self.attn_size], initializer=tanh_init)  # [state_size, A]
                W = tf.get_variable("linearW", [context_size, self.attn_size], initializer=tanh_init)
                b = tf.get_variable("linearB", [self.attn_size], initializer=const)

                attn_features = tf.matmul(states, k)
                y = tf.matmul(context, W) + b

                # Calculating alpha
                y = tf.reshape(y, [batch_size, 1, self.attn_size])
                attn_features = tf.reshape(attn_features, [batch_size, 1, self.attn_size])
                attn_matrix = tf.nn.tanh(y + tf.transpose(attn_features, [1,0,2]))  # [path,1,A]+[1,path,A] -> [path, path, A]

                attn_matrix = tf.reshape(attn_matrix, [batch_size*batch_size, self.attn_size])
                attn_matrix = tf.matmul(attn_matrix, score_weights)
                attn_matrix = tf.reshape(attn_matrix, [batch_size, batch_size]) / 10

                # Do selective softmax
                attn_exp = tf.exp(attn_matrix)
                valid = tf.multiply(attn_exp, A)  # element-wise product
                denom = tf.reduce_sum(valid, axis=1, keep_dims=True)
                attn_values = valid / (denom + 1e-15)  # softmax

                # Calculate context c
                data = tf.matmul(attn_values, states)

            return data, attn_values

    def attentive_combine(self, node, states, A): #D1 attention

        with tf.variable_scope('Attentive_Ensemble'):
            state_size = self.num_units
            context_size = node.get_shape().as_list()[1]
            batch_size = tf.shape(node)[0]
            context = node

            if self.attention == 0:
                A = A/(tf.reduce_sum(A, axis=1, keep_dims=True)+1e-15)
                c = tf.matmul(A, states)
                return c, tf.ones(tf.shape(A))

            else:
                # Attention Mechanism
                score_weights = tf.get_variable("ScoreW", [self.attn_size, 1])  # [A, 1]

                k = tf.get_variable("AttnW", [state_size, self.attn_size], initializer=tanh_init)  # [state_size, A]
                W = tf.get_variable("linearW", [context_size, self.attn_size], initializer=tanh_init)
                b = tf.get_variable("linearB", [self.attn_size], initializer=const)

                attn_features = tf.matmul(states, k)
                y = tf.matmul(context, W) + b

                # Calculating alpha
                y = tf.reshape(y, [batch_size, self.attn_size])
                attn_features = tf.reshape(attn_features, [batch_size, self.attn_size])
                attn_matrix = tf.nn.tanh(
                    y + attn_features)  # [path,1,A]+[1,path,A] -> [path, path, A]

                attn_matrix = tf.reshape(attn_matrix, [batch_size, self.attn_size])
                attn_matrix = tf.matmul(attn_matrix, score_weights)
                attn_matrix = tf.reshape(attn_matrix, [1, batch_size])

                # Do selective softmax
                attn_exp = tf.exp(attn_matrix)
                valid = tf.multiply(A, attn_exp)  # element-wise product
                denom = tf.reduce_sum(valid, axis=1, keep_dims=True)
                attn_values = valid / (denom + 1e-15)  # softmax

                # Calculate context c
                data = tf.matmul(attn_values, states)

            return data, attn_values


    def predict(self, data):
        if not self.config.data_sets._multi_label:
            predictions = tf.nn.softmax(tf.matmul(data, self.W_L) + self.b_L)
        else:
            predictions = tf.sigmoid(tf.matmul(data, self.W_L) + self.b_L)
        return predictions

    def combined_prediction(self, node, neighbor, node_L, neigh_L, keep_prob_out):
        if self.config.combine == 'add':
            x = node + neighbor
        elif self.config.combine == 'mul':
            print('mul')
            x = node * neighbor
        else:
            raise ValueError

        val1 = tf.concat([x, node_L], axis=1)
        val2 = tf.concat([x, neigh_L], axis=1)
        #g_val = tf.concat([tf.tanh(tf.matmul(val1, self.W_g) + self.b_g), tf.tanh(tf.matmul(val2, self.W_g) + self.b_g)], axis=0) #/ 0.25
        #g_val = tf.nn.softmax(g_val, dim=0)
        #prediction = g_val[0] * node_L + g_val[1] * neigh_L
        prediction = tf.reduce_mean([node_L, neigh_L], axis=0)
        return prediction, tf.constant([0.5, 0.5])
        # return prediction, g_val

    def __call__(self, x, state, A, keep_prob_in, keep_prob_out, get_labels=False, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            temp = tf.matmul(x, self.W_xh) #, a_is_sparse=self.config.data_sets.input_is_sparse)
            node = tf.tanh(temp + self.node_bias)
            node = tf.nn.dropout(node, keep_prob_out)
            c = tf.nn.dropout(c, keep_prob_in)

            neighbor, attn_values = self.attentive_combine(node, c, A)
            c_hat = tf.tanh(temp + tf.matmul(neighbor, self.W_hh) + self.c_bias)
            c = c_hat
            h = c

            if get_labels:
                neighbor= tf.nn.dropout(neighbor, keep_prob_out)
                node_L  = self.predict(tf.expand_dims(node[0], axis=0))
                neigh_L = self.predict(tf.expand_dims(neighbor[0], axis=0))
                emb_L   = self.predict(tf.expand_dims(c[0], axis=0))
                L, gating_values  = self.combined_prediction(tf.expand_dims(node[0], axis=0), tf.expand_dims(neighbor[0], axis=0), node_L, neigh_L, keep_prob_out)
                return (c, h), (node_L, neigh_L, emb_L, L), attn_values[0], gating_values
            else:
                return (c, h)


