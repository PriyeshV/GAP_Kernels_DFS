import numpy as np
import tensorflow as tf
from cells import propogation_Cell
from utils import relu_init, tanh_init, zeros, const


class Network(object):

    def __init__(self, config):
        self.config = config
        self.global_step = tf.Variable(0, name="global_step", trainable=False)  # Epoch
        self.cell = None
        self.attn_values = tf.constant(0)
        self.gating_values = tf.constant(0)

    def variable_summaries(self):
        with tf.name_scope('summaries'):
            for k in tf.trainable_variables():
                name = '-'.join(k.name.split('/')[-2:])
                tf.summary.histogram(name, k)

    def get_path_data(self, x_attr, x_lengths, keep_prob_in, keep_prob_out, path_matrix, state=None):

        steps, batch_size = tf.shape(x_attr)[0], tf.shape(x_attr)[1]
        num_units = self.config.mRNN._hidden_size

        # Sparse drop used outside tensor graph
        # with tf.variable_scope('InputDropout'):
        #   x_attr = tf.nn.dropout(x_attr, keep_prob_in)

        with tf.variable_scope('MyCell') as scope:
            inputs      = tf.split(x_attr,      num_or_size_splits=self.config.num_steps, axis=0)
            path_matrix = tf.split(path_matrix, num_or_size_splits=self.config.num_steps, axis=0)

            cell = propogation_Cell(self.config)
            state = (tf.zeros((batch_size, num_units)), tf.zeros((batch_size, num_units)))
            for tstep in range(len(inputs)):
                if tstep == len(inputs) - 1:
                    state, labels, self.attn_values, self.gating_values = cell.__call__(inputs[tstep][0], state, path_matrix[tstep][0], keep_prob_in, keep_prob_out,
                                                  get_labels=True)
                else:
                    state = cell.__call__(inputs[tstep][0], state, path_matrix[tstep][0], keep_prob_in, keep_prob_out,
                                             get_labels=False)
                scope.reuse_variables()

        return state, labels

    def consensus_loss(self, predictions, pred_mean):
        pred_mean = tf.reduce_mean(predictions)
        cross_loss = -1*tf.reduce_mean(tf.multiply(pred_mean, tf.log(1e-10 + predictions)))
        return cross_loss

    def loss(self, predictions, labels, wce):
        if self.config.data_sets._multi_label:
            cross_loss = tf.add(tf.log(1e-10 + predictions) * labels,
                                tf.log(1e-10 + (1 - predictions)) * (1 - labels))
            cross_entropy_label = -1 * tf.reduce_mean(tf.reduce_sum(wce * cross_loss, 1))
        else:
            cross_loss = labels * tf.log(predictions + 1e-10)
            cross_entropy_label = tf.reduce_mean(-tf.reduce_sum(wce * cross_loss, 1))

        return cross_entropy_label

    def L2loss(self):
        # wts = ['W_xh', 'W_hh', 'W_L']
        # with tf.variable_scope('L2_loss'):
        #     if self.config.solver._L2loss:
        #         L2_loss = tf.add_n([tf.nn.l2_loss(v) if v.name.split('/')[-1].split(':')[0] in wts else tf.constant(0.0)
        #                             for v in tf.trainable_variables()])
        with tf.variable_scope('L2_loss'):
            if self.config.solver._L2loss:
                L2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        return L2_loss

    def training(self, loss, optimizer):
        train_op = optimizer.minimize(loss[0])
        return train_op

    def custom_training(self, loss, optimizer, batch_size):

        # gradient accumulation over multiple batches
        # http://stackoverflow.com/questions/42156957/how-to-update-model-parameters-with-accumulated-gradients
        # https://github.com/DrSleep/tensorflow-deeplab-resnet/issues/18#issuecomment-279702843
        #batch_size = tf.Print(batch_size, [batch_size], message="Batch size: ")
        with tf.variable_scope('custom_training'):

            tvs = tf.trainable_variables()
            accum_grads = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
            reset_op = [v.assign(tf.zeros_like(v)) for v in accum_grads]

            gvs = tf.gradients(loss, tvs)  # compute gradients
            accum_op = [accum_grads[i].assign_add(gv) for i, gv in enumerate(gvs)]  # accumulate computed gradients

            normalized_grads = [var/batch_size for var in accum_grads]
            update_op = optimizer.apply_gradients(zip(normalized_grads, tvs))

        return reset_op, accum_op, update_op





