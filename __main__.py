import utils
from config import Config
from parser import Parser
from dataset import DataSet
from network import Network
import eval_performance as perf

import sys
import time
import pickle
import threading
import numpy as np
import tensorflow as tf
from os import path
from copy import deepcopy
from tabulate import tabulate

from tensorflow.python.client import timeline
from tensorflow.python import debug as tf_debug

class DeepDOPE(object):

    def __init__(self, config):
        self.config = config
        self.patience = self.config.patience
        self.learning_rate = self.config.solver.learning_rate

        self.load_data()
        self.setup_arch()
        self.init = tf.global_variables_initializer()
        self.init2 = tf.no_op()

    def setup_arch(self):

        self.entropy = tf.constant(0)
        self.ph_lr, self.ph_keep_prob_in, self.ph_keep_prob_out, self.ph_wce, self.ph_batch_size = self.get_placeholders()

        # Setup Data Queue
        self.ph_x_attr, self.ph_x_lengths, self.ph_y_label, self.ph_node_id, self.ph_path_matrix = self.get_queue_placeholders()
        self.Q, self.enqueue_op, self.dequeue_op = self.setup_data_queues()
        self.x_attr, self.x_lengths, self.y_labels, self.node_id, self.path_matrix  = self.dequeue_op
        self.arch = self.add_network(self.config)
        self.final_state, self.labels = self.arch.get_path_data(self.x_attr, self.x_lengths, self.ph_keep_prob_in, self.ph_keep_prob_out, self.path_matrix)

        self.emb = self.final_state[1] # (c,h)
        (self.att_prediction, self.path_prediction, self.emb_prediction, self.combined_prediction) = self.labels


        with tf.variable_scope('Loss'):
            #secondary losses
            if self.config.solver.path_loss:
                self.path_loss = self.arch.loss(self.path_prediction, self.y_labels,
                                                self.ph_wce) * self.config.solver.path_loss
                tf.summary.scalar('Path_loss', self.path_loss)
            else:
                self.path_loss = tf.constant(0.0)

            if self.config.solver.node_loss:
                self.node_loss = self.arch.loss(self.att_prediction, self.y_labels,
                                                self.ph_wce) * self.config.solver.node_loss
                tf.summary.scalar('Node_loss', self.node_loss)
            else:
                self.node_loss = tf.constant(0.0)

            # Primary Losses
            self.combined_loss  = self.arch.loss(self.combined_prediction, self.y_labels, self.ph_wce)
            self.L2_loss        = self.arch.L2loss() * self.config.solver._L2loss

            self.total_loss = self.L2_loss + self.combined_loss + self.path_loss + self.node_loss #+ self.config.solver.consensus_loss*self.consensus_loss

            tf.summary.scalar('total_loss', tf.reduce_sum(self.total_loss))
            tf.summary.scalar('L2_loss', self.L2_loss)
            tf.summary.scalar('combined_loss', self.combined_loss)

        #Collection of all predictions and losses
        self.predictions    = {'node': self.att_prediction, 'path': self.path_prediction, 'combined': self.combined_prediction}
        self.losses         = [self.node_loss, self.path_loss, self.combined_loss, self.total_loss]

        #Optimizer
        self.optimizer  = self.config.solver._optimizer(self.ph_lr)
        train           = self.arch.custom_training(self.total_loss, self.optimizer, self.config.batch_size)
        self.reset_grads, self.accumulate_op, self.update_op = train

        #Summaries and Checkpoints
        self.saver = tf.train.Saver()
        self.arch.variable_summaries()
        self.summary = tf.summary.merge_all()
        self.step_incr_op = self.arch.global_step.assign(self.arch.global_step + 1)

        self.init2 = tf.global_variables_initializer()

    def load_and_enqueue(self, sess, data):
        for idx, (x_attr, x_lengths, label, node_id, path_matrix) in enumerate(self.dataset.walks_generator(data)):
            feed_dict = self.create_feed_dict([x_attr],[x_lengths], [label], [node_id], [path_matrix])
            sess.run(self.enqueue_op, feed_dict=feed_dict)

    def load_data(self):
        # Get the 'encoded data'
        self.dataset                            = DataSet(self.config)
        self.config.data_sets._len_labels       = self.dataset.n_labels
        self.config.data_sets._len_features     = self.dataset.n_features
        self.config.data_sets._len_features_curr= self.dataset.n_features
        self.config.data_sets._multi_label      = self.dataset.multi_label
        self.config.data_sets._n_nodes          = self.dataset.n_nodes
        self.config.num_steps                   = self.dataset.diameter + 1
        print('--------- Project Path: ' + self.config.codebase_root_path + self.config.project_name)

    def get_queue_placeholders(self):
        # 0th axis should have same size for all tensord in the Queue
        with tf.variable_scope('Queue_placeholders'):
            x_attr_placeholder      = tf.placeholder(tf.float32, name='Input_val', shape=[1, self.config.num_steps, None, None])  # self.config.data_sets._len_features])
            x_lengths_placeholder   = tf.placeholder(tf.int32, name='walk_lengths', shape=[1, None])
            y_label_placeholder     = tf.placeholder(tf.float32, name='Target', shape=[1, 1, self.config.data_sets._len_labels])
            node_id_placeholder     = tf.placeholder(tf.int32, name='node_id', shape=[1])
            path_matrix_placeholder = tf.placeholder(tf.float32, name='path_matrix_inputs', shape=[1, self.config.num_steps, None, None])

        return x_attr_placeholder, x_lengths_placeholder, y_label_placeholder, node_id_placeholder, path_matrix_placeholder

    def get_placeholders(self):
        with tf.variable_scope('Placeholders'):
            lr              = tf.placeholder(tf.float32, name='learning_rate')
            keep_prob_in    = tf.placeholder(tf.float32, name='keep_prob_in')
            keep_prob_out   = tf.placeholder(tf.float32, name='keep_prob_out')
            batch_size      = tf.placeholder(tf.int32, name='batch_size')
            wce_placeholder = tf.placeholder(tf.float32, shape=[self.config.data_sets._len_labels], name='Cross_entropy_weights')
        return lr, keep_prob_in, keep_prob_out, wce_placeholder, batch_size

    def setup_data_queues(self):
        Q = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32, tf.float32, tf.int32, tf.float32])
        enqueue_op = Q.enqueue_many([self.ph_x_attr, self.ph_x_lengths, self.ph_y_label, self.ph_node_id, self.ph_path_matrix])
        dequeue_op = Q.dequeue()
        return Q, enqueue_op, dequeue_op

    def create_feed_dict(self, x_attr, x_lengths, label_batch, node_id, path_matrix):
        feed_dict = {
            self.ph_x_attr: x_attr,
            self.ph_x_lengths: x_lengths,
            self.ph_y_label: label_batch,
            self.ph_node_id: node_id,
            self.ph_batch_size: self.config.batch_size,
            self.ph_path_matrix: path_matrix}
        return feed_dict

    def add_network(self, config):
        return Network(config)

    def add_summaries(self, sess):
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer_train    = tf.summary.FileWriter(self.config.logs_dir + "train", sess.graph)
        summary_writer_val      = tf.summary.FileWriter(self.config.logs_dir + "validation", sess.graph)
        summary_writer_test     = tf.summary.FileWriter(self.config.logs_dir + "test", sess.graph)
        summary_writers         = {'train': summary_writer_train, 'val': summary_writer_val, 'test': summary_writer_test}
        return summary_writers

    def add_metrics(self, metrics):
        """assign and add summary to a metric tensor"""
        for i, metric in enumerate(self.config.metrics):
            tf.summary.scalar(metric, metrics[i])

    def print_metrics(self, inp):
        for idx, item in enumerate(inp):
            print(self.config.metrics[idx], ": ", item)

    def club(self, best_nodes, best_predictions, best_emb):
        total_steps = np.sum(self.dataset.get_nodes('all'))
        new_labels  = np.zeros((total_steps+1, self.config.data_sets._len_labels))
        new_entropy = np.zeros((total_steps+1, 1))
        new_emb     = np.zeros((total_steps+1, self.config.mRNN._hidden_size))
        entr = {}

        for k, nodes in best_nodes.items():
            p = best_predictions[k]['combined']
            new_labels[nodes] = p
            new_emb[nodes] = best_emb[k]

            entr[k] = -np.sum(p * np.log2(p + 1e-15), axis=1, keepdims=True)
            new_entropy[nodes] = entr[k]

        return new_labels, new_emb, new_entropy, entr

    def get_new_label_emb(self, sess):
        feed_dict = {self.ph_keep_prob_in: 1, self.ph_keep_prob_out: 1, self.ph_wce: self.dataset.wce, self.ph_lr: 0}
        data = 'all'
        total_steps = np.sum(self.dataset.get_nodes(data))

        t = threading.Thread(target=self.load_and_enqueue, args=[sess, data])
        t.daemon = True
        t.start()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        step = 0
        new_labels  = np.zeros((total_steps+1, self.config.data_sets._len_labels))
        new_emb     = np.zeros((total_steps+1, self.config.mRNN._hidden_size))
        new_entropy = np.zeros((total_steps+1,1))
        t0 = time.time()
        while step < total_steps:
            step += 1
            node_id, pred_labels, emb, entropy = sess.run([self.node_id, self.combined_prediction, self.emb, self.entropy], feed_dict=feed_dict)

            new_labels[node_id] = pred_labels[0]
            new_emb[node_id] = emb[0]
            new_entropy[node_id][0] = entropy

        coord.request_stop()
        coord.join(threads)

        print("Created new embeddings and label in %.4f sec"%(time.time()-t0))
        return new_labels, new_emb, new_entropy

    def run_epoch(self, sess, data, train_op=None, summary_writer=None, verbose=1, learning_rate=0, get_emb=False):
        train = train_op
        if train_op is None:
            train_op = tf.no_op()
            keep_prob_in = 1
            keep_prob_out = 1
        else:
            keep_prob_in = self.config.mRNN._keep_prob_in
            keep_prob_out = self.config.mRNN._keep_prob_out


        # Set up all variables
        total_steps = np.sum(self.dataset.get_nodes(data))  # Number of Nodes to run through
        verbose = min(verbose, total_steps)
        node_ids, gradients, targets, attn_values, gating_values, emb = np.zeros(total_steps, dtype=int), [], [], [], [], np.zeros((total_steps, self.config.mRNN._hidden_size))
        losses, predictions, metrics, entropy= dict(), dict(), dict(), dict()

        metrics['node'], metrics['path'], metrics['combined'] = [], [], []
        predictions['node'], predictions['path'], predictions['combined'] = np.zeros((total_steps, self.config.data_sets._len_labels)), \
                                                                            np.zeros((total_steps, self.config.data_sets._len_labels)), \
                                                                            np.zeros((total_steps, self.config.data_sets._len_labels))
        losses['node'], losses['path'], losses['combined'], losses['total'] = [], [], [], []

        ########################################################################################################
        feed_dict = {self.ph_keep_prob_in: keep_prob_in, self.ph_keep_prob_out: keep_prob_out,
                     self.ph_wce: self.dataset.wce, self.ph_lr: learning_rate}


        # Reset grad accumulator at the beginning
        sess.run([self.reset_grads], feed_dict=feed_dict)

        #Start Running Queue
        t = threading.Thread(target=self.load_and_enqueue, args=[sess, data])
        t.daemon = True
        t.start()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        #Code profiling
        # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()

        step = 0
        while step < total_steps:
            feed_dict = {self.ph_keep_prob_in: keep_prob_in, self.ph_keep_prob_out: keep_prob_out,
                         self.ph_wce: self.dataset.wce, self.ph_lr: learning_rate}

            if ((step < total_steps - 1) or not self.config.summaries) or summary_writer == None:
                id, grads, t_losses, t_pred_probs, target_label, t_attn_values, t_gating_values, t_entropy, t_emb = \
                    sess.run([self.node_id, train_op, self.losses, self.predictions, self.y_labels,
                              self.arch.attn_values, self.arch.gating_values, self.entropy, self.emb], feed_dict=feed_dict)#, options=options, run_metadata=run_metadata)

            else:
                summary, id, grads, t_losses, t_pred_probs, target_label, t_attn_values, t_gating_values, t_entropy, t_emb = \
                    sess.run([self.summary, self.node_id, train_op, self.losses, self.predictions, self.y_labels,
                              self.arch.attn_values, self.arch.gating_values, self.entropy, self.emb], feed_dict=feed_dict)#, options=options, run_metadata=run_metadata)
                #if summary_writer is not None:
                summary_writer.add_summary(summary, self.arch.global_step.eval(session=sess))
                summary_writer.flush()


            #Saving code profile
            # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            # chrome_trace = fetched_timeline.generate_chrome_trace_format()
            # with open('timeline_02_step_%d.json' % step, 'w') as f:
            #     f.write(chrome_trace)


            node_ids[step] = id

            # Accumulate attention values
            attn_values.append(np.std(t_attn_values[t_attn_values.nonzero()]))
            gating_values.append(np.abs(t_gating_values[0]-t_gating_values[1]))


            # Accumulate losses
            losses['node'].append(t_losses[0])
            losses['path'].append(t_losses[1])
            losses['combined'].append(t_losses[2])
            losses['total'].append(t_losses[3])

            #Accumulate entropy of prediction
            entropy[id] = t_entropy

            #accumulate mebeddings
            if get_emb:
                emb[step] = t_emb[0]

            # Accumulate Predictions
            for k, v in t_pred_probs.items():
                predictions[k][step] = v
            targets.append(np.squeeze(target_label))

            step += 1

            if train and (step % self.config.batch_size == 0 or step == total_steps):
                # Update gradients after batch_size or at the end of the current epoch
                #print("Queue size: ", sess.run([self.Q.size()]))

                batch_size =  self.config.batch_size
                if step == total_steps:
                    batch_size = step%batch_size
                feed_dict[self.ph_batch_size] = batch_size

                sess.run([self.update_op], feed_dict=feed_dict)
                sess.run([self.reset_grads], feed_dict=feed_dict)

                if verbose and self.config.solver.gradients:
                    # get the absolute maximum gradient to each variable
                    gradients.append([np.max(np.abs(item)) for item in grads])
                    print("%d/%d :: " % (step, total_steps), end="")
                    for var, val in zip(['-'.join(k.name.split('/')[-2:]) for k in tf.trainable_variables()],
                                        np.mean(gradients, axis=0)):
                        print("%s :: %.8f  " % (var, val / self.config.batch_size), end="")
                    print("\n")
                sys.stdout.flush()

        coord.request_stop()
        coord.join(threads)

        # Average statistics over batches
        for k in losses.keys():
            losses[k] = np.mean(losses[k])
        for k in metrics.keys():
            _, metrics[k] = perf.evaluate(np.asarray(predictions[k]), np.asarray(targets), multi_label=self.config.data_sets._multi_label)

        #Hack around to store attn and gating aggreagtes
        metrics['combined']['pak'] = np.mean(attn_values)
        metrics['combined']['average_precision'] = np.mean(gating_values)

        #return raw_predictions
        return node_ids, predictions, losses, metrics, np.asarray(attn_values), np.asarray(gating_values), np.mean(list(entropy.values())), emb

    def fit(self, sess, summary_writers):
        patience = self.config.patience
        learning_rate = self.config.solver.learning_rate

        inner_epoch, best_epoch, best_val_loss = 0, 0, 1e6
        nodes       = {'train': None, 'val': None, 'test': None}
        entropy     = {'train': None, 'val': None, 'test': None}
        losses      = {'train': None, 'val': None, 'test': None}
        metrics     = {'train': None, 'val': None, 'test': None}
        attn_values = {'train': None, 'val': None, 'test': None}
        gating_values= {'train': None, 'val': None, 'test': None}
        emb         = {'train': None, 'val': None, 'test': None}
        predictions = {'train': None, 'val': None, 'test': None}

        best_losses, best_metrics, best_predictions, best_attn, best_gate = deepcopy(losses), deepcopy(metrics), deepcopy(predictions), deepcopy(attn_values), deepcopy(gating_values)
        best_entr, best_emb, best_nodes = deepcopy(entropy), deepcopy(emb), deepcopy(nodes)

        while inner_epoch < self.config.max_inner_epochs:
            inner_epoch += 1
            t0 = time.time()
            _, _, losses['train'], metrics['train'], attn_values['train'], gating_values['train'], entropy['train'], _ = self.run_epoch(sess, data='train', train_op=self.accumulate_op,
                                                                                                summary_writer=summary_writers['train'], learning_rate=learning_rate)

            if inner_epoch % self.config.val_epochs_freq == 0:
                nodes['val'], predictions['val'], losses['val'], metrics['val'], attn_values['val'], gating_values['val'], entropy['val'], emb['val'] = \
                    self.run_epoch(sess, data='val', summary_writer=summary_writers['val'], get_emb=True)

                if self.config.run_test:
                    nodes['test'], predictions['test'], losses['test'], metrics['test'], attn_values['test'], gating_values['test'], entropy['test'], emb['test'] = \
                        self.run_epoch(sess, data='test', summary_writer=summary_writers['test'], get_emb=True)
                    self.print_inner_loop_stats(inner_epoch, metrics, losses)

                else:
                    print('Epoch %d: tr_loss = %.2f val_loss %.2f || tr_micro = %.2f, val_micro = %.2f || '
                          'tr_acc = %.2f, val_acc = %.2f || tr_gate = %.5f, tr_attn = %.5f (%.3f)' %
                          (inner_epoch, losses['train']['combined'], losses['val']['combined'],
                           metrics['train']['combined']['micro_f1'], metrics['val']['combined']['micro_f1'],
                           metrics['train']['combined']['accuracy'], metrics['val']['combined']['accuracy'],
                           np.mean(gating_values['train']), np.mean(attn_values['train']), time.time()-t0))

                new_val_loss = losses['val']['combined'] #losses['val']['node'] + losse s['val']['combined'] + losses['val']['path']

                #if new_val_loss < best_val_loss:
                if new_val_loss < (best_val_loss * self.config.improvement_threshold):
                    best_epoch      = inner_epoch
                    best_losses     = losses
                    best_metrics    = metrics
                    best_attn       = attn_values
                    best_gate       = gating_values
                    best_entr       = entropy
                    best_val_loss   = new_val_loss
                    best_predictions= predictions
                    best_emb        = emb
                    best_nodes      = nodes #Nodes are shuffled, need to keep corresponding nodes for best measures

                    self.saver.save(sess, self.config.ckpt_dir + 'inner-last-best')
                    patience = self.config.patience

                else:
                    if patience < 1:
                        # Restore the best parameters
                        self.saver.restore(sess, self.config.ckpt_dir + 'inner-last-best')
                        if learning_rate <= 0.00001:
                            print('Stopping by patience method')
                            break
                        else:
                            learning_rate /= 10
                            patience = self.config.patience
                            print('Learning rate dropped to %.8f' % learning_rate)
                    else:
                        patience -= 1

        print('Best epoch: ', best_epoch)

        # Last gradient update needs to be taken into account and train_op set to None to ignore dropouts
        best_nodes['train'], best_predictions['train'],best_losses['train'], best_metrics['train'], best_attn['train'],\
        best_gate['train'], best_entr['train'], best_emb['train'] = self.run_epoch(sess, data='train', get_emb=True)

        best_nodes['remaining'], best_predictions['remaining'],best_losses['remaining'], best_metrics['remaining'],best_attn['remaining'],\
        best_gate['remaining'], best_entr['remaining'], best_emb['remaining'] = self.run_epoch(sess, data='remaining', get_emb=True)
        # Run Test set
        t0 = time.time()
        if not self.config.run_test:
            best_nodes['test'], best_predictions['test'], best_losses['test'], best_metrics['test'], best_attn['test'],\
            best_gate['test'], best_entr['test'], best_emb['test'] = self.run_epoch(sess, data='test', summary_writer=summary_writers['test'], get_emb=True)
        print("Test time: %0.5f"%(time.time() - t0))

        print('Epoch %d: tr_gate = %.5f, val_gate = %.5f, te_gate = %.5f || tr_attn =  %.5f, val_attn =  %.5f, '
              'te_attn = %.5f, (%.3f)' %
              (inner_epoch, np.mean(best_gate['train']), np.mean(best_gate['val']), np.mean(best_gate['test']),
               np.mean(best_attn['train']), np.mean(best_attn['val']), np.mean(best_attn['test']), time.time() - t0))

        # UPDATE LABEL and embeddings:
        new_labels, new_emb, new_entropy, entr = self.club(best_nodes, best_predictions, best_emb)
        for k,e in entr.items():
            best_entr[k] = np.mean(e)

        self.dataset.update_emb_label(new_emb, new_labels, new_entropy)

        return inner_epoch, best_nodes, best_losses, best_metrics, best_attn, best_gate, best_entr

    def fit_outer(self, sess, summary_writers):

        stats = []
        outer_epoch = 1
        flag = self.config.boot_reset
        patience = 1
        metrics = {'train': None, 'val': None, 'test': None}
        first_run = {'train': None, 'val': None, 'test': None}
        best_val_loss, best_nodes, best_metrics, best_attn_values, best_gating_values = 1e6, None, None, None, None

        while outer_epoch <= self.config.max_outer_epochs:
            print('OUTER_EPOCH: ', outer_epoch)
            if outer_epoch == 2 and flag:  # reset after first bootstrap | Shall we reuse the weights ???
                print("------ Graph Reset | First bootstrap done -----")
                self.dataset.set_feature(False)  # don't feed features from next run onwards\

                tf.reset_default_graph()
                g2 = tf.get_default_graph()
                #with tf.get_default_graph():
                with g2.as_default():
                    self.setup_arch()
                    self.init2 = tf.global_variables_initializer()
                sess = tf.Session(graph=g2)
                sess.run(self.init2)

                flag = False

            with sess.as_default():
                sess.run(self.init2)  # reset all weights after every outer epoch

                # Just to monitor the trainable variables in tf graph
                print([v.name for v in tf.trainable_variables()], "\n")

                start = time.time()
                # Fit the model to predict best possible labels given the current estimates of unlabeled values
                inner_epoch, nodes, losses, metrics, attn_values, gating_values, entr = self.fit(sess, summary_writers)
                metrics['test']['combined']['val_loss'] = losses['test']

                if outer_epoch == 1:
                    first_run = metrics

                # np.save(self.config.results_folder+'label_cache-epoch%d'%(outer_epoch), self.dataset.label_cache)
                # np.save(self.config.results_folder+'emb_cache-epoch%d'%(outer_epoch), self.dataset.emb_cache)

                duration = time.time() - start
                stats.append(
                    np.round([outer_epoch, inner_epoch,
                              losses['train']['combined'], losses['val']['combined'], losses['test']['combined'],
                              metrics['train']['combined']['micro_f1'], metrics['val']['combined']['micro_f1'], metrics['test']['combined']['micro_f1'],
                              metrics['train']['combined']['accuracy'], metrics['val']['combined']['accuracy'], metrics['test']['combined']['accuracy'],
                              duration, entr['test']+entr['val']], decimals=5))

                print('Outer Epoch %d: tr_loss = %.2f, val_loss %.3f te_loss %.3f|| '
                      'tr_micro = %.2f, val_micro = %.2f te_micro = %.3f|| '
                      'tr_acc = %.2f, val_acc = %.2f  te_acc = %.3f (%.3f sec)' %
                      (outer_epoch, losses['train']['combined'], losses['val']['combined'], losses['test']['combined'],
                       metrics['train']['combined']['micro_f1'], metrics['val']['combined']['micro_f1'], metrics['test']['combined']['micro_f1'],
                       metrics['train']['combined']['accuracy'], metrics['val']['combined']['accuracy'], metrics['test']['combined']['accuracy'],
                       duration))

                new_val_loss = entr['test'] + entr['val'] + entr['remaining']
                # new_val_loss = losses['val']['combined'] #+ losses['train']['combined']
                #if patience >= 1 and (new_val_loss < best_val_loss):
                if new_val_loss < best_val_loss:
                    patience = 5
                    best_metrics = metrics
                    best_attn_values = attn_values
                    best_gating_values = gating_values
                    best_val_loss = new_val_loss
                    best_nodes = nodes
                else:
                    patience -= 1
                    if patience < 1:
                        break
                outer_epoch += 1

        headers = ['Epoch', 'I_Epoch', 'TR_LOSS', 'VAL_LOSS', 'TE_LOSS', 'TR_MICRO', 'VAL_MACRO', 'TE_MACRO',
                   'TR_ACC', 'VAL_ACC', 'TE_ACC', 'DURATION', 'ENTR']
        stats_table = tabulate(stats, headers)
        print(stats_table)
        print('Best Test Results || Accuracy %.3f | MICRO %.3f | MACRO %.3f' %
              (best_metrics['test']['combined']['accuracy'], best_metrics['test']['combined']['micro_f1'], best_metrics['test']['combined']['macro_f1']))
        print("\n   All metrics: \n",best_metrics['test'])

        #hack around to store first_run's values
        best_metrics['test']['combined']['ranking_loss'] = first_run['test']['combined']['micro_f1']
        best_metrics['test']['combined']['hamming_loss'] = first_run['test']['combined']['accuracy']
        best_metrics['test']['combined']['coverage']     = first_run['test']['combined']['bae']
        best_metrics['test']['combined']['val_loss']     = best_val_loss


        return stats, best_nodes, best_metrics['test']['combined'], best_attn_values, best_gating_values


    def print_inner_loop_stats(self, inner_epoch, metrics, losses):

        print('Epoch %d: tr_loss = %.2f val_loss %.2f te_loss %.2f ||'
              ' tr_micro = %.2f, val_micro = %.2f te_micro = %.2f|| '
              'tr_acc = %.2f, val_acc = %.2f  te_acc = %.2f ' %
              (inner_epoch, losses['train']['combined'], losses['val']['combined'], losses['test']['combined'],
               metrics['train']['combined']['micro_f1'], metrics['val']['combined']['micro_f1'], metrics['test']['combined']['micro_f1'],
               metrics['train']['combined']['accuracy'], metrics['val']['combined']['accuracy'], metrics['test']['combined']['accuracy']))

        print('########################################################################################')
        print('#~~~~~~~~~~~~~~~~~~~ tr_consensus_loss = %.2f val_consensus_loss %.2f te_consensus_loss %.2f ||\n'
            '#~~~~~~~~~~~~~~~~~~~ tr_node_loss = %.2f val_node_loss %.2f te_node_loss %.2f ||\n'
              '#~~~~~~~~~~~~~~~~~~~ tr_path_loss = %.2f val_path_loss %.2f te_path_loss %.2f ||\n'
              '#~~~~~~~~~~~~~~~~~~~ tr_comb_loss = %.2f val_comb_loss %.2f te_comb_loss %.2f ||\n'              
              '#~~~~~~~~~~~~~~~~~~~ tr_total_loss = %.2f val_total_loss %.2f te_total_loss %.2f' %
              (losses['train']['consensus'], losses['val']['consensus'], losses['test']['consensus'],
               losses['train']['node'], losses['val']['node'], losses['test']['node'],
               losses['train']['path'], losses['val']['path'], losses['test']['path'],
               losses['train']['combined'], losses['val']['combined'], losses['test']['combined'],
               losses['train']['total'], losses['val']['total'], losses['test']['total']))

        print('########################################################################################')
        print('#~~~~~~~~~~~~~~~~~~~ tr_node_acc %.2f val_node_acc %.2f te_node_acc %.2f ||\n'
              '#~~~~~~~~~~~~~~~~~~~ tr_path_acc %.2f val_path_acc %.2f te_path_acc %.2f ||\n'
              '#~~~~~~~~~~~~~~~~~~~ tr_comb_acc %.2f val_comb_acc %.2f te_comb_acc %.2f ' %
              (metrics['train']['node']['accuracy'], metrics['val']['node']['accuracy'], metrics['test']['node']['accuracy'],
               metrics['train']['path']['accuracy'], metrics['val']['path']['accuracy'], metrics['test']['path']['accuracy'],
               metrics['train']['combined']['accuracy'], metrics['val']['combined']['accuracy'], metrics['test']['combined']['accuracy']))


def init_model(config):
    tf.reset_default_graph()
    tf.set_random_seed(1234)
    np.random.seed(1234)
    with tf.variable_scope('DEEP_DOPE', reuse=None) as scope:
        model = DeepDOPE(config)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    sm = tf.train.SessionManager()

    if config.retrain:
        print("Loading model from checkpoint")
        load_ckpt_dir = config.ckpt_dir
    else:
        print("No model loaded from checkpoint")
        load_ckpt_dir = ''
    sess = sm.prepare_session("", init_op=model.init, saver=model.saver, checkpoint_dir=load_ckpt_dir, config=tf_config)
    return model, sess


def train_model(cfg):

    print('############## Training Module ')
    config = deepcopy(cfg)
    model, sess = init_model(config)
    #with sess:
    summary_writers = model.add_summaries(sess)
    return  model.fit_outer(sess, summary_writers)
    #return stats, nodes, test_metrics, attn_values


def main():

    args = Parser().get_parser().parse_args()
    print("=====Configurations=====\n", args)
    cfg = Config(args)
    train_percents = args.percents.split('_')
    folds = args.folds.split('_')

    outer_loop_stats = {}
    attention = {}
    gating = {}
    results = {}
    nodes = {}

    #Create Main directories
    path_prefixes = [cfg.dataset_name, cfg.folder_suffix, cfg.data_sets.label_type]
    utils.create_directory_tree(path_prefixes)

    for train_percent in train_percents:
        cfg.train_percent = train_percent
        path_prefix = path.join(path.join(*path_prefixes), cfg.train_percent)
        utils.check_n_create(path_prefix)

        attention[train_percent] = {}
        gating[train_percent] = {}
        results[train_percent] = {}
        outer_loop_stats[train_percent] = {}
        nodes[train_percent] = {}

        for fold in folds:
            print('Training percent: ', train_percent, ' Fold: ', fold, '---Running')
            cfg.train_fold = fold
            utils.check_n_create(path.join(path_prefix, cfg.train_fold))
            cfg.create_directories(path.join(path_prefix, cfg.train_fold))
            outer_loop_stats[train_percent][fold], nodes[train_percent][fold], results[train_percent][fold],\
            attention[train_percent][fold], gating[train_percent][fold] = train_model(deepcopy(cfg))

            print('Training percent: ', train_percent, ' Fold: ', fold, '---completed')

        utils.remove_directory(path_prefix)
    path_prefixes = [cfg.dataset_name, cfg.folder_suffix, cfg.data_sets.label_type]

    np.save(path.join(*path_prefixes, 'nodes.npy'), nodes)
    np.save(path.join(*path_prefixes, 'results.npy'), results)
    np.save(path.join(*path_prefixes, 'attentions.npy'), attention)
    np.save(path.join(*path_prefixes, 'gating.npy'), gating)
    np.save(path.join(*path_prefixes, 'outer_loop_stats.npy'), outer_loop_stats)

if __name__ == "__main__":
    main()
