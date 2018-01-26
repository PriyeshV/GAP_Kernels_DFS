import numpy as np
from copy import deepcopy
from os import path
import scipy as sp
from genAllWalks import getWalks
from genSubGraph import getSubGraph

class DataSet(object):
    def __init__(self, cfg):
        """Construct a DataSet.
        """
        self.cfg = cfg

        self.diameter, self.degree = self.load_graph()
        self.diameter = min(cfg.max_depth, self.diameter)

        self.all_labels     = self.get_labels(cfg.label_path)
        self.all_features   = self.get_fetaures(cfg.features_path)

        # Increment the positions by 1 and mark the 0th one as False
        self.train_nodes    = np.concatenate(([False], np.load(path.join(cfg.fold_dir, 'train_ids.npy'))))
        self.val_nodes      = np.concatenate(([False], np.load(path.join(cfg.fold_dir, 'val_ids.npy'))))
        self.test_nodes     = np.concatenate(([False], np.load(path.join(cfg.fold_dir, 'test_ids.npy'))))

        if (self.train_nodes*self.val_nodes).any() or (self.val_nodes*self.test_nodes).any() or (self.test_nodes*self.train_nodes).any():
            raise ValueError("Overlap between train/test/validation sets")

        self.n_train_nodes  = np.sum(self.train_nodes)
        self.n_val_nodes    = np.sum(self.val_nodes)
        self.n_test_nodes   = np.sum(self.test_nodes)
        self.n_nodes        = len(self.train_nodes)

        self.change = 0
        self.path_pred_variance = {}
        self.prv_entropy = []
        self.label_cache, self.emb_cache = self.get_emb_label_cache()
        self.wce = self.get_wce()

        self.n_features     = self.all_features.shape[1]
        self.n_labels       = self.all_labels.shape[1]
        self.multi_label    = self.is_multilabel()
        self.gs = getSubGraph(self.cfg)

        self.feature_flag = True
        self.set_feature()
        self.print_statistics()

    def get_walks(self):
        gW = getWalks(self.cfg)
        return gW.walks

    def set_feature(self, flag=True):
        self.feature_flag = flag
        if flag:
            self.cfg.data_sets._len_features_curr  = self.n_features
        else:
            self.cfg.data_sets._len_features_curr = self.cfg.mRNN._hidden_size + self.cfg.data_sets._len_labels
            self.cfg.data_sets.input_is_sparse = False

        print("Feature size set to: ", self.cfg.data_sets._len_features_curr)

    def get_emb_label_cache(self):
        return np.zeros_like(self.all_labels), np.zeros((self.n_nodes, self.cfg.mRNN._hidden_size)) #start from scratch

    def get_fetaures(self, path):
        # a) add feature for dummy node 0 a.k.a <EOS> and <unlabeled>
        # b) increments index of all features by 1, thus aligning it with indices in walks
        all_features = np.load(path)
        all_features = all_features.astype(np.float32, copy=False)  # Required conversion for Python3
        all_features = np.concatenate(([np.zeros(all_features.shape[1])], all_features), 0)

        pos = np.where(all_features != 0)
        self.cfg.data_sets.input_is_sparse = len(pos) < 0.20*np.size(all_features)

        return all_features

    def get_labels(self, path):
        # Labels start with node '0'; Walks_data with node '1'
        # To get corresponding mapping, increment the label node number by 1
        # add label for dummy node 0 a.k.a <EOS> and <unlabeled>
        all_labels = np.load(path)
        all_labels = np.concatenate(([np.zeros(all_labels.shape[1])], all_labels), 0)

        self.inv_labels = {}
        label_size = np.shape(all_labels)[1]
        for idx in range(label_size):
            self.inv_labels[idx] = np.where(all_labels[:,idx]==1)[0]

        return all_labels

    def get_wce(self):
        if self.cfg.solver.wce:
            valid = self.train_nodes + self.val_nodes
            tot = np.dot(valid, self.all_labels)
            wce = 1 / (len(tot) * (tot * 1.0 / np.sum(tot)))
        else:
            wce = np.ones(self.all_labels.shape[1])

        wce[np.isinf(wce)] = 0
        print("Cross-Entropy weights: ", wce)
        return wce

    def load_graph(self):
        diameter = 999
        adjmat = sp.io.loadmat(self.cfg.adjmat_path)['adjmat']
        degree = np.array(sp.sparse.csc_matrix.sum(adjmat, axis=0)).flatten()
        degree = np.concatenate([[0], degree])
        return diameter, np.array(degree)

    def get_degree(self, node_id):
        return self.degree[node_id]

    def is_multilabel(self):
        sum = np.count_nonzero(self.all_labels[self.train_nodes])
        return sum > self.n_train_nodes

    def print_statistics(self):
        print('############### DATASET STATISTICS ####################')
        print(
            'Train Nodes: %d \nVal Nodes: %d \nTest Nodes: %d \nFeatures: %d \nLabels: %d \nMulti-label: %s \nDiameter: %d \nMax Degree: %d \nAverage Degree: %d'\
            % (
            self.n_train_nodes, self.n_val_nodes, self.n_test_nodes, self.n_features, self.n_labels, self.multi_label,
            self.diameter, np.max(self.degree), np.mean(self.degree[1:])))
        print('-----------------------------------------------------\n')

    def get_nodes(self, dataset):
        if dataset == 'train':
            nodes = self.train_nodes
        elif dataset == 'val':
            nodes = self.val_nodes
        elif dataset == 'test':
            nodes = self.test_nodes
        elif dataset == 'remaining':
            nodes = np.logical_not(self.train_nodes + self.val_nodes + self.test_nodes)
        elif dataset == 'all':
            nodes = np.ones(len(self.train_nodes), dtype=bool)
        else:
            raise ValueError

        nodes[0] = False
        return nodes

    def update_emb_label(self, emb, predictions, entropy):
        if (len(self.prv_entropy) == 0) or not self.cfg.cautious_updates:
            # Naive straight forward update
            self.label_cache = predictions
            self.emb_cache = emb
            self.prv_entropy = entropy

        else:
            score = self.prv_entropy/(self.prv_entropy + entropy + 1e-15)

            self.emb_cache   = score*emb + (1-score)*self.emb_cache
            self.label_cache = score*predictions + (1-score)*self.label_cache
            self.prv_entropy = score**2 + (1-score)**2

    def walks_generator(self, data='train', by_max_degree=True, by_prob=True, shuffle=True):
        nodes = np.where(self.get_nodes(data))[0]
        if shuffle:
            indices = np.random.permutation(len(nodes))
            nodes = nodes[indices]

        for node_id in nodes:
            x, path_matrix = self.gs.get_sub(node_id)
            x = np.swapaxes(x, 0, 1)  # convert from (batch x step) to (step x batch)

            temp = np.array(x) > 0  # get locations of all zero inputs as binary matrix
            lengths = np.sum(temp, axis=0)  ### Incorrect for walks starting with 0

            if self.feature_flag:
                x1 = [self.all_features[row] for row in x]  # get features for all data points
            else:
                x1 = [np.concatenate((self.label_cache[row], self.emb_cache[row]), axis=1) for row in x]  # get pseudo labels

            y = [self.all_labels[node_id]]  # get tru labels for Node of interest

            if data == 'train':
                if self.cfg.data_sets.sparse_dropout:
                    # #Sparse dropout
                    x1 = np.array(x1)
                    drop_prob = self.cfg.data_sets.sparse_dropout
                    pos = np.where(x1 != 0)
                    count = len(pos[0])
                    l = np.random.permutation(count)[:int(drop_prob*count)]
                    mask = (pos[0][l], pos[1][l], pos[2][l])

                    x1[mask] = 0
                    x1 /= (1-drop_prob)

                if self.cfg.data_sets.add_noise:
                    # Add gaussian noise to categorical attributes
                    x1 += np.random.normal(0, 0.01, np.shape(x1))

            yield (x1, lengths, y, node_id, path_matrix)