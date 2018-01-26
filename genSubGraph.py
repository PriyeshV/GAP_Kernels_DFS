import numpy as np
from scipy.io import loadmat, savemat
import scipy
import time

class getSubGraph:

    def __init__(self, config):
        self.config = config
        self.max_depth = config.max_depth
        self.adjmat_depth = {}

        try:
            self.adjmat_depth[self.max_depth] = loadmat(self.config.adjmat_path+str(self.max_depth)+'.mat')['adjmat']
            #Load all the required Adj_depth matrices previously created
            for idx in range(self.max_depth, 0, -1):
                self.adjmat_depth[idx] = loadmat(self.config.adjmat_path+str(idx))['adjmat']
                # print('---ADJ\n ',self.adjmat_depth[idx].todense())

        except(FileNotFoundError):
            print("---- Adjacency matrices not found, creating froms scratch...")
            t0 = time.time()
            self.get_adj_depth()
            t1 = time.time()
            print("Depth Adjancency matrices created in %.5f"%(t1-t0))

    def get_adj_depth(self):
        A1 = loadmat(self.config.adjmat_path)['adjmat']
        A1 = scipy.sparse.csr_matrix(A1, dtype=bool) #Assumming Graph is not weighted
        A1 = A1 + A1.T                               #Make it symmetric
        savemat(self.config.adjmat_path+'1', {'adjmat':A1})

        self.adjmat_depth[1] = A1
        for idx in range(2, self.max_depth+1):
            self.adjmat_depth[idx] = self.adjmat_depth[idx-1].dot(self.adjmat_depth[1]) #Reachability
            savemat(self.config.adjmat_path+str(idx)+'.mat', {'adjmat': self.adjmat_depth[idx]})

    def get_sub(self, node, MAXIMUM=1e6):
        node = node -1 #Index for adjmat starts with 0

        nodes_at_depth = {0:[node]}
        path_matrices = {}
        max_neigh = 1

        #Select the Kth order neighbors of the 'node'
        for idx in range(1, self.max_depth+1):
            r, c = self.adjmat_depth[idx].getrow(node).nonzero()
            nodes_at_depth[idx] = c[:min(MAXIMUM, len(c))]

            #keep track of maximum neighbors to assist in padding later
            if len(nodes_at_depth[idx]) > max_neigh:
                max_neigh = len(nodes_at_depth[idx])
        #print('node:', max_neigh)
        # extract a sub_graph between k and k+1th depth neighbors of the 'node'
        for idx in range(self.max_depth, 0, -1):
            cur = nodes_at_depth[idx]
            prv = nodes_at_depth[idx-1]
            path_matrices[idx-1] = self.adjmat_depth[1][prv,:].tocsc()[:,cur].todense()

        # Add padding to path matrices and kth depth nodes
        #-----------------------------------

        # path_matrix for last time step is all zeros
        pm = np.zeros((self.max_depth+1, max_neigh, max_neigh), dtype=int)
        for idx in range(self.max_depth-1, -1, -1):
            pad_size_r = max_neigh - np.shape(path_matrices[idx])[0]
            pad_size_c = max_neigh - np.shape(path_matrices[idx])[1]
            #print(max_neigh, np.shape(path_matrices[idx]))
            pm[self.max_depth - idx] = np.pad(path_matrices[idx], ((0,pad_size_r),(0,pad_size_c)), mode='constant', constant_values=0)

        # Add one to increment node id and make padding '0'
        # At 0th depth, only 'node' is present
        x = np.zeros((self.max_depth+1, max_neigh), dtype=int)
        x[self.max_depth] = np.pad(nodes_at_depth[0], (0, max_neigh-1), mode='constant', constant_values=-1) + 1
        for idx in range(self.max_depth, 0, -1):
            pad_size = max_neigh - len(nodes_at_depth[idx])
            x[self.max_depth - idx] = np.pad(nodes_at_depth[idx], (0,pad_size), mode='constant', constant_values=-1) + 1

        return x.T, pm


def run_test():
    class config:
        max_depth = 2
        adjmat_path = './adjmat_test'

        arr = np.array([[0,1,0,1,1],
                        [1,0,1,1,0],
                        [0,1,0,0,0],
                        [1,0,0,0,1],
                        [0,1,0,0,1]])
        savemat(adjmat_path+'.mat',{'adjmat':arr})

    class config2:
        max_depth = 5
        adjmat_path = './adjmat'
        # adjmat_path = '../BlogCatalog/adjmat'

    cfg = config2()
    gW = getSubGraph(cfg)
    for i in range(1,6):
        gW.get_sub(i)
        #print('\n\n--x, pm \n', gW.get_sub(i))

# run_test()