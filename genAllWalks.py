import numpy as np
from scipy.io import loadmat, savemat
import scipy
import time

class getWalks:

    def __init__(self, config):
        self.config = config
        self.max_depth = config.max_depth
        self.walks = {}

        try:
            print("---- Loading walks from path: %s"%(self.config.walks_dir))
            self.walks = np.load(config.walks_dir+'.npy').item()

        except(FileNotFoundError):
            print("---- No walks found, creating walks...")
            t0 = time.time()

            self.adjmat_depth = self.get_adj_depth()
            t1 = time.time()
            print("Depth Adjancency matrices created in %.5f"%(t1-t0))

            self.create_walks()
            print("All walks created in %.5f" % (time.time() - t1))

            np.save(config.walks_dir, self.walks)
            print("---- Walks saved to: %s"%(self.config.walks_dir))

    def get_adj_depth(self):
        self.adjmat = loadmat(self.config.adjmat_path)['adjmat']
        self.adjmat = self.adjmat + self.adjmat.T #make it symmetric
        adjmat_depth = {}

        # Keep in bool for both space and time efficiency
        adjmat_depth[1] = scipy.sparse.csr_matrix(self.adjmat, dtype='bool')
        adjmat_sum = adjmat_depth[1].copy().toarray() + np.eye(self.adjmat.shape[0])
        adjmat_pow = adjmat_depth[1].copy()

        """
        To get Nth depth nodes only:
        DN = Not(D0 or D1 or .... or D(n-1)) and D.pow(N)
        """
        for idx in range(2, self.max_depth + 1):
            adjmat_pow = adjmat_pow.dot(adjmat_depth[1])

            #Node is reachable in Nth power but wasn't reachable by any power less than N
            temp = np.logical_and(adjmat_pow.toarray(), np.logical_not(adjmat_sum))
            adjmat_depth[idx] = scipy.sparse.csr_matrix(temp, dtype='bool')

            #Keep track of which all nodes have already been assigned lesser depth
            adjmat_sum += temp

        return adjmat_depth

    def create_walks(self):
        total_walks = 0
        """Can be trivially parallelized for faster computation"""
        for idx in range(self.adjmat.shape[0]):
            self.walks[idx+1] = np.empty((0, self.max_depth + 1), int)
            self.genWalks(idx, seed = idx+1, path=[idx+1], curr_depth = 1, max_depth=self.max_depth)

            total_walks += len(self.walks[idx+1])
            if idx%1000 == 0:
                print("%d/%d"%(idx,self.adjmat.shape[0]))
        print("Total %d walks generated"%(total_walks))

    def genWalks(self, node_id, seed, path, curr_depth, max_depth):
        if curr_depth < (max_depth+1):
            _, seed_d_neighs = self.adjmat_depth[curr_depth].getrow(seed-1).nonzero()
            _, imm_neighs = self.adjmat_depth[1].getrow(node_id).nonzero()
            neighs = np.intersect1d(seed_d_neighs, imm_neighs)

            if len(neighs):
                for neigh in neighs:
                    self.genWalks(neigh, seed, np.insert(path, 0, neigh+1), curr_depth+1, max_depth)

            else:
                pad_size = max_depth - curr_depth + 1
                path = np.lib.pad(path, (pad_size, 0), 'constant', constant_values=(0)) #Pad in the beginnings

                # path = np.lib.pad(path, (0, pad_size), 'constant', constant_values=(0)) #Pad in the end
                self.walks[seed] = np.append(self.walks[seed], np.expand_dims(path, 0), axis=0)
                #print(self.walks[seed])
        else:
            self.walks[seed] = np.append(self.walks[seed], np.expand_dims(path, 0), axis=0)


def run_test():
    class config:
        max_depth = 2
        walks_dir = './temp/walks-D'+str(max_depth)
        adjmat_path = './temp/adjmat_test.mat'

        arr = np.array([[0,1,0,1],
                        [1,0,1,1],
                        [0,1,0,0],
                        [1,1,0,0]])

        savemat(adjmat_path,{'adjmat':arr})

    class config2:
        max_depth = 2
        walks_dir = '../Datasets/cora/walks/walks-D'+str(max_depth)
        adjmat_path = '../Datasets/cora/adjmat.mat'

    cfg = config2()
    gW = getWalks(cfg)
    #print(gW.walks)

#run_test()