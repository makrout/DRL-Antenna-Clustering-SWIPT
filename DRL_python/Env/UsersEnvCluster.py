# coding: utf-8

from itertools import *
import itertools
import random
import numpy as np
from scipy.stats import rice


class UsersEnvCluster:
    def __init__(self, M1, M2, snr_M1, snr_M2, n_c=2, H=None, res=10, fixed_channel=False):
        '''
        std_h: the standard deviation of the channel
        std_n: the standard deviation of the noise
        n_t: the number of antenna of the transmitter
        n_r: the number of antenna of the receiver
        P: the power of messages
        '''
        self.M1 = M1
        self.M2 = M2
        self.n_c = n_c
        self.res = res
        self.fixed_channel = fixed_channel

        self.action_dim = (self.M1 - 1) * (self.M2 - 1) + 1
        self.state_dim = (2,)
        self.snr_M1 = snr_M1
        self.snr_M2 = snr_M2

        self.power_M2 = None
        self.std_w_M1 = np.ones((self.M1, 1))

        self.S = None
        self.SINR1 = None
        self.SINR2 = None

        # if the channel is fixed, then H will be generated only once in the constructor
        if self.fixed_channel:
            if H is None:
                self.H = self.compute_Rayleigh_channel()
            else:
                print("Loading H from the passed parameter")
                self.H = H

        self.mapping_M1 = self.create_cluster_mapping(self.M1, self.n_c)
        self.mapping_M2 = self.create_cluster_mapping(self.M2, self.n_c)

        self.final_mapping = self.create_final_mapping(self.mapping_M1, self.mapping_M2)
        self.n_clusters = len(list(self.final_mapping.keys()))

    def __str__(self):
        return "Users Environment with:\n\t" + \
               "M1 : {}\n\tM2 : {}\n\t".format(self.M1, self.M2)

    @staticmethod
    def subsets(arr):
        return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

    def k_subset(self, arr, k):
        s_arr = sorted(arr)
        return list(set([i for i in combinations(self.subsets(arr), k) if sorted(chain(*i)) == s_arr]))

    def create_cluster_mapping(self, n_t, n_c):
        k_partitions = self.k_subset(range(0, n_t), n_c)
        mapping = {}
        for i in range(len(k_partitions)):
            mapping[i] = k_partitions[i]
        return mapping

    def create_final_mapping(self, mapping1, mapping2):
        cartesian_product = list(itertools.product(list(mapping1.values()), list(mapping2.values())))
        mapping = {}
        for k in range(len(cartesian_product)):
            mapping[k] = cartesian_product[k]
        return mapping

    def compute_c(self, H, c):
        c_coeff = []
        for cluster in c:
            c_coeff.append(np.sum(H[list(cluster)]))
        return c_coeff

    # Rayleigh channel
    def compute_Rayleigh_channel(self):
        h = np.random.normal(loc=0., scale=self.std_w_M1, size=(self.M2, self.M1))
        return h

    def create_Rice_channel(self, size_x, size_y, b=2):
        h = rice.rvs(b, size=(size_x, size_y))
        return h

    def compute_state(self, cluster_index=None, W=None):

        # if the channel is not fixed, then H will different at each interaction
        if not self.fixed_channel:
            self.H = self.compute_Rayleigh_channel()

        if cluster_index is None:
            cluster_index = random.choice(list(self.final_mapping.keys()))

        cluster_partition = self.final_mapping[cluster_index]

        cluster_partition_M1, cluster_partition_M2 = cluster_partition

        cluster_partition_M1_broadcasted_0 = [[e] for e in cluster_partition_M1[0]]
        cluster_partition_M1_broadcasted_1 = [[e] for e in cluster_partition_M1[1]]

        H_EH = self.H[cluster_partition_M1_broadcasted_0, cluster_partition_M2[0]]
        H_IT = self.H[cluster_partition_M1_broadcasted_1, cluster_partition_M2[1]]

        H_S1 = self.create_Rice_channel(H_IT.shape[0], H_EH.shape[1])
        H_S2 = self.create_Rice_channel(H_EH.shape[0], H_IT.shape[1])

        power_M1 = np.ones((H_EH.shape[1], 1)) * 10**(self.snr_M1 / 10) * 10**(-3)
        power_M1 /= len(power_M1)

        if self.power_M2 is None:
            self.power_M2 = np.ones((H_IT.shape[1], 1)) * 10**(self.snr_M2 / 10) * 10**(-3)
            self.power_M2 /= len(self.power_M2)
        else:
            self.power_M2 = np.ones((H_IT.shape[1], 1)) * self.SINR2 / self.M2

        # compute the dimension of every W
        dim_W_M1 = np.prod([H_EH.shape[1], H_EH.shape[1]])
        dim_W_M2 = np.prod([H_IT.shape[1], H_IT.shape[1]])

        # Beamforming matrix
        if W is None:
            W = np.random.rand((self.M1 - 1) * (self.M2 - 1) + 1,)

        W_M1 = W[0:dim_W_M1]
        W_M2 = W[dim_W_M1:dim_W_M2 + dim_W_M1]

        W_M1 = W_M1.reshape((H_EH.shape[1], H_EH.shape[1]))
        W_M2 = W_M2.reshape((H_IT.shape[1], H_IT.shape[1]))

        std_awgn_M1 = np.ones((H_IT.shape[0], 1))
        std_awgn_M2 = np.ones((H_EH.shape[0], 1))
        omega1 = np.diag(std_awgn_M1)
        omega2 = np.diag(std_awgn_M2)

        # Compute the SINR 2
        term3 = np.matmul(np.matmul(np.matmul(np.matmul(H_EH, W_M1), np.diag(np.squeeze(power_M1, axis=1))), W_M1.T), H_EH.T)

        self.power_M2 = np.minimum(self.power_M2, power_M1)

        term4 = omega2 + np.matmul(np.matmul(np.matmul(np.matmul(H_S2, W_M2), np.diag(np.squeeze(self.power_M2, axis=1))), W_M2.T), H_S2.T)
        self.SINR2 = np.trace(term3 + term4)

        # Compute the SINR 1
        term1 = omega1 + np.matmul(np.matmul(np.matmul(np.matmul(H_S1, W_M1), np.diag(np.squeeze(power_M1, axis=1))), W_M1.T), H_S1.T)
        term2 = np.matmul(np.matmul(np.matmul(np.matmul(H_IT, W_M2), np.diag(np.squeeze(self.power_M2, axis=1))), W_M2.T), H_IT.T)
        self.SINR1 = np.trace(np.matmul(np.linalg.pinv(term1), term2)) / self.M1

        self.S = np.array([self.SINR1, self.SINR2])
        return self.S.reshape(-1)

    def reset(self):
        s = self.compute_state()
        return s.reshape(-1)

    def fairness_reward(self):
        return np.mean(np.log(1 + self.S))

    def step(self, w_and_c):
        w = w_and_c['a']
        cluster_index = w_and_c['c']

        next_s = self.compute_state(cluster_index, w)
        r = self.fairness_reward()
        done = False
        info = None
        return next_s, r, done, info


if __name__ == "__main__":
    # An example with a random policy
    n_t = 2
    n_r = 2
    snr_M1, snr_M2 = 37, 37
    users_env = UsersEnvCluster(n_t, n_r, snr_M1, snr_M2)

    print("action_dim = ", users_env.action_dim)
    print("state_dim = ", users_env.state_dim)

    s = users_env.reset()
    done = False

    while not done:
        print("-" * 20)
        fake_action = {'a': np.random.rand(n_t, n_r).flatten(), 'c': 0}
        next_s, r, done, _ = users_env.step(fake_action)
        print("a = ", fake_action)
        print("new_state = ", next_s)
        print("reward = ", r)
        input("Press enter to continue ...")
