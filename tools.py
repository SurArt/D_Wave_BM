import numpy as np
import pandas as pd
from minorminer import find_embedding

def make_RBM_topology(num_in, num_vis):  #10, 187

    RBM =[]
    for i in range(num_in):
        for j in range(num_in,num_vis+num_in):
            RBM.append((i,j))
           
    solver = DWaveSampler()
    qbits = solver.edgelist
    return find_embedding(RBM, qbits)        

def make_aver_sigm_v(w, vis_val): #w - array with Ising coeff. , vis_val - vector wint shape (N,1)
    n = len(w)
    if(n > len(vis_val)):
        vector = np.zeros(n).reshape(n,1)
        vector[:n - 1] = vis_val
    else:
        vector = vis_val
    return np.tanh(np.array(w) @ np.array(vector) + np.diag(w) @ (1 - np.array(vector))) # tanh((w - diag(w))*vect +diag(w)*vect(1, 1, ..., 1))

def make_aver_sigm_v_respect_to_prob(w, vis_val_matrix, P): #w(dim_syst, dim_syst) - array with Ising coeff, vis_val_matrix(dim_syst, 25435) - array with columns aver_sigm_v, P(25435, 1) - vector with probability of data sets
    return np.tanh(np.array(w) @ np.array(vis_val_matrix) + np.diag(np.diag(w)) @ (1 - np.array(vis_val_matrix))) @ P

def make_w_from_Ising(Ising_dict_biases, Ising_dict_weights, Num_bits): #required binary quadratic model with numeric variables
    w = np.zeros((Num_bits, Num_bits))
    for weights in Ising_dict_weights:
        w[weights[0]][weights[1]] = Ising_dict_weights[weights]
    for bias in Ising_dict_biases:
        w[bias][bias] = Ising_dict_biases[bias]
    return w


def get_trace_of_first_term(ro: pd.DataFrame, list_sig):
    """
    >>> ro = pd.DataFrame({'prob': [0.4, 0.2, 0.2, 0.2], 1: [1, 1, 1, -1], 2: [1, -1, -1, -1], 'a': [1, 1, -1, -1]})
    >>> list_sig = ['a']
    >>> round(get_trace_of_first_term(ro, list_sig)[list_sig[0]], 3)
    0.2
    """
    return {
        it: ro[ro[it] == 1]['prob'].sum() - ro[ro[it] == -1]['prob'].sum()
        for it in list_sig
    }


def get_trace_of_second_term(ro: pd.DataFrame, list_qb):
    """
    >>> ro = pd.DataFrame({'prob': [0.4, 0.2, 0.2, 0.2], 1: [1, 1, 1, -1], 2: [1, -1, -1, -1], 'a': [1, 1, -1, -1]})
    >>> list_qb = [(1, 2)]
    >>> round(get_trace_of_second_term(ro, list_qb)[list_qb[0]], 3)
    0.2
    >>> list_qb = [(2, 'a')]
    >>> round(get_trace_of_second_term(ro, list_qb)[list_qb[0]], 3)
    0.6
    >>> list_qb = [(1, 2), (2, 'a')]
    >>> round(get_trace_of_second_term(ro, list_qb)[list_qb[0]], 3)
    0.2
    """
    Tr = dict()
    sigm_z_x2 = np.array([[1,  0,  0,  0],
                          [0, -1,  0,  0],
                          [0,  0, -1,  0],
                          [0,  0,  0,  1]])

    for k in list_qb:
        n_00 = ro[(ro[k[0]] == 1) & (ro[k[1]] == 1)]['prob'].sum()
        n_01 = ro[(ro[k[0]] == 1) & (ro[k[1]] == -1)]['prob'].sum()
        n_10 = ro[(ro[k[0]] == -1) & (ro[k[1]] == 1)]['prob'].sum()
        n_11 = ro[(ro[k[0]] == -1) & (ro[k[1]] == -1)]['prob'].sum()
        RO_e = np.diag((n_00, n_01, n_10, n_11))
        Tr[k] = np.trace(sigm_z_x2 @ RO_e)
    return Tr


if __name__ == '__main__':
    import doctest
    doctest.testmod()


