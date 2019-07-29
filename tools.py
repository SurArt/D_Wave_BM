import numpy as np
import pandas as pd


# def make_aver_sigm_v(w, vis_val): #w - QUBO array, pat - vector wint shape (N,1)
#     n = len(w)
#     if n > len(vis_val):
#         vector = np.zeros(n).reshape(n,1)
#         vector[:n - 1] = vis_val
#     else:
#         vector = pat
#     return np.tanh(np.array(w) @ np.array(vector) + np.diag(w) @ (1 - np.array(vector))) # tanh((w - diag(w))*vect +diag(w)*vect(1, 1, ..., 1))


# def make_aver_sigm_v_respect_to_prob(w, vis_val_matrix, P): #w(dim_syst, dim_syst) - QUBO array, vis_val_matrix(dim_syst, 25435) - array with columns aver_sigm_v, P(25435, 1) - vector with probability of data sets
#     return np.tanh(np.array(w) @ np.array(vis_val_matrix) + np.diag(np.diag(w)) @ (1 - np.array(vis_val_matrix))) @ P


def form_Trace(ro: pd.DataFrame, list_qb):
    """
    >>> ro = pd.DataFrame({'prob': [0.4, 0.2, 0.2, 0.2], '1': [1, 1, 1, -1], '2': [1, -1, -1, -1], 'a': [1, 1, -1, -1]})
    >>> list_qb = [('1', '2')]
    >>> round(form_Trace(ro, list_qb)[list_qb[0]], 3)
    0.2
    >>> list_qb = [('2', 'a')]
    >>> round(form_Trace(ro, list_qb)[list_qb[0]], 3)
    0.6
    >>> list_qb = [('1', '2'), ('2', 'a')]
    >>> round(form_Trace(ro, list_qb)[list_qb[0]], 3)
    0.2
    """
    RO = ro
    Tr = dict()
    sigm_z_x2 = np.array([[1,  0,  0,  0],
                          [0, -1,  0,  0],
                          [0,  0, -1,  0],
                          [0,  0,  0,  1]])
    list_qb = list(map(lambda x: (str(x[0]), str(x[1])), list_qb))

    for k in list_qb:
        n_00 = RO[(RO[k[0]] == 1) & (RO[k[1]] == 1)]['prob'].sum()
        n_01 = RO[(RO[k[0]] == 1) & (RO[k[1]] == -1)]['prob'].sum()
        n_10 = RO[(RO[k[0]] == -1) & (RO[k[1]] == 1)]['prob'].sum()
        n_11 = RO[(RO[k[0]] == -1) & (RO[k[1]] == -1)]['prob'].sum()
        RO_e = np.diag((n_00, n_01, n_10, n_11))
        Tr[k] = np.trace(sigm_z_x2 @ RO_e)
    return Tr


if __name__ == '__main__':
    import doctest
    doctest.testmod()


