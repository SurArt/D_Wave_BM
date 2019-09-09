import numpy as np
import pandas as pd
from minorminer import find_embedding
import os

from dwave.system.samplers import DWaveSampler


DATA_DIRECTORY = 'data'

# def get_prob(d: dict(), search_value_i, search_value_j, i, j): # search_value takes value 1 or -1; (i,j) - list of qbits
#     prob = 0
#     for key, values in d.items():
#         if((values[i] == search_value_i) & (values[j] == search_value_j)):
#             prob += d['prob'][int(key)]
#     return prob
        


# def get_trace_of_second_term(ro: dict(), list_qb): 
# #ro = dict(0: [1, 1, -1, 1], 1: [1, -1, 1, 1], 2:[1, 1, 1, 1], 'prob': [0.2, 0.3, 0.5]),  
# #ro[i] is i result of computing with probability ro['prob'][i]    

#     """
#     >>> ro = {0: [1, 1, -1, 1], 1: [1, -1, 1, 1], 2:[1, 1, 1, 1], 'prob': [0.2, 0.3, 0.5]}
#     >>> list_qb = [(1, 2), (0, 2)]
#     >>> list_sig = (0, 1)
#     >>> get_trace_of_second_term(ro, list_qb)
#     {(1, 2): 0.0, (0, 2): 0.6000000000000001}
#     """
    

#     Tr = dict()
#     sigm_z_x2 = np.array([[1,  0,  0,  0],
#                           [0, -1,  0,  0],
#                           [0,  0, -1,  0],
#                           [0,  0,  0,  1]])

#     for k in list_qb:
#         n_00 = get_prob(ro, 1, 1, k[0], k[1])
#         n_01 = get_prob(ro, 1, -1, k[0], k[1])
#         n_10 = get_prob(ro, -1, 1, k[0], k[1])
#         n_11 = get_prob(ro, -1, -1, k[0], k[1])
#         RO_e = np.diag((n_00, n_01, n_10, n_11))
#         Tr[k] = np.trace(sigm_z_x2 @ RO_e)
#     return Tr


# def get_trace_of_first_term(ro: dict(), list_sig, num_qbits):
# #ro = dict(0: [1, 1, -1, 1], 1: [1, -1, 1, 1], 2:[1, 1, 1, 1], 'prob': [0.2, 0.3, 0.5]),  
# #ro[i] is i result of computing with probability ro['prob'][i] 

#     """
#     >>> ro = {0: [1, 1, -1, 1], 1: [1, -1, 1, 1], 2:[1, 1, 1, 1], 'prob': [0.2, 0.3, 0.5]}
#     >>> list_qb = [(1, 2), (0, 2)]
#     >>> list_sig = (0, 1)
#     >>> get_trace_of_second_term(ro, list_qb, 4)
#     {0: 1.0, 1: 0.4, 2: 0.6, 3: 1.0}
#     """



#     res = dict()
#     x = 0
#     for i in range(num_qbits):
#         for key, values in ro.items():
#             if(key != 'prob'):
#                 x += ro['prob'][key] * values[i]
#         res[i] = x
#         x = 0

#     return res


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
    diag_w = np.diag(np.diag(w))
    if w.shape[0] > w.shape[1]:
        diag_w = np.append(diag_w, np.zeros((w.shape[0] - w.shape[1], w.shape[1])), axis=0)
    elif w.shape[0] < w.shape[1]:
        raise NotImplementedError
    return np.tanh(np.array(w) @ np.array(vis_val_matrix) + diag_w @ (1 - np.array(vis_val_matrix))) @ P


def make_w_from_Ising(Ising_dict_biases, Ising_dict_weights, Num_bits): #required binary quadratic model with numeric variables
    w = np.zeros((Num_bits, Num_bits))
    for weights in Ising_dict_weights:
        w[weights[0]][weights[1]] = Ising_dict_weights[weights]
    for bias in Ising_dict_biases:
        w[bias][bias] = Ising_dict_biases[bias]
    return w


def get_trace_of_first_term(ro: pd.DataFrame, list_sig):
    return {
        it: ro[ro[it] == 1]['prob'].sum() - ro[ro[it] == -1]['prob'].sum()
        for it in list_sig
    }


def get_trace_of_second_term(ro: pd.DataFrame, list_qb):
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


def make_pat_with_pr_without_rep_from_pat():
    Data = pd.read_csv(os.path.join(DATA_DIRECTORY, 'Discr_sort_all.csv'), header=0, sep=';')
    Data.time = Data.time.apply(pd.to_datetime)
    Data['weekday'] = Data['time'].apply(pd.Timestamp.weekday)
    pat_columns = list(range(-1, 13))
    pat_columns[0] = 'weekday'
    pat_columns[13] = 'time'

    pat_columns_pr = list(range(-1, 14))
    pat_columns_pr[0] = 'weekday'
    pat_columns_pr[14] = 'prob'
    pat_columns_pr[13] = 'flag'

    pattern = pd.read_csv(os.path.join(DATA_DIRECTORY, 'pattern.csv'), header=0, sep=';')
    del pattern['time']
    pattern['flag'] = 0

    new_pat_dict = dict()
    dict_str = []
    new_pattern = pd.DataFrame(columns=pat_columns)
    del new_pattern['time']
    for num in range(27468):
        column = Data[(Data.index >= num) & (Data.index < num + 12)].value
        line = {i: column[i + num] for i in range(12)}
        line['weekday'] = Data.weekday[num]
        new_pattern = pattern[
            (pattern['0'] == line[0]) & (pattern['1'] == line[1]) & (pattern['2'] == line[2]) &
            (pattern['3'] == line[3]) & (pattern['4'] == line[4]) & (pattern['5'] == line[5]) &
            (pattern['6'] == line[6]) & (pattern['7'] == line[7]) & (pattern['8'] == line[8]) &
            (pattern['9'] == line[9]) & (pattern['10'] == line[10]) & (pattern['11'] == line[11]) &
            (pattern['weekday'] == line['weekday']) & (pattern['flag'] == 0)].copy(deep=True)

        value_of_str = len(new_pattern)
        for index, row in new_pattern.iterrows():
            pattern.at[index, 'flag'] = 1
            new_pattern.at[index, 'prob'] = value_of_str

        if not new_pattern[:1].empty:
            dict_str.append(dict(zip(
                pat_columns_pr, np.array(new_pattern[:1]).reshape(15, ).tolist()
            )))
    new_pattern_end = pd.DataFrame(dict_str)
    new_pattern_end.to_csv(
        os.path.join(DATA_DIRECTORY, 'pat_with_pr_without_rep.csv'),
        sep=';', header=True, index=False
    )

    return new_pattern_end


if __name__ == '__main__':
    import doctest
    doctest.testmod()


