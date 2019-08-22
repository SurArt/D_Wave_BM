#!/usr/bin/env python3

from boltzman import Boltzman, RestrictedBoltzman
from tools import get_trace_of_second_term, get_trace_of_first_term
from tools import make_aver_sigm_v_respect_to_prob, make_w_from_Ising, DATA_DIRECTORY
from accuracy import accuracy

import pandas as pd
import os
import numpy as np

from tqdm import tqdm


def set_schedule(anneal_time=100):
    return [(0, 0), (0.9, anneal_time/2), (1, anneal_time)]


def learn(boltzman: Boltzman, dwave_parameters=None, num_steps=None,
          rho_distribution_file='rho_distribution.csv'):
    dwave_parameters = dwave_parameters if dwave_parameters is not None else {}
    if 'anneal_schedule' not in dwave_parameters and set_schedule() is not None:
        dwave_parameters['anneal_schedule'] = set_schedule(1000)

    data = pd.read_csv(os.path.join(DATA_DIRECTORY, 'bits_with_prob.csv'), header=0, sep=';')
    prob = data['prob'].values
    prob.shape = (prob.shape[0], 1)  # transpose
    prob /= np.sum(prob)

    vis_val_matrix = np.transpose(data.drop('prob', 1).values)

    if isinstance(boltzman, RestrictedBoltzman):
        max_delta = 1000
        steps = 0
        while max_delta > 0.1:
            max_delta = 0
            steps += 1
            print(f"Iteration number: {steps}. ")
            if num_steps is not None and steps > num_steps:
                break

            results = boltzman.run(dwave_parameters)
            p_ro = pd.DataFrame()
            for result in results:
                p_ro = p_ro.append(pd.DataFrame(
                    {'prob': result['occurrences']/100, **result['results']},
                    index=[0]
                ), ignore_index=True, sort=False)
            p_ro.rename(columns={
                i: item
                for i, item in enumerate(data.drop(columns='prob').columns)
            }).to_csv(rho_distribution_file, index=False)
            unclamped_first_term_trace = get_trace_of_first_term(p_ro, boltzman.v_all)
            unclamped_second_term_trace = get_trace_of_second_term(p_ro, boltzman.weights.keys())
            w = make_w_from_Ising(boltzman.biases, boltzman.weights, boltzman.number_all)
            clamped_first_term = make_aver_sigm_v_respect_to_prob(
                w[:, :boltzman.number_in + boltzman.number_out],
                vis_val_matrix, prob
            )
            for i in boltzman.biases:
                # TODO clamped - np.array, unclamped - dict
                delta = (clamped_first_term[i] - unclamped_first_term_trace[i])[0]
                boltzman.biases[i] += boltzman.gradient_velocity*delta
                max_delta = max(max_delta, abs(delta))

            for k1 in range(len(v_in)):
                clamped_first_term = make_aver_sigm_v_respect_to_prob(
                    w[:, :boltzman.number_in + boltzman.number_out],
                    vis_val_matrix,
                    vis_val_matrix[k1, :].reshape((vis_val_matrix.shape[1], 1)) * prob
                )
                for k2 in range(len(v_in), len(v_all)):
                    if (k1, k2) in boltzman.weights or (k2, k1) in boltzman.weights:
                        coupling = (k1, k2) if (k1, k2) in boltzman.weights else (k2, k1)
                    else:
                        continue
                    delta = (clamped_first_term[k2] - unclamped_second_term_trace[coupling])[0]
                    boltzman.weights[coupling] += boltzman.gradient_velocity*delta
                    max_delta = max(max_delta, abs(delta))
            print(f"Current accuracy: {accuracy(boltzman, data)}")
        return boltzman
    else:
        raise NotImplementedError

if __name__ == '__main__':
    from dwave.system.samplers import DWaveSampler
    from config import ENDPOINT, TOKEN, SOLVER
    from topology import EMBEDDING

    sampler = DWaveSampler(endpoint=ENDPOINT, token=TOKEN, solver=SOLVER)
    v_in = [i for i in range(187)]
    v_all = v_in + [i for i in range(187, 187 + 10)]
    weights = {}
    for k2 in range(len(v_in), len(v_all)):
        for k1 in range(len(v_in)):
            weights[(k1, k2)] = 0
    biases = {k: 0 for k in v_all}
    rbm = RestrictedBoltzman(
        weights=weights, biases=biases, v_in=v_in,
        v_all=v_all, sampler=sampler, embedding=EMBEDDING
    )

    # rbm = RestrictedBoltzman(
    #     weights={(0, 1): -2, (0, 2): 0, (1, 2): 0}, biases={0: 1, 1: 1, 2: -1},
    #                          v_in=[0, 2], v_all=[0, 1, 2])
    rbm.num_reads = 5000
    try:
        rbm = learn(rbm, num_steps=2)
    finally:
        print(rbm.biases)
        print(rbm.weights)

