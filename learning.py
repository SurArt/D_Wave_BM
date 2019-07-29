#!/usr/bin/env python3

from boltzman import Boltzman, RestrictedBoltzman
from tools import get_trace_of_second_term, get_trace_of_first_term

import pandas as pd


def set_schedule():
    return


def learn(boltzman: Boltzman, dwave_parameters=None):
    dwave_parameters = dwave_parameters if dwave_parameters is not None else {}
    if 'anneal_schedule' not in dwave_parameters and set_schedule() is not None:
        dwave_parameters['anneal_schedule'] = set_schedule()

    if isinstance(boltzman, RestrictedBoltzman):
        results = boltzman.run(dwave_parameters)
        p_ro = pd.DataFrame(columns=['prob', *results[0]['results'].keys()])
        for result in results:
            p_ro = p_ro.append(pd.DataFrame(
                {'prob': result['occurrences']/100, **result['results']},
                index=[0]
            ), ignore_index=True)
        unclamped_first_term_trace = get_trace_of_first_term(p_ro, boltzman.v_all)
        unclamped_second_term_trace = get_trace_of_second_term(p_ro, boltzman.weights.keys())

    else:
        raise NotImplementedError

if __name__ == '__main__':
    rbm = RestrictedBoltzman(weights={(0, 1): -2, (0, 2): 0, (1, 2): 0}, biases={0: 1, 1: 1, 2: -1},
                             v_in=[0, 2])
    rbm.num_reads = 5000
    learn(rbm)
