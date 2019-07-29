#!/usr/bin/env python3

from boltzman import Boltzman, RestrictedBoltzman


def set_schedule():
    return


def learn(boltzman: Boltzman, dwave_parameters=None):
    dwave_parameters = dwave_parameters if dwave_parameters is not None else {}
    if 'anneal_schedule' not in dwave_parameters and set_schedule() is not None:
        dwave_parameters['anneal_schedule'] = set_schedule()

    if isinstance(boltzman, RestrictedBoltzman):
        results = boltzman.run(dwave_parameters)
        print(results)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    rbm = RestrictedBoltzman(weights={('0', '1'): -2}, biases={'0': 1, '1': 1}, v_in=['0', '1'])
    learn(rbm)
