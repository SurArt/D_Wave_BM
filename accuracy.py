#!/usr/bin/env python3

import pandas as pd
from sampler import NUMBER_TICKS, DISCR_SIZE
from boltzman import RestrictedBoltzman

from tqdm import tqdm
from math import exp, cosh, log


def accuracy(boltzman: RestrictedBoltzman, data: pd.DataFrame):
        from tqdm import tqdm
        for row in tqdm(list(data.drop(columns=['prob']).itertuples())):
            vis_energy = 0
            vis_spins = list(row)[1:]
            for i in range(boltzman.number_vis):
                vis_energy += vis_spins[i] * boltzman.biases.get(i, 0)
            prob = exp(-vis_energy)
            for i in range(boltzman.number_all - boltzman.number_vis):
                b_eff = boltzman.biases[i]
                for j in range(boltzman.number_vis):
                    if (i, j) in boltzman.weights:
                        b_eff += vis_spins[j] * boltzman.weights[(i, j)]
                    elif (j, i) in boltzman.weights:
                        b_eff += vis_spins[j] * boltzman.weights[(j, i)]
                prob *= cosh(b_eff)

            data.at[row.Index, 'exp_prob'] = prob

        data['exp_prob'] = data['exp_prob'] / sum(data['exp_prob'])
        data['log-likelihood'] = data.apply(lambda row: -row.prob*log(row.exp_prob), axis=1)
        kl = sum(data['log-likelihood'])

        del data['log-likelihood']
        del data['exp_prob']
        return kl


if __name__ == "__main__":
    data = pd.read_csv('bits_with_prob.csv', sep=';', header=0)
    v_in = [i for i in range(187)]
    v_all = v_in + [i for i in range(187, 187 + 10)]
    weights = {}
    for k2 in range(len(v_in), len(v_all)):
        for k1 in range(len(v_in)):
            weights[(k1, k2)] = 0
    biases = {k: 0 for k in v_all}
    rbm = RestrictedBoltzman(
        weights=weights, biases=biases, v_in=v_in,
        v_all=v_all
    )
    print(accuracy(rbm, data))
    print(data)
