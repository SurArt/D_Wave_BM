from unittest import TestCase, main

import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

from tools import get_trace_of_first_term, get_trace_of_second_term, make_w_from_Ising
from boltzman import Boltzman


class TestTools(TestCase):
    def test_get_trace_of_first_term(self):
        ro = pd.DataFrame({'prob': [0.4, 0.2, 0.2, 0.2], 1: [1, 1, 1, -1], 2: [1, -1, -1, -1],
                           'a': [1, 1, -1, -1]})
        list_sig = ['a']
        self.assertAlmostEqual(round(get_trace_of_first_term(ro, list_sig)[list_sig[0]], 3), 0.2)

    def test_get_trace_of_second_term(self):
        ro = pd.DataFrame({'prob': [0.4, 0.2, 0.2, 0.2], 1: [1, 1, 1, -1], 2: [1, -1, -1, -1],
                           'a': [1, 1, -1, -1]})
        list_qb = [(1, 2)]
        self.assertAlmostEqual(round(get_trace_of_second_term(ro, list_qb)[list_qb[0]], 3), 0.2)

        list_qb = [(2, 'a')]
        self.assertAlmostEqual(round(get_trace_of_second_term(ro, list_qb)[list_qb[0]], 3), 0.6)

        list_qb = [(1, 2), (2, 'a')]
        self.assertAlmostEqual(round(get_trace_of_second_term(ro, list_qb)[list_qb[0]], 3), 0.2)

    def test_make_w_from_Ising(self):
        boltzman = Boltzman(biases={0: 1, 1: 2}, weights={(0, 1): 3})
        assert_array_equal(
            make_w_from_Ising(boltzman.biases, boltzman.weights, 2),
            np.array([[1, 3], [0, 2]])
        )


if __name__ == '__main__':
    main()
