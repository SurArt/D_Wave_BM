from dimod import BinaryQuadraticModel, SPIN

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite, EmbeddingComposite
from dimod import SimulatedAnnealingSampler, Structured

from output import get_response
from config import TOKEN, ENDPOINT, SOLVER

from math import exp


class Boltzman:
    def __init__(self, weights=None, biases=None, v_in=None, v_out=None, v_all=None,
                 embedding=None, num_reads=100, sampler=None, gradient_velocity=0.1):
        """
        :param weights: dict, связь между битами, например {(0, 1): 2}
        :param biases: dict, линейный коэффициент, например {0: 1, 1: 1}
        :param v_in: list, видимые биты, кодирующие вектор признаков
        :param v_out: list, видимые биты, кодирующие выходное значение
        :param v_all: list, все биты
        :param embedding: dict, карта соответствия битов в машине больцмана и кубитов в D-wave
        :param num_reads: максимальное количество итераций при обучении
        :param sampler: объект, осуществяющий отжиг, например SimulatedAnnealingSampler()
        :param gradient_velocity: длина шага для обновления весов
        """
        self.weights = weights if weights is not None else {}  # {(i, j): 1}
        self.biases = biases if biases is not None else {}  # {i: 1, j: 2}

        self.v_in = v_in if v_in is not None else []
        self.v_out = v_out if v_out is not None else []
        self.v_all = v_all if v_all is not None else self.v_in + self.v_out

        if any(v not in self.v_all for v in [*self.v_in, *self.v_out]):
            raise Exception('v_in and v_out must be in v_all')

        self.number_in = len(self.v_in)
        self.number_out = len(self.v_out)
        self.number_all = len(self.v_all)
        self.number_vis = self.number_in + self.number_out

        self.embedding = embedding  # None or tuple
        self.num_reads = num_reads
        self.sampler = sampler if sampler is not None else SimulatedAnnealingSampler()
        self.gradient_velocity = gradient_velocity

        if isinstance(self.sampler, Structured):
            if self.embedding is None:
                self.sampler = EmbeddingComposite(self.sampler)
            else:
                self.sampler = FixedEmbeddingComposite(self.sampler, self.embedding)

    def run(self, dwave_parameters=None, fixed_var=None, output_qubits=None):
        """
        :param dwave_parameters: dict, parameters for sampler
        :param fixed_var: dict, переменные, для которых известны значения
        :param output_qubits:
        :return: [{'energy': energy, 'results': {0: 1, 2: -1, ...}, 'occurrences': ...}, ...]
        """
        dwave_parameters = dwave_parameters if dwave_parameters is not None else {}
        fixed_var = fixed_var if fixed_var is not None else {}
        if 'num_reads' not in dwave_parameters:
            dwave_parameters['num_reads'] = self.num_reads

        BQM_weights = BinaryQuadraticModel(self.biases, self.weights, offset=0, vartype=SPIN)
        for var, value in fixed_var:
            BQM_weights.fix_variable(var, value)

        response_bm = self.sampler.sample(BQM_weights, **dwave_parameters)
        return get_response(response_bm, qubits=output_qubits)

    def get_prob(self, spins: list, partition_function=1):
        prob = 0
        for i in range(len(self.biases)):
            prob += spins[i] * self.biases.get(i, 0)

        for i in range(len(self.weights)):
            for j in range(i + 1, len(self.weights)):
                if (i, j) in self.weights:
                    prob += spins[i] * spins[j] * self.weights[(i, j)]
                elif (j, i) in self.weights:
                    prob += spins[i] * spins[j] * self.weights[(j, i)]

        return exp(-prob)/partition_function


class RestrictedBoltzman(Boltzman):
    """
    Машина больцмана, в котором биты из скрытого слоя не имеют связей
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        v_hidden = list(filter(lambda x: x not in [*self.v_in, *self.v_all], self.v_all))

        for coupling in self.weights:
            if coupling[0] in v_hidden and coupling[1] in v_hidden:
                raise Exception("It's not a restricted Boltzman machine")
