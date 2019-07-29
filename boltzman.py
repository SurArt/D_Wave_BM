from dimod import BinaryQuadraticModel, SPIN

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite, EmbeddingComposite
from dimod import SimulatedAnnealingSampler, Structured

from output import get_response
from config import TOKEN, ENDPOINT, SOLVER


class Boltzman:
    def __init__(self, weights=None, biases=None, v_in=None, v_out=None, v_all=None,
                 embedding=None, num_reads=100, sampler=None, gradient_velocity=0.1):
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
        dwave_parameters = dwave_parameters if dwave_parameters is not None else {}
        fixed_var = fixed_var if fixed_var is not None else {}
        if 'num_reads' not in dwave_parameters:
            dwave_parameters['num_reads'] = self.num_reads

        BQM_weights = BinaryQuadraticModel(self.biases, self.weights, offset=0, vartype=SPIN)
        for var, value in fixed_var:
            BQM_weights.fix_variable(var, value)

        response_bm = self.sampler.sample(BQM_weights, **dwave_parameters)
        return get_response(response_bm, qubits=output_qubits)


class RestrictedBoltzman(Boltzman):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        v_hidden = list(filter(lambda x: x not in [*self.v_in, *self.v_all], self.v_all))

        for coupling in self.weights:
            if coupling[0] in v_hidden and coupling[1] in v_hidden:
                raise Exception("It's not a restricted Boltzman machine")
