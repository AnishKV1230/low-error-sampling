import decoder_bench
from decoder_bench.generator import gen_surface_circuit
from decoder_bench.dataset_gen import DatasetGen

# Generate surface code circuit
gen_surface_circuit((7, 0.001, 'z'))

# Or use the classes directly
import stim
from decoder_bench.dataset_gen import DatasetGen
from decoder_bench.common.noise import NoiseModel

circuit = stim.Circuit.generated('surface_code:rotated_memory_z', 
                                 distance=7, rounds=7)
circuit = NoiseModel.SI1000(0.001).noisy_circuit(circuit)
generator = DatasetGen(circuit, matchable=True, name='my_dataset')
generator.gen_syndromes(max_iterations=100, num_records=10000)

