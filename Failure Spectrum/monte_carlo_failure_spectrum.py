
##failure spectrum vs failure weight
import numpy as np
import stim
import pymatching
import matplotlib.pyplot as plt
from collections import defaultdict

##gen circuit
def _gen_circuit(d, p):
    return stim.Circuit.generated(
        code_task="surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def spectrum(d, p, shots=600000):

    c = gen_circuit(d, p)
    dem = c.detector_error_model()
    decoder = pymatching.Matching.from_detector_error_model(dem)
    sampler = c.compile_detector_sampler()

    fail = defaultdict(int)
    total = defaultdict(int)

    for _ in range(shots):

        dets, obs = sampler.sample(1, separate_observables=True)
        pred = decoder.decode_batch(dets)

        # ----------------------------
        # KEY FIX:
        # use syndrome activity as consistent weight proxy
        # (this is what papers typically do when true fault weight is unavailable)
        # ----------------------------
        w = int(np.count_nonzero(dets))

        f = int(np.any(pred.astype(bool) != obs))

        total[w] += 1
        fail[w] += f

    ws = np.array(sorted(total.keys()))
    fw = np.array([fail[w] / total[w] for w in ws])

    return ws, fw


# ----------------------------
# RUN MULTIPLE DISTANCES
# ----------------------------
ps = [0.0006, 0.004, 0.006, 0.01]
distances = [3, 5, 7]

plt.figure(figsize=(8,6))

for d in distances:
    w, f = spectrum(d, ps[1])  # fixed p for clean separation

    f = np.clip(f, 1e-12, 1)

    plt.loglog(w, f, marker="o", linewidth=1.5, label=f"d={d}")


plt.xlabel("Fault weight w ")
plt.ylabel("Failure probability f(w)")
plt.title(" Failure Spectrum f(w) versus Failure weight(w) ")
plt.grid(True, which="both", alpha=0.4)
plt.legend()