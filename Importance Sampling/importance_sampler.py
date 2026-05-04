##Estimates Logical vs Phsyical Error Rate using importance sampling
import numpy as np
import stim
import pymatching
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

###generate circuit surface code: rotated memory x
def gen_circuit(d, p):
    return stim.Circuit.generated(
        code_task="surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def run_job(args):
    d, p, min_failures_base, max_shots, adaptive_scale = args

    bias = 3.0 ## importance bias
    p_bias = min(0.25, p * bias)

    circuit = gen_circuit(d, p_bias)
    dem = circuit.detector_error_model()
    decoder = pymatching.Matching.from_detector_error_model(dem) ## uses py matching for decoder
    sampler = circuit.compile_detector_sampler()

    n_sites = 2000 * d

    def log_weight(k, pb):
        return (
            k * np.log(p / pb)
            + (n_sites - k) * np.log((1 - p) / (1 - pb))
        )

    pth = 0.006
    scale = adaptive_scale + (1 - adaptive_scale) * min(1.0, pth / p)
    min_failures = max(10, int(min_failures_base * scale))

    shots = 50_000

    weighted_failures = 0.0
    weighted_total = 0.0
    total_shots = 0

    while True:
        dets, obs = sampler.sample(shots, separate_observables=True)
        preds = decoder.decode_batch(dets)

        failures = np.any(preds.astype(bool) != obs, axis=1)

        ###weighting 
        k_est = n_sites * p_bias
        w = np.exp(log_weight(k_est, p_bias))

        weighted_failures += np.sum(failures) * w
        weighted_total += shots * w
        total_shots += shots

        if weighted_failures >= min_failures or total_shots >= max_shots:
            break

        shots = min(shots * 2, max_shots - total_shots)
        if shots <= 0:
            break

    ##estimator(weighted)
    ler = weighted_failures / weighted_total if weighted_total > 0 else 0.0

    return d, p, ler

def run_sweep(
    distances=(3, 5, 7),
    n_grid=20,
    min_failures=100,
    max_shots=2_00_000, ## larger than 1 million for sharpness
    adaptive_scale=0.5,
    max_workers=10,
):
    ps = np.logspace(-5, -1, n_grid) ## shows zig zags past 10e-5, included to show

    jobs = [
        (d, p, min_failures, max_shots, adaptive_scale)
        for d in distances
        for p in ps
    ]

    results = {d: [None] * n_grid for d in distances}
    p_to_idx = {round(p, 15): i for i, p in enumerate(ps)}

    total = len(jobs)
    print(f"Running {total} jobs across {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as ex: ## used to make it easier on my computer or anyones for that matter
        futures = {ex.submit(run_job, job): job for job in jobs}

        done = 0
        for fut in as_completed(futures):
            d, p, ler = fut.result()
            results[d][p_to_idx[round(p, 15)]] = ler
            done += 1
            print(f"  [{done}/{total}]  d={d}  p={p:.2e}  LER={ler:.3e}")

    return ps, results


def plot(ps, results):
    fig, ax = plt.subplots(figsize=(8, 6))
    styles = {3: ("-", "o"), 5: ("--", "s"), 7: (":", "^")}

    for d, ys in results.items():
        ls, mk = styles[d]
        ys = np.clip(np.array(ys, dtype=float), 1e-16, None)

        ax.loglog(ps, ys, linestyle=ls, marker=mk,
                  linewidth=1.5, markersize=4, label=f"d={d}")

    ax.set_xlim(1e-5, 1e-1)
    ax.set_ylim(1e-8, 1)
    ax.set_xlabel("Physical error rate")
    ax.set_ylabel("Logical error rate (importance-sampled)")
    ax.set_title("Logical vs Physical Error Rate Importance Sampling")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()


ps, results = run_sweep()
plot(ps, results)