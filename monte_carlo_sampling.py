##Logical vs Phsyical Error Rate
import stim
import sinter
import pymatching
import numpy as np
import matplotlib.pyplot as plt

##gen circuit
def gen_circuit(d, p):
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )
    return sinter.Task(
        circuit=circuit,
        decoder="pymatching",
        json_metadata={"d": d, "p": p},
    )


##Monte carlo
def run_sweep(distances=(3, 5, 7), n_grid=20, shots_per_point=1_00_000, max_workers=10):
    ps = np.logspace(-4, -1, n_grid)

    tasks = [gen_circuit(d, p) for d in distances for p in ps]

    print(f"Running {len(tasks)} tasks...")

    stats = sinter.collect(
        tasks=tasks,
        num_workers=max_workers,
        max_shots=shots_per_point,  
        max_errors=None,            
    )

    
    results = {d: [None] * n_grid for d in distances}
    p_to_idx = {round(p, 15): i for i, p in enumerate(ps)}

    for s in stats:
        d = s.json_metadata["d"]
        p = s.json_metadata["p"]
        ler = s.errors / s.shots
        results[d][p_to_idx[round(p, 15)]] = ler
        print(f"d={d}  p={p:.2e}  LER={ler:.3e}")

    return ps, results

##plot
def plot(ps, results):
    fig, ax = plt.subplots(figsize=(8, 6))
    styles = {3: ("-", "o"), 5: ("--", "s"), 7: (":", "^")}

    for d, ys in results.items():
        ls, mk = styles[d]
        ps_plot = [p for p, y in zip(ps, ys) if y and y > 0]
        ys_plot = [y for y in ys if y and y > 0]

        ax.loglog(ps_plot, ys_plot,
                  linestyle=ls, marker=mk,
                  linewidth=1.5, markersize=4,
                  label=f"d={d}")

    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(1e-8, 1)
    ax.set_xlabel("Physical error rate")
    ax.set_ylabel("Logical error rate")
    ax.set_title("Monte Carlo Logical versus Physical Error Rate")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    plt.show()


#runs
ps, results = run_sweep()
plot(ps, results)