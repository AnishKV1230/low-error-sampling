
# Estimates failure spectrum vs failure weight using importance sampling.
# This method is more efficient than direct Monte Carlo sampling for higher weights, as it biases the sampling towards more likely failure events and applies appropriate reweighting to get unbiased estimates.
import numpy as np
import stim
import pymatching
import matplotlib.pyplot as plt
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import linregress


def make_circuit(d, p):
    return stim.Circuit.generated(
        code_task="surface_code:rotated_memory_x",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def wilson_interval(successes_w, n_w, z=1.96):
    """Wilson interval adapted for weighted counts."""
    if n_w <= 0:
        return 0.0, 1.0
    p_hat = min(successes_w / n_w, 1.0)
    denom = 1 + z**2 / n_w
    centre = (p_hat + z**2 / (2 * n_w)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n_w + z**2 / (4 * n_w**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def run_job(args):
    d, p, shots_total, bias = args

    p_bias = min(0.25, p * bias)

    circuit = make_circuit(d, p_bias)
    dem = circuit.detector_error_model()
    decoder = pymatching.Matching.from_detector_error_model(dem)
    dem_sampler = dem.compile_sampler()

    n_mechanisms = dem.num_errors

    # Accumulate per-weight-bin weighted failure/total counts
    fail_w  = defaultdict(float)
    total_w = defaultdict(float)

    remaining = shots_total
    batch = 50_000

    while remaining > 0:
        shots = min(batch, remaining)
        remaining -= shots

        dets, obs, faults = dem_sampler.sample(shots, return_errors=True)
        preds = decoder.decode_batch(dets)

        failures = np.any(preds.astype(bool) != obs, axis=1)
        k_vec = faults.sum(axis=1).astype(int)

        # Per-shot IS reweighting
        log_w = (
            k_vec * np.log(p / p_bias)
            + (n_mechanisms - k_vec) * np.log((1 - p) / (1 - p_bias))
        )
        log_w = np.clip(log_w, -500, 500)
        w = np.exp(log_w)

        for k, wi, fi in zip(k_vec, w, failures):
            total_w[k] += wi
            if fi:
                fail_w[k] += wi

    return d, fail_w, total_w


def compute_spectrum(fail_w, total_w, min_weight_total=10.0):
    ws, fw, lo, hi = [], [], [], []
    upper_bounds = []

    for w in sorted(total_w):
        tw = total_w[w]
        fw_val = fail_w.get(w, 0.0)
        if tw < min_weight_total:
            continue
        if fw_val == 0.0:
            _, u = wilson_interval(0, tw)
            upper_bounds.append((w, u))
        else:
            f = fw_val / tw
            l, u = wilson_interval(fw_val, tw)
            ws.append(w)
            fw.append(f)
            lo.append(f - l)
            hi.append(u - f)

    return np.array(ws), np.array(fw), np.array(lo), np.array(hi), upper_bounds


def fit_powerlaw(ws, fw):
    if len(ws) < 3:
        return None, None
    mask = np.array(fw) < 0.15
    wx = np.array(ws)[mask]
    fy = np.array(fw)[mask]
    if len(wx) < 2:
        wx, fy = np.array(ws)[:4], np.array(fw)[:4]
    if len(wx) < 2:
        return None, None
    slope, intercept, *_ = linregress(np.log10(wx), np.log10(fy))
    return slope, intercept


def run_sweep(
    distances=(3, 5, 7),
    p=0.006,
    shots_per_job=500_000,
    bias=5.0,
    max_workers=6,
):
    jobs = [(d, p, shots_per_job, bias) for d in distances]

    spectra = {}
    print(f"Running {len(jobs)} jobs (bias={bias}, shots={shots_per_job:,})...")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_job, job): job for job in jobs}
        for fut in as_completed(futures):
            d, fail_w, total_w = fut.result()
            spectra[d] = compute_spectrum(fail_w, total_w)
            print(f"  d={d} done — {len(spectra[d][0])} bins with failures")

    return spectra


def plot_spectrum(spectra, p):
    styles = {
        3: ("o", "#5555cc"),
        5: ("s", "#229977"),
        7: ("^", "#cc4400"),
        #styles = {3: ("-", "o"), 5: ("--", "s"), 7: (":", "^")}
    }
    #styles={3: ("-", "r"), 5: ("--", "s"), 7: (":", "^")}

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for d, (ws, fw, lo, hi, upper_bounds) in spectra.items():
        color = styles[d][1]
        marker = styles[d][0]

        # Power-law extrapolation line
        slope, intercept = fit_powerlaw(ws, fw)
        if slope is not None:
            w_min = max(ws.min() if len(ws) else 1, 1)
            w_extrap = np.logspace(np.log10(w_min) - 0.3, np.log10(max(ws.max() if len(ws) else 10, 10)) + 0.8, 300)
            f_extrap = np.clip(10**intercept * w_extrap**slope, 1e-16, 1)
            ax.plot(w_extrap, f_extrap, linestyle=":", color=color, linewidth=1.0, alpha=0.6, zorder=1)

        if len(ws):
            ax.errorbar(
                ws, fw,
                yerr=[lo, hi],
                marker=marker,
                color=color,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.2,
                markersize=5,
                linewidth=1.2,
                capsize=2,
                capthick=0.8,
                elinewidth=0.8,
                label=f"d={d}",
                linestyle="-",
                zorder=2,
            )

        for w_ub, u in upper_bounds:
            ax.annotate(
                "", xy=(w_ub, u * 0.55), xytext=(w_ub, u),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, mutation_scale=7),
                zorder=3,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, 10**2.5)
    ax.set_ylim(1e-8, 1.2)
    ax.set_xlabel("fault weight $w$", fontsize=11)
    ax.set_ylabel("failure spectrum $f(w)$", fontsize=11)
    ax.set_title("Failure Spectrum vs Failure Weight Importance Sampling")
    ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.5, color="gray")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.2, alpha=0.3, color="gray")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="lightgray")
    fig.tight_layout()
    plt.show()


spectra = run_sweep(distances=[3, 5, 7], p=0.006, shots_per_job=500_000, bias=5.0)
plot_spectrum(spectra, p=0.006)