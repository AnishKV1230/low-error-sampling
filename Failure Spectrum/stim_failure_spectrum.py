# This code implements the failure spectrum estimation as a method that calls upon STIM objects and ansatz fitting for a given decoder-bench trace. 
# It loads the check matrix, observable matrix, and priors from the trace, then samples random fault vectors of fixed weight to estimate the failure probability f(w) for each weight w. 
# It fits a 5-parameter ansatz to the measured spectrum points, and uses this fit to reconstruct the logical error rate P(q) as a function of physical error rate q. 
# Finally, it plots the results in a format similar to Figure 2 of the Fail Fast paper, including both the spectrum and the P(q) curve with Monte Carlo points overlaid.
import h5py
import numpy as np
from pathlib import Path
import stim
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from scipy.special import comb as scipy_comb

from decoder_bench.sampler import DecoderState
from decoder_bench.common.build_circuit import dem_to_check_matrices
from decoder_bench.decoders import PyMatchingDecoderImpl

def load_trace(h5_path: str):
    with h5py.File(h5_path, 'r') as f:
        H      = f['check_matrix'][:]   # (336, 5471) — detectors × faults
        A      = f['obs_matrix'][:]     # (1, 5471)   — observables × faults
        priors = f['priors'][:]         # (5471,)     — fault probabilities
    return H, A, priors

#from decoder_bench.decoders import BeliefFindDecoder  # or whichever

def load_decoder(h5_path):
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as f:
        if "circuit" not in f:
            raise KeyError(
                f"{h5_path} does not contain a 'circuit' dataset, "
                "so a full DecoderState cannot be reconstructed for circuit-dependent decoders."
            )

        circuit_data = f["circuit"][()]

        if isinstance(circuit_data, bytes):
            circuit = stim.Circuit(circuit_data.decode("utf-8"))
        else:
            circuit = circuit_data

    dem = circuit.detector_error_model()
    check_matrix, obs_matrix, priors = dem_to_check_matrices(dem)

    ds =  DecoderState(
        check_matrix=check_matrix,
        obs_matrix=obs_matrix,
        priors=priors,
        circuit=circuit,
        dem=dem,
    )

    return PyMatchingDecoderImpl(ds)  # or whichever

def is_failure(e, H, A, decoder):
    s = (H @ e) % 2
    pred = np.asarray(decoder.decode(s)).astype(np.uint8).reshape(-1)
    true_obs = np.asarray((A @ e) % 2).astype(np.uint8).reshape(-1)

    n_faults = H.shape[1]
    n_obs = A.shape[0]

    if pred.size == n_faults:
        pred_obs = np.asarray((A @ pred) % 2).astype(np.uint8).reshape(-1)
        return bool(np.any(pred_obs != true_obs))

    if pred.size == n_obs:
        return bool(np.any(pred != true_obs))

    raise ValueError(
        f"Decoder returned vector of length {pred.size},"
        f"but expected either n_faults={n_faults} or n_obs={n_obs}."
    )


def sample_fixed_weight(
    w: int,
    num_trials: int,
    H: np.ndarray,
    A: np.ndarray,
    decoder,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Returns (num_failures, num_trials)."""
    N = H.shape[1]
    failures = 0
    for _ in range(num_trials):
        e = np.zeros(N, dtype=np.uint8)
        fault_positions = rng.choice(N, size=w, replace=False)
        e[fault_positions] = 1
        if is_failure(e, H, A, decoder):
            failures += 1
    return failures, num_trials

def estimate_failure_spectrum(
    H: np.ndarray,
    A: np.ndarray,
    decoder,
    w_range: list[int],
    trials_per_weight: int,
    seed: int = 0,
) -> dict[int, tuple[float, float]]:
    """
    Returns dict mapping w -> (f_hat, stderr).
    stderr = sqrt(f_hat * (1 - f_hat) / trials).
    """
    rng = np.random.default_rng(seed)
    spectrum = {}
    for w in w_range:
        failures, trials = sample_fixed_weight(w, trials_per_weight, H, A, decoder, rng)
        f_hat = failures / trials
        stderr = np.sqrt(f_hat * (1 - f_hat) / trials) if trials > 0 else 0.0
        spectrum[w] = (f_hat, stderr)
        print(f"w={w:4d}  f(w)={f_hat:.4e}  ±{stderr:.2e}  ({failures}/{trials})")
    return spectrum

def f_ansatz_2(w, w0, f0, a):
    """2 free params (a is fixed in practice to 1 - 1/2^K)."""
    x = (1/a) * f0 * (w / w0) ** w0
    return a * (1 - np.exp(-x))

def f_ansatz_3(w, w0, f0, a, gamma):
    x = (1/a) * f0 * (w / w0) ** gamma
    return a * (1 - np.exp(-x))

def f_ansatz_5(w, w0, f0, a, gamma1, gamma2, wc):
    c = 2
    transition = (1 + (w / wc)**c) / (1 + (w0 / wc)**c)
    power = (w / w0)**gamma1 * transition**((gamma2 - gamma1) / c)
    x = (1/a) * f0 * power
    return a * (1 - np.exp(-x))

from scipy.optimize import curve_fit

def fit_ansatz(
    spectrum: dict[int, tuple[float, float]],
    K: int,
    w0: int,
) -> dict:
    a = 1.0 - 1.0 / (2**K)
    
    weights  = np.array([w for w in spectrum if spectrum[w][0] > 0])
    f_values = np.array([spectrum[w][0] for w in weights])
    f_errors = np.array([max(spectrum[w][1], 1e-15) for w in weights])

    # Fix w0 and a, fit f0, gamma1, gamma2, wc
    def model(w, f0, gamma1, gamma2, wc):
        return f_ansatz_5(w, w0, f0, a, gamma1, gamma2, wc)

    p0     = [f_values[0], w0 * 0.9, w0 * 1.1, weights[len(weights)//2]]
    bounds = ([0, 0.1, 0.1, 1], [1, 50, 50, max(weights)])

    popt, pcov = curve_fit(
        model, weights, f_values,
        p0=p0, bounds=bounds, sigma=f_errors,
        maxfev=10000,
    )
    f0_fit, gamma1_fit, gamma2_fit, wc_fit = popt

    return {
        'w0': w0, 'a': a,
        'f0': f0_fit, 'gamma1': gamma1_fit,
        'gamma2': gamma2_fit, 'wc': wc_fit,
        'pcov': pcov,
    }

from scipy.special import comb as scipy_comb

def p_from_ansatz(params: dict, N: int, q_values: np.ndarray) -> np.ndarray:
    """Compute P(q) = sum_w f_ansatz(w) * C(N,w) * q^w * (1-q)^(N-w)."""
    w0 = params['w0']
    P  = np.zeros_like(q_values, dtype=float)
    w_max = min(N, w0 + 200)   # f(w) saturates well before N; cap for speed
    
    for w in range(w0, w_max + 1):
        fw = f_ansatz_5(
            w, params['w0'], params['f0'], params['a'],
            params['gamma1'], params['gamma2'], params['wc'],
        )
        # Use log-space arithmetic to avoid underflow at very small q
        log_binom = (
            np.log(scipy_comb(N, w, exact=False))
            + w * np.log(q_values)
            + (N - w) * np.log1p(-q_values)
        )
        P += fw * np.exp(log_binom)
    return P

# import json

# def run(
#     h5_path: str,
#     decoder_name: str,
#     w_range: list[int],
#     trials_per_weight: int,
#     w0: int,
#     output_path: str,
#     seed: int = 0,
# ):
#     H, A, priors = load_trace(h5_path)
#     K = A.shape[0]
#     N = H.shape[1]

#     decoder  = load_decoder(h5_path)
#     spectrum = estimate_failure_spectrum(H, A, decoder, w_range, trials_per_weight, seed)
#     params   = fit_ansatz(spectrum, K, w0)

#     q_values = np.logspace(-4, -1, 200)
#     P_ansatz = p_from_ansatz(params, N, q_values)

#     results = {
#         'spectrum':  {str(w): list(v) for w, v in spectrum.items()},
#         'fit_params': {k: v for k, v in params.items() if k != 'pcov'},
#         'q_values':  q_values.tolist(),
#         'P_ansatz':  P_ansatz.tolist(),
#         'N': N, 'K': K, 'w0': w0,
#     }
#     with open(output_path, 'w') as f:
#         json.dump(results, f, indent=2)
#     print(f"Results saved to {output_path}")
#     return results


def run(
    h5_path: str,
    decoder_name: str,
    w_range: list[int],
    trials_per_weight: int,
    w0: int,
    output_path: str,
    seed: int = 0,
):
    H, A, priors = load_trace(h5_path)
    K = A.shape[0]
    N = H.shape[1]

    decoder  = load_decoder(h5_path)
    spectrum = estimate_failure_spectrum(H, A, decoder, w_range, trials_per_weight, seed)
    params   = fit_ansatz(spectrum, K, w0)

    q_values = np.logspace(-4, -1, 300)
    P_ansatz = p_from_ansatz(params, N, q_values)

    # --- NEW: also compute the normalized contribution of each weight to P(q)
    # at three representative q values (as in Figure 2b of the paper).
    # This shows which fault weights dominate at each operating point.
    q_show = [0.0005, 0.001, 0.005]
    contributions = {}
    for q in q_show:
        contribs = []
        for w in range(w0, min(N, w0 + 300)):
            fw = f_ansatz_5(
                float(w), float(params['w0']), params['f0'], params['a'],
                params['gamma1'], params['gamma2'], params['wc'],
            )
            log_term = (
                np.log(scipy_comb(N, w, exact=False))
                + w * np.log(q)
                + (N - w) * np.log(1 - q)
            )
            contribs.append(fw * np.exp(log_term))
        total = sum(contribs)
        contributions[str(q)] = [c / total if total > 0 else 0 for c in contribs]

    results = {
        'spectrum':       {str(w): list(v) for w, v in spectrum.items()},
        'fit_params':     {k: (v.tolist() if hasattr(v, 'tolist') else v)
                           for k, v in params.items() if k != 'pcov'},
        'q_values':       q_values.tolist(),
        'P_ansatz':       P_ansatz.tolist(),
        'contributions':  contributions,
        'w0':             w0,
        'N':              N,
        'K':              K,
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return results


def get_mc_ler(h5_path: str, decoder_name: str) -> tuple[float, float, int]:
    """
    Read a decoder-bench trace and compute the Monte Carlo LER using
    the stored syndromes and observables — no re-sampling needed.

    Returns (p_value, ler, n_shots).
    p_value is read from the priors (the most common non-outlier value).
    """
    H, A, priors = load_trace(h5_path)
    decoder = load_decoder(h5_path)

    with h5py.File(h5_path, 'r') as f:
        syndromes   = f['syndromes'][:].astype(np.uint8)
        observables = f['observables'][:].astype(np.uint8)

    n_shots = syndromes.shape[0]
    failures = 0
    for i in range(n_shots):
        c = decoder.decode(syndromes[i])
        if np.any((A @ c) % 2 != observables[i]):
            failures += 1

    ler = failures / n_shots

    # Infer p from priors: the dominant prior value is the 2-qubit gate rate,
    # which equals p directly in SI1000.
    p_val = float(np.median(priors[priors > 0]))

    return p_val, ler, n_shots


def plot_figure2(
    results_json: str,
    mc_trace_paths: list[str],   # list of h5 files at different p values
    decoder_name: str,
    output_png: str = 'figure2_recreation.png',
):
    """
    Recreates the layout of Figure 2 from the Fail Fast paper:

    Left panel  — f(w): measured spectrum points with error bars,
                  ansatz fit, and normalized weight contributions at
                  three q values.
    Right panel — P(q): ansatz reconstruction curve plus Monte Carlo
                  LER points from multiple traces.
    """
    with open(results_json) as f:
        res = json.load(f)

    # --- Parse results ---
    spectrum    = {int(w): tuple(v) for w, v in res['spectrum'].items()}
    params      = res['fit_params']
    q_values    = np.array(res['q_values'])
    P_ansatz    = np.array(res['P_ansatz'])
    N           = res['N']
    K           = res['K']
    w0          = res['w0']
    a           = params['a']

    # --- Collect Monte Carlo points ---
    mc_points = []   # list of (p, ler, n_shots)
    for path in mc_trace_paths:
        try:
            p, ler, n = get_mc_ler(path, decoder_name)
            if ler > 0:
                mc_points.append((p, ler, n))
                print(f"  MC point: p={p:.4f}  LER={ler:.5f}  ({n} shots)")
            else:
                print(f"  MC point: p={p:.4f}  LER=0 (no failures — skipping)")
        except Exception as e:
            print(f"  Could not process {path}: {e}")

    mc_points.sort(key=lambda x: x[0])

    # --- Build the ansatz f(w) curve for the fit overlay ---
    w_dense = np.linspace(w0, max(spectrum.keys()) * 1.5, 500)
    f_fit   = np.array([
        f_ansatz_5(w, float(params['w0']), params['f0'], params['a'],
                   params['gamma1'], params['gamma2'], params['wc'])
        for w in w_dense
    ])

    # --- Figure layout ---
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={'wspace': 0.35}
    )

    # ================================================================
    # LEFT PANEL — failure spectrum f(w)
    # ================================================================

    # Measured spectrum points
    ws      = np.array(sorted(spectrum.keys()), dtype=float)
    f_meas  = np.array([spectrum[int(w)][0] for w in ws])
    f_err   = np.array([spectrum[int(w)][1] for w in ws])
    nonzero = f_meas > 0

    ax_left.errorbar(
        ws[nonzero], f_meas[nonzero], yerr=f_err[nonzero],
        fmt='o', color='steelblue', markersize=5, linewidth=1.2,
        capsize=3, label='data $f(w)$', zorder=5,
    )

    # Ansatz fit curve
    ax_left.plot(w_dense, f_fit, color='crimson', linewidth=1.8,
                 label=r'$f^{(5)}_\mathrm{ansatz}(w)$', zorder=4)

    # Normalized contribution bars at three q values
    # These show which weights actually matter for P(q) at each operating point
    q_colors = ['#2ca02c', '#ff7f0e', '#9467bd']
    q_labels = [0.0005, 0.001, 0.005]
    for q_val, color in zip(q_labels, q_colors):
        contrib_key = str(q_val)
        if contrib_key in res['contributions']:
            contribs = np.array(res['contributions'][contrib_key])
            w_contrib = np.arange(w0, w0 + len(contribs), dtype=float)
            # Scale to roughly overlay on the f(w) plot
            if contribs.max() > 0:
                scale = f_meas[nonzero].max() * 0.6 / contribs.max()
                ax_left.fill_between(
                    w_contrib, 0, contribs * scale,
                    alpha=0.25, color=color,
                    label=f'contribution $q={q_val}$',
                )

    ax_left.set_xscale('log')
    ax_left.set_yscale('log')
    ax_left.set_xlabel('fault weight $w$', fontsize=12)
    ax_left.set_ylabel('failure spectrum $f(w)$', fontsize=12)
    ax_left.set_title(f'Surface code $d=7$, circuit noise', fontsize=11)
    ax_left.legend(fontsize=8, loc='upper left')
    ax_left.set_ylim(bottom=max(f_meas[nonzero].min() * 0.1, 1e-8))
    ax_left.grid(True, which='both', alpha=0.3, linewidth=0.5)

    # ================================================================
    # RIGHT PANEL — P(q) ansatz vs Monte Carlo
    # ================================================================

    # Ansatz curve
    ax_right.plot(q_values, P_ansatz, color='crimson', linewidth=1.8,
                  label=r'$P_\mathrm{ansatz}(q)$', zorder=4)

    # 1-sigma band from fit uncertainty (propagate pcov if available)
    # For now just draw the central line; add band if pcov is stored
    ax_right.axhline(y=a, color='gray', linewidth=0.8, linestyle='--', alpha=0.5,
                     label=f'$a = {a}$')

    # Monte Carlo points with Clopper-Pearson error bars
    if mc_points:
        mc_p   = np.array([pt[0] for pt in mc_points])
        mc_ler = np.array([pt[1] for pt in mc_points])
        mc_n   = np.array([pt[2] for pt in mc_points])

        # Symmetric Poisson approximation for error bars: sqrt(k)/n
        mc_k   = np.round(mc_ler * mc_n).astype(int)
        mc_err = np.sqrt(np.maximum(mc_k, 1)) / mc_n

        ax_right.errorbar(
            mc_p, mc_ler, yerr=mc_err,
            fmt='ko', markersize=5, linewidth=1.2,
            capsize=3, label='Monte Carlo $\\hat{P}(q)$', zorder=5,
        )

    ax_right.set_xscale('log')
    ax_right.set_yscale('log')
    ax_right.set_xlabel('physical error rate $q$', fontsize=12)
    ax_right.set_ylabel('logical error rate $P(q)$', fontsize=12)
    ax_right.set_title(f'Surface code $d=7$, circuit noise', fontsize=11)
    ax_right.legend(fontsize=9, loc='upper left')
    ax_right.set_xlim(q_values[0], q_values[-1])
    ax_right.grid(True, which='both', alpha=0.3, linewidth=0.5)

    # Annotate the operating point used for the spectrum run
    p_op  = 0.001
    idx   = np.argmin(np.abs(q_values - p_op))
    P_op  = float(P_ansatz[idx])
    ax_right.annotate(
        f'spectrum\nmeasured\nat $q={p_op}$',
        xy=(p_op, P_op), xytext=(p_op * 3, P_op * 5),
        fontsize=7, color='steelblue',
        arrowprops=dict(arrowstyle='->', color='steelblue', lw=0.8),
    )

    fig.suptitle(
        'Failure spectrum and logical error rate — '
        r'$f^{(5)}_\mathrm{ansatz}$ fit  '
        f'($w_0={w0}$, $N={N}$)',
        fontsize=11, y=1.01,
    )

    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_png}")
    plt.show()
    return fig