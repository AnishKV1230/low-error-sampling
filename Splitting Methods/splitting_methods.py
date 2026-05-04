"""
splitting_methods.py  –  Technique III from Beverland et al. "Fail Fast" (arXiv:2511.15177)

Implementation of the splitting method (Algorithm 2) for estimating rare logical
failure rates in QEC systems.  Includes:
  - splitting_method()        : single-seeded Algorithm 2
  - multi_seeded_splitting()  : Algorithm 4 (multi-seeded, paper §5.3)

decoder_bench integration:  see `splitting_from_decoder_bench()` at the bottom.
"""

import math
import numpy as np
import stim
from decoder_bench.sampler import DecoderState
from decoder_bench.common.build_circuit import dem_to_check_matrices


# ─────────────────────────────────────────────────────────────────────────────
# Core primitives
# ─────────────────────────────────────────────────────────────────────────────

def is_failure(e: np.ndarray, H: np.ndarray, A: np.ndarray, decoder) -> bool:
    """Return True if fault bitstring e causes a logical failure."""
    e = np.asarray(e, dtype=np.uint8)
    syndrome = (H @ e) % 2
    correction = decoder.decode(syndrome)
    return bool(np.any((A @ correction) % 2 != (A @ e) % 2))


def log_pi(e: np.ndarray, p: float, N: int) -> float:
    """
    Log-probability of fault bitstring e under iid model with rate p.
    log π(e) = |e|·log(p) + (N−|e|)·log(1−p)
    """
    w = int(np.sum(e))
    return w * math.log(p) + (N - w) * math.log(1.0 - p)


def g(x: float) -> float:
    """Bennett's optimal g-function: g(x) = 1/(1+x)."""
    return 1.0 / (1.0 + x)


# ─────────────────────────────────────────────────────────────────────────────
# DEM / prior extraction  (used by decoder_bench integration)
# ─────────────────────────────────────────────────────────────────────────────

def extract_priors_from_dem(dem: stim.DetectorErrorModel) -> np.ndarray:
    """
    Extract the prior probability array from a stim.DetectorErrorModel.

    Uses dem_to_check_matrices internally to ensure the column ordering
    is identical to the H / A matrices produced by decoder_bench.
    """
    _, _, priors = dem_to_check_matrices(dem)
    return np.asarray(priors, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Seed finding
# ─────────────────────────────────────────────────────────────────────────────

def find_initial_failing_config(
    circuit: stim.Circuit,
    H: np.ndarray,
    A: np.ndarray,
    decoder,
    p0: float,
    max_tries: int = 100_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample from the DEM at error rate p0 until a failing fault bitstring is
    found.  Returns e as a numpy array of shape (N,).

    p0 should be high enough that failures are common – typically 5-10× the
    pseudo-threshold, or wherever standard MC is reliable.
    """
    rng = np.random.default_rng(seed)
    dem = circuit.detector_error_model(decompose_errors=True)
    priors = extract_priors_from_dem(dem)
    N = H.shape[1]

    # Guard: priors length must match H columns
    if len(priors) != N:
        raise ValueError(
            f"DEM yielded {len(priors)} fault columns but H has {N}. "
            "Ensure the circuit and check matrices are built from the same DEM."
        )

    for _ in range(max_tries):
        e = (rng.random(N) < priors).astype(np.uint8)
        if e.sum() == 0:
            continue
        if is_failure(e, H, A, decoder):
            return e

    raise RuntimeError(
        f"No failing configuration found in {max_tries} tries at p0={p0}. "
        "Increase p0 or max_tries."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metropolis step  (Algorithm 3)
# ─────────────────────────────────────────────────────────────────────────────

def metropolis_step(
    e: np.ndarray,
    p: float,
    H: np.ndarray,
    A: np.ndarray,
    decoder,
) -> np.ndarray:
    """
    Single Metropolis update targeting π(E|F) ∝ p^|E| (1−p)^(N−|E|).

    Flip a random bit, accept with probability min(1, π(E')/π(E)), then
    reject the proposal if E' ∉ F  (ensures chain stays in failing set).
    Detailed balance is maintained because rejection at the is-failure check
    is symmetric: both E→E' and E'→E are rejected iff E' ∉ F.
    """
    N = len(e)
    i = np.random.randint(N)
    e_new = e.copy()
    e_new[i] ^= 1

    # Acceptance ratio: min(1, π(e_new)/π(e))
    q = min(1.0, math.exp(log_pi(e_new, p, N) - log_pi(e, p, N)))

    if np.random.rand() < (1.0 - q):
        return e                        # rejected (stay at current state)

    if is_failure(e_new, H, A, decoder):
        return e_new                    # accepted and in failing set
    return e                            # proposed state not failing → stay


# ─────────────────────────────────────────────────────────────────────────────
# g-value computation  (Algorithm 2, lines 11-15)
# ─────────────────────────────────────────────────────────────────────────────

def compute_g_values(
    chain: list,
    p_j: float,
    p_prev: float,
    p_next: float,
    N: int,
) -> tuple:
    """
    For each sample E_α in chain[1:] compute:
      g_minus[α] = g(π_j(E_α) / π_{j−1}(E_α))   → used for ratio r_{j−1}
      g_plus[α]  = g(π_j(E_α) / π_{j+1}(E_α))   → used for ratio r_j
    """
    samples = chain[1:]          # exclude seeding state E_0
    T_j = len(samples)
    g_minus = np.empty(T_j)
    g_plus  = np.empty(T_j)

    for alpha, e in enumerate(samples):
        log_pj   = log_pi(e, p_j,   N)
        log_prev = log_pi(e, p_prev, N)
        log_next = log_pi(e, p_next, N)
        g_minus[alpha] = g(math.exp(log_pj - log_prev))
        g_plus[alpha]  = g(math.exp(log_pj - log_next))

    return g_minus, g_plus


def compute_estimates(
    chain: list,
    p_j: float,
    p_prev: float,
    p_next: float,
    N: int,
    epsilon: float,
    t: int,
) -> dict:
    """
    Algorithm 2 lines 11-16: compute ĝ±, s², σ, Δ and decide convergence.
    """
    g_minus, g_plus = compute_g_values(chain, p_j, p_prev, p_next, N)
    T_j = len(g_minus)

    g_hat_minus = float(np.mean(g_minus))
    g_hat_plus  = float(np.mean(g_plus))

    s2_minus = float(np.var(g_minus, ddof=1)) if T_j > 1 else 0.0
    s2_plus  = float(np.var(g_plus,  ddof=1)) if T_j > 1 else 0.0
    s_minus = math.sqrt(s2_minus)
    s_plus  = math.sqrt(s2_plus)

    # Statistical error σ  (line 13)
    rel_plus  = (s_plus  / g_hat_plus)  if g_hat_plus  > 0 else float('inf')
    rel_minus = (s_minus / g_hat_minus) if g_hat_minus > 0 else float('inf')
    sigma = max(rel_plus, rel_minus) / math.sqrt(T_j)

    # First-half subset estimates  (line 14)
    T_half = math.ceil(T_j / 2)
    g_hat_minus_half = float(np.mean(g_minus[:T_half]))
    g_hat_plus_half  = float(np.mean(g_plus[:T_half]))

    # Mixing discrepancy Δ  (line 15)
    mix_plus  = abs(g_hat_plus  - g_hat_plus_half)  / g_hat_plus  if g_hat_plus  > 0 else float('inf')
    mix_minus = abs(g_hat_minus - g_hat_minus_half) / g_hat_minus if g_hat_minus > 0 else float('inf')
    delta = max(mix_plus, mix_minus)

    threshold = epsilon / math.sqrt(t)
    converged = (sigma + delta) <= threshold

    return {
        'g_hat_plus': g_hat_plus,
        'g_hat_minus': g_hat_minus,
        's_plus': s_plus,
        's_minus': s_minus,
        'sigma': sigma,
        'g_hat_plus_half': g_hat_plus_half,
        'g_hat_minus_half': g_hat_minus_half,
        'delta': delta,
        'threshold': threshold,
        'converged': converged,
        'sigma_plus_delta': sigma + delta,
        'T_j': T_j,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chain extension loop  (Algorithm 2, lines 9-17)
# ─────────────────────────────────────────────────────────────────────────────

def extend_chain_until_converged(
    chain: list,
    p_j: float,
    p_prev: float,
    p_next: float,
    N: int,
    epsilon: float,
    t: int,
    T_init: int,
    H: np.ndarray,
    A: np.ndarray,
    decoder,
    lambda_: float = 2.0,
    verbose: bool = False,
) -> tuple:
    """
    Run Algorithm 2's while loop: extend chain until σ+Δ ≤ ε/√t,
    doubling the extension length each failed convergence check.
    """
    lambda_prime = lambda_

    while len(chain) - 1 < T_init:
        chain.append(metropolis_step(chain[-1], p_j, H, A, decoder))

    while True:
        stats = compute_estimates(chain, p_j, p_prev, p_next, N, epsilon, t)
        if verbose:
            print(
                f"  T_j={stats['T_j']:>7d}  σ={stats['sigma']:.4f}  "
                f"Δ={stats['delta']:.4f}  σ+Δ={stats['sigma_plus_delta']:.4f}  "
                f"thr={stats['threshold']:.4f}  "
                f"{'CONV' if stats['converged'] else '...'}"
            )
        if stats['converged']:
            break
        extension = math.ceil(lambda_prime * T_init)
        lambda_prime *= lambda_
        for _ in range(extension):
            chain.append(metropolis_step(chain[-1], p_j, H, A, decoder))

    return chain, stats


# ─────────────────────────────────────────────────────────────────────────────
# Bennett ratio estimator  (Algorithm 2, lines 18-23)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_ratio(
    chain_j: list,
    chain_prev: list,
    p_j: float,
    p_prev: float,
    N: int,
    n_iterations: int = 3,
) -> tuple:
    """
    Estimate r_{j−1} = P(p_j)/P(p_{j−1}) using Bennett's acceptance ratio.

    Returns (r_hat, c_final).
    """
    samples_j    = chain_j[1:]
    samples_prev = chain_prev[1:]

    if not samples_j or not samples_prev:
        raise ValueError("Both chains must have at least one sample beyond E_0.")

    log_pj   = math.log(p_j)
    log_prev = math.log(p_prev)
    log_1pj  = math.log(1.0 - p_j)
    log_1pr  = math.log(1.0 - p_prev)

    def log_ratio_j_over_prev(e):
        w = int(e.sum())
        return w * (log_pj - log_prev) + (N - w) * (log_1pj - log_1pr)

    log_ratios_j    = np.array([log_ratio_j_over_prev(e) for e in samples_j])
    log_ratios_prev = np.array([log_ratio_j_over_prev(e) for e in samples_prev])

    c = 1.0
    for _ in range(n_iterations):
        log_c = math.log(c) if c > 0 else 0.0
        numerator   = float(np.mean([g(math.exp(log_c  - lr)) for lr in log_ratios_prev]))
        denominator = float(np.mean([g(math.exp(lr - log_c))  for lr in log_ratios_j]))
        if denominator == 0.0:
            raise RuntimeError(
                f"Bennett denominator is zero at c={c:.4e}. "
                "p_j and p_prev may be too far apart – reduce the splitting step."
            )
        c = c * numerator / denominator

    return c, c


def update_probability_estimate(
    P_hat: float,
    chain_j: list,
    chain_prev: list,
    p_j: float,
    p_prev: float,
    N: int,
    j: int,
) -> tuple:
    """Compute r_{j−1} and update P_hat ← P_hat · r_{j−1}."""
    if j < 1:
        return P_hat, 1.0, 1.0
    r_hat, c_final = estimate_ratio(chain_j, chain_prev, p_j, p_prev, N)
    P_hat_new = P_hat * r_hat
    return P_hat_new, r_hat, c_final


# ─────────────────────────────────────────────────────────────────────────────
# Single-seeded splitting  (Algorithm 2)
# ─────────────────────────────────────────────────────────────────────────────

def splitting_method(
    ps: np.ndarray,
    T_init: int,
    P_p0: float,
    H: np.ndarray,
    A: np.ndarray,
    decoder,
    circuit: stim.Circuit,
    epsilon: float = 0.25,
    lambda_: float = 2.0,
    initial_seed: np.ndarray = None,
    seed: int = 42,
    verbose: bool = False,
) -> float:
    """
    Algorithm 2: estimate P(p_target) = P(ps[-1]) via the splitting method.

    Parameters
    ----------
    ps        : Probability sequence [p_0, p_1, ..., p_t].
                p_0 must be in the MC-accessible regime; p_t is the target.
    T_init    : Initial Markov-chain length before the first convergence check.
    P_p0      : Known / Monte Carlo estimate of P(p_0).
    H, A      : Check and action matrices (columns = faults).
    decoder   : decoder_bench Decoder instance with a .decode(syndrome) method.
    circuit   : stim.Circuit used to build H and A.
    epsilon   : Target relative error (paper default 0.25).
    lambda_   : Chain-extension scaling factor (paper default 2).
    initial_seed : Optional pre-found failing configuration at ps[0].
    seed      : RNG seed for find_initial_failing_config.
    verbose   : Print per-level progress.

    Returns
    -------
    P_hat : Estimated logical failure rate at ps[-1].
    """
    N = H.shape[1]
    t = len(ps) - 1
    P_hat = float(P_p0)
    M_prev = None

    np.random.seed(seed)

    if initial_seed is None:
        if verbose:
            print(f"Finding initial failing config at p0={ps[0]:.3e} ...")
        initial_seed = find_initial_failing_config(
            circuit, H, A, decoder, p0=ps[0], seed=seed
        )

    E0 = initial_seed.copy()

    for j in range(t + 1):
        p_j    = ps[j]
        p_prev = ps[j - 1] if j > 0 else ps[j]
        p_next = ps[j + 1] if j < t else ps[j]

        if j == 0:
            seed_e = E0
        else:
            seed_e = M_prev[-1]

        M = [seed_e.copy()]
        M, stats = extend_chain_until_converged(
            chain=M, p_j=p_j, p_prev=p_prev, p_next=p_next,
            N=N, epsilon=epsilon, t=max(1, t), T_init=T_init,
            H=H, A=A, decoder=decoder, lambda_=lambda_, verbose=verbose,
        )

        if j >= 1:
            P_hat, r_hat, _ = update_probability_estimate(
                P_hat, M, M_prev, p_j, p_prev, N, j
            )
            if verbose:
                print(f"  j={j}  p={p_j:.2e}  r={r_hat:.4e}  P_hat={P_hat:.4e}")

        M_prev = M

    return P_hat


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seeded splitting  (Algorithm 4 / paper §5.3)
# ─────────────────────────────────────────────────────────────────────────────

def multi_seeded_splitting(
    ps: np.ndarray,
    T_init: int,
    P_p0: float,
    H: np.ndarray,
    A: np.ndarray,
    decoder,
    circuit: stim.Circuit,
    num_seeds: int = 10,
    epsilon: float = 0.25,
    lambda_: float = 2.0,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """
    Multi-seeded splitting (Beverland et al. §5.3).

    The key idea: at p_0 (MC-accessible regime), draw `num_seeds` independent
    failing configurations via Monte Carlo.  Each seed launches one independent
    run of Algorithm 2 down the probability ladder.  The final estimate is the
    arithmetic mean of the per-seed estimates, and its uncertainty is assessed
    from the standard deviation across seeds.

    This addresses two limitations of single-seeded splitting:
      1. Non-ergodicity: different seeds may belong to disconnected components
         of the failing set; averaging samples all components.
      2. Mixing-time sensitivity: biased initialisation (e.g. a high-weight
         logical) is diluted when many seeds are used.

    Parameters
    ----------
    ps          : Probability sequence [p_0, ..., p_t].
    T_init      : Initial chain length per level per seed.
    P_p0        : MC estimate of P(p_0) (used as the starting normalisation
                  for every seed run).
    H, A        : Check and action matrices.
    decoder     : decoder_bench Decoder with .decode(syndrome) method.
    circuit     : stim.Circuit that generated H and A.
    num_seeds   : Number of independent MC seeds at p_0  (S in the paper).
    epsilon     : Per-run target relative error.
    lambda_     : Chain-extension scaling factor.
    seed        : Master RNG seed; each sub-run gets seed + k.
    verbose     : Per-seed / per-level progress.

    Returns
    -------
    dict with keys:
      P_hat        : Mean logical failure rate estimate across seeds.
      P_hat_seeds  : Array of per-seed estimates (length num_seeds).
      std          : Standard deviation of per-seed estimates.
      rel_std      : Relative standard deviation (std / P_hat).
      seeds_used   : List of initial failing configs that succeeded.
    """
    rng = np.random.default_rng(seed)

    # ── Step 1: collect num_seeds distinct failing configs at p_0 ──────────
    if verbose:
        print(f"Collecting {num_seeds} MC seeds at p_0={ps[0]:.3e} ...")

    N = H.shape[1]
    seeds_found: list = []
    attempt_seed = int(rng.integers(0, 2**31))

    for k in range(num_seeds):
        e0 = find_initial_failing_config(
            circuit, H, A, decoder,
            p0=ps[0],
            max_tries=200_000,
            seed=attempt_seed + k,
        )
        seeds_found.append(e0)
        if verbose:
            print(f"  seed {k+1}/{num_seeds}: weight={int(e0.sum())}")

    # ── Step 2: one Algorithm 2 run per seed ───────────────────────────────
    P_hat_seeds = np.empty(num_seeds)

    for k, e0 in enumerate(seeds_found):
        if verbose:
            print(f"\nSeed {k+1}/{num_seeds}  (initial weight={int(e0.sum())})")
        P_hat_k = splitting_method(
            ps=ps,
            T_init=T_init,
            P_p0=P_p0,
            H=H,
            A=A,
            decoder=decoder,
            circuit=circuit,
            epsilon=epsilon,
            lambda_=lambda_,
            initial_seed=e0,
            seed=int(rng.integers(0, 2**31)),
            verbose=verbose,
        )
        P_hat_seeds[k] = P_hat_k
        if verbose:
            print(f"  → P_hat[{k}] = {P_hat_k:.4e}")

    # ── Step 3: aggregate ──────────────────────────────────────────────────
    P_hat = float(np.mean(P_hat_seeds))
    std    = float(np.std(P_hat_seeds, ddof=1)) if num_seeds > 1 else 0.0
    rel_std = std / P_hat if P_hat > 0 else float('inf')

    return {
        'P_hat':        P_hat,
        'P_hat_seeds':  P_hat_seeds,
        'std':           std,
        'rel_std':       rel_std,
        'seeds_used':    seeds_found,
    }


# ─────────────────────────────────────────────────────────────────────────────
# decoder_bench integration 
# ─────────────────────────────────────────────────────────────────────────────

def splitting_from_decoder_bench(
    decoder_class,
    circuit: stim.Circuit,
    ps: np.ndarray,
    T_init: int,
    P_p0: float,
    num_seeds: int = 1,
    epsilon: float = 0.25,
    lambda_: float = 2.0,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """
    Convenience wrapper that builds a decoder_bench DecoderState from a
    stim.Circuit and runs either single- or multi-seeded splitting.

    Parameters
    ----------
    decoder_class : A decoder_bench Decoder subclass (not an instance).
                    E.g. PyMatchingDecoderImpl, BeliefFindDecoderImpl.
    circuit       : stim.Circuit at the *base* noise level (ps[0]).
                    The DEM is extracted from it to build H, A, and priors.
    ps            : Splitting probability ladder [p_0, ..., p_t].
    T_init        : Initial Markov-chain length.
    P_p0          : MC estimate of P(p_0).
    num_seeds     : Number of independent seeds (1 = single-seeded).
    epsilon       : Target relative error.
    lambda_       : Chain-extension scaling factor.
    seed          : Master RNG seed.
    verbose       : Progress output.

    Returns
    -------
    dict from multi_seeded_splitting (or a compatible dict for num_seeds=1).

    Notes
    -----
    Steps required before calling this function from decoder_bench:
      1. Build the stim.Circuit at p_0 using decoder_bench's build_circuit /
         NoiseModel (e.g. NoiseModel.SI1000(p0).noisy_circuit(base_circuit)).
      2. Choose ps: a geometric sequence from p_0 down to p_target, e.g.
         ps = np.geomspace(p0, p_target, num=t+1).
      3. Obtain P_p0 from decoder_bench's Sampler.collect() at p_0 with
         standard MC (the 'logical_error_rate' key in the returned dict).
      4. Pass the desired decoder class (from decoder_bench.decoders) as
         decoder_class.

    The function automatically:
      - Calls dem_to_check_matrices(circuit.detector_error_model()) to get H, A, priors
      - Instantiates the decoder via DecoderState
      - Runs the requested number of seeds and returns aggregated results
    """
    from decoder_bench.common.build_circuit import dem_to_check_matrices
    from decoder_bench.sampler import DecoderState

    dem = circuit.detector_error_model(decompose_errors=True)
    check_matrix, obs_matrix, priors = dem_to_check_matrices(dem)

    # Convert sparse to dense for the splitting kernel
    H = np.asarray(check_matrix.toarray(), dtype=np.uint8)
    A = np.asarray(obs_matrix.toarray(),   dtype=np.uint8)

    state = DecoderState(
        check_matrix=check_matrix,
        obs_matrix=obs_matrix,
        priors=priors,
        circuit=circuit,
        dem=dem,
    )
    decoder = decoder_class(state)

    if num_seeds == 1:
        P_hat = splitting_method(
            ps=ps, T_init=T_init, P_p0=P_p0,
            H=H, A=A, decoder=decoder, circuit=circuit,
            epsilon=epsilon, lambda_=lambda_, seed=seed, verbose=verbose,
        )
        return {'P_hat': P_hat, 'P_hat_seeds': np.array([P_hat]),
                'std': 0.0, 'rel_std': 0.0, 'seeds_used': []}

    return multi_seeded_splitting(
        ps=ps, T_init=T_init, P_p0=P_p0,
        H=H, A=A, decoder=decoder, circuit=circuit,
        num_seeds=num_seeds, epsilon=epsilon,
        lambda_=lambda_, seed=seed, verbose=verbose,
    )
