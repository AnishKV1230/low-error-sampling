import stim
import numpy as np
from decoder_bench.sampler import DecoderState
from decoder_bench.common.build_circuit import dem_to_check_matrices
from decoder_bench.decoders import PyMatchingDecoderImpl
import math


def is_failure(e: np.ndarray, H: np.ndarray, A: np.ndarray, decoder) -> bool:
    syndrome   = (H @ e) % 2
    correction = decoder.decode(syndrome)       # returns c s.t. H@c == syndrome
    return bool(np.any((A @ correction) % 2 != (A @ e) % 2))

def pi_dist(e: str, p: float) -> float:
    """Calculate the probability of error string e given error rate p."""
    num_errors = e.count('1')
    return (p ** num_errors) * ((1 - p) ** (len(e) - num_errors))

def metropolis_step(e: str, p: float, H, A, decoder) -> str:
    """Metropolis step for error string e with error rate p."""
    e_list = list(e)
    i = np.random.randint(len(e_list))
    e_list[i] = '1' if e_list[i] == '0' else '0'
    e_new = ''.join(e_list)

    q = min(1, pi_dist(e_new, p) / pi_dist(e, p))  # Symmetric proposal distribution
    
    # Calculate acceptance probability
    if np.random.rand() < (1 - q):
        return e  # Randomly return the old error string with probability 1 - q
    else:
        if is_failure(np.array([int(b) for b in e_new], dtype=np.uint8), H, A, decoder):
            return e_new 
        else:
            return e  # Reject and keep the old error string
        
def g(x: float) -> float:
    """Bennett's optimal estimator function."""
    return 1.0 / (1.0 + x)

def splitting_method(
    ps: np.ndarray,
    T_init: int,
    P_p0: float,
    H: np.ndarray,
    A: np.ndarray,
    decoder,
    circuit,
    epsilon: float = 0.25,
    lambda_: float = 2.0,
) -> float:
    """
    Splitting method for estimating P(p_target) = P(ps[-1]).

    ps      : Probability sequence [p_0, p_1, ..., p_t] with p_0 > p_1 > ...
              p_0 must be high enough for Monte Carlo to be reliable.
    T_init  : Initial chain length before convergence check.
    P_p0    : Known or Monte Carlo estimate of P(p_0).
    """
    N = H.shape[1]
    t = len(ps) - 1     # number of ratio estimates needed (paper's t)
    P_hat = P_p0
    M_prev = None

    for j in range(t + 1):
        p_j    = ps[j]
        p_prev = ps[j-1] if j > 0 else None
        p_next = ps[j+1] if j < t else None

        # Seed: at j=0 sample from Monte Carlo; at j>0 use final state of M_prev
        if j == 0:
            E0 = find_initial_failing_config(
                circuit, H, A, decoder, p0=p_j
            )
        else:
            E0 = M_prev[-1]   # final state of previous chain seeds this one

        M = [E0]

        # Extend chain until convergence criterion is met
        M, stats = extend_chain_until_converged(
            chain=M,
            p_j=p_j,
            p_prev=p_prev if p_prev is not None else p_j,   # unused at j=0
            p_next=p_next if p_next is not None else p_j,   # unused at j=t
            N=N,
            epsilon=epsilon,
            t=t,
            T_init=T_init,
            metropolis_step_fn=metropolis_step,
            lambda_=lambda_,
        )

        # Compute ratio and update estimate (skipped at j=0)
        if j >= 1:
            P_hat, r_hat, c_final = update_probability_estimate(
                P_hat=P_hat,
                chain_j=M,
                chain_prev=M_prev,
                p_j=p_j,
                p_prev=p_prev,
                N=N,
                j=j,
            )
            print(f"  j={j}  p={p_j:.2e}  r={r_hat:.4e}  P_hat={P_hat:.4e}")

        M_prev = M

    return P_hat

            

def find_initial_failing_config(
    circuit: stim.Circuit,
    H: np.ndarray,
    A: np.ndarray,
    decoder,
    p0: float,
    max_shots: int = 100_000,
    batch_size: int = 1_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Run Monte Carlo at p0 until a failing fault bitstring is found.
    Returns e as a numpy array of shape (N,).

    p0 should be high enough that failures are common — typically
    5-10x the threshold, or wherever your standard MC is reliable.
    For a d=7 surface code at p=0.001 threshold, use p0=0.005–0.01.
    """
    # Build a noisy circuit at p0 and extract its DEM
    from decoder_bench.common.noise import NoiseModel
    noisy = NoiseModel.SI1000(p0).noisy_circuit(circuit)
    dem   = noisy.detector_error_model(decompose_errors=True)

    sampler = dem.compile_sampler(seed=seed)
    shots_run = 0

    while shots_run < max_shots:
        # record_errors=True returns which fault mechanisms fired
        errors, syndromes, observables = sampler.sample(
            batch_size,
            separate_observables=True,
            record_errors=True,
        )
        # errors shape: (batch_size, N) — each row is a fault bitstring e
        # syndromes shape: (batch_size, M)
        # observables shape: (batch_size, K)

        for i in range(batch_size):
            e   = errors[i].astype(np.uint8)
            syn = syndromes[i].astype(np.uint8)
            obs = observables[i].astype(np.uint8)

            # Quick consistency check on first shot
            if shots_run == 0 and i == 0:
                assert np.all((H @ e) % 2 == syn), \
                    "H @ e != syndrome — DEM columns don't match check matrix"

            c = decoder.decode(syn)
            if np.any((A @ c) % 2 != obs):
                print(f"  Found E0 after {shots_run + i + 1} shots "
                      f"(weight={int(e.sum())})")
                return e

        shots_run += batch_size
        print(f"  {shots_run}/{max_shots} shots, no failure yet "
              f"(try higher p0 if this repeats)")

    raise RuntimeError(
        f"No failing configuration found in {max_shots} shots at p0={p0}. "
        f"Increase p0 or max_shots."
    )


def log_pi(e: np.ndarray, p: float, N: int) -> float:
    """
    Log probability of fault bitstring e under the Metropolis distribution.
    log pi(e) = |e| * log(p) + (N - |e|) * log(1 - p)
    Working in log space avoids underflow at small p.
    """
    w = int(e.sum())
    return w * math.log(p) + (N - w) * math.log(1 - p)


def compute_g_values(
    chain: list[np.ndarray],
    p_j: float,
    p_prev: float,      
    p_next: float,      
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each sample E_alpha in the chain (excluding E_0 at index 0),
    compute the g values needed for both ratio estimates:

        g_minus[alpha] = g( pi_j(E_alpha) / pi_{j-1}(E_alpha) )
                       = g( exp(log_pi(E, p_j) - log_pi(E, p_{j-1})) )
        g_plus[alpha]  = g( pi_j(E_alpha) / pi_{j+1}(E_alpha) )

    These map to:
        g_minus  →  used in ⟨ĝ-⟩_j  →  contributes to ratio r_{j-1}
        g_plus   →  used in ⟨ĝ+⟩_j  →  contributes to ratio r_j

    Returns arrays of length T_j (chain length minus the seed E_0).
    """
    samples = chain[1:]   # exclude E_0 per Algorithm 2
    T_j     = len(samples)

    g_minus = np.empty(T_j)
    g_plus  = np.empty(T_j)

    for alpha, e in enumerate(samples):
        log_pj   = log_pi(e, p_j,   N)
        log_prev = log_pi(e, p_prev, N)
        log_next = log_pi(e, p_next, N)

        # ratio pi_j / pi_{j-1}  — if p_j < p_{j-1} this is < 1 (downward splitting)
        ratio_minus = math.exp(log_pj - log_prev)
        ratio_plus  = math.exp(log_pj - log_next)

        g_minus[alpha] = g(ratio_minus)
        g_plus[alpha]  = g(ratio_plus)

    return g_minus, g_plus


def compute_estimates(
    chain: list[np.ndarray],
    p_j: float,
    p_prev: float,
    p_next: float,
    N: int,
    epsilon: float,
    t: int,
) -> dict:
    """
    Implements Algorithm 2 lines 11–15.

    Returns a dict containing all intermediate quantities and the
    convergence decision (sigma + delta) vs the threshold epsilon / sqrt(t).

    Parameters
    ----------
    chain   : Metropolis chain at level j, including E_0 at index 0.
    p_j     : Physical error rate at level j.
    p_prev  : Physical error rate at level j-1  (higher, since we split down).
    p_next  : Physical error rate at level j+1  (lower).
    N       : Number of fault mechanisms.
    epsilon : Target total relative error for the full algorithm (paper uses 0.25).
    t       : Total number of splitting levels, used to set per-level threshold.
    """
    g_minus, g_plus = compute_g_values(chain, p_j, p_prev, p_next, N)
    T_j = len(g_minus)   # = len(chain) - 1

    # ------------------------------------------------------------------
    # Line 11 — sample mean estimates
    # ⟨ĝ-⟩_j and ⟨ĝ+⟩_j
    # ------------------------------------------------------------------
    g_hat_minus = float(np.mean(g_minus))
    g_hat_plus  = float(np.mean(g_plus))

    # ------------------------------------------------------------------
    # Line 12 — sample variances  (denominator T_j - 1, unbiased)
    # s²± = 1/(T_j-1) * sum( (g(ratio) - ⟨ĝ±⟩)² )
    # ------------------------------------------------------------------
    s2_minus = float(np.var(g_minus, ddof=1)) if T_j > 1 else 0.0
    s2_plus  = float(np.var(g_plus,  ddof=1)) if T_j > 1 else 0.0

    s_minus = math.sqrt(s2_minus)
    s_plus  = math.sqrt(s2_plus)

    # ------------------------------------------------------------------
    # Line 13 — statistical error sigma
    # σ = max( s+ / ⟨ĝ+⟩, s- / ⟨ĝ-⟩ ) / sqrt(T_j)
    # This is the relative standard error of the mean for both ratio
    # estimates, taking the worse of the two.
    # ------------------------------------------------------------------
    rel_err_plus  = (s_plus  / g_hat_plus)  if g_hat_plus  > 0 else float('inf')
    rel_err_minus = (s_minus / g_hat_minus) if g_hat_minus > 0 else float('inf')

    sigma = max(rel_err_plus, rel_err_minus) / math.sqrt(T_j)

    # ------------------------------------------------------------------
    # Line 14 — first-half subset estimates
    # ⟨ĝ'±⟩_j = mean of g values from the first ceil(T_j / 2) samples
    # If these differ significantly from the full-chain mean, the chain
    # has not mixed: early samples are biased by the starting state.
    # ------------------------------------------------------------------
    T_half = math.ceil(T_j / 2)

    g_hat_minus_half = float(np.mean(g_minus[:T_half]))
    g_hat_plus_half  = float(np.mean(g_plus[:T_half]))

    # ------------------------------------------------------------------
    # Line 15 — mixing discrepancy delta
    # Δ = max( |⟨ĝ+⟩ - ⟨ĝ'+⟩| / ⟨ĝ+⟩,  |⟨ĝ-⟩ - ⟨ĝ'-⟩| / ⟨ĝ-⟩ )
    # Small when T_j >> mixing time.
    # ------------------------------------------------------------------
    mix_plus  = (abs(g_hat_plus  - g_hat_plus_half)  / g_hat_plus)  \
                if g_hat_plus  > 0 else float('inf')
    mix_minus = (abs(g_hat_minus - g_hat_minus_half) / g_hat_minus) \
                if g_hat_minus > 0 else float('inf')

    delta = max(mix_plus, mix_minus)

    # ------------------------------------------------------------------
    # Line 16 — convergence check
    # The per-level threshold is epsilon / sqrt(t), so that errors
    # combine incoherently across t levels to give total error < epsilon.
    # ------------------------------------------------------------------
    threshold         = epsilon / math.sqrt(t)
    converged         = (sigma + delta) <= threshold

    return {
        # Full-chain estimates
        'g_hat_plus':        g_hat_plus,
        'g_hat_minus':       g_hat_minus,
        # Variances
        's_plus':            s_plus,
        's_minus':           s_minus,
        # Statistical error
        'sigma':             sigma,
        # Half-chain estimates
        'g_hat_plus_half':   g_hat_plus_half,
        'g_hat_minus_half':  g_hat_minus_half,
        # Mixing discrepancy
        'delta':             delta,
        # Convergence
        'threshold':         threshold,
        'converged':         converged,
        'sigma_plus_delta':  sigma + delta,
        'T_j':               T_j,
    }


def extend_chain_until_converged(
    chain: list[np.ndarray],
    p_j: float,
    p_prev: float,
    p_next: float,
    N: int,
    epsilon: float,
    t: int,
    T_init: int,
    metropolis_step_fn,           # callable: (e, p_j) -> e_new
    lambda_: float = 2.0,         # scaling factor λ from Algorithm 2 line 2
) -> tuple[list[np.ndarray], dict]:
    """
    Implements the while loop of Algorithm 2 (lines 9–17).

    Runs the Metropolis chain until the convergence criterion is met,
    doubling the extension length each time it is not (λ' *= λ).

    Parameters
    ----------
    chain               : Initial chain at level j, with E_0 at index 0.
                          Extended in-place and also returned.
    metropolis_step_fn  : Wraps your metropolis_step; signature (e, p) -> e.
    lambda_             : Scaling factor — each failed convergence check
                          adds ceil(λ' * T_init) more steps, with λ' growing
                          as λ^(number of extensions).

    Returns
    -------
    chain   : Extended chain (same object, mutated).
    stats   : Final dict from compute_estimates.
    """
    lambda_prime = lambda_          # line 8: initialise λ' ← λ

    # Ensure we have at least T_init samples before the first check
    while len(chain) - 1 < T_init:
        e_new = metropolis_step_fn(chain[-1], p_j)
        chain.append(e_new)

    while True:
        stats = compute_estimates(chain, p_j, p_prev, p_next, N, epsilon, t)

        T_j = stats['T_j']
        print(
            f"  T_j={T_j:>7d}  σ={stats['sigma']:.4f}  "
            f"Δ={stats['delta']:.4f}  "
            f"σ+Δ={stats['sigma_plus_delta']:.4f}  "
            f"threshold={stats['threshold']:.4f}  "
            f"{'CONVERGED' if stats['converged'] else 'extending...'}"
        )

        if stats['converged']:
            break

        # Line 16: T_j ← T_j + ceil(λ' · T_init);  λ' ← λ · λ'
        extension    = math.ceil(lambda_prime * T_init)
        lambda_prime = lambda_ * lambda_prime

        for _ in range(extension):
            e_new = metropolis_step_fn(chain[-1], p_j)
            chain.append(e_new)

    return chain, stats

def estimate_ratio(
    chain_j: list[np.ndarray],
    chain_prev: list[np.ndarray],
    p_j: float,
    p_prev: float,
    N: int,
    n_iterations: int = 3,
) -> tuple[float, float]:
    """
    Algorithm 2 lines 18-23.

    Estimates the ratio r_{j-1} = P(p_j) / P(p_{j-1}) using samples
    from both chains via Bennett's acceptance ratio method.

    The estimator is:

        r_{j-1} = c * mean( g( c * pi_{j-1}(E') / pi_j(E') ) )   [over chain_{j-1}]
                      -----------------------------------------------
                      mean( g( pi_j(E)  / (c * pi_{j-1}(E)) ) )   [over chain_j]

    where g(x) = 1/(1+x) and c is iterated to minimise estimator variance.
    The optimal c equals r_{j-1} itself, so we converge by substituting
    the current estimate back — 3 iterations is sufficient per the paper.

    Parameters
    ----------
    chain_j    : Chain at level j   — [E0, E1, ..., E_{T_j}].
                 E0 is excluded from the sum (line 18 footnote).
    chain_prev : Chain at level j-1 — [E0, E'1, ..., E'_{T_{j-1}}].
                 E0 is excluded from the sum.
    p_j        : Error rate at level j   (lower, since we split downward).
    p_prev     : Error rate at level j-1 (higher).
    N          : Number of fault mechanisms.
    n_iterations: Number of times to refine c (paper uses 3).

    Returns
    -------
    r_hat  : Estimated ratio P(p_j) / P(p_{j-1}).
    c_final: Final value of c (useful for diagnostics).
    """
    # Exclude E_0 seed from both chains (lines 18, 21)
    samples_j    = chain_j[1:]
    samples_prev = chain_prev[1:]

    if len(samples_j) == 0 or len(samples_prev) == 0:
        raise ValueError(
            "Both chains must have at least one sample beyond E_0. "
            f"Got {len(samples_j)} from chain_j, "
            f"{len(samples_prev)} from chain_prev."
        )

    # Pre-compute log-probability differences for every sample.
    # log[ pi_j(E) / pi_{j-1}(E) ] = (log p_j - log p_{j-1}) * |E|
    #                                + (log(1-p_j) - log(1-p_{j-1})) * (N - |E|)
    # This difference depends only on the weight of E, not on E itself,
    # so we compute it once per sample and reuse it across c iterations.
    log_pj   = math.log(p_j)
    log_prev = math.log(p_prev)
    log_1pj  = math.log(1 - p_j)
    log_1pr  = math.log(1 - p_prev)

    def log_ratio_j_over_prev(e: np.ndarray) -> float:
        """log[ pi_j(e) / pi_{j-1}(e) ]"""
        w = int(e.sum())
        return w * (log_pj - log_prev) + (N - w) * (log_1pj - log_1pr)

    # Pre-compute for all samples in both chains
    log_ratios_j    = np.array([log_ratio_j_over_prev(e) for e in samples_j])
    log_ratios_prev = np.array([log_ratio_j_over_prev(e) for e in samples_prev])

    # ------------------------------------------------------------------
    # Lines 19-23 — iterate c to minimise estimator variance
    # ------------------------------------------------------------------
    c = 1.0   # line 19: initialise c ← 1

    for i in range(n_iterations):   # line 20: for i ∈ {1, 2, 3}

        log_c = math.log(c) if c > 0 else 0.0

        # Numerator: mean over chain_{j-1} of g( c * pi_{j-1}(E') / pi_j(E') )
        #          = mean over chain_{j-1} of g( c / (pi_j/pi_{j-1}) )
        #          = mean of g( exp(log_c - log_ratio) )
        # log_ratios_prev holds log[ pi_j / pi_{j-1} ] for each E' in chain_{j-1}
        # so  c * pi_{j-1}/pi_j = c * exp(-log_ratio) = exp(log_c - log_ratio)
        numerator_args   = np.exp(log_c - log_ratios_prev)
        numerator        = float(np.mean([g(x) for x in numerator_args]))

        # Denominator: mean over chain_j of g( pi_j(E) / (c * pi_{j-1}(E)) )
        #            = mean of g( exp(log_ratio - log_c) )
        denominator_args = np.exp(log_ratios_j - log_c)
        denominator      = float(np.mean([g(x) for x in denominator_args]))

        if denominator == 0.0:
            raise RuntimeError(
                f"Denominator is zero at c-iteration {i}. "
                "This usually means p_j and p_prev are too far apart — "
                "reduce the splitting step size so the two distributions overlap."
            )

        # Line 21: r_hat_{j-1} = c * numerator / denominator
        r_hat = c * numerator / denominator

        print(
            f"    c-iteration {i+1}: c={c:.6f}  "
            f"numer={numerator:.6f}  denom={denominator:.6f}  "
            f"r_hat={r_hat:.6f}"
        )

        # Line 22: c ← r_hat
        c = r_hat

    return r_hat, c


def update_probability_estimate(
    P_hat: float,
    chain_j: list[np.ndarray],
    chain_prev: list[np.ndarray],
    p_j: float,
    p_prev: float,
    N: int,
    j: int,
) -> tuple[float, float, float]:
    """
    Algorithm 2 lines 18-25 as a single call.

    Computes the ratio estimate and multiplies it into the running
    probability estimate. Skipped when j < 2 (no previous chain yet).

    Parameters
    ----------
    P_hat      : Current running estimate of P(p_j-1) before this update.
    j          : Current splitting level index (1-based to match the paper).

    Returns
    -------
    P_hat_new  : Updated P_hat = old P_hat * r_{j-1}.
    r_hat      : The ratio estimate P(p_j) / P(p_{j-1}).
    c_final    : Final c value (for diagnostics).
    """
    # Line 18: if j >= 2
    if j < 2:
        return P_hat, 1.0, 1.0

    r_hat, c_final = estimate_ratio(
        chain_j, chain_prev, p_j, p_prev, N
    )

    # Line 24: P_hat ← P_hat * r_{j-1}
    P_hat_new = P_hat * r_hat

    print(
        f"  Level j={j}: r_hat={r_hat:.6e}  "
        f"P_hat={P_hat:.6e} → {P_hat_new:.6e}"
    )

    return P_hat_new, r_hat, c_final