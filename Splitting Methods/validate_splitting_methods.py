import numpy as np
import math
from collections import defaultdict
from splitting import is_failure, metropolis_step, estimate_ratio, update_probability_estimate, extend_chain_until_converged


# ---------------------------------------------------------------------------
# Analytical ground truth for the 3-qubit repetition code
# ---------------------------------------------------------------------------

def p_exact_repetition(p: float) -> float:
    """
    Exact logical error rate for the 3-qubit repetition code with a
    min-weight decoder under i.i.d. bit-flip noise.

    The decoder fails iff 2 or more qubits are flipped:
        P(p) = C(3,2)*p^2*(1-p) + C(3,3)*p^3
             = 3p^2(1-p) + p^3
    """
    return 3 * p**2 * (1 - p) + p**3


def make_repetition_system():
    """3-qubit repetition code matrices and decoder (reused from earlier tests)."""
    H = np.array([[1, 1, 0],
                  [0, 1, 1]], dtype=np.uint8)
    A = np.array([[1, 1, 1]], dtype=np.uint8)

    class RepDecoder:
        def decode(self, syndrome):
            best, best_w = np.zeros(3, dtype=np.uint8), 4
            for bits in range(8):
                c = np.array([(bits >> i) & 1 for i in range(3)], dtype=np.uint8)
                if np.all((H @ c) % 2 == syndrome):
                    w = int(c.sum())
                    if w < best_w:
                        best, best_w = c.copy(), w
            return best

    return H, A, RepDecoder()


# ---------------------------------------------------------------------------
# Test 1 — P(p0) bootstrap from Monte Carlo
# ---------------------------------------------------------------------------

def test_p0_estimate():
    """
    The splitting method uses P(p0) as its starting estimate.
    Verify Monte Carlo at p0=0.3 matches the analytical value.
    """
    H, A, dec = make_repetition_system()
    p0 = 0.3
    n_trials = 50_000
    rng = np.random.default_rng(42)

    failures = 0
    for _ in range(n_trials):
        e = (rng.random(3) < p0).astype(np.uint8)
        if is_failure(e, H, A, dec):
            failures += 1

    mc_ler   = failures / n_trials
    exact    = p_exact_repetition(p0)
    rel_err  = abs(mc_ler - exact) / exact

    print(f"\nTest 1 — P(p0) bootstrap")
    print(f"  p0={p0},  MC={mc_ler:.5f},  exact={exact:.5f},  rel_err={rel_err:.3f}")
    assert rel_err < 0.05, \
        f"Monte Carlo P(p0) off by {rel_err:.1%} — need more trials or check is_failure"
    print("  PASSED")
    return mc_ler


# ---------------------------------------------------------------------------
# Test 2 — single ratio estimate recovers known ratio
# ---------------------------------------------------------------------------

def test_single_ratio():
    """
    For the repetition code, P(p_j)/P(p_{j-1}) is analytically known.
    Verify estimate_ratio recovers it within tolerance.
    """
    H, A, dec = make_repetition_system()
    N = 3

    p_prev = 0.3
    p_j    = 0.15
    exact_ratio = p_exact_repetition(p_j) / p_exact_repetition(p_prev)

    # Build chains by running Metropolis at each level
    def mstep(e, p):
        return metropolis_step(e, p, H, A, dec)

    # Seed from a known failing config
    E0 = np.array([1, 0, 1], dtype=np.uint8)   # in F

    chain_prev = [E0.copy()]
    for _ in range(20_000):
        chain_prev.append(mstep(chain_prev[-1], p_prev))

    chain_j = [chain_prev[-1].copy()]   # seed j from end of j-1
    for _ in range(20_000):
        chain_j.append(mstep(chain_j[-1], p_j))

    r_hat, c = estimate_ratio(chain_j, chain_prev, p_j, p_prev, N)
    rel_err   = abs(r_hat - exact_ratio) / exact_ratio

    print(f"\nTest 2 — single ratio estimate")
    print(f"  p_prev={p_prev}, p_j={p_j}")
    print(f"  exact ratio={exact_ratio:.6f},  r_hat={r_hat:.6f},  rel_err={rel_err:.3f}")
    assert rel_err < 0.10, \
        f"Ratio estimate off by {rel_err:.1%} — check estimate_ratio or chain length"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 3 — full splitting method recovers P(p_target)
# ---------------------------------------------------------------------------

def test_full_splitting_recovers_exact():
    """
    Run the full splitting method on the repetition code from p0=0.3
    down to p_target=0.05 and compare against the analytical P(p_target).

    This is the end-to-end verification. If this passes, the ratio
    accumulation, chain seeding, and convergence check are all working.
    """
    H, A, dec = make_repetition_system()
    N = 3

    # Build probability sequence using the paper's heuristic (Eq. 18)
    # p_{j+1} = p_j * 2^{-1/sqrt(w_j)},  w_j = max(D/2, p_j * N)
    p0    = 0.3
    D     = 2    # onset weight for repetition code
    ps    = [p0]
    while ps[-1] > 0.04:
        p_curr = ps[-1]
        w_j    = max(D / 2, p_curr * N)
        p_next = p_curr * (2 ** (-1.0 / math.sqrt(w_j)))
        ps.append(p_next)
    ps = np.array(ps)

    print(f"\nTest 3 — full splitting method")
    print(f"  p sequence ({len(ps)} levels): "
          f"{' → '.join(f'{p:.4f}' for p in ps)}")

    p_target = float(ps[-1])
    exact    = p_exact_repetition(p_target)

    # Bootstrap P(p0) from Monte Carlo
    P_p0 = test_p0_estimate()   # reuse Test 1

    def mstep(e, p):
        return metropolis_step(e, p, H, A, dec)

    # Run splitting — using simplified inline version without circuit dependency
    P_hat  = P_p0
    M_prev = None

    for j in range(len(ps)):
        p_j    = ps[j]
        p_prev = ps[j-1] if j > 0 else None
        p_next = ps[j+1] if j < len(ps)-1 else None

        E0 = M_prev[-1] if M_prev is not None else np.array([1, 0, 1], dtype=np.uint8)

        M = [E0.copy()]
        M, stats = extend_chain_until_converged(
            chain=M,
            p_j=p_j,
            p_prev=p_prev if p_prev else p_j,
            p_next=p_next if p_next else p_j,
            N=N,
            epsilon=0.25,
            t=len(ps)-1,
            T_init=5_000,
            metropolis_step_fn=mstep,
        )

        if j >= 1:
            P_hat, r_hat, _ = update_probability_estimate(
                P_hat, M, M_prev, p_j, p_prev, N, j
            )

        M_prev = M

    rel_err = abs(P_hat - exact) / exact
    print(f"\n  p_target={p_target:.5f}")
    print(f"  P_hat (splitting) = {P_hat:.6e}")
    print(f"  P_exact           = {exact:.6e}")
    print(f"  Relative error    = {rel_err:.3f}  (expect < 0.25 given ε=0.25)")
    assert rel_err < 0.5, \
        f"Splitting estimate off by {rel_err:.1%}. " \
        f"Expected < 50% for a single run with ε=0.25."
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 4 — multiple runs reduce variance (statistical consistency)
# ---------------------------------------------------------------------------

def test_variance_across_runs():
    """
    Run the splitting method 10 times with different seeds.
    The estimates should scatter around the exact value with std
    proportional to ε * P(p_target).

    This catches bugs where the estimate is biased rather than just noisy —
    a biased estimator will have all 10 runs on the same side of the truth.
    """
    H, A, dec  = make_repetition_system()
    N          = 3
    p0, p_j    = 0.3, 0.1
    exact      = p_exact_repetition(p_j)
    estimates  = []

    print(f"\nTest 4 — variance across 10 independent runs")
    print(f"  exact P(p={p_j}) = {exact:.6f}")

    for run_idx in range(10):
        rng = np.random.default_rng(run_idx * 100)

        # Monte Carlo P(p0)
        mc_fails = sum(
            1 for _ in range(20_000)
            if is_failure((rng.random(N) < p0).astype(np.uint8), H, A, dec)
        )
        P_p0 = mc_fails / 20_000

        # Two-level split: p0 → p_j directly
        def mstep(e, p, seed=run_idx):
            return metropolis_step(e, p, H, A, dec)

        E0 = np.array([1, 0, 1], dtype=np.uint8)

        M_prev = [E0.copy()]
        for _ in range(10_000):
            M_prev.append(mstep(M_prev[-1], p0))

        M_j = [M_prev[-1].copy()]
        for _ in range(10_000):
            M_j.append(mstep(M_j[-1], p_j))

        r_hat, _ = estimate_ratio(M_j, M_prev, p_j, p0, N)
        P_hat    = P_p0 * r_hat
        estimates.append(P_hat)
        print(f"  run {run_idx+1:>2d}: P_hat={P_hat:.6f}  r={r_hat:.5f}")

    estimates = np.array(estimates)
    mean_est  = estimates.mean()
    std_est   = estimates.std()
    bias      = (mean_est - exact) / exact

    # Count how many estimates are on each side of the truth
    above = int((estimates > exact).sum())
    below = 10 - above

    print(f"\n  mean={mean_est:.6f}  std={std_est:.6f}  "
          f"bias={bias:+.3f}  above/below={above}/{below}")

    assert abs(bias) < 0.15, \
        f"Estimator is biased by {bias:.1%} — all runs systematically " \
        f"{'above' if bias > 0 else 'below'} the truth. " \
        f"Check ratio direction (p_j vs p_prev) or chain seed ordering."
    assert above > 1 and below > 1, \
        f"All or nearly all runs on one side of truth ({above} above, {below} below) " \
        f"— likely a systematic bias in the estimator."
    print("  PASSED")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for test in [
        test_p0_estimate,
        test_single_ratio,
        test_full_splitting_recovers_exact,
        test_variance_across_runs,
    ]:
        print(f"\n{'='*55}\n  {test.__name__}\n{'='*55}")
        try:
            test()
        except AssertionError as e:
            print(f"  FAILED: {e}")
        except Exception as e:
            import traceback
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()