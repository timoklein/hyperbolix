"""Probe 1: which digamma direction preserves E[log ||v_spatial||] under concatenation."""

import numpy as np
from scipy.special import digamma

rng = np.random.default_rng(0)
S = 400_000

print(f"{'N':>3} {'d':>4} {'target':>9} {'hcat':>9} {'shipped':>9} {'fixed':>9} {'s_fix':>8} {'s_ship':>8} {'k=s*sqrtN':>10}")
for N, d in [(9, 32), (9, 16), (9, 64), (4, 32), (2, 32), (9, 3), (4, 2), (2, 8)]:
    v = rng.standard_normal((S, N, d))  # N Gaussian blocks, unit component variance
    target = np.mean(np.log(np.linalg.norm(v[:, 0, :], axis=-1)))
    hcat = np.mean(np.log(np.linalg.norm(v.reshape(S, N * d), axis=-1)))
    s_fix = np.exp(0.5 * (digamma(d / 2) - digamma(N * d / 2)))
    s_ship = np.exp(0.5 * (digamma(N * d / 2) - digamma(d / 2)))
    fixed = hcat + np.log(s_fix)
    shipped = hcat + np.log(s_ship)
    print(
        f"{N:>3} {d:>4} {target:>9.4f} {hcat:>9.4f} {shipped:>9.4f} {fixed:>9.4f} "
        f"{s_fix:>8.4f} {s_ship:>8.4f} {s_fix * np.sqrt(N):>10.4f}"
    )

# closed form: for chi_k, E[log chi_k] = 0.5*(digamma(k/2) + log 2)
print("\nclosed form check (unit-variance Gaussian, dim k):")
for k in [8, 32, 288]:
    v = rng.standard_normal((200_000, k))
    print(
        f"  k={k:>4} empirical {np.mean(np.log(np.linalg.norm(v, axis=-1))):.5f} closed {0.5 * (digamma(k / 2) + np.log(2)):.5f}"
    )
