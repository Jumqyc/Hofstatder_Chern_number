"""Reference implementation extracted verbatim from Hofstadter.ipynb (validation only)."""
import numpy as np
from math import gcd
from functools import lru_cache


@lru_cache(maxsize=None)
def mod_inv(a, m):
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1


def hofstadter_band(p: int, q: int, sep: int = 100):
    if gcd(p, q) != 1:
        raise ValueError("p and q must be coprime.")
    kx = np.linspace(-np.pi, np.pi, sep)
    ky = np.linspace(-np.pi / q, np.pi / q, sep)
    H = np.zeros((sep, sep, q, q), dtype=complex)
    for m in range(q):
        H[..., m, m] = 2 * np.cos(ky[:, None] + 2 * np.pi * m * p / q)
        H[..., m, (m + 1) % q] = np.exp(-1j * kx[None, :])
        H[..., (m + 1) % q, m] = np.exp(1j * kx[None, :])
    band = np.linalg.eigvalsh(H)
    bandtop = np.max(band, axis=(0, 1))
    bandbottom = np.min(band, axis=(0, 1))
    return bandtop, bandbottom


def coprime_pairs(n):
    for q in range(1, n + 1):
        for p in range(1, q):
            if gcd(p, q) == 1:
                yield (p, q)


def compute_cherns(mu, bandtop, bandbottom, p, q):
    M = len(mu)
    r = np.sum(bandtop < mu[:, None], axis=1)

    mask_lower = (r == 0) & (mu < bandbottom[0])
    mask_upper = (r == q) & (mu > bandtop[-1])

    mid_mask = (r > 0) & (r < q)
    mask_inner = np.zeros(M, dtype=bool)
    if np.any(mid_mask):
        r_mid = r[mid_mask]
        top_low = bandtop[r_mid - 1]
        bot_high = bandbottom[r_mid]
        gap_open = top_low < bot_high
        in_gap = (top_low < mu[mid_mask]) & (mu[mid_mask] < bot_high)
        mask_inner[mid_mask] = gap_open & in_gap

    inv_p = mod_inv(p, q)
    half = q // 2
    j = np.arange(1, q)
    t = (inv_p * j) % q
    t[t > half] -= q
    k_for_r = np.zeros(q + 1, dtype=int)
    k_for_r[1:q] = t
    chern = np.full(M, 999, dtype=int)
    chern[mask_lower] = 0
    chern[mask_upper] = 0
    chern[mask_inner] = k_for_r[r[mask_inner]]
    return chern


def _helper(args):
    p, q, sep, mu = args
    bandtop, bandbottom = hofstadter_band(p, q, sep)
    return bandtop, bandbottom, compute_cherns(mu, bandtop, bandbottom, p, q)


def main():
    import sys, time, json, argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--q_max", type=int, default=40)
    ap.add_argument("--sep", type=int, default=100)
    ap.add_argument("--n_mu", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=str, default="reference/ref_result.npz")
    args = ap.parse_args()

    from multiprocessing import Pool

    pairs = list(coprime_pairs(args.q_max))
    pairs.sort(key=lambda x: x[0] / x[1])
    mu = np.linspace(-4, 4, args.n_mu)

    tasks = [(p, q, args.sep, mu) for p, q in pairs]

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        results = pool.map(_helper, tasks)
    dt = time.perf_counter() - t0
    print(f"python wall time: {dt:.2f} s", flush=True)

    tops = [r[0] for r in results]
    bots = [r[1] for r in results]
    cherns = np.array([r[2] for r in results], dtype=int)
    chern_numbers = np.where(cherns == 999, np.nan, cherns)

    import pickle
    with open(args.out, "wb") as f:
        pickle.dump({"pairs": np.array(pairs), "mu": mu, "bandtop": tops,
                     "bandbottom": bots, "chern_numbers": chern_numbers,
                     "time": dt}, f)
    print(f"saved {args.out}  pairs={len(pairs)}  shape={chern_numbers.shape}")


if __name__ == "__main__":
    main()
