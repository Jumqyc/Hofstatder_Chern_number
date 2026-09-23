// hofstadter.cpp
// ---------------------------------------------------------------------------
// A C++ rewrite of Hofstadter.ipynb:
//   1. Builds the Hofstadter (Harper) Hamiltonian H(kx, ky) [q x q complex
//      Hermitian] on a (sep x sep) Brillouin-zone grid and diagonalises it.
//   2. For every coprime pair (p, q) with 1 <= p < q <= q_max (sorted by
//      phi = p/q) it computes the band edges
//         bandtop[i]    = max over (kx, ky) of eigenvalue i
//         bandbottom[i] = min over (kx, ky) of eigenvalue i
//   3. Evaluates the Chern number of every insulating gap on a mu grid of
//      n_mu points (exactly the mask/Diophantine logic of the notebook),
//      with 999 meaning "no gap" (NaN in the notebook plot).
//   4. Writes the results (pairs, chern matrix, band edges) and renders the
//      butterfly plot (coolwarm_r colourmap, log|C| normalisation, gray NaN)
//      to a PNG, reproducing the notebook figure.
//
// Parallelised with OpenMP over the (p,q) pairs; diagonalisation through
// Eigen's SelfAdjointEigenSolver (complex Hermitian, eigenvalues only).
// ---------------------------------------------------------------------------
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

using cd = std::complex<double>;
namespace chrono = std::chrono;

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
struct Pair {
    int p, q;
};

// All coprime pairs 1 <= p < q <= n  (as enumerated in the notebook), then
// sorted by the ratio p/q exactly like `pairs.sort(key=lambda x: x[0]/x[1])`.
static std::vector<Pair> coprime_pairs_sorted(int n) {
    std::vector<Pair> out;
    for (int q = 1; q <= n; ++q)
        for (int p = 1; p < q; ++p)
            if (std::gcd(p, q) == 1) out.push_back({p, q});
    std::sort(out.begin(), out.end(), [](const Pair& a, const Pair& b) {
        return double(a.p) / double(a.q) < double(b.p) / double(b.q);
    });
    return out;
}

// Literal translation of the notebook's extended-Euclid modular inverse.
static int mod_inv(int a, int m0) {
    int m = m0, x0 = 0, x1 = 1;
    while (a > 1) {
        int qq = a / m;          // floor division (a, m >= 0)
        int m_new = a % m;       // python: m, a = a % m, m
        int a_new = m;
        int x0_new = x1 - qq * x0;  // python: x0, x1 = x1 - q*x0, x0
        int x1_new = x0;
        a = a_new; m = m_new; x0 = x0_new; x1 = x1_new;
    }
    return x1 < 0 ? x1 + m0 : x1;
}

// kx grid points whose spectra are pairwise distinct.  The spectrum of H is
// invariant under kx -> kx + 2*pi/q (magnetic Brillouin zone) and kx -> -kx,
// so on the sep-point grid the (sep-1)/2 lower-left representatives collapse
// to ceil(P/2) classes, with P = (sep-1)/gcd(q, sep-1).  One index per class
// is returned (always in [0, (sep-1)/2]), so the four-fold (kx,ky) orbit
// sampling in band_edges still covers every spectrum of the full grid.
static std::vector<int> kx_orbit_reps(int q, int sep) {
    const int n = sep - 1;
    const int P = n / std::gcd(q, n);
    std::vector<int> reps;
    reps.reserve((size_t)(P / 2 + 1));
    for (int i = 0; i <= n / 2; ++i) {
        const int m = i % P;
        const int r = std::min(m, (P - m) % P);
        if (i == r) reps.push_back(i);
    }
    return reps;
}

// ---------------------------------------------------------------------------
// core physics: band edges of the Hofstadter Hamiltonian for pair (p,q)
// ---------------------------------------------------------------------------
// Band edges of H(kx,ky) over a (sep x sep) grid.  kx in [-pi,pi],
// ky in [-pi/q, pi/q] (numpy linspace, endpoint-inclusive) and
//   H[m][m]          = 2 cos(ky + 2*pi*m*p/q)
//   H[m][(m+1)%q]    = exp(-i kx)
//   H[(m+1)%q][m]    = exp(+i kx)
// Only the distinct kx representatives in `kx_reps` are solved (the omitted
// points have identical spectra).  The diagonal is precomputed once per ky
// instead of once per (kx,ky).
// out top/bot are length q; eigenvalues are sorted ascending at every k point.
// prof (only compiled with -DHOF_PROFILE): [total s, eig-solver s, #solves]
static void band_edges_dense(int p, int q, int sep,
                             const std::vector<int>& kx_reps, double* top,
                             double* bot, double* prof = nullptr) {
    (void)prof;
    const double twopi = 2.0 * M_PI;
    const double dkx = twopi / (sep - 1.0);
    const double dky = twopi / double(q) / (sep - 1.0);

#ifdef HOF_PROFILE
    double t_total = 0.0, t_eig = 0.0;
    long long n_solve = 0;
    double w0 = omp_get_wtime();
#endif

    for (int m = 0; m < q; ++m) {
        top[m] = -std::numeric_limits<double>::infinity();
        bot[m] = std::numeric_limits<double>::infinity();
    }

    // theta_m = 2*pi*m*p/q (mod 2 pi)  [used in the diagonal cos]
    std::vector<double> th(q);
    for (int m = 0; m < q; ++m) th[m] = twopi * double((long long)(m * p) % q) / q;

    // diagonal values for every ky grid point, shared by all kx
    std::vector<double> diag_table((size_t)sep * q);
    for (int j = 0; j < sep; ++j) {
        const double ky = -M_PI / q + j * dky;
        double* dj = &diag_table[(size_t)j * q];
        for (int m = 0; m < q; ++m) dj[m] = 2.0 * std::cos(ky + th[m]);
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(q);
    Eigen::MatrixXcd H(q, q);

    for (int i : kx_reps) {
        // Four-fold k-space symmetry of the spectrum (per band index, because
        // eigenvalues are sorted ascending and every orbit member shares the
        // same spectrum):
        //   * H(-kx, ky) = conj(H(kx, ky))       -> spectrum even in kx
        //   * H(-kx,-ky) = H(kx, ky)^T (Bloch)   -> spectrum even in (kx,ky)
        // Together: spectrum(kx,ky) = spectrum(±kx, ±ky).  Only one
        // representative per orbit {(i,j),(si,j),(i,sj),(si,sj)} is needed,
        // with si = sep-1-i, sj = sep-1-j, so sampling the lower-left
        // quadrant (i <= si and j <= sj) is exact and 4x cheaper.
        const int si = sep - 1 - i;
        const double kx = -M_PI + i * dkx;
        const cd z = cd(std::cos(kx), -std::sin(kx));   // exp(-i kx)
        const cd zc = std::conj(z);
        for (int j = 0; j < sep; ++j) {
            if (si == i) {              // odd sep: middle column
                const int sj = sep - 1 - j;
                if (sj < j) continue;
            } else if (j > sep - 1 - j) {
                continue;               // sample only j <= sep-1-j
            }
            const double* diag = &diag_table[(size_t)j * q];

            // fill the Hermitian matrix (Eigen's complex solver reads the full
            // matrix, so both triangles must be written and the rest zeroed)
            H.setZero();
            for (int m = 0; m < q; ++m) {
                H.coeffRef(m, m) = diag[m];
                const int mm = (m + 1) % q;
                H.coeffRef(m, mm) = z;
                H.coeffRef(mm, m) = zc;
            }
#ifdef HOF_PROFILE
            double tq = omp_get_wtime();
            es.compute(H, Eigen::EigenvaluesOnly);
            t_eig += omp_get_wtime() - tq;
            ++n_solve;
#else
            es.compute(H, Eigen::EigenvaluesOnly);
#endif
            const auto& ev = es.eigenvalues();
            for (int m = 0; m < q; ++m) {
                const double e = ev[m];
                if (e > top[m]) top[m] = e;
                if (e < bot[m]) bot[m] = e;
            }
        }
    }
#ifdef HOF_PROFILE
    t_total = omp_get_wtime() - w0;
    if (prof) {
        prof[0] = t_total;
        prof[1] = t_eig;
        prof[2] = (double)n_solve;
    }
#endif
}

// ---------------------------------------------------------------------------
// structured solver: the q x q cyclic tridiagonal H(kx,ky) is bordered by its
// leading (q-1)x(q-1) block B, an open chain with diagonal a_0..a_{q-2} and
// unit hopping.  With B = Q diag(mu) Q^T and
//     w_i = gamma * Q_{i,0} + Q_{i,q-2},    gamma = exp(-i q kx),
// the eigenvalues of H are the roots of the scalar secular equation
//     s(lam) = (a_{q-1} - lam) - sum_i |w_i|^2 / (mu_i - lam) = 0.
// s is strictly decreasing between the poles mu_i, so it has exactly one root
// in each interval (-inf,mu_0), (mu_0,mu_1), ..., (mu_{q-2},inf).  A bracketed
// Newton iteration with a bisection safeguard finds all q roots in O(q) per
// evaluation.  B is diagonalised once per ky and reused for every kx, so the
// per-kx work is O(q^2) instead of the dense O(q^3).
// ---------------------------------------------------------------------------
static inline double hof_secular(double lam, double a_last,
                                 const Eigen::VectorXd& mu,
                                 const Eigen::VectorXd& w2,
                                 double* absum = nullptr) {
    double sum = 0.0, asum = 0.0;
    for (int i = 0; i < mu.size(); ++i) {
        const double t = w2[i] / (mu[i] - lam);
        sum += t;
        asum += std::abs(t);
    }
    if (absum) *absum = asum;
    return (a_last - lam) - sum;
}
static inline double hof_secular_deriv(double lam, const Eigen::VectorXd& mu,
                                       const Eigen::VectorXd& w2) {
    double sum = 0.0;
    for (int i = 0; i < mu.size(); ++i) {
        const double d = mu[i] - lam;
        sum += w2[i] / (d * d);
    }
    return -1.0 - sum;
}
// good starting point: two-term quadratic for the outer intervals, weighted
// average of the two neighbouring poles for the inner ones
static double hof_initial_guess(int kk, int n, double a_last,
                                const Eigen::VectorXd& mu,
                                const Eigen::VectorXd& w2, double lo,
                                double hi) {
    double x;
    if (kk == 0) {
        const double A = w2[0], s0 = a_last + mu[0], p0 = a_last * mu[0] - A;
        x = 0.5 * (s0 - std::sqrt(std::max(0.0, s0 * s0 - 4.0 * p0)));
    } else if (kk == n) {
        const double A = w2[n - 1], s0 = a_last + mu[n - 1],
                     p0 = a_last * mu[n - 1] - A;
        x = 0.5 * (s0 + std::sqrt(std::max(0.0, s0 * s0 - 4.0 * p0)));
    } else {
        const double A = w2[kk - 1], B = w2[kk];
        x = (A * mu[kk] + B * mu[kk - 1]) / (A + B);
    }
    if (!(x > lo && x < hi)) x = 0.5 * (lo + hi);
    return x;
}
static double hof_solve_interval(double lo, double hi, double a_last,
                                 const Eigen::VectorXd& mu,
                                 const Eigen::VectorXd& w2, int kk, int n,
                                 double x0) {
    double x = (x0 > lo && x0 < hi)
                   ? x0
                   : hof_initial_guess(kk, n, a_last, mu, w2, lo, hi);
    for (int it = 0; it < 100; ++it) {
        double absum = 0.0;
        const double fx = hof_secular(x, a_last, mu, w2, &absum);
        if (std::abs(fx) <= 1e-14 * (1.0 + absum)) break;
        if (fx > 0) lo = x; else hi = x;
        if (hi - lo <= 1e-15 * std::max(1.0, std::abs(x))) break;
        const double d = hof_secular_deriv(x, mu, w2);
        double xn = x - fx / d;
        if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
        x = xn;
    }
    return x;
}
static void band_edges_structured(int p, int q, int sep,
                                  const std::vector<int>& kx_reps,
                                  double* top, double* bot, double* prof) {
    (void)prof;
    const double twopi = 2.0 * M_PI;
    const double dkx = twopi / (sep - 1.0);
    const double dky = twopi / double(q) / (sep - 1.0);
#ifdef HOF_PROFILE
    double t_total = 0.0, t_eig = 0.0;
    long long n_solve = 0;
    double w0 = omp_get_wtime();
#endif
    for (int m = 0; m < q; ++m) {
        top[m] = -std::numeric_limits<double>::infinity();
        bot[m] = std::numeric_limits<double>::infinity();
    }
    std::vector<double> th(q);
    for (int m = 0; m < q; ++m)
        th[m] = twopi * double((long long)(m * p) % q) / q;

    const int n = q - 1;
    Eigen::VectorXd dm(n), em(n - 1);
    Eigen::VectorXd mu(n), w2(n);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esb(n);
    std::vector<double> a(q), roots(q), prev(q);
    // dense fallback for the rare degenerate case
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(q);
    Eigen::MatrixXcd H(q, q);
    const double nan = std::numeric_limits<double>::quiet_NaN();

    for (int j = 0; j <= (sep - 1) / 2; ++j) {
        const double ky = -M_PI / q + j * dky;
        for (int m = 0; m < q; ++m) a[m] = 2.0 * std::cos(ky + th[m]);
#ifdef HOF_PROFILE
        const double te0 = omp_get_wtime();
#endif
        for (int m = 0; m < n; ++m) dm[m] = a[m];
        for (int m = 0; m + 1 < n; ++m) em[m] = 1.0;
        esb.computeFromTridiagonal(dm, em, Eigen::ComputeEigenvectors);
        for (int i = 0; i < n; ++i) mu[i] = esb.eigenvalues()[i];
#ifdef HOF_PROFILE
        t_eig += omp_get_wtime() - te0;
#endif
        const double amin = *std::min_element(a.begin(), a.end());
        const double amax = *std::max_element(a.begin(), a.end());
        bool first = true;
        for (int i : kx_reps) {
            const double kx = -M_PI + i * dkx;
            const cd gamma(std::cos(q * kx), -std::sin(q * kx));
            double w2min = std::numeric_limits<double>::infinity();
            for (int k = 0; k < n; ++k) {
                const cd w = gamma * esb.eigenvectors()(0, k) +
                             esb.eigenvectors()(n - 1, k);
                w2[k] = std::norm(w);
                if (w2[k] < w2min) w2min = w2[k];
            }
            bool ok = (w2min > 1e-300);
            if (ok) {
                roots[0] = hof_solve_interval(amin - 2.0, mu[0], a[q - 1],
                                              mu, w2, 0, n, first ? nan : prev[0]);
                for (int k = 1; k < n; ++k)
                    roots[k] = hof_solve_interval(mu[k - 1], mu[k], a[q - 1],
                                                  mu, w2, k, n,
                                                  first ? nan : prev[k]);
                roots[n] = hof_solve_interval(mu[n - 1], amax + 2.0, a[q - 1],
                                              mu, w2, n, n, first ? nan : prev[n]);
                for (int k = 0; k < q; ++k)
                    if (!std::isfinite(roots[k])) ok = false;
            }
            if (!ok) {
                const cd z(std::cos(kx), -std::sin(kx));
                const cd zc = std::conj(z);
                H.setZero();
                for (int m = 0; m < q; ++m) {
                    H.coeffRef(m, m) = a[m];
                    const int mm = (m + 1) % q;
                    H.coeffRef(m, mm) = z;
                    H.coeffRef(mm, m) = zc;
                }
                es.compute(H, Eigen::EigenvaluesOnly);
                for (int k = 0; k < q; ++k) roots[k] = es.eigenvalues()[k];
            }
            for (int k = 0; k < q; ++k) {
                const double e = roots[k];
                if (e > top[k]) top[k] = e;
                if (e < bot[k]) bot[k] = e;
            }
            prev = roots;
            first = false;
#ifdef HOF_PROFILE
            ++n_solve;
#endif
        }
    }
#ifdef HOF_PROFILE
    t_total = omp_get_wtime() - w0;
    if (prof) {
        prof[0] = t_total;
        prof[1] = t_eig;
        prof[2] = (double)n_solve;
    }
#endif
}

// dispatch: structured solver for q >= 3, dense reference/fallback otherwise
static void band_edges(int p, int q, int sep, const std::vector<int>& kx_reps,
                       double* top, double* bot, bool use_structured,
                       double* prof = nullptr) {
    if (use_structured && q >= 3)
        band_edges_structured(p, q, sep, kx_reps, top, bot, prof);
    else
        band_edges_dense(p, q, sep, kx_reps, top, bot, prof);
}

// ---------------------------------------------------------------------------
// Chern numbers on a mu grid (verbatim notebook logic)
// ---------------------------------------------------------------------------
// r(mu) = number of bands whose top edge lies below mu.
// Returns the chern row (length M): 0 below/above the spectrum, k_for_r[r]
// inside the r-th open gap, and 999 when mu is not in a gap.
static void compute_cherns(const std::vector<double>& mu, const double* bandtop,
                           const double* bandbottom, int p, int q,
                           int* chern_out) {
    const int M = (int)mu.size();
    std::vector<int> r(M);
    for (int a = 0; a < M; ++a) {
        int cnt = 0;
        for (int m = 0; m < q; ++m)
            if (bandtop[m] < mu[a]) ++cnt;
        r[a] = cnt;
    }

    // Diophantine helper k_for_r (same construction as the notebook)
    int inv_p = mod_inv(p, q);
    int half = q / 2;
    std::vector<int> k_for_r(q + 1, 0);
    for (int j = 1; j < q; ++j) {
        int t = (inv_p * j) % q;
        if (t > half) t -= q;
        k_for_r[j] = t;
    }

    for (int a = 0; a < M; ++a) {
        const double mu_a = mu[a];
        int val = 999;  // default: no gap
        if (r[a] == 0 && mu_a < bandbottom[0]) val = 0;
        else if (r[a] == q && mu_a > bandtop[q - 1]) val = 0;
        else if (r[a] > 0 && r[a] < q) {
            const double top_low = bandtop[r[a] - 1];
            const double bot_high = bandbottom[r[a]];
            const bool gap_open = top_low < bot_high;
            const bool in_gap = (top_low < mu_a) && (mu_a < bot_high);
            if (gap_open && in_gap) val = k_for_r[r[a]];
        }
        chern_out[a] = val;
    }
}

// ---------------------------------------------------------------------------
// output helpers
// ---------------------------------------------------------------------------
static void write_csv(const std::string& path, const std::vector<int>& data,
                      int npairs, int M) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::perror(path.c_str()); std::exit(1); }
    for (int row = 0; row < npairs; ++row) {
        for (int c = 0; c < M; ++c)
            std::fprintf(f, "%s%d", c ? "," : "", data[(size_t)row * M + c]);
        std::fputc('\n', f);
    }
    std::fclose(f);
}

// ---------------------------------------------------------------------------
// butterfly renderer (notebook plotting semantics, coolwarm_r + log|C|)
// ---------------------------------------------------------------------------
#include "colormap.h"  // COOLWARM_R_LUT: 256 {r,g,b} bytes generated from matplotlib

static void render_png(const char* path, const std::vector<int>& chern,
                       int npairs, int M, int scale) {
    // statistics over the non-999 entries
    double cmin = 1e300, cmax = -1e300;
    for (int v : chern) {
        if (v != 999) {
            if (v < cmin) cmin = v;
            if (v > cmax) cmax = v;
        }
    }
    if (cmin > cmax) {
        std::fprintf(stderr, "all chern values are NaN, nothing to plot\n");
        return;
    }
    const double max_abs = std::max(std::fabs(cmin), std::fabs(cmax));
    const double vmax_trans = std::log1p(max_abs);  // log1p in C++ == numpy log1p

    const int W = npairs * scale, Hgt = M * scale;
    std::vector<unsigned char> img((size_t)W * Hgt * 3);

    for (int row = 0; row < M; ++row) {          // mu ascending
        const double mu_t = row;                 // (used only for readability)
        (void)mu_t;
        for (int col = 0; col < npairs; ++col) {
            const int v = chern[(size_t)col * M + row];
            unsigned char r, g, b;
            if (v == 999) {
                r = g = b = 128;                 // matplotlib gray for NaN
            } else {
                const double f = std::copysign(std::log1p(std::fabs(double(v))),
                                               double(v));
                double t = 0.5 * (f / vmax_trans + 1.0);  // in [0,1]
                int idx = (int)std::lround(t * 255.0);
                if (idx < 0) idx = 0;
                if (idx > 255) idx = 255;
                r = COOLWARM_R_LUT[idx][0];
                g = COOLWARM_R_LUT[idx][1];
                b = COOLWARM_R_LUT[idx][2];
            }
            for (int sy = 0; sy < scale; ++sy)
                for (int sx = 0; sx < scale; ++sx) {
                    size_t px = ((size_t)(M - 1 - row) * scale + sy) * W
                              + ((size_t)col * scale + sx);
                    img[px * 3 + 0] = r;
                    img[px * 3 + 1] = g;
                    img[px * 3 + 2] = b;
                }
        }
    }
    if (!stbi_write_png(path, W, Hgt, 3, img.data(), W * 3))
        std::fprintf(stderr, "failed to write %s\n", path);
    else
        std::printf("wrote %s (%d x %d)\n", path, W, Hgt);
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
static double get_arg(const std::vector<std::string>& argv, const char* name,
                      double dflt) {
    for (size_t i = 1; i < argv.size(); ++i)
        if (argv[i] == name && i + 1 < argv.size())
            return std::atof(argv[i + 1].c_str());
    return dflt;
}
static std::string get_arg(const std::vector<std::string>& argv, const char* name,
                           const char* dflt) {
    for (size_t i = 1; i < argv.size(); ++i)
        if (argv[i] == name && i + 1 < argv.size()) return argv[i + 1];
    return dflt ? dflt : "";
}
static bool has_flag(const std::vector<std::string>& argv, const char* name) {
    for (size_t i = 1; i < argv.size(); ++i)
        if (argv[i] == name) return true;
    return false;
}

int main(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
    const int q_max = (int)get_arg(args, "--q_max", 40.0);
    const int sep = (int)get_arg(args, "--sep", 100.0);
    const int n_mu = (int)get_arg(args, "--n_mu", 1000.0);
    int threads = (int)get_arg(args, "--threads", 0.0);
    const std::string outdir = get_arg(args, "--outdir", "out");
    std::string png_path = get_arg(args, "--png", "");
    const std::string edges_path = get_arg(args, "--edges", "");
    const int scale = (int)get_arg(args, "--scale", 2);
    const bool use_structured = !has_flag(args, "--dense");

#ifdef _OPENMP
    if (threads <= 0) threads = omp_get_max_threads();
    omp_set_num_threads(threads);
#else
    threads = 1;
#endif
    if (has_flag(args, "--help") || has_flag(args, "-h")) {
        std::printf("usage: hofstadter [--q_max N] [--sep N] [--n_mu N]"
                    " [--threads N] [--outdir D] [--png P] [--edges F]\n"
                    "  [--scale S] [--no-png] [--dense]\n");
        return 0;
    }

    std::filesystem::create_directories(outdir);

    const std::vector<Pair> pairs = coprime_pairs_sorted(q_max);
    const int npairs = (int)pairs.size();
    std::printf("q_max=%d sep=%d n_mu=%d pairs=%d threads=%d solver=%s\n",
                q_max, sep, n_mu, npairs, threads,
                use_structured ? "structured" : "dense");

    std::vector<double> mu(n_mu);
    const double dmu = 8.0 / (n_mu - 1.0);
    for (int i = 0; i < n_mu; ++i) mu[i] = -4.0 + i * dmu;

    // band edges, flat storage: per pair, q values of top then q of bottom
    std::vector<double> edges((size_t)npairs * 2 * q_max);
    std::vector<int> chern((size_t)npairs * n_mu, 999);

    // p <-> q-p symmetry: H^(q-p)(kx,ky) = H^(p)(kx,-ky) and the ky grid is
    // symmetric, so the two pairs have identical band edges.  Solve only the
    // canonical p < q/2 pairs and copy the edges to their mirrors; the Chern
    // numbers still use each pair's own p (cheap).
    std::vector<int> mirror(npairs, -1);
    {
        const int stride = q_max + 1;
        std::vector<int> index_of((size_t)stride * stride, -1);
        for (int idx = 0; idx < npairs; ++idx)
            index_of[(size_t)pairs[idx].p * stride + pairs[idx].q] = idx;
        for (int idx = 0; idx < npairs; ++idx) {
            const Pair& pr = pairs[idx];
            const int p2 = pr.q - pr.p;
            mirror[idx] = (p2 == pr.p)
                              ? idx
                              : index_of[(size_t)p2 * stride + pr.q];
        }
    }
    long long n_solved = 0, n_mirrored = 0;

    const auto t0 = chrono::steady_clock::now();
#ifdef HOF_PROFILE
    std::vector<double> prof((size_t)npairs * 3, 0.0);
    std::vector<double>* profp = &prof;
#else
    std::vector<double>* profp = nullptr;
#endif
#pragma omp parallel for schedule(dynamic, 1)
    for (int idx = 0; idx < npairs; ++idx) {
        const Pair& pr = pairs[idx];
        if (pr.p * 2 > pr.q) continue;      // mirror: filled in below
        double* top = &edges[(size_t)idx * 2 * q_max];
        double* bot = top + q_max;
        const std::vector<int> reps = kx_orbit_reps(pr.q, sep);
        band_edges(pr.p, pr.q, sep, reps, top, bot, use_structured,
                   profp ? &(*profp)[(size_t)idx * 3] : nullptr);
    }
    // Copy mirrored band edges and evaluate every pair's Chern numbers.
#pragma omp parallel for schedule(dynamic, 1)
    for (int idx = 0; idx < npairs; ++idx) {
        const Pair& pr = pairs[idx];
        double* top = &edges[(size_t)idx * 2 * q_max];
        double* bot = top + q_max;
        if (pr.p * 2 > pr.q) {              // mirror: reuse canonical edges
            const double* src = &edges[(size_t)mirror[idx] * 2 * q_max];
            std::copy(src, src + 2 * q_max, top);
        }
        compute_cherns(mu, top, bot, pr.p, pr.q,
                       &chern[(size_t)idx * n_mu]);
    }
    for (int idx = 0; idx < npairs; ++idx) {
        if (pairs[idx].p * 2 > pairs[idx].q) ++n_mirrored; else ++n_solved;
    }
    std::printf("solved pairs=%lld (mirrored=%lld)\n", n_solved, n_mirrored);
    const double compute_s =
        chrono::duration<double>(chrono::steady_clock::now() - t0).count();

#ifdef HOF_PROFILE
    {
        double sum_tot = 0, sum_eig = 0;
        long long sum_n = 0;
        for (int idx = 0; idx < npairs; ++idx) {
            if (pairs[idx].p * 2 > pairs[idx].q) continue;  // mirrored
            sum_tot += prof[(size_t)idx * 3 + 0];
            sum_eig += prof[(size_t)idx * 3 + 1];
            sum_n += (long long)prof[(size_t)idx * 3 + 2];
        }
        std::printf("PROFILE wall=%.3f s | sum(pair cpu)=%.3f s | "
                    "eig-solver=%.3f s (%.1f%%) | solve-calls=%lld "
                    "(avg %.2f us/solve)\n",
                    compute_s, sum_tot, sum_eig,
                    100.0 * sum_eig / (sum_tot > 0 ? sum_tot : 1.0), sum_n,
                    sum_tot > 0 ? sum_tot / sum_n * 1e6 : 0.0);
        // per-pair breakdown (p,q,total_s,eig_s,nsolve)
        FILE* pf_ = std::fopen((outdir + "/prof.csv").c_str(), "w");
        for (int idx = 0; idx < npairs; ++idx) {
            const Pair& pr = pairs[idx];
            std::fprintf(pf_, "%d,%d,%.6f,%.6f,%.0f\n", pr.p, pr.q,
                         prof[(size_t)idx * 3 + 0], prof[(size_t)idx * 3 + 1],
                         prof[(size_t)idx * 3 + 2]);
        }
        std::fclose(pf_);
        std::printf("wrote %s/prof.csv\n", outdir.c_str());
    }
#endif

    std::printf("computation wall time: %.3f s\n", compute_s);

    // ---- write text outputs ----
    FILE* meta = std::fopen((outdir + "/meta.txt").c_str(), "w");
    std::fprintf(meta, "q_max=%d\nsep=%d\nn_mu=%d\nthreads=%d\nnpairs=%d\n",
                 q_max, sep, n_mu, threads, npairs);
    std::fprintf(meta, "mu_min=-4\nmu_max=4\n");
    std::fprintf(meta, "wall_time=%.3f\n", compute_s);
    std::fclose(meta);

    FILE* pf = std::fopen((outdir + "/pairs.csv").c_str(), "w");
    for (const Pair& pr : pairs) std::fprintf(pf, "%d,%d\n", pr.p, pr.q);
    std::fclose(pf);

    write_csv(outdir + "/chern.csv", chern, npairs, n_mu);

    if (!edges_path.empty()) {
        FILE* ef = std::fopen(edges_path.c_str(), "w");
        for (int idx = 0; idx < npairs; ++idx) {
            const Pair& pr = pairs[idx];
            std::fprintf(ef, "%d %d", pr.p, pr.q);
            const double* e = &edges[(size_t)idx * 2 * q_max];
            for (int m = 0; m < pr.q; ++m) std::fprintf(ef, " %.17g", e[m]);
            for (int m = 0; m < pr.q; ++m) std::fprintf(ef, " %.17g", e[q_max + m]);
            std::fputc('\n', ef);
        }
        std::fclose(ef);
        std::printf("wrote %s\n", edges_path.c_str());
    }

    std::printf("wrote %s/chern.csv, %s/pairs.csv, %s/meta.txt\n", outdir.c_str(),
                outdir.c_str(), outdir.c_str());

    // ---- optional figure ----
    if (png_path.empty()) png_path = outdir + "/Chern_numbers.png";
    if (!has_flag(args, "--no-png"))
        render_png(png_path.c_str(), chern, npairs, n_mu, scale);

    return 0;
}
