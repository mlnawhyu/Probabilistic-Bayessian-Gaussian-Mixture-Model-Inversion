"""
inversion_utils.py — Utilities untuk Bayesian AVO inversion (NB6).
GPU-accelerated via gpu_utils (auto-fallback ke CPU).

Components
----------
1. Aki-Richards angle-dependent reflectivity coefficient matrix
2. Wavelet → Toeplitz convolution operator
3. Forward operator G = W · A · D
4. Bayesian linearized inversion (Buland-Omre 2003)
5. Vertical exponential covariance Σ_m
6. GMM-based petrophysical inversion (Grana 2010 eq. 5-7)
7. Calibration helpers
8. Lateral post-processing
"""
from __future__ import annotations

from typing import Tuple, Dict, List, Optional

import numpy as np
from scipy.linalg import toeplitz, cho_factor, cho_solve
from scipy.ndimage import gaussian_filter

# GPU support — auto-fallback ke CPU jika tidak ada
try:
    from gpu_utils import (
        GPU_AVAILABLE, xp, asgpu, asnumpy,
        gpu_cholesky, gpu_cho_solve, gpu_solve, gpu_inv,
        gpu_percentile, gpu_default_rng, free_gpu_memory
    )
except ImportError:
    GPU_AVAILABLE = False
    xp = np
    def asgpu(a): return np.asarray(a)
    def asnumpy(a): return np.asarray(a)
    def gpu_cholesky(A, lower=True, ridge=0.0):
        if ridge > 0: A = A + ridge * np.eye(A.shape[0])
        c, low = cho_factor(A, lower=lower)
        return np.tril(c) if (lower == low) else np.tril(c.T)
    def gpu_cho_solve(L, b, lower=True): return cho_solve((L, lower), b)
    def gpu_solve(A, b): return np.linalg.solve(A, b)
    def gpu_inv(A): return np.linalg.inv(A)
    def gpu_percentile(a, q, axis=None): return np.percentile(a, q, axis=axis)
    def gpu_default_rng(seed=None): return np.random.default_rng(seed)
    def free_gpu_memory(): pass

EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════
#                       AKI-RICHARDS COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════════
def aki_richards_matrix(theta_deg: np.ndarray,
                         vp_bg: np.ndarray, vs_bg: np.ndarray
                         ) -> np.ndarray:
    """
    Aki-Richards 3-term linearized reflectivity matrix.

    For each angle θ and each interface i, compute:
      R(θ) = a(θ) · ΔlnVP/2 + b(θ) · ΔlnVS/2 + c(θ) · Δlnρ/2

    where (with γ = VS/VP background):
      a(θ) = 1 + tan²(θ)
      b(θ) = -8 γ² sin²(θ)
      c(θ) = 1 - 4 γ² sin²(θ)

    Parameters
    ----------
    theta_deg : (NA,) angles in degrees
    vp_bg, vs_bg : (NT,) background VP, VS at each sample (interfaces use mean)

    Returns
    -------
    A : (NA, NT-1, 3) coefficient tensor — for each angle, each interface,
        the [a, b, c] terms used in the differential operator.
    """
    NA = len(theta_deg)
    NT = len(vp_bg)
    NI = NT - 1   # number of interfaces

    # Background γ at interface = mean of the two adjacent samples
    vp_int = 0.5 * (vp_bg[:-1] + vp_bg[1:])
    vs_int = 0.5 * (vs_bg[:-1] + vs_bg[1:])
    gamma  = vs_int / np.maximum(vp_int, EPS)
    gamma2 = gamma**2

    A = np.zeros((NA, NI, 3))
    for i_a, theta in enumerate(np.radians(theta_deg)):
        a = 0.5 * (1 + np.tan(theta)**2)
        b = -4.0 * gamma2 * np.sin(theta)**2
        c = 0.5 * (1 - 4.0 * gamma2 * np.sin(theta)**2)
        A[i_a, :, 0] = a   # ΔlnVP coefficient
        A[i_a, :, 1] = b   # ΔlnVS coefficient
        A[i_a, :, 2] = c   # Δlnρ coefficient
    return A


# ═══════════════════════════════════════════════════════════════════════════
#                       WAVELET TOEPLITZ MATRIX
# ═══════════════════════════════════════════════════════════════════════════
def make_wavelet_matrix(wavelet: np.ndarray, n_out: int) -> np.ndarray:
    """
    Build (n_out × n_out) Toeplitz convolution matrix for given wavelet.
    Wavelet should be zero-phase (or pre-shifted) with center at len(w)//2.
    """
    w = np.asarray(wavelet, float)
    nw = len(w)
    half = nw // 2

    # Pad/truncate to fit n_out
    col = np.zeros(n_out)
    row = np.zeros(n_out)
    for i, wi in enumerate(w):
        lag = i - half
        if lag >= 0 and lag < n_out:
            col[lag] = wi
        if -lag >= 0 and -lag < n_out:
            row[-lag] = wi
    return toeplitz(col, row)


# ═══════════════════════════════════════════════════════════════════════════
#                       FORWARD OPERATOR G
# ═══════════════════════════════════════════════════════════════════════════
def build_forward_operator(theta_deg: np.ndarray,
                            vp_bg: np.ndarray, vs_bg: np.ndarray,
                            wavelet: np.ndarray
                            ) -> np.ndarray:
    """
    Build the full forward operator G mapping log-elastic m to seismic d.

    Model vector m ∈ R^{3·NT}: [ln VP_1..NT, ln VS_1..NT, ln ρ_1..NT]
    Data vector  d ∈ R^{NA·(NT-1)}: stacked angle gathers

    G = block_stack_over_angles( W · A_θ · D )
        where D is the (NT-1)×NT first-difference operator.

    Returns
    -------
    G : (NA·(NT-1), 3·NT) matrix
    """
    NA = len(theta_deg)
    NT = len(vp_bg)
    NI = NT - 1
    NM = 3 * NT      # model dimension
    ND = NA * NI     # data dimension

    A = aki_richards_matrix(theta_deg, vp_bg, vs_bg)   # (NA, NI, 3)

    # Differential operator: D[i, i] = -1, D[i, i+1] = +1
    D = np.zeros((NI, NT))
    for i in range(NI):
        D[i, i]   = -1.0
        D[i, i+1] =  1.0

    W = make_wavelet_matrix(wavelet, NI)  # (NI, NI)

    G = np.zeros((ND, NM))
    for i_a in range(NA):
        # For this angle, build (NI, NM) sub-matrix
        # R(θ) = a·D·ln_VP + b·D·ln_VS + c·D·ln_ρ
        # then convolve with W
        a_diag = A[i_a, :, 0]
        b_diag = A[i_a, :, 1]
        c_diag = A[i_a, :, 2]
        R_block = np.hstack([a_diag[:, None] * D,
                             b_diag[:, None] * D,
                             c_diag[:, None] * D])   # (NI, 3·NT)
        G[i_a*NI:(i_a+1)*NI, :] = W @ R_block
    return G


# ═══════════════════════════════════════════════════════════════════════════
#                   VERTICAL EXPONENTIAL COVARIANCE Σ_m
# ═══════════════════════════════════════════════════════════════════════════
def vertical_covariance(NT: int, dt: float, corr_length_s: float,
                         sigma0: np.ndarray
                         ) -> np.ndarray:
    """
    Construct Σ_m = Σ₀ ⊗ exp(-|τ|/L) (Buland-Omre 2003).

    Parameters
    ----------
    NT : number of time samples
    dt : sample interval (s)
    corr_length_s : vertical correlation length L (s)
    sigma0 : (3, 3) prior covariance of [ln VP, ln VS, ln ρ]

    Returns
    -------
    Sigma_m : (3·NT, 3·NT) covariance matrix
    """
    # Time-correlation matrix
    t = np.arange(NT) * dt
    tau = np.abs(t[:, None] - t[None, :])
    C_t = np.exp(-tau / max(corr_length_s, 1e-6))

    # Σ_m = Σ₀ ⊗ C_t (Kronecker, but block-stacked layout)
    Sigma_m = np.zeros((3 * NT, 3 * NT))
    for i in range(3):
        for j in range(3):
            Sigma_m[i*NT:(i+1)*NT, j*NT:(j+1)*NT] = sigma0[i, j] * C_t
    return Sigma_m


def diag_noise(NA: int, NI: int, noise_var: float,
                noise_scale_per_angle: List[float]
                ) -> np.ndarray:
    """Diagonal noise covariance Σ_e per angle."""
    sigma_e = np.zeros(NA * NI)
    for i_a in range(NA):
        sigma_e[i_a*NI:(i_a+1)*NI] = noise_var * noise_scale_per_angle[i_a]**2
    return np.diag(sigma_e)


# ═══════════════════════════════════════════════════════════════════════════
#               BAYESIAN LINEARIZED INVERSION (per trace)
# ═══════════════════════════════════════════════════════════════════════════
def bayes_invert_one_trace(d: np.ndarray, G: np.ndarray,
                            mu_prior: np.ndarray, Sigma_m: np.ndarray,
                            Sigma_e: np.ndarray,
                            return_post_cov: bool = False,
                            ridge: float = 1e-8
                            ) -> Dict:
    """
    Closed-form Bayesian linearized AVO inversion (Buland-Omre 2003).

    m_post = μ + Σ_m G^T (G Σ_m G^T + Σ_e)^{-1} (d - G μ)
    Σ_post = Σ_m - Σ_m G^T (G Σ_m G^T + Σ_e)^{-1} G Σ_m

    Returns
    -------
    dict with 'mu_post' (NM,), and optionally 'Sigma_post' (NM, NM).
    """
    H = G @ Sigma_m @ G.T + Sigma_e
    H += ridge * np.trace(H) / H.shape[0] * np.eye(H.shape[0])
    try:
        c, low = cho_factor(H, lower=True)
    except np.linalg.LinAlgError:
        # Fall back to small extra ridge
        H += 1e-6 * np.trace(H) / H.shape[0] * np.eye(H.shape[0])
        c, low = cho_factor(H, lower=True)

    innov = d - G @ mu_prior
    alpha = cho_solve((c, low), innov)
    mu_post = mu_prior + Sigma_m @ G.T @ alpha

    out = {"mu_post": mu_post}
    if return_post_cov:
        K = cho_solve((c, low), G @ Sigma_m)
        Sigma_post = Sigma_m - Sigma_m @ G.T @ K
        # Ensure symmetry
        Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)
        out["Sigma_post"] = Sigma_post
    return out


def extract_post_std(Sigma_post: np.ndarray, NT: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract per-component posterior standard deviations from Σ_post.

    Returns
    -------
    sigma_lnVP, sigma_lnVS, sigma_lnRho : (NT,) each
    """
    diag = np.diag(Sigma_post)
    sigma_lnVP  = np.sqrt(np.maximum(diag[:NT],          0.0))
    sigma_lnVS  = np.sqrt(np.maximum(diag[NT:2*NT],      0.0))
    sigma_lnRho = np.sqrt(np.maximum(diag[2*NT:3*NT],    0.0))
    return sigma_lnVP, sigma_lnVS, sigma_lnRho


# ═══════════════════════════════════════════════════════════════════════════
#                   GMM-BASED PETROPHYSICAL INVERSION
#               (Grana & Della Rossa 2010, eq. 5-7)
# ═══════════════════════════════════════════════════════════════════════════
def gmm_conditional_params(gmm, m_idx: List[int], r_idx: List[int]) -> List[Dict]:
    """
    Pre-compute conditional GMM parameters per component.

    For each component k with full joint covariance:
       Σ_k = [[Σ_RR, Σ_Rm], [Σ_mR, Σ_mm]]

    Conditional R | m for component k:
       μ_R|m^k = μ_R^k + A_k (m - μ_m^k),  with A_k = Σ_Rm Σ_mm^{-1}
       Σ_R|m^k = Σ_RR - A_k Σ_mR

    Parameters
    ----------
    gmm : fitted sklearn.mixture.GaussianMixture
    m_idx : indices of elastic features in gmm.means_ (e.g. [3, 4, 5] for VP,VS,ρ)
    r_idx : indices of petro features (e.g. [0, 1, 2] for PHIE, VSH, SW)

    Returns
    -------
    list of dicts per component: {alpha, mu_m, mu_R, Sig_mm, Sig_mm_inv, A,
                                   Sig_R_m, mu_full, Sig_full}
    """
    K = gmm.n_components
    params = []
    for k in range(K):
        mu_full = gmm.means_[k]
        Sig_full = gmm.covariances_[k]
        mu_m = mu_full[m_idx]
        mu_R = mu_full[r_idx]
        Sig_mm = Sig_full[np.ix_(m_idx, m_idx)]
        Sig_RR = Sig_full[np.ix_(r_idx, r_idx)]
        Sig_Rm = Sig_full[np.ix_(r_idx, m_idx)]
        Sig_mR = Sig_full[np.ix_(m_idx, r_idx)]
        try:
            Sig_mm_inv = np.linalg.inv(Sig_mm + 1e-9 * np.eye(len(m_idx)))
        except np.linalg.LinAlgError:
            Sig_mm_inv = np.linalg.pinv(Sig_mm)
        A = Sig_Rm @ Sig_mm_inv
        Sig_R_m = Sig_RR - A @ Sig_mR
        Sig_R_m = 0.5 * (Sig_R_m + Sig_R_m.T)
        params.append({
            "alpha": float(gmm.weights_[k]),
            "mu_m": mu_m, "mu_R": mu_R,
            "Sig_mm": Sig_mm, "Sig_mm_inv": Sig_mm_inv,
            "A": A, "Sig_R_m": Sig_R_m,
        })
    return params


def gmm_responsibilities(m: np.ndarray, gmm_params: List[Dict]) -> np.ndarray:
    """
    Compute λ_k(m) responsibilities for sample(s) m in elastic space.

    Parameters
    ----------
    m : (D_m,) or (Ns, D_m)
    gmm_params : output of gmm_conditional_params

    Returns
    -------
    lam : (K,) or (Ns, K)
    """
    if m.ndim == 1:
        m_2d = m[None, :]
    else:
        m_2d = m

    K = len(gmm_params)
    Ns = m_2d.shape[0]
    log_pdf = np.zeros((Ns, K))
    for k, p in enumerate(gmm_params):
        diff = m_2d - p["mu_m"]                            # (Ns, D_m)
        maha = np.einsum("ij,jk,ik->i", diff, p["Sig_mm_inv"], diff)
        try:
            sign, logdet = np.linalg.slogdet(p["Sig_mm"])
        except np.linalg.LinAlgError:
            logdet = 0.0
            sign = 1.0
        D_m = m_2d.shape[1]
        log_pdf[:, k] = (np.log(p["alpha"] + EPS)
                         - 0.5 * D_m * np.log(2 * np.pi)
                         - 0.5 * logdet
                         - 0.5 * maha)
    log_pdf -= log_pdf.max(axis=1, keepdims=True)
    pdf = np.exp(log_pdf)
    lam = pdf / pdf.sum(axis=1, keepdims=True)
    if m.ndim == 1:
        return lam[0]
    return lam


def petro_posterior(m: np.ndarray, gmm_params: List[Dict]
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute petrophysical posterior P(R | m) at sample(s) m.

    Returns
    -------
    R_mean : (Ns, D_R) posterior mean
    R_var  : (Ns, D_R) posterior variance (diagonal of Σ_R|m^k summed properly)
    lam    : (Ns, K) responsibilities
    """
    if m.ndim == 1:
        m = m[None, :]
    Ns, D_m = m.shape
    K = len(gmm_params)
    D_R = gmm_params[0]["mu_R"].shape[0]

    lam = gmm_responsibilities(m, gmm_params)        # (Ns, K)
    R_mean_k = np.zeros((Ns, K, D_R))
    for k, p in enumerate(gmm_params):
        diff = m - p["mu_m"]
        R_mean_k[:, k, :] = p["mu_R"] + diff @ p["A"].T

    # Mixture mean
    R_mean = np.einsum("ij,ijk->ik", lam, R_mean_k)

    # Mixture variance: E[E[R|m,k]² ] - E[R|m]² + Σ λ_k Σ_R|m^k
    R_diff_k = R_mean_k - R_mean[:, None, :]            # (Ns, K, D_R)
    R_var = np.zeros((Ns, D_R))
    for k, p in enumerate(gmm_params):
        diag_R_m = np.diag(p["Sig_R_m"])
        R_var += lam[:, k:k+1] * (diag_R_m[None, :] + R_diff_k[:, k, :]**2)

    return R_mean, R_var, lam


def petro_ensemble_samples(m: np.ndarray, gmm_params: List[Dict],
                            n_samples: int, rng: Optional[np.random.Generator] = None
                            ) -> np.ndarray:
    """
    Draw ensemble samples from P(R | m) via mixture sampling — VECTORIZED.

    Returns
    -------
    R_samples : (n_samples, Ns, D_R)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if m.ndim == 1:
        m = m[None, :]
    Ns, D_m = m.shape
    K = len(gmm_params)
    D_R = gmm_params[0]["mu_R"].shape[0]

    lam = gmm_responsibilities(m, gmm_params)        # (Ns, K)

    # Pre-compute per-component conditional means/cholesky for ALL samples at once
    # mu_R|m^k for each (j, k): shape (Ns, K, D_R)
    cond_mean_all = np.zeros((Ns, K, D_R))
    chol_R_m_k = np.zeros((K, D_R, D_R))               # per-component Cholesky
    for k, p in enumerate(gmm_params):
        cond_mean_all[:, k, :] = p["mu_R"] + (m - p["mu_m"]) @ p["A"].T
        try:
            L = np.linalg.cholesky(p["Sig_R_m"] + 1e-12 * np.eye(D_R))
        except np.linalg.LinAlgError:
            # SPD failed → diagonal fallback
            L = np.diag(np.sqrt(np.maximum(np.diag(p["Sig_R_m"]), 0.0)))
        chol_R_m_k[k] = L

    # Vectorized sampling: per s and j, pick component k_chosen[s,j], then sample
    cum = np.cumsum(lam, axis=1)                       # (Ns, K)
    u = rng.random((n_samples, Ns))                    # (S, Ns)
    # k_chosen[s, j] = component index for sample s at location j
    k_chosen = (u[:, :, None] < cum[None, :, :]).argmax(axis=2)   # (S, Ns)

    # Generate iid normal samples
    z = rng.standard_normal((n_samples, Ns, D_R))      # (S, Ns, D_R)

    # Compute samples: R[s,j] = cond_mean_all[j, k] + chol_R_m_k[k] @ z[s,j]
    # Vectorize: gather per (s,j) the mean and chol
    cond_mean_picked = np.take_along_axis(
        cond_mean_all[None, :, :, :].repeat(n_samples, axis=0),     # (S, Ns, K, D_R)
        k_chosen[:, :, None, None].repeat(D_R, axis=3),
        axis=2).squeeze(2)                                          # (S, Ns, D_R)

    chol_picked = chol_R_m_k[k_chosen]                              # (S, Ns, D_R, D_R)

    # Apply Cholesky: chol @ z, vectorized via einsum
    R_samp = cond_mean_picked + np.einsum("snij,snj->sni", chol_picked, z)

    return R_samp


# ═══════════════════════════════════════════════════════════════════════════
#                       CALIBRATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def calibrate_sigma0_from_logs(well_dframes: Dict, top_zone: str,
                                 bot_zone: str, well_tops: Dict,
                                 col_VP: str = "VP", col_VS: str = "VS",
                                 col_RHO: str = "RHOB"
                                 ) -> np.ndarray:
    """
    Compute prior covariance Σ₀ from well logs in zone of interest.
    """
    log_data = []
    for wname, df in well_dframes.items():
        tops = well_tops.get(wname, {})
        z_top = tops.get(top_zone)
        z_bot = tops.get(bot_zone)
        if z_top is None or z_bot is None:
            continue
        if "DEPTH" in df.columns:
            mask = (df["DEPTH"] >= min(z_top, z_bot)) & \
                   (df["DEPTH"] <= max(z_top, z_bot))
            sub = df.loc[mask, [col_VP, col_VS, col_RHO]].dropna()
            if len(sub) > 10:
                m_log = np.column_stack([np.log(sub[col_VP].values),
                                         np.log(sub[col_VS].values),
                                         np.log(sub[col_RHO].values)])
                log_data.append(m_log)
    if not log_data:
        return np.diag([0.0034, 0.0042, 0.0015])
    M = np.vstack(log_data)
    return np.cov(M.T)


def calibrate_corr_length_from_logs(well_dframes: Dict, dt_s: float,
                                      max_lag_ms: float = 40.0,
                                      col_VP: str = "VP",
                                      col_TIME: str = "TIME"
                                      ) -> float:
    """
    Estimate vertical correlation length L by fitting exponential to
    autocorrelation of log VP (in TWT domain).
    Returns L in seconds.
    """
    L_estimates = []
    for wname, df in well_dframes.items():
        if col_VP not in df.columns or col_TIME not in df.columns:
            continue
        sub = df[[col_VP, col_TIME]].dropna()
        if len(sub) < 50:
            continue
        # Resample uniform in time
        t_uni = np.arange(sub[col_TIME].min(), sub[col_TIME].max(), dt_s)
        if len(t_uni) < 30:
            continue
        ln_vp = np.interp(t_uni, sub[col_TIME].values, np.log(sub[col_VP].values))
        ln_vp = ln_vp - ln_vp.mean()
        # Autocorrelation
        n_lag_max = int(max_lag_ms * 1e-3 / dt_s)
        ac = np.correlate(ln_vp, ln_vp, mode="full")
        ac = ac[len(ln_vp)-1:len(ln_vp)+n_lag_max]
        ac = ac / max(ac[0], EPS)
        # Fit exponential exp(-|τ|/L) → log(ac) = -τ/L
        lag_t = np.arange(len(ac)) * dt_s
        valid = (ac > 0.05) & (lag_t > 0)
        if valid.sum() < 5:
            continue
        log_ac = np.log(ac[valid])
        slope = np.polyfit(lag_t[valid], log_ac, 1)[0]
        L = -1.0 / max(-slope, EPS)
        if 0.001 < L < 0.05:   # sane range: 1-50 ms
            L_estimates.append(L)
    if not L_estimates:
        return 0.006   # 6 ms default
    return float(np.median(L_estimates))


def calibrate_noise_from_residuals(d_obs: np.ndarray, d_syn: np.ndarray,
                                     noise_zone_mask: Optional[np.ndarray] = None
                                     ) -> float:
    """
    Estimate noise variance from residuals in a quiet zone.

    Parameters
    ----------
    d_obs, d_syn : same shape, observed and synthetic gathers
    noise_zone_mask : boolean mask indicating noise-only samples

    Returns
    -------
    noise_var : scalar variance estimate
    """
    res = d_obs - d_syn
    if noise_zone_mask is not None:
        res = res[noise_zone_mask]
    res = res[np.isfinite(res)]
    if len(res) < 10:
        return 5e-3
    return float(np.var(res))


# ═══════════════════════════════════════════════════════════════════════════
#                   LATERAL POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════
def structure_oriented_filter(cube: np.ndarray,
                                sigma_trace: float = 2.0,
                                sigma_time: float = 1.0
                                ) -> np.ndarray:
    """
    Anisotropic Gaussian smoothing for lateral coherence.
    Larger sigma_trace → more lateral smoothing; smaller sigma_time → preserve resolution.

    Preserves NaN mask.
    """
    nan_mask = np.isnan(cube)
    cube_filled = np.where(nan_mask, np.nanmean(cube), cube)
    smooth = gaussian_filter(cube_filled, sigma=(sigma_trace, sigma_time))
    smooth[nan_mask] = np.nan
    return smooth


def lateral_blend(cube_inv: np.ndarray, cube_lfm: np.ndarray,
                   lambda_blend: float = 0.25
                   ) -> np.ndarray:
    """
    Hybrid blend: λ × LFM + (1−λ) × inverted result.
    λ=0 → fully inverted, λ=1 → fully LFM.
    """
    return (1 - lambda_blend) * cube_inv + lambda_blend * cube_lfm


# ═══════════════════════════════════════════════════════════════════════════
#                   AUTO-PERMUTE GMM COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════
def auto_permute_gmm(gmm, feature_names: List[str],
                       target_order: List[str] = ("Shale", "Wet Sand", "HC Sand"),
                       vsh_idx: Optional[int] = None,
                       phie_idx: Optional[int] = None,
                       sw_idx: Optional[int] = None
                       ) -> List[int]:
    """
    Auto-detect facies-component mapping based on petrophysical means.

    Heuristic
    ---------
    Shale    : highest VSH
    HC Sand  : lowest SW (among non-shale)
    Wet Sand : remaining

    Returns
    -------
    permutation : list[int] — gmm.means_[permutation[k]] = component for facies k
                              in target_order [Shale, WetSand, HCSand]
    """
    if vsh_idx is None and "VSH" in feature_names:
        vsh_idx = feature_names.index("VSH")
    if phie_idx is None and "PHIE" in feature_names:
        phie_idx = feature_names.index("PHIE")
    if sw_idx is None and "SW" in feature_names:
        sw_idx = feature_names.index("SW")

    K = gmm.n_components
    means = gmm.means_

    # Identify shale = highest VSH
    if vsh_idx is not None:
        shale_k = int(np.argmax(means[:, vsh_idx]))
    else:
        shale_k = 0

    remaining = [k for k in range(K) if k != shale_k]

    # Among remaining: HC Sand = lowest SW
    if sw_idx is not None and len(remaining) > 1:
        sws = [means[k, sw_idx] for k in remaining]
        hc_k = remaining[int(np.argmin(sws))]
    else:
        hc_k = remaining[0] if remaining else 1

    wet_k = [k for k in remaining if k != hc_k]
    wet_k = wet_k[0] if wet_k else 0

    return [shale_k, wet_k, hc_k][:K]


def apply_permutation_to_gmm(gmm, perm: List[int]):
    """Permute GMM means/covariances/weights in-place."""
    gmm.means_       = gmm.means_[perm].copy()
    gmm.covariances_ = gmm.covariances_[perm].copy()
    gmm.weights_     = gmm.weights_[perm].copy()
    if hasattr(gmm, "precisions_"):
        gmm.precisions_ = gmm.precisions_[perm].copy()
    if hasattr(gmm, "precisions_cholesky_"):
        gmm.precisions_cholesky_ = gmm.precisions_cholesky_[perm].copy()
    return gmm


# ═══════════════════════════════════════════════════════════════════════════
#                 GPU-ACCELERATED BATCH KERNELS
#  Dipakai di NB6 untuk kecepatan tinggi pada banyak trace × banyak sampel
# ═══════════════════════════════════════════════════════════════════════════
def gpu_build_forward_G_batch(vp_bg_batch: np.ndarray,
                                vs_bg_batch: np.ndarray,
                                rho_bg_batch: np.ndarray,
                                theta_rad: np.ndarray,
                                wavelet: np.ndarray,
                                ridge: float = 0.0):
    """
    Bangun forward operator G untuk BANYAK trace sekaligus di GPU.

    Parameters
    ----------
    vp_bg_batch, vs_bg_batch, rho_bg_batch : (N_TRACES, NT) — background prior
    theta_rad : (NTHETA,) angles in radian
    wavelet   : (Lw,) wavelet samples (zero-phase, centered)

    Returns
    -------
    G_batch : (N_TRACES, NTHETA*(NT-1), 3*NT)  — operator per trace
              Type sesuai xp module (cupy.ndarray jika GPU, np.ndarray jika CPU)
    """
    vp = asgpu(vp_bg_batch)
    vs = asgpu(vs_bg_batch)
    rho = asgpu(rho_bg_batch)
    theta_rad = asgpu(theta_rad)
    w = asgpu(wavelet)

    N, NT = vp.shape
    NTHETA = theta_rad.shape[0]
    NI = NT - 1
    Lw = w.shape[0]
    cz = Lw // 2

    # Build Toeplitz wavelet matrix W (NI, NI) — sama untuk semua trace
    col = xp.zeros(NI, dtype=vp.dtype)
    row = xp.zeros(NI, dtype=vp.dtype)
    for i in range(NI):
        kc = cz + i
        kr = cz - i
        if 0 <= kc < Lw: col[i] = w[kc]
        if 0 <= kr < Lw: row[i] = w[kr]
    # Build Toeplitz manually (CuPy doesn't have scipy.linalg.toeplitz)
    W = xp.zeros((NI, NI), dtype=vp.dtype)
    for i in range(NI):
        for j in range(NI):
            k = cz + (i - j)
            if 0 <= k < Lw:
                W[i, j] = w[k]

    # Diff matrix D (NI, NT) — sama untuk semua trace
    D = xp.zeros((NI, NT), dtype=vp.dtype)
    rows = xp.arange(NI)
    D[rows, rows]   = -0.5
    D[rows, rows+1] =  0.5

    # WD = W @ D  (NI, NT)
    WD = W @ D

    # vp_mid, vs_mid per trace: (N, NI)
    vp_mid = 0.5 * (vp[:, 1:] + vp[:, :-1])
    vs_mid = 0.5 * (vs[:, 1:] + vs[:, :-1])
    vpvs2  = (vs_mid / (vp_mid + 1e-9)) ** 2          # (N, NI)

    # AVO weights per (trace, theta, sample)
    sin2 = xp.sin(theta_rad) ** 2                       # (NTHETA,)
    tan2 = xp.tan(theta_rad) ** 2                       # (NTHETA,)
    a_th = 0.5 * (1.0 + tan2)                            # (NTHETA,) — independent of sample
    # b[trace, theta, sample] = -4 * vpvs2[trace, sample] * sin2[theta]
    b_tts = -4.0 * vpvs2[:, None, :] * sin2[None, :, None]   # (N, NTHETA, NI)
    c_tts =  0.5 * (1.0 - 4.0 * vpvs2[:, None, :] * sin2[None, :, None])

    # Build G per trace
    NM = 3 * NT
    G = xp.zeros((N, NTHETA * NI, NM), dtype=vp.dtype)
    for ith in range(NTHETA):
        ro_st, ro_en = ith * NI, (ith + 1) * NI
        # VP block: a (scalar per theta) * WD (NI, NT)
        G[:, ro_st:ro_en, 0:NT]      = a_th[ith] * WD[None, :, :]
        # VS block: b (N, NI) * WD (NI, NT) → broadcast: (N, NI, NT)
        G[:, ro_st:ro_en, NT:2*NT]   = b_tts[:, ith, :, None] * WD[None, :, :]
        # RHO block
        G[:, ro_st:ro_en, 2*NT:3*NT] = c_tts[:, ith, :, None] * WD[None, :, :]

    return G


# Jalankan cell ini untuk patch langsung ke modul (tanpa edit file .py)
import inversion_utils, numpy as np

def gpu_bayes_invert_batch(d_obs_batch: np.ndarray,
                             vp_pri_batch: np.ndarray,
                             vs_pri_batch: np.ndarray,
                             rho_pri_batch: np.ndarray,
                             theta_rad: np.ndarray,
                             wavelet,                          # ← ndarray | List[ndarray]
                             sigma0: np.ndarray,
                             corr_len_ms: float,
                             noise_var,                        # ← float | Dict[int,float]
                             noise_scale_per_angle: List[float],
                             dt_ms: float,
                             ridge: float = 1e-8,
                             wd_prod_list=None):               # ← [1] TAMBAHAN BARU
    """
    GPU-accelerated batch Bayesian inversion (Buland-Omre 2003).
    Mendukung group wavelet (Near/Mid/Far) via wd_prod_list.
    """
    # Transfer inputs ke GPU — tidak berubah
    d_obs        = asgpu(d_obs_batch.astype(np.float64))
    vp           = asgpu(vp_pri_batch.astype(np.float64))
    vs           = asgpu(vs_pri_batch.astype(np.float64))
    rho          = asgpu(rho_pri_batch.astype(np.float64))
    theta        = asgpu(theta_rad.astype(np.float64))
    sigma0_g     = asgpu(sigma0.astype(np.float64))
    noise_scales = asgpu(np.array(noise_scale_per_angle, dtype=np.float64))

    N, NT    = vp.shape
    NTHETA   = theta.shape[0]
    NI       = NT - 1
    NM       = 3 * NT
    NDATA    = NTHETA * NI

    # Sigma_m — tidak berubah
    t       = xp.arange(NT) * dt_ms
    dT      = xp.abs(t[:, None] - t[None, :])
    C_t     = xp.exp(-dT / max(corr_len_ms, 1e-3))
    Sigma_m = xp.zeros((NM, NM), dtype=xp.float64)
    for i in range(3):
        for j in range(3):
            Sigma_m[i*NT:(i+1)*NT, j*NT:(j+1)*NT] = sigma0_g[i, j] * C_t

    # ── [2] Sigma_e: scalar×scale² (asli) ATAU dict per-angle (group wav) ──
    sigma_e_diag = xp.zeros(NDATA, dtype=xp.float64)
    for ith in range(NTHETA):
        if isinstance(noise_var, dict):
            th_int = int(np.degrees(theta_rad[ith]))
            nv = float(noise_var.get(th_int, 5e-3))
        else:
            nv = float(noise_var) * float(noise_scale_per_angle[ith]) ** 2
        sigma_e_diag[ith*NI:(ith+1)*NI] = nv
    # ──────────────────────────────────────────────────────────────────────

    # ── [3] Forward operator G: wd_prod_list (group wav) ATAU wavelet tunggal
    if wd_prod_list is not None:
        # Bangun G menggunakan WD pre-built per angle — GPU batched
        # wd_prod_list[ith] shape: (NI, NT), sudah = W_ith @ D
        vp_mid  = 0.5 * (vp[:, 1:] + vp[:, :-1])          # (N, NI)
        vs_mid  = 0.5 * (vs[:, 1:] + vs[:, :-1])
        vpvs2   = (vs_mid / (vp_mid + 1e-9)) ** 2
        sin2    = xp.sin(theta) ** 2                         # (NTHETA,)
        tan2    = xp.tan(theta) ** 2

        G_batch = xp.zeros((N, NDATA, NM), dtype=xp.float64)
        for ith in range(NTHETA):
            ro_st, ro_en = ith * NI, (ith + 1) * NI
            WD = asgpu(wd_prod_list[ith].astype(np.float64))  # (NI, NT)

            a = 0.5 * (1.0 + tan2[ith])                      # scalar
            b = -4.0 * vpvs2 * sin2[ith]                     # (N, NI)
            c =  0.5 * (1.0 - 4.0 * vpvs2 * sin2[ith])

            # VP block : a · WD  →  broadcast ke (N, NI, NT)
            G_batch[:, ro_st:ro_en,      0:NT] = a * WD[None, :, :]
            # VS block : b[:,i] · WD[i,:]
            G_batch[:, ro_st:ro_en,    NT:2*NT] = b[:, :, None] * WD[None, :, :]
            # RHO block
            G_batch[:, ro_st:ro_en, 2*NT:3*NT] = c[:, :, None] * WD[None, :, :]
    else:
        # Perilaku asli: single wavelet
        wav = wavelet[0] if isinstance(wavelet, list) else wavelet
        G_batch = gpu_build_forward_G_batch(
            asnumpy(vp), asnumpy(vs), asnumpy(rho),
            asnumpy(theta), wav)
    # ──────────────────────────────────────────────────────────────────────

    # Prior log-space — tidak berubah
    mu_pri = xp.concatenate([
        xp.log(xp.maximum(vp,  1.0)),
        xp.log(xp.maximum(vs,  1.0)),
        xp.log(xp.maximum(rho, 0.001)),
    ], axis=1)   # (N, NM)

    # Solve per trace — tidak berubah
    VP_post  = xp.zeros((N, NT), dtype=xp.float64)
    VS_post  = xp.zeros((N, NT), dtype=xp.float64)
    RHO_post = xp.zeros((N, NT), dtype=xp.float64)
    VP_std   = xp.zeros((N, NT), dtype=xp.float64)
    VS_std   = xp.zeros((N, NT), dtype=xp.float64)
    RHO_std  = xp.zeros((N, NT), dtype=xp.float64)
    diag_Sm  = xp.diag(Sigma_m)

    for it in range(N):
        G     = G_batch[it]
        d     = d_obs[it].reshape(-1)
        m_pri = mu_pri[it]
        inn   = d - G @ m_pri

        SmGT = Sigma_m @ G.T
        H    = G @ SmGT
        idx  = xp.arange(NDATA)
        H[idx, idx] += sigma_e_diag
        H[idx, idx] += ridge * (xp.trace(H) / NDATA)

        try:
            L          = gpu_cholesky(H, lower=True, ridge=0.0)
            Hinv_inn   = gpu_cho_solve(L, inn,         lower=True)
            Hinv_GSm   = gpu_cho_solve(L, G @ Sigma_m, lower=True)
            diag_post  = diag_Sm - xp.einsum("ij,ji->i", SmGT, Hinv_GSm)
        except Exception:
            Hinv_inn  = gpu_solve(H + 1e-6 * xp.eye(NDATA), inn)
            diag_post = diag_Sm

        m_post = m_pri + SmGT @ Hinv_inn
        VP_post[it]  = xp.exp(m_post[     :NT  ])
        VS_post[it]  = xp.exp(m_post[NT   :2*NT])
        RHO_post[it] = xp.exp(m_post[2*NT :3*NT])

        diag_post   = xp.clip(diag_post, 1e-10, None)
        VP_std[it]  = VP_post[it]  * xp.sqrt(diag_post[     :NT  ])
        VS_std[it]  = VS_post[it]  * xp.sqrt(diag_post[NT   :2*NT])
        RHO_std[it] = RHO_post[it] * xp.sqrt(diag_post[2*NT :3*NT])

    free_gpu_memory()
    return {
        "VP":      asnumpy(VP_post),
        "VS":      asnumpy(VS_post),
        "RHO":     asnumpy(RHO_post),
        "VP_std":  asnumpy(VP_std),
        "VS_std":  asnumpy(VS_std),
        "RHO_std": asnumpy(RHO_std),
    }


def gpu_petro_inversion_batch(M_batch: np.ndarray,
                                gmm_params: List[Dict],
                                n_qsamp: int = 500,
                                clip_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                                rng_seed: int = 42):
    """
    GPU batch petrophysical inversion (Step 2 Grana 2010).

    Parameters
    ----------
    M_batch : (N, NS, 3) elastic samples per trace [VP, VS, RHO]
              N = num traces, NS = samples per trace
    gmm_params : output of gmm_conditional_params (3-component GMM)
    n_qsamp : number of Monte Carlo samples for P10/P90

    Returns
    -------
    Dict berisi: PHIE_mean, VSH_mean, SW_mean (N, NS),
                  PHIE_std, VSH_std, SW_std (N, NS),
                  PHIE_P10, PHIE_P90, VSH_P10, VSH_P90, SW_P10, SW_P90,
                  LFC_PROB (N, NS, K), LFC_MAP (N, NS)
    Semua sebagai NumPy array di CPU.
    """
    if clip_bounds is None:
        clip_bounds = {"PHIE": (0.001, 0.45), "VSH": (0.001, 1.0), "SW": (0.001, 1.0)}

    M_g = asgpu(M_batch.astype(np.float64))     # (N, NS, 3)
    N, NS, _ = M_g.shape
    K = len(gmm_params)
    D_R = gmm_params[0]["mu_R"].shape[0]

    # Pre-compute GMM params at GPU
    mu_m = xp.stack([asgpu(p["mu_m"]) for p in gmm_params])               # (K, 3)
    mu_R = xp.stack([asgpu(p["mu_R"]) for p in gmm_params])               # (K, D_R)
    A_k  = xp.stack([asgpu(p["A"])    for p in gmm_params])               # (K, D_R, 3)
    Sig_mm_inv = xp.stack([asgpu(p["Sig_mm_inv"]) for p in gmm_params])   # (K, 3, 3)
    Sig_mm = xp.stack([asgpu(p["Sig_mm"]) for p in gmm_params])           # (K, 3, 3)
    Sig_R_m = xp.stack([asgpu(p["Sig_R_m"]) for p in gmm_params])         # (K, D_R, D_R)
    alpha = xp.array([p["alpha"] for p in gmm_params], dtype=xp.float64)  # (K,)

    # Cholesky for each Sig_R_m (constant per k)
    chol_R_m = xp.zeros((K, D_R, D_R), dtype=xp.float64)
    for k in range(K):
        try:
            chol_R_m[k] = xp.linalg.cholesky(Sig_R_m[k] + 1e-12 * xp.eye(D_R))
        except Exception:
            chol_R_m[k] = xp.diag(xp.sqrt(xp.maximum(xp.diag(Sig_R_m[k]), 0.0)))

    # log-pdf p(m | k) for each (trace, sample, k)
    # diff[N, NS, K, 3] = M[N, NS, 1, 3] - mu_m[K, 3]
    diff = M_g[:, :, None, :] - mu_m[None, None, :, :]                    # (N, NS, K, 3)
    # mahalanobis: einsum
    maha = xp.einsum("nskc,kcd,nskd->nsk", diff, Sig_mm_inv, diff)        # (N, NS, K)

    # log-determinant Sig_mm per k
    sign_k, logdet_k = xp.linalg.slogdet(Sig_mm)                          # (K,) each
    log_pi = xp.log(xp.array(2 * np.pi))
    log_pdf = (xp.log(alpha + 1e-300)[None, None, :]
               - 0.5 * 3 * log_pi
               - 0.5 * logdet_k[None, None, :]
               - 0.5 * maha)                                                # (N, NS, K)

    log_pdf = log_pdf - log_pdf.max(axis=2, keepdims=True)
    pdf = xp.exp(log_pdf)
    LAM = pdf / pdf.sum(axis=2, keepdims=True)                              # (N, NS, K)

    # Per-component conditional mean: μ_R|m^k = μ_R^k + A_k · (m − μ_m^k)
    # cond_mean[N, NS, K, D_R] = mu_R[K, D_R] + einsum(A_k, diff)
    cond_mean = mu_R[None, None, :, :] + xp.einsum("kdc,nskc->nskd", A_k, diff)
    # (N, NS, K, D_R)

    # Mixture mean: E[R|m] = Σ_k λ_k · μ_R|m^k
    R_mean = xp.einsum("nsk,nskd->nsd", LAM, cond_mean)                     # (N, NS, D_R)

    # Mixture variance:
    # Var[R|m] = Σ_k λ_k (Σ_R|m^k + (μ_R|m^k − E[R|m])²)
    diff_sq = (cond_mean - R_mean[:, :, None, :]) ** 2                       # (N, NS, K, D_R)
    diag_RM = xp.diagonal(Sig_R_m, axis1=1, axis2=2)                         # (K, D_R)
    R_var = xp.einsum("nsk,nskd->nsd", LAM, diag_RM[None, None, :, :] + diff_sq)
    R_std = xp.sqrt(xp.maximum(R_var, 0.0))                                  # (N, NS, D_R)

    # MAP class
    LFC_MAP = xp.argmax(LAM, axis=2)                                         # (N, NS)

    # Apply clip
    _R_FEAT_ORDER = ["PHIE", "VSH", "SW", "LR", "mR", "RD"]
    LO_R = xp.array([clip_bounds.get(_R_FEAT_ORDER[i], (-1e30, 1e30))[0] for i in range(D_R)])
    HI_R = xp.array([clip_bounds.get(_R_FEAT_ORDER[i], (-1e30, 1e30))[1] for i in range(D_R)])
    R_mean = xp.clip(R_mean, LO_R, HI_R)

    # P10/P90 via vectorized ensemble sampling
    rng = gpu_default_rng(rng_seed)
    cum = xp.cumsum(LAM, axis=2)                                             # (N, NS, K)

    # Component picks: u (S, N, NS) → k_chosen (S, N, NS)
    u = rng.random((n_qsamp, N, NS))
    k_chosen = (u[:, :, :, None] < cum[None, :, :, :]).argmax(axis=3)        # (S, N, NS)

    # Generate iid normals
    z = rng.standard_normal((n_qsamp, N, NS, D_R))                            # (S, N, NS, D_R)

    # Pick mean & chol per (s, n, ns)
    # cond_mean shape (N, NS, K, D_R)
    # We need: cond_mean_picked[s, n, ns, d] = cond_mean[n, ns, k_chosen[s,n,ns], d]
    # Using fancy indexing:
    n_idx = xp.arange(N)[None, :, None]                                       # (1, N, 1)
    ns_idx = xp.arange(NS)[None, None, :]                                     # (1, 1, NS)
    cond_mean_picked = cond_mean[n_idx, ns_idx, k_chosen, :]                  # (S, N, NS, D_R)
    chol_picked = chol_R_m[k_chosen]                                          # (S, N, NS, D_R, D_R)

    # R_samp[s,n,ns] = cond_mean_picked + chol @ z
    R_samp = cond_mean_picked + xp.einsum("snisd,snid->snis", chol_picked, z) \
        if False else cond_mean_picked + xp.einsum("snijd,snij->snid", chol_picked, z) \
        if False else None
    # Simpler: use einsum with explicit index names
    R_samp = cond_mean_picked + xp.einsum("snikd,snid->snik".replace("ik", "kj").replace("ki", "jk"),
                                            chol_picked, z) if False else None

    # Build R_samp manually (cleaner)
    R_samp = xp.zeros((n_qsamp, N, NS, D_R), dtype=xp.float64)
    # For each sample s and (n, ns), R_samp = cond_mean + L @ z
    # We can do this with einsum:
    # chol_picked has shape (S, N, NS, D_R, D_R); z has shape (S, N, NS, D_R)
    # output: R_samp[s, n, ns, i] = sum_j chol_picked[s, n, ns, i, j] * z[s, n, ns, j]
    R_samp = cond_mean_picked + xp.einsum("snikj,snij->snik", chol_picked, z)
    # Clip
    R_samp = xp.clip(R_samp, LO_R, HI_R)

    # Percentiles
    R_P10 = xp.percentile(R_samp, 10, axis=0)                                 # (N, NS, D_R)
    R_P90 = xp.percentile(R_samp, 90, axis=0)                                 # (N, NS, D_R)

    # Transfer balik ke CPU
    result = {
        "PHIE_mean": asnumpy(R_mean[..., 0]),
        "VSH_mean":  asnumpy(R_mean[..., 1]),
        "SW_mean":   asnumpy(R_mean[..., 2]),
        "PHIE_std":  asnumpy(R_std[..., 0]),
        "VSH_std":   asnumpy(R_std[..., 1]),
        "SW_std":    asnumpy(R_std[..., 2]),
        "PHIE_P10":  asnumpy(R_P10[..., 0]),
        "PHIE_P90":  asnumpy(R_P90[..., 0]),
        "VSH_P10":   asnumpy(R_P10[..., 1]),
        "VSH_P90":   asnumpy(R_P90[..., 1]),
        "SW_P10":    asnumpy(R_P10[..., 2]),
        "SW_P90":    asnumpy(R_P90[..., 2]),
        "LFC_PROB":  asnumpy(LAM),
        "LFC_MAP":   asnumpy(LFC_MAP).astype(np.int8),
    }
    free_gpu_memory()
    return result
