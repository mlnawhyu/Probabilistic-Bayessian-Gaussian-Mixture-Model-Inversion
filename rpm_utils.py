"""
rpm_utils.py — Rock Physics Model utilities untuk NB2 dan NB6.

Pipeline RPM:
  1. Hertz-Mindlin contact model → K_HM, G_HM
  2. Soft / Stiff Sand Hashin-Shtrikman bounds
  3. Contact Cement Model (Dvorkin-Nur 1996)
  4. Sand classification: Cemented / Stiff / Transition / Soft / Sub-Soft
  5. K_dry, G_dry interpolation per sand class
  6. VRH mineral mixing (Quartz + Clay)
  7. Brie fluid mixing (Brine + Gas)
  8. Gassmann fluid substitution → VP_sat, VS_sat, RHO_sat
"""
from __future__ import annotations

from typing import Tuple, Dict, Optional

import numpy as np
from scipy.interpolate import interp1d

EPS = 1e-9


# ═══════════════════════════════════════════════════════════════════════════
#                       HERTZ-MINDLIN CONTACT MODEL
# ═══════════════════════════════════════════════════════════════════════════
def hertz_mindlin(K_min: float, G_min: float, phi_c: float,
                   n_coord: float, P_MPa: float, nu_min: float
                   ) -> Tuple[float, float]:
    """
    Hertz-Mindlin contact model.

    Returns
    -------
    K_HM, G_HM : (GPa, GPa) bulk and shear modulus at critical porosity
    """
    P_GPa = P_MPa * 1e-3
    G_HM = ((n_coord**2 * (1 - phi_c)**2 * G_min**2 * P_GPa) /
            (18 * np.pi**2 * (1 - nu_min)**2)) ** (1 / 3)
    K_HM = (2 / 3) * G_HM * (1 + nu_min) / (1 - 2 * nu_min)
    return K_HM, G_HM


# ═══════════════════════════════════════════════════════════════════════════
#               SOFT / STIFF SAND HASHIN-SHTRIKMAN BOUNDS
# ═══════════════════════════════════════════════════════════════════════════
def soft_sand_K(K_HM: float, G_HM: float, K_min: float, f_HM: float) -> float:
    """Soft sand (lower HS) bulk modulus."""
    num = f_HM / (K_HM + 4/3 * G_HM + EPS) + (1 - f_HM) / (K_min + 4/3 * G_HM + EPS)
    return 1.0 / num - 4/3 * G_HM


def soft_sand_G(G_HM: float, G_min: float, f_HM: float) -> float:
    """Soft sand shear modulus."""
    z = G_HM / 6 * (9 * G_HM + 8 * G_HM) / (G_HM + 2 * G_HM + EPS)
    num = f_HM / (G_HM + z + EPS) + (1 - f_HM) / (G_min + z + EPS)
    return 1.0 / num - z


def stiff_sand_K(K_HM: float, G_min: float, K_min: float, f_HM: float) -> float:
    """Stiff sand (upper HS) bulk modulus."""
    num = f_HM / (K_HM + 4/3 * G_min + EPS) + (1 - f_HM) / (K_min + 4/3 * G_min + EPS)
    return 1.0 / num - 4/3 * G_min


def stiff_sand_G(G_HM: float, G_min: float, K_min: float, f_HM: float) -> float:
    """Stiff sand shear modulus."""
    z = G_min / 6 * (9 * K_min + 8 * G_min) / (K_min + 2 * G_min + EPS)
    num = f_HM / (G_HM + z + EPS) + (1 - f_HM) / (G_min + z + EPS)
    return 1.0 / num - z


# ═══════════════════════════════════════════════════════════════════════════
#                   CONTACT CEMENT MODEL (Dvorkin-Nur 1996)
# ═══════════════════════════════════════════════════════════════════════════
def contact_cement(phi: np.ndarray, phi_c: float, c_cement: float,
                   K_min: float, G_min: float,
                   K_cem: float, G_cem: float,
                   nu_min: float, nu_cem: float,
                   n_coord: float = 9.0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dvorkin-Nur contact cement model (Scheme 2: cement at grain contacts).

    Returns
    -------
    K_ccm, G_ccm : array of bulk/shear modulus at given phi
    """
    Mc = K_cem + 4/3 * G_cem
    Lambda_n = 2 * G_cem / (np.pi * G_min) * \
        (1 - nu_min) * (1 - nu_cem) / (1 - 2 * nu_cem + EPS)
    Lambda_t = G_cem / (np.pi * G_min)

    alpha = ((2/3) * (phi_c - phi) / (1 - phi_c + EPS)) ** 0.5
    alpha = np.clip(alpha, 1e-6, 1.0)

    A_n = -0.024153 * Lambda_n**(-1.3646)
    B_n =  0.20405  * Lambda_n**(-0.89008)
    C_n =  0.00024649 * Lambda_n**(-1.9864)
    S_n = A_n * alpha**2 + B_n * alpha + C_n

    A_t = -1e-2 * (2.26 * nu_min**2 + 2.07 * nu_min + 2.3) * \
          Lambda_t**(0.079 * nu_min**2 + 0.1754 * nu_min - 1.342)
    B_t =  (0.0573 * nu_min**2 + 0.0937 * nu_min + 0.202) * \
          Lambda_t**(0.0274 * nu_min**2 + 0.0529 * nu_min - 0.8765)
    C_t =  1e-4 * (9.654 * nu_min**2 + 4.945 * nu_min + 3.1) * \
          Lambda_t**(0.01867 * nu_min**2 + 0.4011 * nu_min - 1.8186)
    S_t = A_t * alpha**2 + B_t * alpha + C_t

    K_ccm = (1/6) * n_coord * (1 - phi_c) * Mc * S_n
    G_ccm = (3/5) * K_ccm + (3/20) * n_coord * (1 - phi_c) * G_cem * S_t

    return K_ccm, G_ccm


# ═══════════════════════════════════════════════════════════════════════════
#                       VRH MINERAL MIXING
# ═══════════════════════════════════════════════════════════════════════════
def vrh_mixing(vsh: np.ndarray, K_qtz: float, G_qtz: float,
               K_clay: float, G_clay: float,
               rho_qtz: float, rho_clay: float
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Voigt-Reuss-Hill average untuk campuran quartz-clay.

    Returns
    -------
    K_min, G_min, rho_min : VRH average for each sample (length len(vsh))
    """
    vsh = np.clip(vsh, 0.0, 1.0)
    f_qtz, f_clay = 1.0 - vsh, vsh

    # Voigt
    K_V = f_qtz * K_qtz + f_clay * K_clay
    G_V = f_qtz * G_qtz + f_clay * G_clay
    # Reuss
    K_R = 1.0 / (f_qtz / (K_qtz + EPS) + f_clay / (K_clay + EPS))
    G_R = 1.0 / (f_qtz / (G_qtz + EPS) + f_clay / (G_clay + EPS))
    # Hill average
    K_H = 0.5 * (K_V + K_R)
    G_H = 0.5 * (G_V + G_R)
    rho_min = f_qtz * rho_qtz + f_clay * rho_clay

    return K_H, G_H, rho_min


# ═══════════════════════════════════════════════════════════════════════════
#                       BRIE FLUID MIXING
# ═══════════════════════════════════════════════════════════════════════════
def brie_fluid_mixing(sw: np.ndarray, K_brine: float, K_gas: float,
                       rho_brine: float, rho_gas: float, brie_e: float = 7.0
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Brie's mixing law untuk K_fluid; volumetric for rho_fluid.

    Returns
    -------
    K_fl, rho_fl : array
    """
    sw = np.clip(sw, 0.0, 1.0)
    K_fl = (K_brine - K_gas) * sw**brie_e + K_gas
    rho_fl = sw * rho_brine + (1 - sw) * rho_gas
    return K_fl, rho_fl


# ═══════════════════════════════════════════════════════════════════════════
#                   GASSMANN FLUID SUBSTITUTION
# ═══════════════════════════════════════════════════════════════════════════
def gassmann_substitute(K_dry: float, G_dry: float,
                         K_min: float, K_fl: float, phi: float
                         ) -> Tuple[float, float]:
    """
    Biot-Gassmann fluid substitution.

    Returns
    -------
    K_sat, G_sat : (GPa, GPa)
    """
    if K_dry >= K_min:
        K_dry = K_min - 1e-3

    a = (1 - K_dry / (K_min + EPS)) ** 2
    b = phi / (K_fl + EPS) + (1 - phi) / (K_min + EPS) - K_dry / (K_min**2 + EPS)
    K_sat = K_dry + a / (b + EPS)
    G_sat = G_dry  # Gassmann: G unchanged
    return K_sat, G_sat


# ═══════════════════════════════════════════════════════════════════════════
#                   SAND CLASSIFICATION (5-class scheme)
# ═══════════════════════════════════════════════════════════════════════════
def classify_sand(phi: float, vp: Optional[float] = None,
                   vs: Optional[float] = None,
                   K_HM: Optional[float] = None,
                   phi_c: float = 0.37) -> int:
    """
    Heuristic 5-class sand classification (only meaningful for sand facies).

    Codes
    -----
    0 = Cemented   (phi < 0.10)
    1 = Stiff      (0.10 <= phi < 0.20)
    2 = Transition (0.20 <= phi < 0.28)
    3 = Soft       (0.28 <= phi < 0.34)
    4 = Sub-soft   (phi >= 0.34)
    """
    if not np.isfinite(phi):
        return -1
    if phi < 0.10:
        return 0
    elif phi < 0.20:
        return 1
    elif phi < 0.28:
        return 2
    elif phi < 0.34:
        return 3
    else:
        return 4


SAND_CLASS_NAMES = {0: "Cemented", 1: "Stiff Sand", 2: "Transition",
                    3: "Soft Sand", 4: "Sub-Soft"}


# ═══════════════════════════════════════════════════════════════════════════
#                   FULL FORWARD MODEL — single sample
# ═══════════════════════════════════════════════════════════════════════════
def forward_rpm_single(phi: float, vsh: float, sw: float,
                        rp_cfg: Dict
                        ) -> Tuple[float, float, float]:
    """
    Single-sample forward RPM: (phi, vsh, sw) → (VP, VS, RHO_sat).

    rp_cfg expected keys (from CFG['rock_physics']):
      K_qtz, K_clay, mu_qtz, mu_clay, rho_qtz, rho_clay
      K_brine, K_gas, rho_brine, rho_gas
      brie_e, P_eff_MPa, n_coord, phi_c
      [optional] nu_min, K_cem, G_cem, nu_cem, c_cement
    """
    K_qtz   = rp_cfg["K_qtz"]
    K_clay  = rp_cfg["K_clay"]
    G_qtz   = rp_cfg.get("mu_qtz",  rp_cfg.get("G_qtz", 45.0))
    G_clay  = rp_cfg.get("mu_clay", rp_cfg.get("G_clay", 9.0))
    rho_qtz  = rp_cfg["rho_qtz"]
    rho_clay = rp_cfg["rho_clay"]
    K_brine  = rp_cfg["K_brine"]
    K_gas    = rp_cfg["K_gas"]
    rho_brine = rp_cfg["rho_brine"]
    rho_gas   = rp_cfg["rho_gas"]
    brie_e   = rp_cfg.get("brie_e", 7.0)
    P_MPa    = rp_cfg["P_eff_MPa"]
    n_coord  = rp_cfg.get("n_coord", 9.0)
    phi_c    = rp_cfg["phi_c"]
    nu_min   = rp_cfg.get("nu_min", 0.08)

    # 1) VRH mineral
    K_min, G_min, rho_min = vrh_mixing(np.array([vsh]), K_qtz, G_qtz,
                                       K_clay, G_clay, rho_qtz, rho_clay)
    K_min, G_min, rho_min = float(K_min[0]), float(G_min[0]), float(rho_min[0])

    # 2) Hertz-Mindlin
    K_HM, G_HM = hertz_mindlin(K_min, G_min, phi_c, n_coord, P_MPa, nu_min)

    # 3) Soft sand frame at phi
    if phi >= phi_c - 1e-3:
        K_dry, G_dry = K_HM * (1 - phi / phi_c), G_HM * (1 - phi / phi_c)
    else:
        f_HM = phi / phi_c
        K_dry = soft_sand_K(K_HM, G_HM, K_min, f_HM)
        G_dry = soft_sand_G(G_HM, G_min, f_HM)
    K_dry = max(K_dry, 1e-3)
    G_dry = max(G_dry, 1e-3)

    # 4) Brie fluid
    K_fl, rho_fl = brie_fluid_mixing(np.array([sw]), K_brine, K_gas,
                                      rho_brine, rho_gas, brie_e)
    K_fl, rho_fl = float(K_fl[0]), float(rho_fl[0])

    # 5) Gassmann
    K_sat, G_sat = gassmann_substitute(K_dry, G_dry, K_min, K_fl, phi)

    # 6) VP, VS, RHO
    rho_sat = (1 - phi) * rho_min + phi * rho_fl
    vp = np.sqrt((K_sat + 4/3 * G_sat) / (rho_sat + EPS)) * 1000.0  # m/s
    vs = np.sqrt(G_sat / (rho_sat + EPS)) * 1000.0
    return float(vp), float(vs), float(rho_sat)


def forward_rpm_array(phi: np.ndarray, vsh: np.ndarray, sw: np.ndarray,
                       rp_cfg: Dict
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized forward RPM. Returns (VP, VS, RHO_sat) arrays."""
    n = len(phi)
    vp_out = np.full(n, np.nan)
    vs_out = np.full(n, np.nan)
    rho_out = np.full(n, np.nan)
    for i in range(n):
        if not (np.isfinite(phi[i]) and np.isfinite(vsh[i]) and np.isfinite(sw[i])):
            continue
        try:
            vp_out[i], vs_out[i], rho_out[i] = forward_rpm_single(
                float(phi[i]), float(vsh[i]), float(sw[i]), rp_cfg)
        except Exception:
            pass
    return vp_out, vs_out, rho_out


# ═══════════════════════════════════════════════════════════════════════════
#                   GENERATE RPT CURVES (untuk crossplot overlay)
# ═══════════════════════════════════════════════════════════════════════════
def generate_rpt_curves(rp_cfg: Dict,
                         sw_values: list = (1.0, 0.7, 0.3, 0.0),
                         vsh_values: list = (0.0, 0.5),
                         phi_range: Tuple[float, float] = (0.001, 0.40),
                         n_phi: int = 80
                         ) -> Dict:
    """
    Generate RPT curves untuk overlay di crossplot.
    Curves di-organize per fluid case (Sw=1.0 brine, Sw=0.0 gas, dst).

    Returns
    -------
    dict {
      'phi'  : (n_phi,) phi axis
      'curves' : list of dict {sw, vsh, vp, vs, rho_sat, ai, vpvs, label, color, ls}
    }
    """
    phi = np.linspace(phi_range[0], phi_range[1], n_phi)
    curves = []
    color_by_sw = {1.0: "#1f77b4", 0.7: "#5b9bd5", 0.3: "#e07b00", 0.0: "#d62728"}
    ls_by_vsh   = {0.0: "-", 0.5: "--"}

    for sw_val in sw_values:
        for vsh_val in vsh_values:
            sw_arr  = np.full_like(phi, sw_val)
            vsh_arr = np.full_like(phi, vsh_val)
            vp, vs, rho = forward_rpm_array(phi, vsh_arr, sw_arr, rp_cfg)
            ai   = vp * rho
            vpvs = np.where(vs > 0, vp / vs, np.nan)
            curves.append({
                "phi": phi,
                "vp":  vp,  "vs": vs,   "rho_sat": rho,
                "ai":  ai,  "vpvs": vpvs,
                "sw":  sw_val, "vsh": vsh_val,
                "label": f"Sw={sw_val:.1f}, Vsh={vsh_val:.1f}",
                "color": color_by_sw.get(sw_val, "gray"),
                "ls":    ls_by_vsh.get(vsh_val, "-"),
                "alpha": 0.85,
                "lw":    1.3,
            })
    return {"phi": phi, "curves": curves}
