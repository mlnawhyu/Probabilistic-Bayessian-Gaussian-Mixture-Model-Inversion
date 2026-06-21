"""
plot_utils.py — Template plot terstandardisasi untuk pipeline NB1-NB6.

9-track order (revised per user spec):
  T1 : GR              — fill 0→GR dengan colormap gist_heat
  T2 : RHOB + NPHI     — crossover fill YELLOW ketika RHOB < NPHI (gas X-over)
  T3 : RS + RD         — log scale; fallback RT
  T4 : VP + VS         — measured solid, model dotted
  T5 : VSH + PHIE      — darkgreen 0→VSH, yellow VSH→1, cyan 0→PHIE
  T6 : SW              — merah, magenta fill SW<0.5
  T7 : LITHO           — facies colormap
  T8 : PI
  T9 : SI

Extra functions:
  plot_facies_comparison_2track  — Rule-based vs GMM side-by-side
  plot_sensitivity_analysis      — AI vs Vp/Vs + PI vs SI, bersih tanpa RPT
  plot_crossplot                 — RPT + density contour + stats (unchanged)
  plot_log_histograms
  plot_seismic_section
  plot_inversion_section
  plot_gmm_diagnostic
"""
from __future__ import annotations
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

# ─────────────────────── Color palette ───────────────────────
LITHO_COLORS = {
    1: "#6B4F3A",   # Shale     — coklat
    2: "#4CA3DD",   # Wet Sand  — biru muda
    3: "#F4A82E",   # HC Sand   — orange
}
LITHO_LABELS = {1: "Shale", 2: "Wet Sand", 3: "HC Sand"}

LITHO_CMAP = ListedColormap(["#FFFFFF",
                              LITHO_COLORS[1],
                              LITHO_COLORS[2],
                              LITHO_COLORS[3]])

WELL_COLORS = {
    "Poseidon-1": "#1f77b4",
    "Boreas-1":   "#d62728",
}


# ─────────────────────── Helpers ───────────────────────
def _resolve_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _gist_heat_fill(ax, z, gr_vals, gr_max, alpha=0.88):
    """
    Fill from x=0 to x=GR with horizontal gist_heat gradient.
    Uses pcolormesh: rows = depth samples, cols = x-bands.
    """
    cmap   = matplotlib.colormaps.get_cmap("gist_heat")
    n_band = 256
    x_lo, x_hi = 0.0, max(float(gr_max) * 1.05, 1.0)

    x_edges = np.linspace(x_lo, x_hi, n_band + 1)
    x_ctrs  = 0.5 * (x_edges[:-1] + x_edges[1:])

    # prepend ghost row so pcolormesh has len(z)+1 y-edges
    y_edges = np.empty(len(z) + 1)
    dz      = abs(z[1] - z[0]) if len(z) > 1 else 1.0
    y_edges[0]  = z[0] - dz
    y_edges[1:] = z

    # color array: normalised position [0,1] in x
    color_2d = np.tile((x_ctrs - x_lo) / (x_hi - x_lo), (len(z), 1))  # (nz, nbands)

    # mask: only paint where x < gr_val[row]
    gr_clipped = np.clip(gr_vals, x_lo, x_hi)
    for i in range(len(z)):
        color_2d[i, x_ctrs >= gr_clipped[i]] = np.nan

    pcm = ax.pcolormesh(x_edges, y_edges, color_2d,
                        cmap=cmap, vmin=0.0, vmax=1.0,
                        alpha=alpha, shading="flat",
                        zorder=0, rasterized=True)
    return pcm


# ═══════════════════════════════════════════════════════════════════════════
#                    9-TRACK COMPOSITE WELL LOG PLOT
# ═══════════════════════════════════════════════════════════════════════════
def plot_well_logs_9track(
    df: pd.DataFrame,
    well_id: str = "Well",
    tops: Optional[Dict[str, float]] = None,
    depth_col: Optional[str] = None,
    depth_label: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 15),
    title_suffix: str = "",
    show: bool = True,
):
    """
    9-track composite log.
    T1 GR | T2 RHOB/NPHI | T3 RS/RD | T4 VP/VS | T5 VSH/PHIE | T6 SW | T7 LITHO | T8 PI | T9 SI
    """
    if depth_col is None:
        depth_col = _resolve_col(df, ["DEPTH","DEPT","MD","TIME","TWT"])
        if depth_col is None:
            raise ValueError("Tidak ada kolom DEPTH/TIME")
    if depth_label is None:
        depth_label = "Depth (m)" if depth_col in ("DEPTH","DEPT","MD") else "TWT (s)"

    z    = df[depth_col].values
    z_mn = float(np.nanmin(z))
    z_mx = float(np.nanmax(z))

    track_titles = ["GR", "RHOB / NPHI", "RS / RD",
                    "VP / VS", "VSH / PHIE", "SW",
                    "Lithology", "PI", "SI"]

    fig  = plt.figure(figsize=figsize)
    gs   = gridspec.GridSpec(1, 9, wspace=0.06)
    axes = [fig.add_subplot(gs[0, i]) for i in range(9)]

    # ══ T1: GR — gist_heat gradient fill ══
    ax     = axes[0]
    gr_col = _resolve_col(df, ["GR"])
    if gr_col:
        gr    = df[gr_col].values.copy()
        gr_ok = np.where(np.isfinite(gr), gr, 0.0)
        gr_mx = float(np.nanpercentile(gr, 99))
        _gist_heat_fill(ax, z, gr_ok, gr_mx, alpha=0.88)
        ax.plot(gr, z, color="#2ca02c", lw=0.7, zorder=3)
        ax.set_xlabel("GR (API)", color="#2ca02c", fontsize=8)
        ax.set_xlim(0, gr_mx * 1.05)
        ax.tick_params(axis="x", colors="#2ca02c", labelsize=7)
    ax.set_ylabel(depth_label, fontsize=10)

    # ══ T2: RHOB (merah inv) + NPHI (biru inv) + crossover fill YELLOW ══
    ax       = axes[1]
    rhob_col = _resolve_col(df, ["RHOB"])
    nphi_col = _resolve_col(df, ["NPHI"])

    RHOB_LO, RHOB_HI = 1.9, 2.9   # physical axis limits
    NPHI_LO, NPHI_HI = 0.45, -0.15

    if rhob_col and nphi_col:
        rhob = df[rhob_col].values.copy()
        nphi = df[nphi_col].values.copy()

        # normalise to [0,1] using the same "left-to-right" convention:
        # RHOB: low density (gas) = large positive value on its own scale
        #       but in standard display, RHOB axis is inverted (high RHOB at left)
        # NPHI: high porosity at left as well
        # Gas crossover: RHOB drops (moves right on inverted axis) past NPHI
        rhob_norm = np.clip((RHOB_HI - rhob) / (RHOB_HI - RHOB_LO), 0.0, 1.0)
        nphi_norm = np.clip((NPHI_HI - nphi) / (NPHI_HI - NPHI_LO), 0.0, 1.0)

        # crossover = RHOB_norm < nphi_norm (RHOB has moved to lower density side)
        crossover = (rhob_norm > nphi_norm) & np.isfinite(rhob) & np.isfinite(nphi)

        # Primary: RHOB axis (inverted: high values at left = 1.95, low at right = 3.05)
        ax.plot(rhob, z, color="#d62728", lw=0.8, zorder=3, label="RHOB")
        ax.set_xlabel("RHOB (g/cc)", color="#d62728", fontsize=8)
        ax.set_xlim(RHOB_HI, RHOB_LO)   # inverted
        ax.tick_params(axis="x", colors="#d62728", labelsize=7)

        # Twin: NPHI axis (also inverted so high NPHI at left)
        axt2 = ax.twiny()
        axt2.plot(nphi, z, color="#1f77b4", lw=0.8, zorder=3, label="NPHI")
        axt2.set_xlabel("NPHI (frac)", color="#1f77b4", fontsize=8)
        axt2.set_xlim(NPHI_HI, NPHI_LO)   # inverted
        axt2.tick_params(axis="x", colors="#1f77b4", labelsize=7)
        axt2.spines["top"].set_position(("outward", 18))

        # crossover fill: convert NPHI to RHOB scale then fill_betweenx
        nphi_as_rhob = RHOB_HI - nphi_norm * (RHOB_HI - RHOB_LO)
        ax.fill_betweenx(z, rhob, nphi_as_rhob,
                         where=crossover,
                         color="#FFE000", alpha=0.65, zorder=2,
                         label="Gas X-over")

    elif rhob_col:
        ax.plot(df[rhob_col].values, z, color="#d62728", lw=0.8)
        ax.set_xlim(RHOB_HI, RHOB_LO)
        ax.set_xlabel("RHOB (g/cc)", fontsize=8)
    elif nphi_col:
        ax.plot(df[nphi_col].values, z, color="#1f77b4", lw=0.8)
        ax.set_xlabel("NPHI (frac)", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No\nRHOB/NPHI", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
    ax.tick_params(labelsize=7)

    # ══ T3: RS/RD log scale ══
    ax = axes[2]
    rd_col = _resolve_col(df, ["RD","RDEEP","RDEP"])
    rs_col = _resolve_col(df, ["RS","RSHAL"])
    rt_col = _resolve_col(df, ["RT","ILD"])
    plotted_r = False
    if rd_col:
        ax.semilogx(df[rd_col].clip(lower=0.01), z,
                    color="#000000", lw=0.7, label="RD", zorder=3)
        plotted_r = True
    if rs_col:
        ax.semilogx(df[rs_col].clip(lower=0.01), z,
                    color="#888888", lw=0.6, ls="--", label="RS", zorder=3)
        plotted_r = True
    if not plotted_r and rt_col:
        ax.semilogx(df[rt_col].clip(lower=0.01), z,
                    color="#000000", lw=0.7, label="RT", zorder=3)
        plotted_r = True
    if plotted_r:
        ax.set_xlabel("Resistivity (Ω·m)", fontsize=8)
        ax.set_xlim(0.2, 2000)
        ax.legend(fontsize=6, loc="lower left")
    else:
        ax.text(0.5, 0.5, "No\nresistivity", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
    ax.tick_params(labelsize=7)

    # ══ T4: VP + VS (measured solid, model dotted) ══
    ax           = axes[3]
    vp_col       = _resolve_col(df, ["VP"])
    vp_model_col = _resolve_col(df, ["VP_GASSMANN","Vp_model","VP_MODEL"])
    vs_col       = _resolve_col(df, ["VS"])
    vs_model_col = _resolve_col(df, ["VS_GASSMANN","Vs_model","VS_MODEL"])

    if vp_col:
        ax.plot(df[vp_col].values, z, color="#1f77b4", lw=0.8, label="VP")
    if vp_model_col:
        ax.plot(df[vp_model_col].values, z, color="#1f77b4",
                lw=0.65, ls=":", alpha=0.85, label="VP model")
    ax.set_xlabel("VP (m/s)", color="#1f77b4", fontsize=8)
    ax.set_xlim(1500, 6500)
    ax.tick_params(axis="x", colors="#1f77b4", labelsize=7)

    if vs_col or vs_model_col:
        axt4 = ax.twiny()
        if vs_col:
            axt4.plot(df[vs_col].values, z, color="#d62728", lw=0.8, label="VS")
        if vs_model_col:
            axt4.plot(df[vs_model_col].values, z, color="#d62728",
                      lw=0.65, ls=":", alpha=0.85, label="VS model")
        axt4.set_xlabel("VS (m/s)", color="#d62728", fontsize=8)
        axt4.set_xlim(500, 4000)
        axt4.tick_params(axis="x", colors="#d62728", labelsize=7)
        axt4.spines["top"].set_position(("outward", 18))

    # ══ T5: VSH + PHIE — color fill ══
    ax       = axes[4]
    vsh_col  = _resolve_col(df, ["VSH"])
    phie_col = _resolve_col(df, ["PHIE","PHIT"])

    if vsh_col:
        vsh   = df[vsh_col].values.copy()
        valid = np.isfinite(vsh)
        ax.plot(vsh, z, color="#000000", lw=0.75, zorder=3)
        ax.fill_betweenx(z, 0,   vsh,  where=valid,
                         color="#1F6F00", alpha=0.50, zorder=1)
        ax.fill_betweenx(z, vsh, 1.0, where=valid,
                         color="#FFE066", alpha=0.60, zorder=1)
    ax.set_xlabel("VSH (frac)", fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.tick_params(labelsize=7)

    if phie_col:
        axt5 = ax.twiny()
        phie   = df[phie_col].values.copy()
        validp = np.isfinite(phie)
        axt5.plot(phie, z, color="#1f77b4", lw=0.85, zorder=4)
        axt5.fill_betweenx(z, 0, phie, where=validp,
                           color="#00CED1", alpha=0.52, zorder=2)
        axt5.set_xlabel("PHIE (frac)", color="#1f77b4", fontsize=8)
        axt5.set_xlim(0.45, 0.0)   # right-to-left: 0 at left
        axt5.tick_params(axis="x", colors="#1f77b4", labelsize=7)
        axt5.spines["top"].set_position(("outward", 18))

    # ══ T6: SW — magenta fill SW<0.5 ══
    ax     = axes[5]
    sw_col = _resolve_col(df, ["SW"])
    if sw_col:
        sw   = df[sw_col].values.copy()
        hc_m = (sw < 0.5) & np.isfinite(sw)
        ax.plot(sw, z, color="#d62728", lw=0.8, zorder=3)
        if hc_m.sum() > 0:
            ax.fill_betweenx(z, sw, 0.5, where=hc_m,
                             color="#FF1493", alpha=0.55, zorder=2)
        ax.axvline(0.5, color="#FF1493", ls="--", lw=0.6, alpha=0.70)
    ax.set_xlabel("SW (frac)", color="#d62728", fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis="x", colors="#d62728", labelsize=7)

    # ══ T7: LITHO ══
    ax        = axes[6]
    litho_col = _resolve_col(df, ["LITHO","LITHO_CODE"])
    if litho_col:
        litho = df[litho_col].values
        for code, color in LITHO_COLORS.items():
            mask = litho == code
            if mask.sum() > 0:
                ax.fill_betweenx(z, 0, 1, where=mask,
                                 color=color, alpha=0.88, zorder=1)
        legend_patches = [Patch(facecolor=c, label=LITHO_LABELS[code])
                          for code, c in LITHO_COLORS.items()]
        ax.legend(handles=legend_patches, fontsize=6,
                  loc="lower center", framealpha=0.85)
    else:
        ax.text(0.5, 0.5, "No\nLITHO", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
    ax.set_xlim(0, 1)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # ══ T8: PI ══
    ax     = axes[7]
    pi_col = _resolve_col(df, ["PI","PI_GASSMANN","AI"])
    if pi_col:
        ax.plot(df[pi_col].values, z, color="#2E8B57", lw=0.8)
    ax.set_xlabel("PI (m/s·g/cc)", fontsize=8)
    ax.tick_params(labelsize=7)

    # ══ T9: SI ══
    ax     = axes[8]
    si_col = _resolve_col(df, ["SI","SI_GASSMANN"])
    if si_col:
        ax.plot(df[si_col].values, z, color="#8B008B", lw=0.8)
    ax.set_xlabel("SI (m/s·g/cc)", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── common formatting ──
    for i, ax in enumerate(axes):
        ax.set_title(track_titles[i], fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.28, lw=0.4)
        ax.set_ylim(z_mx, z_mn)
        if i > 0:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.tick_params(axis="y", labelsize=7)

    # ── formation tops ──
    if tops:
        for fm_name, fm_d in tops.items():
            fm_d = float(fm_d)
            if z_mn <= fm_d <= z_mx:
                for ax in axes:
                    ax.axhline(fm_d, color="black", lw=0.7, ls="-",
                               alpha=0.60, zorder=10)
                axes[0].text(0.02, fm_d, f" {fm_name}",
                             transform=axes[0].get_yaxis_transform(),
                             fontsize=6, va="center", ha="left",
                             bbox=dict(boxstyle="round,pad=0.15",
                                       fc="yellow", alpha=0.70, lw=0))

    fig.suptitle(f"Composite well log — {well_id}{title_suffix}",
                 fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    if show:
        plt.show()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
#          2-TRACK FACIES COMPARISON: Rule-based vs GMM
# ═══════════════════════════════════════════════════════════════════════════
def plot_facies_comparison_2track(
    well_dframes: Dict[str, pd.DataFrame],
    well_tops: Optional[Dict[str, Dict]] = None,
    depth_col: str = "DEPTH",
    litho_rule_col: str = "LITHO",
    litho_gmm_col:  str = "LITHO_GMM",
    figsize: Tuple[float, float] = (4, 15),
    title: str = "Facies comparison — Rule-based vs GMM",
    show: bool = True,
):
    """
    2 tracks per well: kiri = Rule-based LITHO, kanan = GMM LITHO.
    Sumur ditampilkan berdampingan.
    """
    n_wells = len(well_dframes)
    n_cols  = n_wells * 2
    fig, axes = plt.subplots(1, n_cols,
                              figsize=(figsize[0] * n_wells, figsize[1]))
    if n_cols == 1:
        axes = [axes]
    axes = list(axes)

    col_idx = 0
    for well_id, df in well_dframes.items():
        if depth_col not in df.columns:
            col_idx += 2
            continue

        z    = df[depth_col].values
        z_mn = float(np.nanmin(z))
        z_mx = float(np.nanmax(z))
        tops = (well_tops or {}).get(well_id, {})

        for track_idx, (col_name, track_title, bg) in enumerate([
            (litho_rule_col, "Rule-based", "#fff8f5"),
            (litho_gmm_col,  "GMM",        "#f2f5ff"),
        ]):
            ax = axes[col_idx + track_idx]
            ax.set_facecolor(bg)

            if col_name in df.columns:
                litho = df[col_name].values
                for code, color in LITHO_COLORS.items():
                    mask = litho == code
                    if mask.sum() > 0:
                        ax.fill_betweenx(z, 0, 1, where=mask,
                                         color=color, alpha=0.90, zorder=1)

            # formation tops
            for fm_name, fm_d in tops.items():
                fm_d = float(fm_d)
                if z_mn <= fm_d <= z_mx:
                    ax.axhline(fm_d, color="black", lw=0.9, alpha=0.65, zorder=5)
                    if track_idx == 0:
                        ax.text(0.03, fm_d, f" {fm_name}",
                                transform=ax.get_yaxis_transform(),
                                fontsize=6, va="center",
                                bbox=dict(boxstyle="round,pad=0.1",
                                          fc="white", alpha=0.75, lw=0))

            ax.set_xlim(0, 1)
            ax.set_ylim(z_mx, z_mn)
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_title(f"{well_id}\n{track_title}", fontsize=9,
                         fontweight="bold",
                         color="#8B0000" if track_idx == 0 else "#00008B")
            ax.grid(True, alpha=0.22)
            if col_idx + track_idx > 0:
                ax.tick_params(axis="y", left=False, labelleft=False)
            else:
                ax.set_ylabel("Depth (m)", fontsize=9)

        col_idx += 2

    legend_patches = [Patch(facecolor=c, label=LITHO_LABELS[code],
                            edgecolor="gray", lw=0.5)
                      for code, c in LITHO_COLORS.items()]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=3, fontsize=9, framealpha=0.90,
               bbox_to_anchor=(0.5, 0.00))

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1.0])
    if show:
        plt.show()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
#          SENSITIVITY ANALYSIS — bersih, tanpa RPT curves
# ═══════════════════════════════════════════════════════════════════════════
def plot_sensitivity_analysis(
    well_dframes: Dict[str, pd.DataFrame],
    figsize: Tuple[float, float] = (14, 6),
    title: str = "Sensitivity analysis",
    show: bool = True,
):
    """
    Dua crossplot sensitivity bersih (tanpa RPT):
      Kiri  : AI vs Vp/Vs  — per facies + density contour + error bars + stats
      Kanan : PI vs SI      — per facies + density contour + error bars + stats
    """
    df_all = pd.concat(
        [df.assign(WELL=wid) for wid, df in well_dframes.items()],
        ignore_index=True
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    def _scatter_panel(ax, sub, x_col, y_col, xlabel, ylabel, panel_title):
        """Generic scatter + KDE + errorbar + stats box."""
        if sub is None or len(sub) < 5:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return

        # KDE contour fill + lines
        try:
            xy  = np.vstack([sub[x_col].values, sub[y_col].values])
            kde = gaussian_kde(xy, bw_method=0.22)
            xg  = np.linspace(sub[x_col].quantile(0.01), sub[x_col].quantile(0.99), 80)
            yg  = np.linspace(sub[y_col].quantile(0.01), sub[y_col].quantile(0.99), 80)
            Xg, Yg = np.meshgrid(xg, yg)
            Zg = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
            ax.contourf(Xg, Yg, Zg, levels=8, cmap="Greys", alpha=0.16, zorder=0)
            ax.contour( Xg, Yg, Zg, levels=8, colors="gray",
                        alpha=0.45, linewidths=0.55, zorder=1)
        except Exception:
            pass

        # Scatter + error bars
        stat_lines = [f"{panel_title} — per-facies (mean ± σ):"]
        for code, color in LITHO_COLORS.items():
            mask = sub["LITHO_GMM"] == code
            if mask.sum() < 3:
                continue
            ax.scatter(sub.loc[mask, x_col], sub.loc[mask, y_col],
                       c=color, s=11, alpha=0.52, edgecolors="none",
                       label=f"{LITHO_LABELS[code]} (n={mask.sum()})", zorder=2)
            mx, my = sub.loc[mask, x_col].mean(), sub.loc[mask, y_col].mean()
            sx, sy = sub.loc[mask, x_col].std(),  sub.loc[mask, y_col].std()
            ax.errorbar(mx, my, xerr=sx, yerr=sy,
                        fmt="o", color=color, ms=7,
                        elinewidth=1.3, capsize=3.5, zorder=5,
                        markeredgecolor="black", markeredgewidth=0.5)
            stat_lines.append(
                f"  {LITHO_LABELS[code]:8s}: "
                f"{x_col.split('_')[0]}={mx:.0f}±{sx:.0f}  "
                f"{y_col.split('_')[0]}={my:.0f}±{sy:.0f}  n={mask.sum()}")

        ax.text(0.02, 0.98, "\n".join(stat_lines),
                transform=ax.transAxes, fontsize=7, va="top", ha="left",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          alpha=0.88, ec="lightgray", lw=0.5))

        # HC Sand reference lines (P75 boundary)
        hc = sub[sub["LITHO_GMM"] == 3]
        if len(hc) > 5:
            ax.axvline(hc[x_col].quantile(0.75), color=LITHO_COLORS[3],
                       lw=1.1, ls="--", alpha=0.70)
            ax.axhline(hc[y_col].quantile(0.75), color=LITHO_COLORS[3],
                       lw=1.1, ls="--", alpha=0.70)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(panel_title + "\n(sensitivity — no RPT)", fontsize=11,
                     fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.88)
        ax.grid(True, alpha=0.28, lw=0.4)

    # Panel kiri: AI vs Vp/Vs
    ai_col   = _resolve_col(df_all, ["PI"])
    vpvs_col = _resolve_col(df_all, ["VPVS"])
    if ai_col and vpvs_col and "LITHO_GMM" in df_all.columns:
        sub_l = df_all[[ai_col, vpvs_col, "LITHO_GMM"]].dropna()
        _scatter_panel(axes[0], sub_l, ai_col, vpvs_col,
                       f"AI — {ai_col} (m/s·g/cc)", "Vp/Vs", "AI vs Vp/Vs")
    else:
        axes[0].text(0.5, 0.5, "AI / VPVS column\nnot found",
                     ha="center", va="center", transform=axes[0].transAxes)

    # Panel kanan: PI vs SI
    pi_col = _resolve_col(df_all, ["PI"])
    si_col = _resolve_col(df_all, ["SI"])
    if pi_col and si_col and "LITHO_GMM" in df_all.columns:
        sub_r = df_all[[pi_col, si_col, "LITHO"]].dropna()
        _scatter_panel(axes[1], sub_r, pi_col, si_col,
                       f"PI — {pi_col} (m/s·g/cc)",
                       f"SI — {si_col} (m/s·g/cc)", "PI vs SI")
    else:
        axes[1].text(0.5, 0.5, "PI / SI column\nnot found",
                     ha="center", va="center", transform=axes[1].transAxes)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if show:
        plt.show()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
#          STANDARD CROSSPLOT — RPT + density + stats (unchanged)
# ═══════════════════════════════════════════════════════════════════════════
def plot_crossplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_by: str = "LITHO",
    rpt_curves: Optional[List[Dict]] = None,
    density: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10, 8),
    annotate_stats: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sub = df[[x_col, y_col, color_by]].dropna() if color_by in df.columns \
        else df[[x_col, y_col]].dropna()

    if color_by in sub.columns:
        for code, color in LITHO_COLORS.items():
            mask = sub[color_by] == code
            if mask.sum() > 0:
                ax.scatter(sub.loc[mask, x_col], sub.loc[mask, y_col],
                           c=color, s=30, alpha=0.6, edgecolors="none",
                           label=f"{LITHO_LABELS[code]} (n={mask.sum()})",
                           zorder=2)
    else:
        ax.scatter(sub[x_col], sub[y_col], c="steelblue",
                   s=30, alpha=0.6, edgecolors="none", zorder=2)

    if rpt_curves:
        for curve in rpt_curves:
            ax.plot(curve["x"], curve["y"],
                    color=curve.get("color","black"),
                    ls=curve.get("ls","-"),
                    lw=curve.get("lw", 1.4),
                    label=curve.get("label",""),
                    alpha=curve.get("alpha", 0.9),
                    zorder=3)

    if annotate_stats and color_by in sub.columns:
        stat_lines = ["Per-facies mean ± σ:"]
        for code in sorted(sub[color_by].unique()):
            if code in LITHO_LABELS:
                fac = sub[sub[color_by] == code]
                if len(fac) > 5:
                    mx, my = fac[x_col].mean(), fac[y_col].mean()
                    sx, sy = fac[x_col].std(),  fac[y_col].std()
                    stat_lines.append(
                        f"{LITHO_LABELS[code]:8s}: "
                        f"({mx:.2f}±{sx:.2f}, {my:.3f}±{sy:.3f})  n={len(fac)}"
                    )
        if len(stat_lines) > 1:
            ax.text(0.02, 0.98, "\n".join(stat_lines),
                    transform=ax.transAxes, fontsize=7,
                    va="top", ha="left", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4",
                              fc="white", alpha=0.85, ec="gray", lw=0.4))

    ax.set_xlabel(xlabel or x_col, fontsize=10)
    ax.set_ylabel(ylabel or y_col, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.30, lw=0.4)
    ax.legend(fontsize=7, loc="best", framealpha=0.85)

    if show:
        plt.tight_layout()
        plt.show()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
#                 HISTOGRAM per log (multi-well overlay)
# ═══════════════════════════════════════════════════════════════════════════
def plot_log_histograms(
    well_dframes: Dict[str, pd.DataFrame],
    cols: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (16, 8),
    title: str = "Distribusi log per sumur",
    show: bool = True,
):
    if cols is None:
        cols = ["GR","VP","VS","RHOB","NPHI","PHIE","VSH","SW"]
    n    = len(cols)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(cols):
        ax = axes[i]
        for wid, dfw in well_dframes.items():
            if col in dfw.columns:
                v = dfw[col].dropna().values
                if v.size:
                    ax.hist(v, bins=40, alpha=0.55,
                            color=WELL_COLORS.get(wid,"gray"),
                            label=f"{wid} (n={v.size})",
                            edgecolor="black", lw=0.3)
        ax.set_xlabel(col, fontsize=9)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.28)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if show:
        plt.show()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
#                   SEISMIC SECTION DISPLAY
# ═══════════════════════════════════════════════════════════════════════════
def plot_seismic_section(
    cube, twt_axis, trace_axis=None, title="Seismic section",
    horizons=None, horizon_styles=None, well_traces=None,
    cmap="RdBu_r", vmin=None, vmax=None,
    figsize=(16, 7), cbar_label="Amplitude", ax=None, show=True,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if trace_axis is None:
        trace_axis = np.arange(cube.shape[0])
    if vmin is None or vmax is None:
        v = float(np.nanpercentile(np.abs(cube), 98))
        vmin, vmax = -v, v
    im = ax.imshow(cube.T, aspect="auto",
                   extent=[trace_axis[0], trace_axis[-1],
                           twt_axis[-1], twt_axis[0]],
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    if horizons:
        for hname, hz in horizons.items():
            style = (horizon_styles or {}).get(hname, {"color":"black","lw":2.0,"ls":"-"})
            ax.plot(trace_axis[:len(hz)], hz, **style, label=hname)
    if well_traces:
        for wname, t_idx in well_traces.items():
            ax.axvline(t_idx, color="white", lw=2.5, alpha=0.9, zorder=4)
            ax.axvline(t_idx, color="black", lw=1.2, ls="--", alpha=0.95, zorder=5)
            ax.text(t_idx, twt_axis[0], f" {wname}", fontsize=9,
                    fontweight="bold", va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              alpha=0.85, ec="black", lw=0.4))
    ax.set_xlabel("Trace index", fontsize=10)
    ax.set_ylabel("TWT (s)" if twt_axis.max() < 100 else "TWT (ms)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if horizons:
        ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label=cbar_label)
    if show:
        plt.tight_layout(); plt.show()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
#               INVERSION SECTION
# ═══════════════════════════════════════════════════════════════════════════
def plot_inversion_section(
    cube, twt_axis, trace_axis=None, title="Inversion result", cbar_label="",
    horizons=None, horizon_styles=None, well_traces=None,
    cmap="viridis", vmin=None, vmax=None,
    figsize=(16, 6), ax=None, show=True,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if trace_axis is None:
        trace_axis = np.arange(cube.shape[0])
    im = ax.imshow(cube.T, aspect="auto",
                   extent=[trace_axis[0], trace_axis[-1],
                           twt_axis[-1], twt_axis[0]],
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    if horizons:
        for hname, hz in horizons.items():
            style = (horizon_styles or {}).get(hname, {"color":"black","lw":0.9,"ls":"-"})
            ax.plot(trace_axis[:len(hz)], hz, **style, label=hname)
    if well_traces:
        for wname, t_idx in well_traces.items():
            ax.axvline(t_idx, color="white", lw=2.0, alpha=0.85, zorder=4)
            ax.axvline(t_idx, color="black", lw=1.0, ls="--", alpha=0.95, zorder=5)
            ax.text(t_idx, twt_axis[0], f" {wname}", fontsize=8,
                    fontweight="bold", va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="white", alpha=0.85, ec="black", lw=0.3))
    ax.set_xlabel("Trace index", fontsize=10)
    ax.set_ylabel("TWT (s)" if twt_axis.max() < 100 else "TWT (ms)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if horizons:
        ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label=cbar_label)
    if show:
        plt.tight_layout(); plt.show()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
#                GMM DIAGNOSTIC — PCA projection
# ═══════════════════════════════════════════════════════════════════════════
def plot_gmm_diagnostic(
    X, feature_names, gmm, litho_labels=None,
    title="GMM cluster diagnostic",
    figsize=(14, 5), show=True,
):
    from sklearn.decomposition import PCA
    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    n_p   = 2 if litho_labels is not None else 1
    fig, axes = plt.subplots(1, n_p, figsize=figsize)
    if n_p == 1:
        axes = [axes]

    gmm_pred = gmm.predict(X)
    ax = axes[0]
    for k in range(gmm.n_components):
        m = gmm_pred == k
        if m.sum() > 0:
            ax.scatter(X_pca[m, 0], X_pca[m, 1],
                       c=list(LITHO_COLORS.values())[k % 3],
                       s=8, alpha=0.50, label=f"GMM comp {k} ({m.sum()})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=9)
    ax.set_title("GMM components (PCA)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.28)

    if litho_labels is not None:
        ax = axes[1]
        for code, color in LITHO_COLORS.items():
            m = litho_labels == code
            if m.sum() > 0:
                ax.scatter(X_pca[m, 0], X_pca[m, 1],
                           c=color, s=8, alpha=0.50,
                           label=f"{LITHO_LABELS[code]} ({m.sum()})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=9)
        ax.set_title("Rule-based litho (reference)", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.28)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes
