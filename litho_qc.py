"""
litho_qc.py
-----------
Utility klasifikasi litofasies rule-based + QC visualisasi untuk pipeline NB2.

Fungsi:
    classify_lithology_rule(df, lith_cfg)      -> df + kolom LITHO, LITHO_LBL
    plot_lithology_qc(df, well_id, lith_cfg)   -> 5-track: GR | Resis | VSH | SW | Litho
    plot_lithology_histogram(litho_summary)    -> bar chart distribusi per sumur

Konvensi warna konsisten dengan plot_utils.py:
    Shale    = #8b6f47   Wet Sand = #5ba3d0   HC Sand = #f0a830
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

# -----------------------------------------------------------------------------
# Color palette (sinkron dgn plot inversi)
# -----------------------------------------------------------------------------
LITHO_COLORS = {"Shale": "#8b6f47", "Wet Sand": "#5ba3d0", "HC Sand": "#f0a830"}
LITHO_ORDER  = ["Shale", "Wet Sand", "HC Sand"]
LITHO_CMAP   = ListedColormap([LITHO_COLORS[k] for k in LITHO_ORDER])
LITHO_NORM   = BoundaryNorm([0.5, 1.5, 2.5, 3.5], LITHO_CMAP.N)


# =============================================================================
# 1. KLASIFIKASI RULE-BASED  (tetap kompatibel dgn kode lama)
# =============================================================================
def classify_lithology_rule(df, lith_cfg):
    """
    Rule-based 3-kelas: Shale / Wet Sand / HC Sand.

    Cutoff:
        Shale : VSH > vsh_cutoff   (fallback: GR > P{gr_percentile})
        HC    : (NOT shale) AND SW < sw_cutoff
        Wet   : sisanya

    Menyimpan cutoff aktual ke `df.attrs['litho_cutoffs']` agar bisa
    di-overlay di QC plot tanpa perlu hitung ulang.
    """
    df = df.copy()
    cutoffs = {}

    # ---- Shale gate ----
    if "VSH" in df.columns and df["VSH"].notna().any():
        vsh_cut = float(lith_cfg["vsh_cutoff"])
        cond_shale = df["VSH"] > vsh_cut
        cutoffs["mode"], cutoffs["vsh_cut"] = "VSH", vsh_cut
    elif "GR" in df.columns and df["GR"].notna().any():
        gr_cut = float(df["GR"].dropna().quantile(lith_cfg["gr_percentile"] / 100))
        cond_shale = df["GR"] > gr_cut
        cutoffs["mode"], cutoffs["gr_cut"] = "GR", gr_cut
    else:
        cond_shale = pd.Series(True, index=df.index)
        cutoffs["mode"] = "FALLBACK_ALL_SHALE"

    # GR-percentile selalu disimpan utk overlay (jika GR tersedia)
    if "GR" in df.columns and df["GR"].notna().any():
        cutoffs["gr_p_overlay"] = float(
            df["GR"].dropna().quantile(lith_cfg["gr_percentile"] / 100))
        cutoffs["gr_percentile"] = lith_cfg["gr_percentile"]

    # ---- HC gate ----
    sw_cut = float(lith_cfg["sw_cutoff"])
    sw_series = df.get("SW", pd.Series(1.0, index=df.index))
    cond_hc = (~cond_shale) & (sw_series < sw_cut)
    cutoffs["sw_cut"] = sw_cut

    litho_int = np.where(cond_shale, 1, np.where(cond_hc, 3, 2)).astype(np.int8)
    label_map = {1: "Shale", 2: "Wet Sand", 3: "HC Sand"}

    df["LITHO"]     = litho_int
    df["LITHO_LBL"] = pd.Categorical(
        [label_map[x] for x in litho_int], categories=LITHO_ORDER)
    df.attrs["litho_cutoffs"] = cutoffs
    return df


# =============================================================================
# 2. QC PLOT 5-TRACK : GR | RES | VSH | SW | LITHO
# =============================================================================
def plot_lithology_qc(df, well_id, lith_cfg, depth_col=None,
                       tops=None, depth_range=None, savepath=None):
    """
    QC track untuk mengevaluasi cutoff klasifikasi.

    Parameters
    ----------
    df          : DataFrame hasil classify_lithology_rule()
    well_id     : str    — judul plot
    lith_cfg    : dict   — CFG['lithology']
    depth_col   : str    — auto-detect ('DEPTH'/'DEPT'/'MD') bila None
    tops        : dict   — WELL_TOPS[well_id] = {fm_name: depth}
    depth_range : tuple  — (zmin, zmax) zoom ke interval target
    savepath    : str    — path PNG output (opsional)
    """
    # ----- depth column auto-detect -----
    if depth_col is None:
        for c in ["DEPTH", "DEPT", "MD", "TVD"]:
            if c in df.columns:
                depth_col = c; break
        if depth_col is None:
            raise KeyError("Tidak ada kolom DEPTH/DEPT/MD/TVD di DataFrame.")

    if depth_range is not None:
        zmin, zmax = depth_range
        df = df[(df[depth_col] >= zmin) & (df[depth_col] <= zmax)].copy()

    d   = df[depth_col].to_numpy()
    cut = df.attrs.get("litho_cutoffs", {})
    vsh_cut = cut.get("vsh_cut", lith_cfg["vsh_cutoff"])
    sw_cut  = cut.get("sw_cut",  lith_cfg["sw_cutoff"])
    gr_cut  = cut.get("gr_p_overlay", None)
    gr_pct  = cut.get("gr_percentile", lith_cfg.get("gr_percentile", 35))

    fig, axes = plt.subplots(
        1, 5, figsize=(9, 15), sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.5]})

    title = f"QC Lithology Classification — {well_id}"
    sub   = (f"mode={cut.get('mode','?')}  |  "
             f"VSH cut={vsh_cut:.2f}  |  SW cut={sw_cut:.2f}")
    if gr_cut is not None:
        sub += f"  |  GR P{gr_pct}={gr_cut:.1f}"
    fig.suptitle(f"{title}\n{sub}", fontsize=12, fontweight="bold")

    # ---------- Track 1: GR ----------
    ax = axes[0]
    if "GR" in df.columns:
        gr = df["GR"].to_numpy()
        ax.plot(gr, d, color="black", lw=0.6)
        gr_xmax = np.nanpercentile(gr, 99.5)
        if gr_cut is not None:
            ax.axvline(gr_cut, color="red", ls="--", lw=1.2)
            ax.fill_betweenx(d, gr_cut, gr, where=(gr > gr_cut),
                              color=LITHO_COLORS["Shale"], alpha=0.40,
                              interpolate=True, label=f"Shale (GR>P{gr_pct})")
            ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim(0, gr_xmax); ax.set_xlabel("GR (API)")
    ax.set_ylabel(f"{depth_col} (m)"); ax.invert_yaxis(); ax.grid(alpha=0.3)

    # ---------- Track 2: Resistivity (log scale) ----------
    ax = axes[1]
    rd_col = next((c for c in ["RD", "RT", "RES", "ILD", "LLD"]
                   if c in df.columns), None)
    if rd_col is not None:
        rd = df[rd_col].to_numpy()
        ax.plot(rd, d, color="purple", lw=0.6)
        ax.set_xscale("log"); ax.set_xlim(0.2, 2000)
        ax.set_xlabel(f"{rd_col} (Ω·m)")
    else:
        ax.text(0.5, 0.5, "No Resistivity", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_xlabel("RES")
    ax.grid(alpha=0.3, which="both")

    # ---------- Track 3: VSH ----------
    ax = axes[2]
    if "VSH" in df.columns:
        vsh = df["VSH"].to_numpy()
        ax.plot(vsh, d, color="black", lw=0.6)
        ax.axvline(vsh_cut, color="red", ls="--", lw=1.2,
                   label=f"cut={vsh_cut:.2f}")
        ax.fill_betweenx(d, vsh_cut, vsh, where=(vsh > vsh_cut),
                         color=LITHO_COLORS["Shale"], alpha=0.45,
                         interpolate=True)
        ax.set_xlim(0, 1); ax.set_xlabel("VSH (v/v)")
        ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    # ---------- Track 4: SW ----------
    ax = axes[3]
    if "SW" in df.columns:
        sw = df["SW"].to_numpy()
        ax.plot(sw, d, color="black", lw=0.6)
        ax.axvline(sw_cut, color="red", ls="--", lw=1.2,
                   label=f"cut={sw_cut:.2f}")
        # zona HC (SW<cut) — shade orange
        ax.fill_betweenx(d, 0, sw, where=(sw < sw_cut),
                         color=LITHO_COLORS["HC Sand"], alpha=0.55,
                         interpolate=True, label="HC zone")
        ax.set_xlim(0, 1); ax.set_xlabel("SW (v/v)")
        ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    # ---------- Track 5: LITHO column ----------
    ax = axes[4]
    if "LITHO" in df.columns and len(d) > 0:
        litho = df["LITHO"].to_numpy().reshape(-1, 1)
        ax.imshow(litho, aspect="auto", cmap=LITHO_CMAP, norm=LITHO_NORM,
                   extent=[0, 1, d.max(), d.min()], interpolation="nearest")
        ax.set_xticks([]); ax.set_xlabel("LITHO")

    # ---------- Formation tops overlay ----------
    if tops is not None and well_id in tops:
        for fm_name, fm_depth in tops[well_id].items():
            if depth_range is not None and not (zmin <= fm_depth <= zmax):
                continue
            for a in axes:
                a.axhline(fm_depth, color="black", lw=1.2, alpha=0.7)
            axes[0].text(0.02, fm_depth, fm_name, fontsize=10,
                         va="bottom",
                         transform=axes[0].get_yaxis_transform(),
                         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none",
                                   pad=1))

    # ---------- Litho legend ----------
    handles = [Patch(facecolor=LITHO_COLORS[k], label=k) for k in LITHO_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.005), frameon=True, fontsize=10)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# =============================================================================
# 3. HISTOGRAM DISTRIBUSI LITOFASIES
# =============================================================================
def plot_lithology_histogram(litho_summary, savepath=None):
    """
    Bar chart distribusi 3 kelas — sisi kiri count, sisi kanan persentase.

    Parameters
    ----------
    litho_summary : dict
        {well_id: {"Shale": n, "Wet Sand": n, "HC Sand": n}, ...}
    """
    wells   = list(litho_summary.keys())
    classes = LITHO_ORDER
    n_w     = len(wells)
    x       = np.arange(len(classes))
    width   = 0.8 / max(n_w, 1)

    fig, (ax_cnt, ax_pct) = plt.subplots(1, 2, figsize=(12, 4.5))

    for i, w in enumerate(wells):
        counts = np.array([litho_summary[w][c] for c in classes], dtype=float)
        total  = counts.sum()
        pct    = (counts / total * 100) if total > 0 else counts
        colors = [LITHO_COLORS[c] for c in classes]
        offset = (i - (n_w - 1) / 2) * width
        # alpha berbeda per sumur agar mudah dibedakan
        alpha  = 0.55 + 0.45 * (i / max(n_w - 1, 1))

        b1 = ax_cnt.bar(x + offset, counts, width, color=colors,
                        edgecolor="black", lw=0.7, alpha=alpha, label=w)
        b2 = ax_pct.bar(x + offset, pct, width, color=colors,
                        edgecolor="black", lw=0.7, alpha=alpha, label=w)

        for rect, v in zip(b1, counts):
            ax_cnt.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height(), f"{int(v)}",
                        ha="center", va="bottom", fontsize=8)
        for rect, v in zip(b2, pct):
            ax_pct.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height(), f"{v:.1f}%",
                        ha="center", va="bottom", fontsize=8)

    for ax, ttl, ylab in [
            (ax_cnt, "Distribusi Litofasies (Count)",  "N samples"),
            (ax_pct, "Distribusi Litofasies (%)",       "Persentase (%)")]:
        ax.set_xticks(x); ax.set_xticklabels(classes)
        ax.set_ylabel(ylab); ax.set_title(ttl, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, ls="--")

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# =============================================================================
# 4. HELPER: jalankan klasifikasi + QC sekaligus untuk semua sumur
# =============================================================================
def run_litho_qc_pipeline(WELL_DFRAMES, CFG, WELL_TOPS=None,
                           depth_range=None, save_dir=None):
    """
    Loop drop-in pengganti blok klasifikasi di NB2.
    Memodifikasi WELL_DFRAMES (in-place), return litho_summary.
    """
    print("\n" + "=" * 72)
    print(f"  LITHOFACIES CLASSIFICATION (RULE-BASED) + QC")
    print(f"  VSH cut={CFG['lithology']['vsh_cutoff']}  |  "
          f"SW cut={CFG['lithology']['sw_cutoff']}")
    print("=" * 72)

    litho_summary = {}
    for well_id, df_w in WELL_DFRAMES.items():
        df_c   = classify_lithology_rule(df_w, CFG["lithology"])
        counts = df_c["LITHO_LBL"].value_counts()
        litho_summary[well_id] = {
            lbl: int(counts.get(lbl, 0)) for lbl in LITHO_ORDER}

        mode = df_c.attrs.get("litho_cutoffs", {}).get("mode", "?")
        print(f"  [{well_id}]  mode={mode:<5}  "
              f"Shale={litho_summary[well_id]['Shale']:>5}  "
              f"WetSand={litho_summary[well_id]['Wet Sand']:>5}  "
              f"HCSand={litho_summary[well_id]['HC Sand']:>5}")

        WELL_DFRAMES[well_id] = df_c

        sp = (f"{save_dir}/litho_qc_{well_id}.png"
              if save_dir is not None else None)
        plot_lithology_qc(df_c, well_id, CFG["lithology"],
                           tops=WELL_TOPS, depth_range=depth_range, savepath=sp)

    sp = f"{save_dir}/litho_histogram.png" if save_dir is not None else None
    plot_lithology_histogram(litho_summary, savepath=sp)

    print("\n  ✓ Litho classification + QC plots done")
    return litho_summary
