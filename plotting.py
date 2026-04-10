import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from CoolProp.CoolProp import PropsSI
import matplotlib.patheffects as pe

def plot_logph_cycles(
    ref: str,
    t,
    cycle_ph,
    *,
    t_idx=None,
    at_time=None,
    t_start=None,
    t_end=None,
    every_s=None,                 # e.g. 60.0
    plot_dome=True,

    # Isotherms + grid
    isotherms: bool = True,
    iso_Ts_C=None,                # e.g. [-10, 0, 10, 20, 30, 40]
    n_iso: int = 10,               # used when iso_Ts_C is None
    iso_style: str = "--",
    iso_lw: float = 1.0,
    iso_alpha: float = 0.45,
    iso_labels: bool = True,
    grid: str = "dashed",         # "none" | "dashed" | "light"

    # ---- Auto-scaling without changing the call site ----
    figsize=None,                 # None => automatic (larger default)
    xlim=None,                    # (xmin, xmax) in kJ/kg; None => automatic
    ylim=None,                    # (ymin, ymax) in bar;   None => automatic

    title=None,
    save_path=None,
    show=True
):
    """
    t:        (nt,)
    cycle_ph: (nt,4,2) or list of 4x2 per time:
              cycle_ph[i] = [[p1,h1],[p2,h2],[p3,h3],[p4,h4]] with p[Pa], h[J/kg]
    """

    def _label_angle(ax, x, y, i, clamp=80):
        """
        Curve angle in screen coordinates, normalized to [-90, 90],
        with optional clamping to +/- clamp.
        """
        i0 = max(i - 1, 0)
        i1 = min(i + 1, len(x) - 1)

        p0 = ax.transData.transform((x[i0], y[i0]))
        p1 = ax.transData.transform((x[i1], y[i1]))

        ang = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))

        ang = (ang + 180.0) % 360.0 - 180.0  # -> [-180, 180]
        if ang > 90.0:
            ang -= 180.0
        elif ang < -90.0:
            ang += 180.0

        if clamp is not None:
            ang = max(-clamp, min(clamp, ang))

        return ang

    show_legend = not all(v is None for v in (t if hasattr(t, '__iter__') else [t]))
    t = np.asarray(t, dtype=float).ravel()

    arr = np.asarray(cycle_ph, dtype=object)
    if arr.ndim == 1:
        arr = np.stack([np.asarray(row, dtype=float) for row in arr], axis=0)
    else:
        arr = np.asarray(arr, dtype=float)
    cycle_ph = arr

    if cycle_ph.ndim != 3 or cycle_ph.shape[1:] != (4, 2):
        raise ValueError(f"cycle_ph has shape {cycle_ph.shape}, expected (nt,4,2).")

    # --- Time selection ---
    if t_idx is not None:
        idxs = np.atleast_1d(t_idx).astype(int)
        idxs = np.array([i if i >= 0 else len(t) + i for i in idxs], dtype=int)
    elif at_time is not None:
        targets = np.atleast_1d(at_time).astype(float)
        idxs = np.array([np.abs(t - tau).argmin() for tau in targets], dtype=int)
    elif (t_start is not None) or (t_end is not None):
        if t_start is None: t_start = float(t[0])
        if t_end   is None: t_end   = float(t[-1])
        if every_s is None:
            idxs = np.where((t >= t_start) & (t <= t_end))[0]
        else:
            targets = np.arange(float(t_start), float(t_end) + 1e-12, float(every_s))
            idxs = np.array([np.abs(t - tau).argmin() for tau in targets], dtype=int)
            idxs = np.unique(idxs)
    else:
        idxs = np.array([0], dtype=int)

    # --- Auto-limits (if not explicitly set) ---
    if xlim is None or ylim is None:
        auto_xlim, auto_ylim = _auto_limits_logph_from_cycle(cycle_ph[idxs])
        if xlim is None:
            xlim = auto_xlim
        if ylim is None:
            ylim = auto_ylim

    # Isotherms need pmin/pmax in Pa: prefer (auto/given) y-limits
    if ylim is not None:
        pmin_pa = float(ylim[0]) * 1e5
        pmax_pa = float(ylim[1]) * 1e5
    else:
        # Fallback
        p_sel = cycle_ph[idxs, :, 0].ravel()
        p_sel = p_sel[np.isfinite(p_sel)]
        pmin_pa = max(float(np.min(p_sel)) * 0.7, 1.0) if p_sel.size else 1e4
        pmax_pa = float(np.max(p_sel)) * 1.3 if p_sel.size else 1e7

    # --- Figure ---
    if figsize is None:
        figsize = (11, 7)  # larger default without changing call sites
    fig, ax = plt.subplots(figsize=figsize)

    iso_c = "0.0"
    dome_c = iso_c

    # Saturation dome
    if plot_dome:
        hL, hV, p = _sat_dome_ph(ref)
        ax.plot(hL/1000.0, p/1e5,color=dome_c,linewidth=1.2)
        ax.plot(hV/1000.0, p/1e5,color=dome_c,linewidth=1.2)

    # Isotherms
    if isotherms:
        if iso_Ts_C is None:
            # Tmin/Tmax from selected points (via P,H -> T)
            Ts = []
            for i in idxs:
                for k in range(4):
                    pPa = float(cycle_ph[i, k, 0])
                    hJ  = float(cycle_ph[i, k, 1])
                    if not (np.isfinite(pPa) and np.isfinite(hJ)):
                        continue
                    try:
                        Ts.append(float(PropsSI("T", "P", pPa, "H", hJ, ref) - 273.15))
                    except Exception:
                        pass
            if len(Ts) >= 2:
                Tmin, Tmax = min(Ts) - 5.0, max(Ts) + 5.0
                Ts_C = np.linspace(Tmin, Tmax, int(n_iso))
                Ts_C = [5.0 * round(x/5.0) for x in Ts_C]
                Ts_C = sorted(set(Ts_C))
            else:
                Ts_C = [-30, -20, -10, 0, 10, 20, 30, 40]
        else:
            Ts_C = list(iso_Ts_C)

        iso_lines = _compute_isotherms_ph(ref, pmin_pa, pmax_pa, Ts_C, nP=90)
        # --- Label position as fixed pressure in log scale (0..1) ---
        iso_label_pfrac = 0.05  # 0.75 => higher placement; adjust as needed
        iso_label_every = 2  # label every nth isotherm

        # Optional: readable label background
        label_bbox = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.6)

        # Target pressure (in Pa) along log(p) between pmin_pa and pmax_pa
        p_label_pa = 10.0 ** (
                np.log10(pmin_pa) + iso_label_pfrac * (np.log10(pmax_pa) - np.log10(pmin_pa))
        )

        arrow_kw = dict(arrowstyle="-", lw=0.8, color=iso_c, alpha=0.9)
        text_pe = [pe.withStroke(linewidth=3, foreground="white")]  # white stroke around text

        axis_fontsize = 14
        label_fontsize = 11
        label_weight = "normal"
        label_color = "black"

        for j, (T_C, h_vap, p_vap, h_liq, p_liq) in enumerate(iso_lines):

            # Stagger to avoid overlap (offset in points)
            dy = 9 + (j % 4) * 9  # 2, 9, 16, 23 pt
            dxR, dxL = -4, -4

            if h_vap is not None and p_vap is not None:
                ax.plot(h_vap / 1000.0, p_vap / 1e5,
                        color=iso_c, linestyle=iso_style, linewidth=iso_lw, alpha=iso_alpha)

                if iso_labels and (j % iso_label_every == 0) and len(h_vap) > 0:
                    pv = np.asarray(p_vap, dtype=float)
                    hv = np.asarray(h_vap, dtype=float)
                    m = np.isfinite(pv) & np.isfinite(hv) & (pv > 0.0)
                    pv, hv = pv[m], hv[m]

                    if pv.size:
                        # Index near target pressure (in log scale)
                        iR = int(np.nanargmin(np.abs(np.log(pv) - np.log(p_label_pa))))

                        xR = hv[iR] / 1000.0
                        yR = pv[iR] / 1e5

                        # optional: place a small marker on the curve
                        ax.plot([xR], [yR], marker="o", markersize=2.5, color=iso_c, alpha=0.9)

                        rotR = 0

                        ax.annotate(f"{T_C:.0f}°C",
                                    xy=(xR, yR),
                                    xytext=(dxR, dy), textcoords="offset points",
                                    ha="left", va="bottom",
                                    fontsize=label_fontsize, fontweight=label_weight, color=label_color,
                                    bbox=label_bbox,
                                    path_effects=text_pe,
                                    arrowprops=arrow_kw,
                                    rotation=rotR, rotation_mode="anchor",
                                    clip_on=True)

            if h_liq is not None and p_liq is not None:
                ax.plot(h_liq / 1000.0, p_liq / 1e5,
                        color=iso_c, linestyle=iso_style, linewidth=iso_lw, alpha=iso_alpha)

                if iso_labels and (j % iso_label_every == 0) and len(h_liq) > 0:
                    pl = np.asarray(p_liq, dtype=float)
                    hl = np.asarray(h_liq, dtype=float)
                    m = np.isfinite(pl) & np.isfinite(hl) & (pl > 0.0)
                    pl, hl = pl[m], hl[m]

                    if pl.size:
                        iL = int(np.nanargmin(np.abs(np.log(pl) - np.log(p_label_pa))))

                        xL = hl[iL] / 1000.0
                        yL = pl[iL] / 1e5

                        ax.plot([xL], [yL], marker="o", markersize=2.5, color=iso_c, alpha=0.9)

                        rotL = 0 #_label_angle(ax, hl / 1000.0, pl / 1e5, iL)  # optional

                        ax.annotate(f"{T_C:.0f}°C",
                                    xy=(xL, yL),
                                    xytext=(dxL, dy), textcoords="offset points",
                                    ha="right", va="bottom",
                                    fontsize=label_fontsize, fontweight=label_weight, color=label_color,
                                    bbox=label_bbox,
                                    path_effects=text_pe,
                                    arrowprops=arrow_kw,
                                    rotation=rotL, rotation_mode="anchor",
                                    clip_on=True)

    # Cycle(s)
    label_cycle_idx = int(idxs[-1])  # Show point labels 1..4 only on the last plotted cycle

    for i in idxs:
        ph = cycle_ph[i]
        pPa = ph[:, 0]
        hJ = ph[:, 1]

        p_plot = np.r_[pPa, pPa[0]] / 1e5
        h_plot = np.r_[hJ, hJ[0]] / 1000.0

        ax.plot(h_plot, p_plot, marker="o", label=f"t={t[i]:g} s" if show_legend else None)

        # Draw point labels only once
        if int(i) == label_cycle_idx:
            for k in range(4):
                ax.annotate(str(k + 1),
                            (hJ[k] / 1000.0, pPa[k] / 1e5),
                            bbox=label_bbox,
                            fontsize=label_fontsize, fontweight=label_weight, color=label_color,
                            textcoords="offset points", xytext=(5, 5))

    ax.set_yscale("log")
    ax.set_xlabel("h [kJ/kg]",
                  fontsize=axis_fontsize)
    ax.set_ylabel("p [bar]",
                  fontsize=axis_fontsize)
    ax.tick_params(axis="both", which="both", labelsize=axis_fontsize)

    # Apply auto/manual limits
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    _grid_style(ax, grid)

    if show_legend:
        ax.legend()
    ax.set_title(title if title else f"log(p)-h Diagram ({ref})",
                  fontsize=axis_fontsize)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax

def _auto_limits_logph_from_cycle(cycle_ph_sel, *, x_pad_rel=0.5, x_pad_abs=15.0, y_pad_factor=1.5):
    """
    cycle_ph_sel: (nsel, 4, 2) with p[Pa], h[J/kg]
    Returns: (xlim_kJkg, ylim_bar)
    """
    ph = np.asarray(cycle_ph_sel, dtype=float)
    p_bar = ph[:, :, 0] / 1e5
    h_kj  = ph[:, :, 1] / 1000.0

    p_bar = p_bar[np.isfinite(p_bar)]
    h_kj  = h_kj[np.isfinite(h_kj)]

    if p_bar.size == 0 or h_kj.size == 0:
        return None, None

    hmin, hmax = float(np.min(h_kj)), float(np.max(h_kj))
    pmin, pmax = float(np.min(p_bar[p_bar > 0])) if np.any(p_bar > 0) else 0.1, float(np.max(p_bar))

    dh = hmax - hmin
    pad_x = max(x_pad_abs, x_pad_rel * dh) if dh > 1e-9 else x_pad_abs
    xlim = (hmin - pad_x, hmax + pad_x)

    # log-y: multiplicative padding is more stable
    ylo = max(pmin / y_pad_factor, 1e-6)
    yhi = pmax * y_pad_factor
    ylim = (ylo, yhi)

    return xlim, ylim

def _sat_dome_ph(ref: str, n: int = 400):
    T_tr = PropsSI("Ttriple", ref)
    T_cr = PropsSI("Tcrit", ref)
    T = np.linspace(T_tr + 1.0, T_cr - 1.0, n)

    p  = np.array([PropsSI("P", "T", Ti, "Q", 0, ref) for Ti in T], dtype=float)
    hL = np.array([PropsSI("H", "T", Ti, "Q", 0, ref) for Ti in T], dtype=float)
    hV = np.array([PropsSI("H", "T", Ti, "Q", 1, ref) for Ti in T], dtype=float)
    return hL, hV, p

def _compute_isotherms_ph(ref: str, pmin: float, pmax: float, Ts_C, nP: int = 90):
    lines = []
    for T_C in Ts_C:
        T_K = float(T_C) + 273.15
        try:
            p_sat = float(PropsSI("P", "T", T_K, "Q", 0, ref))
        except Exception:
            continue

        # Vapor side (p <= p_sat)
        h_vap = p_vap = None
        p_hi_vap = min(p_sat * 0.999, pmax)
        if pmin < p_hi_vap:
            pv = np.geomspace(max(pmin, 1.0), p_hi_vap, nP)
            hv = []
            for p in pv:
                try:
                    hv.append(float(PropsSI("H", "T", T_K, "P", float(p), ref)))
                except Exception:
                    hv.append(np.nan)
            hv = np.asarray(hv, dtype=float)
            m = np.isfinite(hv)
            if np.any(m):
                h_vap, p_vap = hv[m], pv[m]

        # Liquid side (p >= p_sat)
        h_liq = p_liq = None
        p_lo_liq = max(p_sat * 1.001, pmin)
        if p_lo_liq < pmax:
            pl = np.geomspace(p_lo_liq, pmax, nP)
            hl = []
            for p in pl:
                try:
                    hl.append(float(PropsSI("H", "T", T_K, "P", float(p), ref)))
                except Exception:
                    hl.append(np.nan)
            hl = np.asarray(hl, dtype=float)
            m = np.isfinite(hl)
            if np.any(m):
                h_liq, p_liq = hl[m], pl[m]

        if (h_vap is not None) or (h_liq is not None):
            lines.append((float(T_C), h_vap, p_vap, h_liq, p_liq))
    return lines

def _grid_style(ax, grid: str):
    grid = (grid or "dashed").lower()
    if grid in ("none", "off", "false", "0"):
        ax.grid(False)
    elif grid in ("dashed", "dash"):
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)
    elif grid in ("light", "thin"):
        ax.grid(True, which="both", linestyle="-", linewidth=0.4, alpha=0.2)
    else:
        raise ValueError("grid must be 'none', 'dashed', or 'light'.")