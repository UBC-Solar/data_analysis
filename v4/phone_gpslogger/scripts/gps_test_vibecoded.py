#!/usr/bin/env python3
"""

Disclaimer: This file was written by Claude Sonnet 4.6

To run:
cd v4/phone_gpslogger
uv sync
uv run python python scripts/gps_analyzer.py data/20260331_pixel7.csv data/20260331_note10pro.csv --labels "Pixel 7" "Note 10 Pro"
-Jonah


GPS Logger Analyzer & Device Comparator
Analyzes and plots GPS data from the FDroid GPSLogger app.

Single-file mode:   python gps_analyzer.py track.csv
Multi-file compare: python gps_analyzer.py pixel7.csv note10pro.csv --labels "Pixel 7" "Note 10 Pro"
Skip individual:    python gps_analyzer.py *.csv --no-individual
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


# ── Palette ────────────────────────────────────────────────────────────────
DARK   = "#0f1117"
MID    = "#1c1f2b"
PANEL  = "#232637"
TEXT   = "#e8eaf6"
MUTED  = "#6c7a9c"
GREEN  = "#39d353"
WARM   = "#ff6b35"

# Per-device (gps_colour, network_colour)
DEVICE_PALETTES = [
    ("#00c8ff", "#ff6b35"),   # device 0: cyan GPS, orange network
    ("#b48eff", "#39d353"),   # device 1: purple GPS, green network
    ("#f5c542", "#ff4fa3"),   # device 2: yellow GPS, pink network
    ("#4fe3c1", "#ff7043"),   # device 3+
]


# ── Data loading ───────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_csv(path: Path) -> pd.DataFrame:
    """Load a GPSLogger CSV and enrich with derived columns."""
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.dropna(subset=["lat", "lon"]).sort_values("time").reset_index(drop=True)

    # Normalise provider
    if "provider" in df.columns:
        df["provider"] = df["provider"].fillna("unknown").str.lower().str.strip()
    else:
        df["provider"] = "unknown"

    # Speed: logged m/s -> km/h; network rows often have no speed logged
    if "speed" in df.columns:
        df["speed_kmh"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0) * 3.6
    else:
        dt   = df["time"].diff().dt.total_seconds().fillna(1).clip(lower=0.001)
        dist = haversine(df["lat"].shift().fillna(df["lat"]),
                         df["lon"].shift().fillna(df["lon"]),
                         df["lat"], df["lon"]).fillna(0)
        df["speed_kmh"] = (dist / dt) * 3.6

    # Elapsed minutes from first fix
    df["elapsed_min"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds() / 60

    # Cumulative distance recomputed from coordinates (monotone, comparable)
    seg = haversine(df["lat"].shift().fillna(df["lat"]),
                    df["lon"].shift().fillna(df["lon"]),
                    df["lat"], df["lon"]).fillna(0)
    df["cum_dist_km"] = seg.cumsum() / 1000

    for col in ("satellites", "accuracy", "hdop", "vdop", "pdop"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_stats(df: pd.DataFrame, label: str) -> dict:
    duration_s = (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds()
    elev = df["elevation"].dropna() if "elevation" in df.columns else pd.Series(dtype=float)
    gain = float(np.sum(np.clip(np.diff(elev.values), 0, None))) if len(elev) > 1 else float("nan")
    loss = float(np.sum(np.clip(np.diff(elev.values), None, 0))) if len(elev) > 1 else float("nan")

    gps = df[df["provider"] == "gps"]
    net = df[df["provider"] == "network"]

    return {
        "label":        label,
        "start":        df["time"].iloc[0].strftime("%H:%M:%S UTC"),
        "duration":     f"{int(duration_s//3600):02d}h {int((duration_s%3600)//60):02d}m {int(duration_s%60):02d}s",
        "points":       len(df),
        "gps_points":   len(gps),
        "net_points":   len(net),
        "distance_km":  df["cum_dist_km"].iloc[-1],
        "avg_speed":    gps["speed_kmh"].mean() if len(gps) else df["speed_kmh"].mean(),
        "max_speed":    df["speed_kmh"].max(),
        "elev_min":     float(elev.min()) if len(elev) else float("nan"),
        "elev_max":     float(elev.max()) if len(elev) else float("nan"),
        "elev_gain_m":  gain,
        "elev_loss_m":  abs(loss) if not np.isnan(loss) else float("nan"),
        "avg_acc_gps":  gps["accuracy"].mean()   if ("accuracy"   in df.columns and len(gps)) else float("nan"),
        "avg_acc_net":  net["accuracy"].mean()   if ("accuracy"   in df.columns and len(net)) else float("nan"),
        "avg_sats":     gps["satellites"].mean() if ("satellites" in df.columns and len(gps)) else float("nan"),
        "avg_hdop":     gps["hdop"].mean()       if ("hdop"       in df.columns and len(gps)) else float("nan"),
    }


# ── Shared helpers ─────────────────────────────────────────────────────────

def fv(v, fmt):
    """Format a float value, returning 'n/a' if NaN."""
    return "n/a" if (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(MID)
    ax.grid(color=MID, linewidth=0.5, linestyle="--", alpha=0.6)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=5)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8)


def colorbar_style(cb):
    cb.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
    cb.outline.set_edgecolor(MID)


# ── Single-device figure ───────────────────────────────────────────────────

def make_single_figure(df: pd.DataFrame, s: dict, palette_idx: int = 0):
    gps_col, net_col = DEVICE_PALETTES[palette_idx % len(DEVICE_PALETTES)]
    label = s["label"]
    gps = df[df["provider"] == "gps"]
    net = df[df["provider"] == "network"]

    fig = plt.figure(figsize=(16, 11), facecolor=DARK)
    fig.suptitle(f"GPS Track — {label}", color=TEXT, fontsize=14,
                 fontweight="bold", y=0.97)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           left=0.06, right=0.97, top=0.92, bottom=0.07,
                           hspace=0.50, wspace=0.42)
    ax_map   = fig.add_subplot(gs[:2, :2])
    ax_elev  = fig.add_subplot(gs[2, :2])
    ax_speed = fig.add_subplot(gs[0, 2])
    ax_acc   = fig.add_subplot(gs[1, 2])
    ax_stats = fig.add_subplot(gs[2, 2])

    # ── Map ────────────────────────────────────────────────────────────────
    ax_map.set_facecolor(PANEL)
    style_ax(ax_map, "Track Map  (GPS = speed-coloured · Network = ◆)",
             "Longitude", "Latitude")

    lats = df["lat"].values
    lons = df["lon"].values

    if len(gps) >= 2:
        spd  = gps["speed_kmh"].values
        pts  = np.array([gps["lon"].values, gps["lat"].values]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        vmax = max(df["speed_kmh"].max(), 1)
        norm = Normalize(vmin=0, vmax=vmax)
        lc   = LineCollection(segs, cmap="plasma", norm=norm, linewidth=2, alpha=0.9)
        lc.set_array(spd[:-1])
        ax_map.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax_map, pad=0.01, shrink=0.75)
        cb.set_label("Speed km/h (GPS)", color=MUTED, fontsize=8)
        colorbar_style(cb)
    elif len(gps) == 1:
        ax_map.scatter(gps["lon"], gps["lat"], s=40, color=gps_col, zorder=5)

    if not net.empty:
        ax_map.scatter(net["lon"], net["lat"], s=30, color=net_col,
                       marker="D", zorder=5, label="Network", alpha=0.85)

    ax_map.scatter([lons[0]],  [lats[0]],  s=100, color=GREEN, marker="o", zorder=7, label="Start")
    ax_map.scatter([lons[-1]], [lats[-1]], s=100, color=WARM,  marker="s", zorder=7, label="End")
    ax_map.legend(fontsize=8, facecolor=MID, edgecolor=MID, labelcolor=TEXT, loc="upper left")

    pad_lat = max((lats.max()-lats.min())*0.12, 0.001)
    pad_lon = max((lons.max()-lons.min())*0.12, 0.001)
    ax_map.set_xlim(lons.min()-pad_lon, lons.max()+pad_lon)
    ax_map.set_ylim(lats.min()-pad_lat, lats.max()+pad_lat)

    # ── Elevation ──────────────────────────────────────────────────────────
    style_ax(ax_elev, "Elevation Profile", "Distance (km)", "Elevation (m)")
    if "elevation" in df.columns and df["elevation"].notna().any():
        elev = df["elevation"].interpolate()
        x    = df["cum_dist_km"].values
        ax_elev.fill_between(x, elev, elev.min()-5, color=gps_col, alpha=0.18)
        ax_elev.plot(x, elev, color=gps_col, linewidth=1.5)
        if not net.empty:
            for _, row in net.iterrows():
                ax_elev.axvline(row["cum_dist_km"], color=net_col,
                                linewidth=0.6, alpha=0.35)
    else:
        ax_elev.text(0.5, 0.5, "No elevation data", transform=ax_elev.transAxes,
                     ha="center", va="center", color=MUTED)

    # ── Speed ──────────────────────────────────────────────────────────────
    style_ax(ax_speed, "Speed over Time", "Elapsed (min)", "Speed (km/h)")
    if not gps.empty:
        ax_speed.plot(gps["elapsed_min"], gps["speed_kmh"],
                      color=gps_col, linewidth=1.3, label="GPS", alpha=0.9)
        ax_speed.fill_between(gps["elapsed_min"], gps["speed_kmh"],
                              color=gps_col, alpha=0.14)
    if not net.empty:
        ax_speed.scatter(net["elapsed_min"], net["speed_kmh"],
                         color=net_col, s=18, label="Network", zorder=5)
    ax_speed.legend(fontsize=7, facecolor=MID, edgecolor=MID, labelcolor=TEXT)

    # ── Accuracy / Satellites ──────────────────────────────────────────────
    style_ax(ax_acc, "GPS Quality", "Elapsed (min)", "Accuracy (m)")
    if "accuracy" in df.columns:
        if not gps.empty:
            ax_acc.plot(gps["elapsed_min"], gps["accuracy"],
                        color=gps_col, linewidth=1.1, label="Acc GPS")
        if not net.empty:
            ax_acc.scatter(net["elapsed_min"], net["accuracy"],
                           color=net_col, s=18, label="Acc Net", zorder=5)
    h2, l2 = [], []
    if "satellites" in df.columns and not gps.empty:
        ax2 = ax_acc.twinx()
        ax2.plot(gps["elapsed_min"], gps["satellites"],
                 color=GREEN, linewidth=1.0, linestyle="--", label="Sats")
        ax2.set_ylabel("Satellites", color=GREEN, fontsize=8)
        ax2.tick_params(axis="y", colors=GREEN, labelsize=7)
        ax2.set_facecolor("none")
        for sp in ax2.spines.values():
            sp.set_edgecolor(MID)
        h2, l2 = ax2.get_legend_handles_labels()
    h1, l1 = ax_acc.get_legend_handles_labels()
    ax_acc.legend(h1+h2, l1+l2, fontsize=7, facecolor=MID, edgecolor=MID, labelcolor=TEXT)

    # ── Stats ──────────────────────────────────────────────────────────────
    ax_stats.set_facecolor(PANEL)
    for sp in ax_stats.spines.values():
        sp.set_edgecolor(MID)
    ax_stats.set_xticks([]); ax_stats.set_yticks([])
    ax_stats.set_title("Summary", color=TEXT, fontsize=9, fontweight="bold", pad=5)

    rows = [
        ("Start",           s["start"]),
        ("Duration",        s["duration"]),
        ("GPS / Net pts",   f"{s['gps_points']} / {s['net_points']}"),
        ("Distance",        fv(s["distance_km"],  "{:.2f} km")),
        ("Avg spd (GPS)",   fv(s["avg_speed"],    "{:.1f} km/h")),
        ("Max speed",       fv(s["max_speed"],    "{:.1f} km/h")),
        ("Elev range",      f"{fv(s['elev_min'],'{:.0f}')} – {fv(s['elev_max'],'{:.0f}')} m"),
        ("Elev ↑ / ↓",     f"+{fv(s['elev_gain_m'],'{:.0f}')} / −{fv(s['elev_loss_m'],'{:.0f}')} m"),
        ("Avg acc GPS",     fv(s["avg_acc_gps"],  "{:.1f} m")),
        ("Avg acc Net",     fv(s["avg_acc_net"],  "{:.1f} m")),
        ("Avg satellites",  fv(s["avg_sats"],     "{:.1f}")),
        ("Avg HDOP",        fv(s["avg_hdop"],     "{:.2f}")),
    ]
    for i, (lbl, val) in enumerate(rows):
        y = 0.97 - i * 0.080
        ax_stats.text(0.03, y, lbl+":", transform=ax_stats.transAxes,
                      color=MUTED, fontsize=7.5, va="top")
        ax_stats.text(0.97, y, val, transform=ax_stats.transAxes,
                      color=TEXT, fontsize=7.5, va="top", ha="right", fontweight="bold")

    return fig


# ── Multi-device comparison figure ────────────────────────────────────────

def make_comparison_figure(datasets):
    """datasets: list of (df, stats_dict)"""
    n        = len(datasets)
    palettes = [DEVICE_PALETTES[i % len(DEVICE_PALETTES)] for i in range(n)]
    labels   = [s["label"] for _, s in datasets]

    fig = plt.figure(figsize=(18, 13), facecolor=DARK)
    fig.suptitle("Device Comparison — " + " vs ".join(labels),
                 color=TEXT, fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           left=0.05, right=0.97, top=0.94, bottom=0.06,
                           hspace=0.52, wspace=0.44)
    ax_map   = fig.add_subplot(gs[:2, :2])
    ax_elev  = fig.add_subplot(gs[:2, 2:])
    ax_speed = fig.add_subplot(gs[2, 0])
    ax_acc   = fig.add_subplot(gs[2, 1])
    ax_sats  = fig.add_subplot(gs[2, 2])
    ax_stats = fig.add_subplot(gs[2, 3])

    # ── Overlay map ────────────────────────────────────────────────────────
    ax_map.set_facecolor(PANEL)
    style_ax(ax_map, "Track Overlay  (solid = GPS · ◆ = Network)", "Longitude", "Latitude")

    all_lats, all_lons = [], []
    legend_patches = []

    for i, (df, s) in enumerate(datasets):
        gps_col, net_col = palettes[i]
        gps = df[df["provider"] == "gps"]
        net = df[df["provider"] == "network"]

        if len(gps) >= 2:
            pts  = np.array([gps["lon"].values, gps["lat"].values]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc   = LineCollection(segs, linewidth=2.0, alpha=0.82, color=gps_col)
            ax_map.add_collection(lc)
        elif len(gps) == 1:
            ax_map.scatter(gps["lon"], gps["lat"], s=30, color=gps_col, zorder=5)

        if not net.empty:
            ax_map.scatter(net["lon"], net["lat"], s=22, color=net_col,
                           marker="D", alpha=0.75, zorder=5)

        ax_map.scatter([df["lon"].iloc[0]],  [df["lat"].iloc[0]],
                       s=90, color=gps_col, marker="o", zorder=7)
        ax_map.scatter([df["lon"].iloc[-1]], [df["lat"].iloc[-1]],
                       s=90, color=gps_col, marker="s", zorder=7)

        all_lats.extend(df["lat"].tolist())
        all_lons.extend(df["lon"].tolist())
        legend_patches.append(mpatches.Patch(color=gps_col, label=f"{s['label']} GPS"))
        if not net.empty:
            legend_patches.append(mpatches.Patch(color=net_col, label=f"{s['label']} Network"))

    ax_map.legend(handles=legend_patches, fontsize=8, facecolor=MID,
                  edgecolor=MID, labelcolor=TEXT, loc="upper left")
    all_lats = np.array(all_lats); all_lons = np.array(all_lons)
    pad_lat = max((all_lats.max()-all_lats.min())*0.12, 0.001)
    pad_lon = max((all_lons.max()-all_lons.min())*0.12, 0.001)
    ax_map.set_xlim(all_lons.min()-pad_lon, all_lons.max()+pad_lon)
    ax_map.set_ylim(all_lats.min()-pad_lat, all_lats.max()+pad_lat)

    # ── Elevation overlay ──────────────────────────────────────────────────
    style_ax(ax_elev, "Elevation Profile", "Distance (km)", "Elevation (m)")
    has_elev = False
    for i, (df, s) in enumerate(datasets):
        gps_col, _ = palettes[i]
        if "elevation" in df.columns and df["elevation"].notna().any():
            elev = df["elevation"].interpolate()
            ax_elev.plot(df["cum_dist_km"], elev, color=gps_col,
                         linewidth=1.5, label=s["label"], alpha=0.85)
            has_elev = True
    if has_elev:
        ax_elev.legend(fontsize=8, facecolor=MID, edgecolor=MID, labelcolor=TEXT)
    else:
        ax_elev.text(0.5, 0.5, "No elevation data", transform=ax_elev.transAxes,
                     ha="center", va="center", color=MUTED)

    # ── Speed ──────────────────────────────────────────────────────────────
    style_ax(ax_speed, "Speed (GPS only)", "Elapsed (min)", "km/h")
    for i, (df, s) in enumerate(datasets):
        gps_col, _ = palettes[i]
        gps = df[df["provider"] == "gps"]
        if not gps.empty:
            ax_speed.plot(gps["elapsed_min"], gps["speed_kmh"],
                          color=gps_col, linewidth=1.2, label=s["label"], alpha=0.85)
    ax_speed.legend(fontsize=7, facecolor=MID, edgecolor=MID, labelcolor=TEXT)

    # ── Accuracy ───────────────────────────────────────────────────────────
    style_ax(ax_acc, "Fix Accuracy", "Elapsed (min)", "Accuracy (m)")
    for i, (df, s) in enumerate(datasets):
        gps_col, net_col = palettes[i]
        gps = df[df["provider"] == "gps"]
        net = df[df["provider"] == "network"]
        if "accuracy" in df.columns:
            if not gps.empty:
                ax_acc.plot(gps["elapsed_min"], gps["accuracy"],
                            color=gps_col, linewidth=1.1, label=f"{s['label']} GPS")
            if not net.empty:
                ax_acc.scatter(net["elapsed_min"], net["accuracy"],
                               color=net_col, s=14, label=f"{s['label']} Net", zorder=5)
    ax_acc.legend(fontsize=6.5, facecolor=MID, edgecolor=MID, labelcolor=TEXT)

    # ── Satellites ─────────────────────────────────────────────────────────
    style_ax(ax_sats, "Satellite Count", "Elapsed (min)", "# Sats")
    for i, (df, s) in enumerate(datasets):
        gps_col, _ = palettes[i]
        gps = df[df["provider"] == "gps"]
        if "satellites" in df.columns and not gps.empty:
            ax_sats.plot(gps["elapsed_min"], gps["satellites"],
                         color=gps_col, linewidth=1.1, label=s["label"])
    ax_sats.legend(fontsize=7, facecolor=MID, edgecolor=MID, labelcolor=TEXT)

    # ── Stats table ────────────────────────────────────────────────────────
    ax_stats.set_facecolor(PANEL)
    for sp in ax_stats.spines.values():
        sp.set_edgecolor(MID)
    ax_stats.set_xticks([]); ax_stats.set_yticks([])
    ax_stats.set_title("Side-by-Side", color=TEXT, fontsize=9, fontweight="bold", pad=5)

    stat_rows = [
        ("Duration",      [s["duration"]                           for _,s in datasets]),
        ("GPS pts",       [str(s["gps_points"])                    for _,s in datasets]),
        ("Net pts",       [str(s["net_points"])                    for _,s in datasets]),
        ("Distance",      [fv(s["distance_km"],  "{:.2f} km")      for _,s in datasets]),
        ("Avg spd GPS",   [fv(s["avg_speed"],    "{:.1f} km/h")    for _,s in datasets]),
        ("Max speed",     [fv(s["max_speed"],    "{:.1f} km/h")    for _,s in datasets]),
        ("Avg acc GPS",   [fv(s["avg_acc_gps"],  "{:.1f} m")       for _,s in datasets]),
        ("Avg acc Net",   [fv(s["avg_acc_net"],  "{:.1f} m")       for _,s in datasets]),
        ("Avg sats",      [fv(s["avg_sats"],     "{:.1f}")         for _,s in datasets]),
        ("Avg HDOP",      [fv(s["avg_hdop"],     "{:.2f}")         for _,s in datasets]),
    ]

    col_w = 1.0 / (n + 1)
    # header
    ax_stats.text(0.02, 0.97, "Metric", transform=ax_stats.transAxes,
                  color=MUTED, fontsize=7, va="top", fontweight="bold")
    for j, (_, s) in enumerate(datasets):
        gps_col = palettes[j][0]
        ax_stats.text(0.02 + (j+1)*col_w, 0.97, s["label"],
                      transform=ax_stats.transAxes,
                      color=gps_col, fontsize=7, va="top", fontweight="bold", ha="center")

    for row_i, (metric, vals) in enumerate(stat_rows):
        y = 0.89 - row_i * 0.088
        ax_stats.text(0.02, y, metric, transform=ax_stats.transAxes,
                      color=MUTED, fontsize=6.8, va="top")
        for j, val in enumerate(vals):
            gps_col = palettes[j][0]
            ax_stats.text(0.02 + (j+1)*col_w, y, val,
                          transform=ax_stats.transAxes,
                          color=TEXT, fontsize=6.8, va="top", ha="center", fontweight="bold")

    return fig


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze & compare GPSLogger CSV files."
    )
    parser.add_argument("files", nargs="+", metavar="CSV")
    parser.add_argument("--labels", "-l", nargs="*", default=None,
                        help="Device labels in file order, e.g. 'Pixel 7' 'Note 10 Pro'")
    parser.add_argument("--no-individual", action="store_true",
                        help="Skip per-device plots; only produce comparison")
    parser.add_argument("--out", "-o", default=None,
                        help="Output directory for PNGs (default: same dir as first input)")
    parser.add_argument("--show", action="store_true",
                        help="Open interactive plot windows")
    args = parser.parse_args()

    paths  = [Path(f) for f in args.files]
    labels = list(args.labels or [])
    while len(labels) < len(paths):
        labels.append(paths[len(labels)].stem)

    datasets = []
    for p, lbl in zip(paths, labels):
        if not p.exists():
            print(f"[SKIP] Not found: {p}", file=sys.stderr)
            continue
        print(f"Loading {p.name}  →  {lbl}")
        df = load_csv(p)
        s  = compute_stats(df, lbl)
        datasets.append((df, s))
        prov = df["provider"].value_counts().to_dict()
        print(f"  {s['points']} pts {prov}  |  {s['distance_km']:.2f} km  |  "
              f"{s['duration']}  |  GPS acc avg {fv(s['avg_acc_gps'], '{:.1f}')} m")

    if not datasets:
        print("No valid files loaded.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out) if args.out else paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Individual plots
    if not args.no_individual:
        for i, (df, s) in enumerate(datasets):
            fig = make_single_figure(df, s, palette_idx=i)
            out = out_dir / f"{s['label'].replace(' ','_')}_analysis.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
            print(f"  Saved → {out}")
            if args.show:
                plt.show()
            plt.close(fig)

    # Comparison (auto-enabled for ≥2 devices)
    if len(datasets) >= 2:
        fig = make_comparison_figure(datasets)
        name = "_vs_".join(s["label"].replace(" ", "_") for _, s in datasets)
        out  = out_dir / f"{name}_comparison.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
        print(f"  Saved → {out}")
        if args.show:
            plt.show()
        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
