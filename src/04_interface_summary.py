#!/usr/bin/env python3
"""Summarize interface occupancy and merge five 4x4 inch plots into one panel.

Args:
    --input: folder with the interface TSV outputs.
"""
###################################### dependencies
import argparse
import csv
import datetime
import resource
import time
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import yaml
from tqdm import tqdm
###################################### constants
NAME = "interface_summary"
SEED = 42
DEFAULTS = {"input": None, "output": None, "seed": SEED, "occ_cutoff": 0.5, "burial_cutoff": 0.25, "contact_min_models": 1, "min_contact_models": 2, "top_residues": 10, "top_pairs": 10}
TYPES = ("hbond", "salt", "hydro")
TYPE_LABEL = {"hbond": "H-bond", "salt": "Salt bridge", "hydro": "Hydrophobic"}
TYPE_COLOR = {"hbond": "#276FBF", "salt": "#C44536", "hydro": "#D49A20"}
TYPE_MARKER = {"hbond": "o", "salt": "^", "hydro": "s"}
SOURCE_COLORS = ["#276FBF", "#C44536", "#D49A20", "#2E7D32", "#7B1FA2", "#00838F"]
PANEL_LAYOUT = (2, 3)
PANEL_LETTERS = "ABCDE"
FIG = (4, 4)
DPI = 300
FONT = 8
SMALL_FONT = 6
###################################### helpers
def save_config(path, params):
    if not path:
        return
    path = Path(path)
    data = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    data[NAME] = params
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
def read_tsv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))
def write_tsv(path, columns, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        writer.writerows(rows)
def number(row, key, default=0.0):
    try:
        value = float(row[key])
        return value if np.isfinite(value) else default
    except (ValueError, TypeError, KeyError):
        return default
def residue_label(key):
    return f"{key[0]}{key[1]} {key[2]}"
def pair_label(key):
    return f"{residue_label(key[:3])} – {residue_label(key[3:])}"
def parse_partner(token):
    resid, resname = token[1:].split(" ", 1)
    return (token[0], int(resid), resname)
def type_handles(extra=()):
    return [Line2D([], [], marker=TYPE_MARKER[k], color=TYPE_COLOR[k], ls="", markersize=5, label=TYPE_LABEL[k]) for k in TYPES] + list(extra)
def style_axis(ax):
    ax.tick_params(axis="both", labelsize=SMALL_FONT, length=2, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
def new_figure():
    return plt.subplots(figsize=FIG, layout="constrained")
def finish_figure(fig, path, rasters):
    fig.savefig(path, format="svg")
    fig.set_dpi(DPI)
    fig.canvas.draw()
    rasters.append(np.asarray(fig.canvas.buffer_rgba()).copy())
    plt.close(fig)
    print(f"[PYTHON-INFO] wrote {path}")
###################################### aggregation
def aggregate(complex_rows, residue_rows, pair_rows, params):
    model_source = {r["structure"]: r["source"] for r in complex_rows}
    models = set(model_source)
    n_models = len(models)
    if not n_models:
        raise SystemExit("[PYTHON-INFO] no complex measurements found")
    sources = sorted(set(model_source.values()))
    source_models = {s: {m for m, v in model_source.items() if v == s} for s in sources}
    burial_values = defaultdict(list)
    residue_seen = defaultdict(lambda: defaultdict(set))
    contact_models = defaultdict(set)
    pair_seen = defaultdict(lambda: defaultdict(set))
    pair_scores = defaultdict(list)
    def burial(key):
        values = burial_values.get(key, [])
        return float(np.mean(values)) if values else float("nan")
    for r in tqdm(residue_rows, desc="Aggregating residues", unit="row"):
        name = r["structure"]
        if name not in models:
            continue
        key = (r["chain"], int(r["resid"]), r["resname"])
        value = number(r, "rel_dSASA", float("nan"))
        if np.isfinite(value):
            burial_values[key].append(value)
        residue_seen[key]["contact"].add(name)
        if r["chain"] == "A":
            for token in filter(None, r["partners"].split(";")):
                contact_models[key + parse_partner(token)].add(name)
    for r in tqdm(pair_rows, desc="Aggregating pairs", unit="row"):
        name = r["structure"]
        if name not in models:
            continue
        ka = (r["chain_a"], int(r["resid_a"]), r["resname_a"])
        kb = (r["chain_b"], int(r["resid_b"]), r["resname_b"])
        kp = ka + kb
        counts = {"hbond": number(r, "n_hbond"), "salt": number(r, "n_salt_bridge"), "hydro": number(r, "n_hydrophobic_contacts")}
        for kind, count in counts.items():
            if count:
                residue_seen[ka][kind].add(name)
                residue_seen[kb][kind].add(name)
                pair_seen[kp][kind].add(name)
        residue_seen[ka]["any"].add(name)
        residue_seen[kb]["any"].add(name)
        pair_seen[kp]["any"].add(name)
        pair_scores[kp].append(number(r, "interaction_score"))
    min_contact = params["min_contact_models"]
    residues = []
    for key, seen in residue_seen.items():
        if len(seen["contact"]) < min_contact:
            continue
        occ = {k: len(seen[k]) / n_models for k in TYPES}
        residues.append({"key": key, "occ": occ, "any": len(seen["any"]) / n_models, "contact": len(seen["contact"]) / n_models, "n": len(seen["any"]), "burial": burial(key), "dominant": max(TYPES, key=occ.get) if any(occ.values()) else "none", "by_src": {s: len(seen["any"] & source_models[s]) / len(source_models[s]) for s in sources}})
    residues.sort(key=lambda r: (-r["contact"], -r["any"], -(r["burial"] if np.isfinite(r["burial"]) else -1)))
    pairs = []
    for key in [k for k, v in contact_models.items() if len(v) >= min_contact]:
        seen = pair_seen[key]
        occ = {k: len(seen[k]) / n_models for k in TYPES}
        bs = [b for b in (burial(key[:3]), burial(key[3:])) if np.isfinite(b)]
        b = float(np.mean(bs)) if bs else float("nan")
        any_occ = len(seen["any"]) / n_models
        pairs.append({"key": key, "occ": occ, "any": any_occ, "contact": len(contact_models[key]) / n_models, "n": len(seen["any"]), "score": float(np.mean(pair_scores[key])) if pair_scores.get(key) else 0.0, "burial": b, "dominant": max(TYPES, key=occ.get) if any(occ.values()) else "none", "stable": bool(any_occ >= params["occ_cutoff"] and np.isfinite(b) and b >= params["burial_cutoff"])})
    pairs.sort(key=lambda r: (-r["contact"], -r["any"], -r["score"]))
    return residues, pairs, contact_models, sources, n_models
###################################### plot A binary contact map
def plot_contact_map(contact_models, min_models, rasters, path):
    binary = [k for k, v in contact_models.items() if len(v) >= min_models]
    rows = sorted({k[:3] for k in binary}, key=lambda k: k[1])
    cols = sorted({k[3:] for k in binary}, key=lambda k: k[1])
    ri = {k: i for i, k in enumerate(rows)}
    ci = {k: j for j, k in enumerate(cols)}
    matrix = np.zeros((max(1, len(rows)), max(1, len(cols))))
    for k in binary:
        matrix[ri[k[:3]], ci[k[3:]]] = 1
    fig, ax = new_figure()
    ax.imshow(matrix, cmap="Greys", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    finish_figure(fig, path, rasters)
###################################### plot B complex stability
def plot_complex(complexes, sources, params, rasters, path):
    fig, ax = new_figure()
    colors = {s: SOURCE_COLORS[i % len(SOURCE_COLORS)] for i, s in enumerate(sources)}
    for s in sources:
        rows = [r for r in complexes if r["source"] == s]
        ax.scatter([number(r, "dASA_A2") for r in rows], [number(r, "frac_dASA") for r in rows], s=10, color=colors[s], edgecolor="white", linewidth=0.3, alpha=0.9, label=s)
    ax.axhline(params["burial_cutoff"], ls="--", color="0.4", lw=0.6)
    ax.set_xlabel("dASA (Å²)", fontsize=FONT, labelpad=3)
    ax.set_ylabel("Fraction buried", fontsize=FONT, labelpad=3)
    ax.set_title("Complex stability", fontsize=FONT + 1, loc="left")
    ax.grid(alpha=0.2, lw=0.5)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=SMALL_FONT, handletextpad=0.3, markerscale=1.2)
    finish_figure(fig, path, rasters)
###################################### plot C burial vs occupancy
def plot_scatter(residues, n_models, params, stats, rasters, path):
    fig, ax = new_figure()
    points = [r for r in residues if np.isfinite(r["burial"])]
    oc, bc = params["occ_cutoff"], params["burial_cutoff"]
    if points:
        ymax = max(0.5, max(r["burial"] for r in points)) + 0.08
        ax.axvspan(oc, 1.3, color="#2e7d32", alpha=0.055, zorder=0)
        ax.axvline(oc, ls="--", color="0.4", lw=0.6)
        ax.axhline(bc, ls="--", color="0.4", lw=0.6)
        rng = np.random.default_rng(params["seed"])
        for r in points:
            ax.scatter(r["any"] + rng.uniform(-0.006, 0.006), r["burial"] + rng.uniform(-0.002, 0.002), s=12 + 60 * r["n"] / max(1, n_models), marker=TYPE_MARKER.get(r["dominant"], "o"), color=TYPE_COLOR.get(r["dominant"], "0.5"), edgecolor="white", linewidth=0.3, alpha=0.9, zorder=3)
        stable = sorted((r for r in points if r["any"] >= oc and r["burial"] >= bc), key=lambda r: (-r["any"], -r["burial"]))
        for r in stable[:5]:
            ax.annotate(residue_label(r["key"]), (r["any"], r["burial"]), xytext=(3, 2), textcoords="offset points", fontsize=SMALL_FONT - 1)
        ax.set_ylim(0, ymax)
    ax.set_xlim(-0.04, 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Occupancy of any interaction", fontsize=FONT, labelpad=3)
    ax.set_ylabel("Mean relative dSASA", fontsize=FONT, labelpad=3)
    ax.set_title(f"Burial vs occupancy · {n_models} models\n{stats['label']}", fontsize=FONT, loc="left")
    style_axis(ax)
    ax.grid(alpha=0.2, lw=0.5, zorder=0)
    frequencies = sorted({r["n"] for r in points}) or [0]
    legend_values = sorted({frequencies[0], frequencies[len(frequencies) // 2], frequencies[-1]})
    size_handles = [Line2D([], [], marker="o", ls="", color="0.55", markersize=np.sqrt(12 + 60 * n / max(1, n_models)), label=f"{n}/{n_models}") for n in legend_values]
    type_legend = ax.legend(handles=type_handles(), fontsize=SMALL_FONT - 1, frameon=False, loc="upper left", borderaxespad=0.2, handletextpad=0.3)
    ax.add_artist(type_legend)
    ax.legend(handles=size_handles, title="Models", title_fontsize=SMALL_FONT, fontsize=SMALL_FONT - 1, frameon=False, loc="lower right", borderaxespad=0.2, labelspacing=0.4)
    finish_figure(fig, path, rasters)
###################################### plot D residue occupancy
def plot_residues(residues, top, n_models, stats, rasters, path):
    blocks = [[r for r in residues if r["key"][0] == chain][:top] for chain in ("A", "B")]
    rows = blocks[0] + [None] + blocks[1]
    labels = [TYPE_LABEL[k] for k in TYPES] + ["Contact"]
    matrix = np.full((len(rows), len(labels)), np.nan)
    for i, r in enumerate(rows):
        if r is not None:
            matrix[i] = [r["occ"][k] for k in TYPES] + [r["contact"]]
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    fig, ax = new_figure()
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for i, j in zip(*np.where(np.isfinite(matrix))):
        v = matrix[i, j]
        ax.text(j, i, f"{v:.0%}" if v else "·", ha="center", va="center", fontsize=SMALL_FONT - 1, color="white" if v > 0.6 else "0.15")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([residue_label(r["key"]) if r else "" for r in rows], fontsize=SMALL_FONT - 1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=SMALL_FONT - 1, rotation=45, ha="left")
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    ax.set_title(f"Residue occupancy · {n_models} models\n{stats['label']}", fontsize=FONT, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    finish_figure(fig, path, rasters)
###################################### plot E pair occupancy
def plot_pairs(pairs, top, n_models, stats, rasters, path):
    selected = pairs[:top]
    ys = np.arange(len(selected))[::-1]
    offsets = {"hbond": 0.2, "salt": 0, "hydro": -0.2}
    fig, ax = new_figure()
    for r, y in zip(selected, ys):
        ax.barh(y, r["any"], height=0.66, color="0.88", zorder=1)
        for k in TYPES:
            if r["occ"][k] > 0:
                ax.barh(y + offsets[k], r["occ"][k], height=0.16, color=TYPE_COLOR[k], zorder=2)
        ax.scatter(r["contact"], y, marker="|", s=30, color="black", linewidths=0.8, zorder=3)
        ax.text(min(r["any"] + 0.02, 1.04), y, f"{r['any']:.0%}", va="center", fontsize=SMALL_FONT - 1)
    ax.set_yticks(ys)
    ax.set_yticklabels([pair_label(r["key"]) for r in selected], fontsize=SMALL_FONT - 1)
    for tick, r in zip(ax.get_yticklabels(), selected):
        if r["stable"]:
            tick.set_fontweight("bold")
    ax.set_xlim(0, 1.13)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Occupancy", fontsize=FONT, labelpad=3)
    ax.set_title(f"Residue-pair occupancy · {n_models} models\n{stats['label']}", fontsize=FONT, loc="left")
    ax.grid(axis="x", alpha=0.22, lw=0.5)
    style_axis(ax)
    extra = [Patch(facecolor="0.88", label="Any"), Line2D([], [], marker="|", color="black", ls="", markersize=6, label="Contact ≤8 Å")]
    ax.legend(handles=type_handles(extra), fontsize=SMALL_FONT - 1, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), handletextpad=0.3, columnspacing=0.6)
    finish_figure(fig, path, rasters)
###################################### merged panel
def plot_panel(rasters, out, name):
    rows, cols = PANEL_LAYOUT
    fig, axes = plt.subplots(rows, cols, figsize=(FIG[0] * cols, FIG[1] * rows), layout="constrained")
    for ax, img, letter in zip(axes.ravel(), rasters, PANEL_LETTERS):
        ax.imshow(img)
        ax.set_axis_off()
        ax.text(0.01, 0.99, letter, transform=ax.transAxes, fontsize=FONT + 3, fontweight="bold", va="top")
    for ax in axes.ravel()[len(rasters):]:
        ax.set_axis_off()
    path = out / f"{name}_interface_panel.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"[PYTHON-INFO] wrote {path}")
###################################### main
def main():
    start = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--seed", type=int)
    p.add_argument("--occ-cutoff", type=float)
    p.add_argument("--burial-cutoff", type=float)
    p.add_argument("--contact-min-models", type=int)
    p.add_argument("--min-contact-models", type=int)
    p.add_argument("--top-residues", type=int)
    p.add_argument("--top-pairs", type=int)
    p.add_argument("--yaml-in")
    p.add_argument("--yaml-out", default="configs/param_configs.yaml")
    args = p.parse_args()
    params = {**DEFAULTS}
    if args.yaml_in:
        with open(args.yaml_in) as handle:
            params.update((yaml.safe_load(handle) or {}).get(NAME, {}))
    for key in ("input", "output", "seed", "occ_cutoff", "burial_cutoff", "contact_min_models", "min_contact_models", "top_residues", "top_pairs"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    if not params["input"]:
        p.error("--input is required unless supplied through --yaml-in")
    plt.rcParams.update({"font.size": FONT, "axes.titlesize": FONT + 1, "axes.labelsize": FONT, "xtick.labelsize": SMALL_FONT, "ytick.labelsize": SMALL_FONT, "legend.fontsize": SMALL_FONT - 1, "svg.fonttype": "none"})
    np.random.seed(params["seed"])
    folder = Path(params["input"]).resolve()
    name = folder.name
    data_folder = folder / "interfaces" if (folder / "interfaces").is_dir() else folder
    out = Path(params["output"]).resolve() if params["output"] else folder / "interface_summary"
    out.mkdir(parents=True, exist_ok=True)
    paths = [data_folder / f"{name}_{s}.tsv" for s in ("complex_scores", "residue_scores", "pymol_pairs")]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"[PYTHON-INFO] missing input TSV: {path}")
    print(f"[PYTHON-INFO] group={name} input={data_folder} output={out}")
    complexes, residues_raw, pairs_raw = (read_tsv(path) for path in paths)
    residues, pairs, contact_models, sources, n_models = aggregate(complexes, residues_raw, pairs_raw, params)
    stats = {"residues": len(residues), "contacts": len(pairs)}
    stats["label"] = f"{stats['residues']} residues · {stats['contacts']} contacts in ≥{params['min_contact_models']} models"
    residue_cols = ["chain", "resid", "resname", "n_models", "occ_hbond", "occ_salt", "occ_hydro", "occ_any", "occ_contact", "mean_rel_dsasa", "dominant"] + [f"occ_any_{s}" for s in sources] + ["of_models"]
    residue_out = [[*r["key"], r["n"], *[round(r["occ"][k], 4) for k in TYPES], round(r["any"], 4), round(r["contact"], 4), "" if not np.isfinite(r["burial"]) else round(r["burial"], 4), r["dominant"], *[round(r["by_src"][s], 4) for s in sources], n_models] for r in residues]
    pair_cols = ["chain_a", "resid_a", "resname_a", "chain_b", "resid_b", "resname_b", "n_models", "occ_hbond", "occ_salt", "occ_hydro", "occ_any", "occ_contact", "mean_score", "mean_rel_dsasa", "dominant", "stabilizing", "of_models"]
    pair_out = [[*r["key"], r["n"], *[round(r["occ"][k], 4) for k in TYPES], round(r["any"], 4), round(r["contact"], 4), round(r["score"], 3), "" if not np.isfinite(r["burial"]) else round(r["burial"], 4), r["dominant"], int(r["stable"]), n_models] for r in pairs]
    write_tsv(out / f"{name}_residue_frequencies.tsv", residue_cols, residue_out)
    write_tsv(out / f"{name}_pair_frequencies.tsv", pair_cols, pair_out)
    rasters = []
    plot_contact_map(contact_models, params["contact_min_models"], rasters, out / f"{name}_A_contact_map.svg")
    plot_complex(complexes, sources, params, rasters, out / f"{name}_B_complex_stability.svg")
    plot_scatter(residues, n_models, params, stats, rasters, out / f"{name}_C_burial_vs_occupancy.svg")
    plot_residues(residues, params["top_residues"], n_models, stats, rasters, out / f"{name}_D_residue_occupancy.svg")
    plot_pairs(pairs, params["top_pairs"], n_models, stats, rasters, out / f"{name}_E_pair_occupancy.svg")
    plot_panel(rasters, out, name)
    print(f"[PYTHON-INFO] models={n_models} residues={len(residues)} pairs={len(pairs)} stabilizing_pairs={sum(r['stable'] for r in pairs)}")
    save_config(args.yaml_out, params)
    print(f"[PYTHON-INFO] {datetime.datetime.now():%Y-%m-%d %H:%M:%S} exec={time.time()-start:.2f}s maxmem={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}MB")
if __name__ == "__main__":
    main()

