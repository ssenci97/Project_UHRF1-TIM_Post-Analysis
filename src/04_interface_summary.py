#!/usr/bin/env python3
"""Summarize interface interaction frequencies and generate plots."""

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
DEFAULTS = {"input": None, "output": None, "seed": SEED, "occ_cutoff": 0.5, "burial_cutoff": 0.25, "top_residues": 15, "top_pairs": 12}
TYPES = ("hbond", "salt", "hydro")
TYPE_LABEL = {"hbond": "H-bond", "salt": "Salt bridge", "hydro": "Hydrophobic"}
TYPE_COLOR = {"hbond": "#1f5fa8", "salt": "#c0392b", "hydro": "#d4a017"}
TYPE_MARKER = {"hbond": "o", "salt": "^", "hydro": "s"}
DPI = 300

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

def number(row, key, default=0.0):
    try:
        value = float(row[key])
        return value if np.isfinite(value) else default
    except (ValueError, TypeError, KeyError):
        return default

def write_tsv(path, columns, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        writer.writerows(rows)

def residue_label(key):
    return f"{key[0]}{key[1]} {key[2]}"

def pair_label(key):
    return f"{key[0]}{key[1]} {key[2]} – {key[3]}{key[4]} {key[5]}"

def type_handles(extra_any=False):
    handles = [Line2D([], [], marker=TYPE_MARKER[k], color=TYPE_COLOR[k], ls="", markersize=5, label=TYPE_LABEL[k]) for k in TYPES]
    if extra_any:
        handles.append(Patch(facecolor="0.88", label="Any interaction"))
    return handles

def finish_figure(fig, path):
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[PYTHON-INFO] wrote {path}")

###################################### residue occupancy
def draw_residue_heatmap(ax, records, chain, top, burial_cutoff):
    selected = [r for r in records if r["key"][0] == chain][:top]
    if not selected:
        ax.text(0.5, 0.5, f"No interactions for chain {chain}", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_axis_off()
        return selected
    matrix = np.array([[r["occ"][k] for k in TYPES] for r in selected])
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.0%}" if v else "·", ha="center", va="center", fontsize=6, color="white" if v > 0.6 else "0.15")
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels([residue_label(r["key"]) for r in selected], fontsize=6)
    ax.set_xticks(range(len(TYPES)))
    ax.set_xticklabels([TYPE_LABEL[k] for k in TYPES], fontsize=6)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"Chain {chain}", fontsize=8, loc="left", pad=19)
    ax.set_xlabel("Interaction occupancy", fontsize=6)
    ax.xaxis.set_label_position("top")
    return selected

def plot_residues(records, top, out, n_models, burial_cutoff):
    chains = sorted({r["key"][0] for r in records})
    if not chains:
        chains = ["none"]
    for chain in chains:
        fig, ax = plt.subplots(figsize=(4.5, max(2.5, min(top, sum(r["key"][0] == chain for r in records)) * 0.24 + 1.3)))
        fig.subplots_adjust(left=0.32, right=0.98, top=0.82, bottom=0.12)
        if chain == "none":
            ax.text(0.5, 0.5, "No residue interactions", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            draw_residue_heatmap(ax, records, chain, top, burial_cutoff)
        fig.suptitle(f"Residue occupancy — chain {chain} ({n_models} models)", fontsize=8, x=0.02, ha="left")
        finish_figure(fig, out / f"{out.parent.name}_A_residue_occupancy_chain_{chain}.svg")
    return chains

###################################### pair occupancy
def draw_pairs(ax, records, top):
    selected = records[:top]
    if not selected:
        ax.text(0.5, 0.5, "No residue pairs detected", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        return
    ys = np.arange(len(selected))[::-1]
    offsets = {"hbond": 0.19, "salt": 0, "hydro": -0.19}
    for r, y in zip(selected, ys):
        ax.barh(y, r["any"], height=0.62, color="0.88", zorder=1)
        for k in TYPES:
            if r["occ"][k] > 0:
                ax.barh(y + offsets[k], r["occ"][k], height=0.15, color=TYPE_COLOR[k], zorder=2)
        ax.text(min(r["any"] + 0.015, 1.04), y, f"{r['any']:.0%}", va="center", fontsize=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([pair_label(r["key"]) for r in selected], fontsize=5)
    for tick, r in zip(ax.get_yticklabels(), selected):
        if r["stable"]:
            tick.set_fontweight("bold")
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Occupancy (fraction of models)", fontsize=6)
    ax.tick_params(labelsize=5)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.legend(handles=type_handles(True), fontsize=5, ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01))

def plot_pairs(records, top, path, n_models):
    fig, ax = plt.subplots(figsize=(5, max(2.5, min(top, len(records)) * 0.28 + 1.5)))
    fig.subplots_adjust(left=0.43, right=0.95, top=0.82, bottom=0.12)
    draw_pairs(ax, records, top)
    fig.suptitle(f"B. Residue pairs ({n_models} models)", fontsize=7, x=0.02, ha="left")
    finish_figure(fig, path)

###################################### burial versus occupancy
def draw_scatter(ax, records, n_models, params):
    points = [r for r in records if np.isfinite(r["burial"])]
    if not points:
        ax.text(0.5, 0.5, "No burial data", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        return
    oc, bc = params["occ_cutoff"], params["burial_cutoff"]
    ymax = max(0.5, max(r["burial"] for r in points)) + 0.08
    ax.axvspan(oc, 1.3, color="#2e7d32", alpha=0.05)
    ax.axvline(oc, ls="--", color="0.4", lw=0.7)
    ax.axhline(bc, ls="--", color="0.4", lw=0.7)
    rng = np.random.default_rng(params["seed"])
    for r in points:
        ax.scatter(r["any"] + rng.uniform(-0.008, 0.008), r["burial"] + rng.uniform(-0.003, 0.003), s=18 + 100 * r["n"] / max(1, n_models), marker=TYPE_MARKER.get(r["dominant"], "x"), color=TYPE_COLOR.get(r["dominant"], "0.5"), edgecolor="white", linewidth=0.3, alpha=0.9)
    stable = sorted((r for r in points if r["any"] >= oc and r["burial"] >= bc), key=lambda r: (-r["any"], -r["burial"]))
    for r in stable[:8]:
        ax.annotate(residue_label(r["key"]), (r["any"], r["burial"]), xytext=(4, 2), textcoords="offset points", fontsize=5)
    ax.set_xlim(-0.05, 1.3)
    ax.set_ylim(0, ymax)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Occupancy of any interaction", fontsize=6)
    ax.set_ylabel("Mean relative dSASA (burial)", fontsize=6)
    ax.tick_params(labelsize=5)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=type_handles(), fontsize=5, frameon=False, loc="upper left")
    frequencies = sorted({r["n"] for r in points})
    legend_values = sorted(set([frequencies[0], frequencies[len(frequencies) // 2], frequencies[-1]]))
    size_handles = [plt.scatter([], [], s=18 + 100 * n / max(1, n_models), color="0.55", edgecolor="white", linewidth=0.3, label=f"{n} models ({n / n_models:.0%})") for n in legend_values]
    ax.legend(handles=size_handles, title="Frequency", title_fontsize=5, fontsize=5, frameon=False, loc="lower right")
    ax.add_artist(ax.legend(handles=type_handles(), fontsize=5, frameon=False, loc="upper left"))

def plot_scatter(records, path, n_models, params):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    fig.subplots_adjust(left=0.17, right=0.96, top=0.85, bottom=0.14)
    draw_scatter(ax, records, n_models, params)
    fig.suptitle(f"C. Burial vs occupancy ({n_models} models)", fontsize=7, x=0.02, ha="left")
    finish_figure(fig, path)

###################################### merged figure
def plot_merged(records, pairs, chains, out, n_models, params):
    chain_data = {chain: [r for r in records if r["key"][0] == chain][:params["top_residues"]] for chain in chains if chain != "none"}
    n_chains = max(1, len(chain_data))
    heatmap_height = max(2.0, min(params["top_residues"], max((len(v) for v in chain_data.values()), default=1)) * 0.24 + 0.9)
    fig = plt.figure(figsize=(13, max(7, n_chains * heatmap_height + 5)))
    grid = fig.add_gridspec(n_chains + 2, 2, height_ratios=[heatmap_height] * n_chains + [0.7, 4], width_ratios=[1.05, 1.2], hspace=0.65, wspace=0.55)
    for i, chain in enumerate(chain_data):
        ax = fig.add_subplot(grid[i, :])
        draw_residue_heatmap(ax, records, chain, params["top_residues"], params["burial_cutoff"])
    pair_ax = fig.add_subplot(grid[n_chains:, 0])
    draw_pairs(pair_ax, pairs, params["top_pairs"])
    scatter_ax = fig.add_subplot(grid[n_chains:, 1])
    draw_scatter(scatter_ax, records, n_models, params)
    fig.suptitle(f"Interface summary — {out.parent.name} ({n_models} models)", fontsize=11, y=0.995)
    finish_figure(fig, out / f"{out.parent.name}_D_merged_interface_summary.svg")

###################################### aggregation
def aggregate(complex_rows, residue_rows, pair_rows, params):
    model_source = {r["structure"]: r["source"] for r in complex_rows}
    models = set(model_source)
    n_models = len(models)
    if not n_models:
        raise SystemExit("[PYTHON-INFO] no complex measurements found")
    sources = sorted(set(model_source.values()))
    residue_burial = defaultdict(list)
    for r in residue_rows:
        if r["structure"] in models:
            value = number(r, "rel_dSASA", float("nan"))
            if np.isfinite(value):
                residue_burial[(r["chain"], r["resid"], r["resname"])].append(value)

    def burial(key):
        values = residue_burial.get(key, [])
        return float(np.mean(values)) if values else float("nan")

    residue_seen = defaultdict(lambda: defaultdict(set))
    pair_seen = defaultdict(lambda: defaultdict(set))
    pair_scores = defaultdict(list)
    for r in tqdm(pair_rows, desc="Aggregating interactions", unit="row"):
        name = r["structure"]
        if name not in models:
            continue
        ka = (r["chain_a"], r["resid_a"], r["resname_a"])
        kb = (r["chain_b"], r["resid_b"], r["resname_b"])
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

    residues = []
    for key, seen in residue_seen.items():
        occ = {k: len(seen[k]) / n_models for k in TYPES}
        dominant = max(TYPES, key=lambda k: occ[k])
        any_occ = len(seen["any"]) / n_models
        b = burial(key)
        residues.append({"key": key, "occ": occ, "any": any_occ, "n": len(seen["any"]), "burial": b, "dominant": dominant if max(occ.values()) else "none", "by_src": {s: len(seen["any"] & {m for m in models if model_source[m] == s}) / max(1, sum(v == s for v in model_source.values())) for s in sources}})
    residues.sort(key=lambda r: (-r["any"], -(r["burial"] if np.isfinite(r["burial"]) else -1)))

    pairs = []
    for key, seen in pair_seen.items():
        occ = {k: len(seen[k]) / n_models for k in TYPES}
        dominant = max(TYPES, key=lambda k: occ[k])
        bs = [burial(key[:3]), burial(key[3:])]
        bs = [b for b in bs if np.isfinite(b)]
        b = float(np.mean(bs)) if bs else float("nan")
        any_occ = len(seen["any"]) / n_models
        score = float(np.mean(pair_scores[key]))
        pairs.append({"key": key, "occ": occ, "any": any_occ, "n": len(seen["any"]), "score": score, "burial": b, "dominant": dominant if max(occ.values()) else "none", "stable": bool(any_occ >= params["occ_cutoff"] and np.isfinite(b) and b >= params["burial_cutoff"])})
    pairs.sort(key=lambda r: (-r["any"], -r["score"]))
    return residues, pairs, sources, n_models

###################################### main
def main():
    start = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--seed", type=int)
    p.add_argument("--occ-cutoff", type=float)
    p.add_argument("--burial-cutoff", type=float)
    p.add_argument("--top-residues", type=int)
    p.add_argument("--top-pairs", type=int)
    p.add_argument("--yaml-in")
    p.add_argument("--yaml-out", default="configs/param_configs.yaml")
    args = p.parse_args()

    params = {**DEFAULTS}
    if args.yaml_in:
        with open(args.yaml_in) as handle:
            params.update((yaml.safe_load(handle) or {}).get(NAME, {}))
    for key in ("input", "output", "seed", "occ_cutoff", "burial_cutoff", "top_residues", "top_pairs"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    if not params["input"]:
        p.error("--input is required unless supplied through --yaml-in")

    np.random.seed(params["seed"])
    folder = Path(params["input"]).resolve()
    name = folder.name
    data_folder = folder / "interfaces" if (folder / "interfaces").is_dir() else folder
    out = Path(params["output"]).resolve() if params["output"] else folder / "interface_summary"
    out.mkdir(parents=True, exist_ok=True)
    complex_path = data_folder / f"{name}_complex_scores.tsv"
    residue_path = data_folder / f"{name}_residue_scores.tsv"
    pair_path = data_folder / f"{name}_pymol_pairs.tsv"
    for path in (complex_path, residue_path, pair_path):
        if not path.is_file():
            raise SystemExit(f"[PYTHON-INFO] missing input TSV: {path}")

    print(f"[PYTHON-INFO] group={name} input={data_folder} output={out}")
    complexes, residues_raw, pairs_raw = read_tsv(complex_path), read_tsv(residue_path), read_tsv(pair_path)
    residues, pairs, sources, n_models = aggregate(complexes, residues_raw, pairs_raw, params)

    residue_cols = ["chain", "resid", "resname", "n_models", "occ_hbond", "occ_salt", "occ_hydro", "occ_any", "mean_rel_dsasa", "dominant"] + [f"occ_any_{s}" for s in sources] + ["of_models"]
    residue_out = [[*r["key"], r["n"], *[round(r["occ"][k], 4) for k in TYPES], round(r["any"], 4), "" if not np.isfinite(r["burial"]) else round(r["burial"], 4), r["dominant"], *[round(r["by_src"][s], 4) for s in sources], n_models] for r in residues]
    pair_cols = ["chain_a", "resid_a", "resname_a", "chain_b", "resid_b", "resname_b", "n_models", "occ_hbond", "occ_salt", "occ_hydro", "occ_any", "mean_score", "mean_rel_dsasa", "dominant", "stabilizing", "of_models"]
    pair_out = [[*r["key"], r["n"], *[round(r["occ"][k], 4) for k in TYPES], round(r["any"], 4), round(r["score"], 3), "" if not np.isfinite(r["burial"]) else round(r["burial"], 4), r["dominant"], int(r["stable"]), n_models] for r in pairs]

    write_tsv(out / f"{name}_residue_frequencies.tsv", residue_cols, residue_out)
    write_tsv(out / f"{name}_pair_frequencies.tsv", pair_cols, pair_out)
    chains = plot_residues(residues, params["top_residues"], out, n_models, params["burial_cutoff"])
    plot_pairs(pairs, params["top_pairs"], out / f"{name}_B_pair_occupancy.svg", n_models)
    plot_scatter(residues, out / f"{name}_C_burial_vs_occupancy.svg", n_models, params)
    plot_merged(residues, pairs, chains, out, n_models, params)

    print(f"[PYTHON-INFO] models={n_models} residues={len(residues)} pairs={len(pairs)} stabilizing_pairs={sum(r['stable'] for r in pairs)}")
    save_config(args.yaml_out, params)
    print(f"[PYTHON-INFO] {datetime.datetime.now():%Y-%m-%d %H:%M:%S} exec={time.time()-start:.2f}s maxmem={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}MB")

if __name__ == "__main__":
    main()

