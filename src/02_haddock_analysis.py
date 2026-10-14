#!/usr/bin/env python3
"""Match HADDOCK scores to PDB files and plot model scores."""
###################################### dependencies
import argparse
import datetime
import resource
import re
import time
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import yaml
from tqdm import tqdm

###################################### constants
NAME = "haddock_analysis"
SEED = 42
DEFAULTS = {"input": None, "output": None, "capri_tsv": "data/capri_ss_all.tsv", "complex": None, "seed": SEED}
COLOR_RULES = (("abin", "salmon", "teal", "abin"), ("model", "grey", "black", "model"), ("pdbeval", "red", "blue", "pdbeval"))

###################################### helpers
def source_style(stem):
    low = stem.lower()
    for key, a, b, label in COLOR_RULES:
        if key in low:
            return a, b, label
    return "steelblue", "purple", "other"

def normalize(value):
    return re.sub(r"\s+", "", str(value or ""))

def save_config(path, params):
    if not path:
        return
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {} if path.exists() else {}
    data[NAME] = params
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))

def load_rows(path):
    rows = []
    with open(path, encoding="utf-8-sig") as handle:
        header = handle.readline().split()
        required = ("score", "complex", "pdb_path")
        missing = [key for key in required if key not in header]
        if missing:
            raise SystemExit(f"[PYTHON-INFO] missing CAPRI columns {missing}; found {header}")
        ix = {key: header.index(key) for key in required}
        for line in handle:
            fields = line.split()
            if len(fields) <= max(ix.values()):
                continue
            try:
                score = float(fields[ix["score"]])
            except ValueError:
                score = float("nan")
            complex_name, pdb_path = fields[ix["complex"]], fields[ix["pdb_path"]]
            if not complex_name or pdb_path == "-":
                continue
            filename = Path(pdb_path.replace("\\", "/")).name
            rows.append({"complex": complex_name, "complex_norm": normalize(complex_name), "pdb_path": pdb_path, "filename": filename, "stem": Path(filename).stem, "score": score})
    print(f"[PYTHON-INFO] loaded {len(rows)} HADDOCK rows from {path}")
    return rows

def match_scores(files, rows, complex_name):
    target = normalize(complex_name)
    scores = {}
    local = {p.stem: p.name for p in files}
    for row in rows:
        if target and target not in row["complex_norm"] and row["complex_norm"] not in target:
            continue
        stem = row["stem"]
        matches = [name for name in local if stem == name or name.endswith("_" + stem) or stem.endswith("_" + name)]
        if matches and np.isfinite(row["score"]):
            best = max(matches, key=len)
            scores[best] = row["score"]
    print(f"[PYTHON-INFO] matched {len(scores)}/{len(files)} local PDB files")
    return scores

def plot_scores(rows, out_dir, complex_name=None):
    groups = defaultdict(list)
    for row in rows:
        if np.isfinite(row["score"]) and (not complex_name or normalize(row["complex"]) == normalize(complex_name)):
            groups[row["complex"]].append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, group in tqdm(sorted(groups.items()), desc="plotting"):
        group = sorted(group, key=lambda r: r["score"])
        labels, values = [r["stem"] for r in group], [r["score"] for r in group]
        fig, ax = plt.subplots(figsize=(max(4, min(16, len(labels) * 0.15)), 4))
        colors = [source_style(label)[0] for label in labels]
        ax.bar(np.arange(len(labels)), values, color=colors, width=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_ylabel("HADDOCK score (lower is better)")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        legend = {}
        for label in labels:
            color, _, category = source_style(label)
            legend.setdefault(category, color)
        ax.legend(handles=[Patch(color=c, label=k) for k, c in legend.items()], fontsize=6, frameon=False)
        ax.set_title(f"{name}: {len(labels)} models", loc="left", fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}_haddock_scores.svg"
        fig.savefig(path, format="svg")
        plt.close(fig)
        print(f"[PYTHON-INFO] wrote {path}")

###################################### main
def main():
    start = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--capri-tsv")
    p.add_argument("--complex")
    p.add_argument("--yaml-in")
    p.add_argument("--yaml-out", default="configs/param_configs.yaml")
    args = p.parse_args()
    params = {**DEFAULTS}
    if args.yaml_in:
        with open(args.yaml_in) as handle:
            params.update((yaml.safe_load(handle) or {}).get(NAME, {}))
    for key in ("input", "output", "capri_tsv", "complex"):
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)
    if not params["input"]:
        raise SystemExit("[PYTHON-INFO] --input is required")
    np.random.seed(params["seed"])
    folder = Path(params["input"])
    files = sorted(folder.glob("*.pdb"))
    rows = load_rows(params["capri_tsv"])
    name = params["complex"] or folder.resolve().name
    scores = match_scores(files, rows, name)
    out = Path(params["output"] or Path("data/pdbs_by_group") / name / "aligned" / "haddock_scores")
    plot_scores(rows, out, name)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "haddock_matched_scores.tsv", "w") as handle:
        handle.write("structure\thaddock_score\n")
        for stem, score in sorted(scores.items()):
            handle.write(f"{stem}\t{score:.6g}\n")
    print(f"[PYTHON-INFO] wrote {out / 'haddock_matched_scores.tsv'}")
    save_config(args.yaml_out, params)
    print(f"[PYTHON-INFO] {datetime.datetime.now():%Y-%m-%d %H:%M:%S} exec={time.time()-start:.2f}s maxmem={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}MB")

if __name__ == "__main__":
    main()


