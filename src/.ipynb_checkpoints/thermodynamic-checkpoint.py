#!/usr/bin/env python3
"""Pairwise chain interaction stability and residue-contact thermodynamic estimates for PDB or mmCIF files.

Args:
    --input: path to a .pdb or .cif file, or a directory containing mixed .pdb and .cif files.
    --yaml-in: optional yaml with parameters nested under the script name.
    --yaml-out: optional yaml to store the script name and parameters.
"""

###################################### dependencies
import argparse, csv, datetime, glob, os, random, resource, time
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.Polypeptide import protein_letters_3to1_extended
from Bio.PDB.SASA import ShrakeRupley
from tqdm import tqdm

###################################### constants
NAME = os.path.splitext(os.path.basename(__file__))[0]
DEFAULTS = {"input": None, "seed": 42, "ca_cutoff": 12.0, "gamma": 0.025, "out_dir": "data/thermodynamic"}
HYDRO = {"A":0.62,"R":-2.53,"N":-0.78,"D":-0.90,"C":0.29,"Q":-0.85,"E":-0.74,"G":0.48,"H":-0.40,"I":1.38,"L":1.06,"K":-1.50,"M":0.64,"F":1.19,"P":0.12,"S":-0.18,"T":-0.05,"W":0.81,"Y":0.26,"V":1.08}
TO1 = {k.upper(): v for k, v in protein_letters_3to1_extended.items()}
EXTS = (".pdb", ".cif", ".mmcif")
PAIR_COLS = ["structure", "chain_a", "chain_b", "n_res_a", "n_res_b", "n_contacts", "iface_res", "dASA_A2", "dG_solv", "dG_contacts", "mean_contact_e", "min_contact_e", "top3"]
CONTACT_COLS = ["structure", "chain_a", "chain_b", "resname_a", "resid_a", "resname_b", "resid_b", "ca_dist_A", "hydro_a", "hydro_b", "contact_e"]
FIG_SIZE = (6, 6)
BINS = 40
PLOT_NAME = "thermodynamic_plots.svg"

###################################### arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--yaml-in")
parser.add_argument("--yaml-out")
args = parser.parse_args()

###################################### configuration
t0 = time.time()
cfg = {}
if args.yaml_in:
    with open(args.yaml_in) as f:
        cfg = (yaml.safe_load(f) or {}).get(NAME, {})
params = {**DEFAULTS, **cfg}
if args.input:
    params["input"] = args.input
assert params["input"], "[PYTHON-INFO] no input provided"
random.seed(params["seed"]); np.random.seed(params["seed"])
files = sorted(f for f in glob.glob(os.path.join(params["input"], "*")) if f.lower().endswith(EXTS)) if os.path.isdir(params["input"]) else [params["input"]]
assert files, "[PYTHON-INFO] no .pdb or .cif files found"
print(f"[PYTHON-INFO] structures={len(files)} cutoff={params['ca_cutoff']}A")

###################################### io
def load(path):
    with open(path) as f:
        head = f.read(256)
    cif = path.lower().endswith((".cif", ".mmcif")) or head.lstrip().startswith("data_")
    return (MMCIFParser if cif else PDBParser)(QUIET=True).get_structure("s", path)

###################################### chains
def build(model):
    chains = {}
    for c in model:
        res = [r for r in c if r.id[0] == " " and "CA" in r]
        if res:
            ca = np.array([r["CA"].coord for r in res])
            h = np.array([HYDRO.get(TO1.get(r.get_resname().upper(), "X"), 0.0) for r in res])
            chains[c.id] = (c, ca, h, res)
    return chains

###################################### sasa
sr = ShrakeRupley()
def area(ents):
    m = Model(0)
    for e in ents:
        m.add(e.copy())
    sr.compute(m, level="A")
    return sum(a.sasa for a in m.get_atoms())

###################################### pairs
def contacts(ca_a, ha, ca_b, hb, cut):
    d = np.linalg.norm(ca_a[:, None] - ca_b[None], axis=-1)
    i, j = np.where(d <= cut)
    e = -0.5 * (ha[i] + hb[j]) * (1 - d[i, j] / cut)
    return i, j, d[i, j], e

###################################### analysis
pair_rows, contact_rows = [], []
for path in tqdm(files, desc="structures"):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        chains = build(load(path)[0])
        solo = {k: area([v[0]]) for k, v in chains.items()}
        for a, b in combinations(chains, 2):
            _, ca_a, ha, ra = chains[a]
            _, ca_b, hb, rb = chains[b]
            i, j, d, e = contacts(ca_a, ha, ca_b, hb, params["ca_cutoff"])
            dasa = solo[a] + solo[b] - area([chains[a][0], chains[b][0]])
            dg_solv = -params["gamma"] * dasa
            top = np.argsort(e)[:3]
            top_s = ", ".join(f"{ra[i[k]].get_resname()}{ra[i[k]].id[1]}-{rb[j[k]].get_resname()}{rb[j[k]].id[1]}:{e[k]:.2f}" for k in top)
            pair_rows.append([stem, a, b, len(ra), len(rb), len(i), len(set(i)) + len(set(j)), round(dasa, 2), round(dg_solv, 3), round(float(e.sum()), 4), round(float(e.mean()) if e.size else 0.0, 4), round(float(e.min()) if e.size else 0.0, 4), top_s])
            for k in range(len(i)):
                contact_rows.append([stem, a, b, ra[i[k]].get_resname(), ra[i[k]].id[1], rb[j[k]].get_resname(), rb[j[k]].id[1], round(float(d[k]), 3), round(float(ha[i[k]]), 3), round(float(hb[j[k]]), 3), round(float(e[k]), 4)])
            print(f"[PYTHON-INFO] {stem} {a}-{b} contacts={len(i)} iface_res={len(set(i)) + len(set(j))} dASA={dasa:.1f}A2 dG_solv={dg_solv:.2f} dG_contacts={float(e.sum()):.2f}")
    except Exception as ex:
        print(f"[PYTHON-INFO] skipped {path}: {ex}")

###################################### output
os.makedirs(params["out_dir"], exist_ok=True)
for name, cols, rows in [("thermodynamic_pairs.tsv", PAIR_COLS, pair_rows), ("thermodynamic_contacts.tsv", CONTACT_COLS, contact_rows)]:
    with open(os.path.join(params["out_dir"], name), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(cols); w.writerows(rows)
print(f"[PYTHON-INFO] wrote pairs={len(pair_rows)} contacts={len(contact_rows)} to {params['out_dir']}")

###################################### plots
if pair_rows:
    fig, ax = plt.subplots(2, 1, figsize=FIG_SIZE, constrained_layout=True)
    ax[0].scatter([r[7] for r in pair_rows], [r[9] for r in pair_rows], s=12)
    ax[0].set_xlabel("dASA (A^2)"); ax[0].set_ylabel("dG_contacts"); ax[0].set_title(f"Interface area vs contact term (cutoff {params['ca_cutoff']} A)")
    if contact_rows:
        ax[1].hist([r[10] for r in contact_rows], bins=BINS)
    ax[1].set_xlabel("contact energy"); ax[1].set_ylabel("count"); ax[1].set_title("Per-contact energy distribution")
    fig.savefig(os.path.join(params["out_dir"], PLOT_NAME), format="svg")
    plt.close(fig)
    print(f"[PYTHON-INFO] wrote plot {os.path.join(params['out_dir'], PLOT_NAME)}")

###################################### config output
if args.yaml_out:
    out = {}
    if os.path.exists(args.yaml_out):
        with open(args.yaml_out) as f:
            out = yaml.safe_load(f) or {}
    out[NAME] = params
    os.makedirs(os.path.dirname(args.yaml_out) or ".", exist_ok=True)
    with open(args.yaml_out, "w") as f:
        yaml.safe_dump(out, f)
print(f"[PYTHON-INFO] {datetime.datetime.now():%Y-%m-%d %H:%M:%S} exec={time.time() - t0:.2f}s maxmem={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}MB")