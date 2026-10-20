#!/usr/bin/env python3
"""Complex and residue interface metrics with buried surface area, relative buried area, heavy-atom contacts, minimum distances, hydrophobicity-weighted contact energies, ensemble frequencies and PyMOL-detected inter-chain residue pairs (hydrogen bonds, salt bridges, hydrophobic contacts), written as three TSVs and one 3x3 summary SVG panel.
Args:
    --input: directory containing harmonized .pdb models with chains A and B.
    --output: directory where the TSVs and SVG are written.
    --yaml-in: optional yaml with parameters nested under the script name.
    --yaml-out: optional yaml to store the script name and parameters.
"""
###################################### dependencies
import argparse, csv, datetime, multiprocessing, os, random, resource, time
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymol
import yaml
from Bio.PDB import PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.Polypeptide import protein_letters_3to1_extended
from Bio.PDB.SASA import ShrakeRupley
from tqdm import tqdm
###################################### constants
NAME = os.path.splitext(os.path.basename(__file__))[0]
DEFAULTS = {"input": None, "output": "data/measurements", "contact_cutoff": 8, "score_cutoff": 12.0, "block": 1000, "bins": 30, "panel_size": 4, "workers": None, "seed": 42, "plots": True, "hbond_cutoff": 3.5, "salt_cutoff": 4.0, "hydro_cutoff": 4.5, "w_hbond": 1.0, "w_salt": 1.5, "w_hydro": 0.1}
EXTS = (".pdb",)
HYDRO = {"A":0.62,"R":-2.53,"N":-0.78,"D":-0.90,"C":0.29,"Q":-0.85,"E":-0.74,"G":0.48,"H":-0.40,"I":1.38,"L":1.06,"K":-1.50,"M":0.64,"F":1.19,"P":0.12,"S":-0.18,"T":-0.05,"W":0.81,"Y":0.26,"V":1.08}
TO1 = {k.upper(): v for k, v in protein_letters_3to1_extended.items()}
TIEN = {"ALA":129,"ARG":274,"ASN":195,"ASP":193,"CYS":167,"GLN":225,"GLU":223,"GLY":104,"HIS":224,"ILE":197,"LEU":201,"LYS":236,"MET":224,"PHE":240,"PRO":159,"SER":155,"THR":172,"TRP":285,"TYR":263,"VAL":174}
POS_SEL = "(resn LYS and name NZ) or (resn ARG and name NH1+NH2+NE)"
NEG_SEL = "(resn ASP and name OD1+OD2) or (resn GLU and name OE1+OE2)"
COMPLEX_COLS = ["structure", "len_res_a", "len_res_b", "len_atoms_a", "len_atoms_b", "dASA_A2", "initial_sasa_A2", "frac_dASA", "n_contacts", "iface_res_a", "iface_res_b", "min_dist_A", "contact_score"]
RESIDUE_COLS = ["structure", "chain", "resid", "resname", "dSASA_A2", "isolated_sasa_A2", "rel_dSASA", "n_partners", "min_dist_A", "contact_energy", "energy_per_partner", "partners", "single_freq", "pair_freq"]
PYMOL_PAIR_COLS = ["structure", "chain_a", "resid_a", "resname_a", "chain_b", "resid_b", "resname_b", "n_hbond", "n_salt_bridge", "n_hydrophobic_atoms", "min_dist_A", "interaction_score"]
CIDX = {c: i for i, c in enumerate(COMPLEX_COLS)}
RIDX = {c: i for i, c in enumerate(RESIDUE_COLS)}
PIDX = {c: i for i, c in enumerate(PYMOL_PAIR_COLS)}
CMD = None
###################################### arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output")
parser.add_argument("--yaml-in")
parser.add_argument("--yaml-out", default="configs/param_configs.yaml")
args = parser.parse_args()
###################################### configuration
t0 = time.time()
cfg = {}
if args.yaml_in:
    with open(args.yaml_in) as f:
        cfg = (yaml.safe_load(f) or {}).get(NAME, {})
params = {**DEFAULTS, **cfg}
params.update({k: v for k, v in vars(args).items() if k in ("input", "output") and v is not None})
assert params["input"], "[PYTHON-INFO] no input provided"
random.seed(params["seed"]); np.random.seed(params["seed"])
nproc = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
workers = params["workers"] or nproc
files = sorted(p for p in Path(params["input"]).iterdir() if p.suffix.lower() in EXTS)
assert files, "[PYTHON-INFO] no .pdb files found"
in_name = Path(params["input"]).resolve().name
Path(params["output"]).mkdir(parents=True, exist_ok=True)
print(f"[PYTHON-INFO] structures={len(files)} detected_cpus={nproc} workers={workers} contact_cutoff={params['contact_cutoff']}A score_cutoff={params['score_cutoff']}A hbond_cutoff={params['hbond_cutoff']}A salt_cutoff={params['salt_cutoff']}A hydro_cutoff={params['hydro_cutoff']}A")
###################################### measurements
sr = ShrakeRupley()
def std(chain):
    return [r for r in chain if r.id[0] == " "]
def heavy(res):
    c, o = [], []
    for i, r in enumerate(res):
        for a in r:
            if (a.element or a.get_name()[:1]).strip().upper() != "H":
                c.append(a.coord); o.append(i)
    return np.array(c, float).reshape(-1, 3), np.array(o, int)
def sasa(chains):
    m = Model(0)
    for c in chains:
        m.add(c.copy())
    sr.compute(m, level="A")
    return sum(a.sasa for a in m.get_atoms())
def residue_sasa(chains):
    m = Model(0)
    for c in chains:
        m.add(c.copy())
    sr.compute(m, level="R")
    return {(c.id, r.id[1]): r.sasa for c in m for r in c if r.id[0] == " "}
def complex_stats(xa, xb):
    n, score, dmin = 0, 0.0, np.inf
    ha, hb = np.zeros(len(xa), bool), np.zeros(len(xb), bool)
    cc, sc, blk = params["contact_cutoff"], params["score_cutoff"], params["block"]
    for s in range(0, len(xa), blk):
        d = np.linalg.norm(xa[s:s + blk, None] - xb[None], axis=-1)
        m = d <= cc
        n += int(m.sum())
        w = (d <= sc) & (d > 0)
        score += float((1.0 / d[w]).sum())
        dmin = min(dmin, float(d.min()))
        ha[s:s + blk] |= m.any(axis=1)
        hb |= m.any(axis=0)
    return n, score, dmin, ha, hb
def residue_min(xa, oa, xb, ob, na, nb):
    out = np.full((na, nb), np.inf, np.float32)
    blk = params["block"]
    for s in range(0, len(xa), blk):
        d = np.linalg.norm(xa[s:s + blk, None] - xb[None], axis=-1).astype(np.float32)
        np.minimum.at(out, (np.repeat(oa[s:s + blk], len(ob)), np.tile(ob, d.shape[0])), d.ravel())
    return out
def measure(name, A, B):
    ra, rb = std(A), std(B)
    xa, oa = heavy(ra); xb, ob = heavy(rb)
    if not len(xa) or not len(xb):
        raise ValueError(f"{name}: no heavy atoms in chain A or B")
    sa, sb, sab = sasa([A]), sasa([B]), sasa([A, B])
    dasa, init = sa + sb - sab, sa + sb
    frac = dasa / init if init > 0 else 0.0
    n, score, dmin_c, ha, hb = complex_stats(xa, xb)
    crow = [name, len(ra), len(rb), len(xa), len(xb), round(dasa, 2), round(init, 2), round(frac, 4), n, len({int(oa[k]) for k in np.where(ha)[0]}), len({int(ob[k]) for k in np.where(hb)[0]}), round(dmin_c, 3), round(score, 4)]
    dmin = residue_min(xa, oa, xb, ob, len(ra), len(rb))
    contact = dmin <= params["contact_cutoff"]
    mask = dmin <= params["score_cutoff"]
    ha_h = np.array([HYDRO.get(TO1.get(r.get_resname().upper(), "X"), 0.0) for r in ra])
    hb_h = np.array([HYDRO.get(TO1.get(r.get_resname().upper(), "X"), 0.0) for r in rb])
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.where(mask, -0.5 * (ha_h[:, None] + hb_h[None]) * (1 - dmin / params["score_cutoff"]), 0.0)
    iso_a, iso_b, cpx = residue_sasa([A]), residue_sasa([B]), residue_sasa([A, B])
    views = (
        ("A", ra, iso_a, e.sum(axis=1), contact.sum(axis=1), dmin.min(axis=1)),
        ("B", rb, iso_b, e.sum(axis=0), contact.sum(axis=0), dmin.min(axis=0)),
    )
    rows = []
    for cid, res, iso, en_sum, npart, dm in views:
        for k, r in enumerate(res):
            if npart[k] == 0:
                continue
            key = (cid, r.id[1])
            dsasa = iso[key] - cpx[key]
            maxa = TIEN.get(r.get_resname().upper())
            rel = dsasa / maxa if maxa else float("nan")
            en = float(en_sum[k])
            if cid == "A":
                plist = [f"B{rb[j].id[1]} {rb[j].get_resname()}" for j in np.where(contact[k])[0]]
            else:
                plist = [f"A{ra[i].id[1]} {ra[i].get_resname()}" for i in np.where(contact[:, k])[0]]
            rows.append([name, cid, r.id[1], r.get_resname(), round(float(dsasa), 2), round(float(iso[key]), 2), round(float(rel), 4), int(npart[k]), round(float(dm[k]), 3), round(en, 4), round(en / npart[k], 4), ";".join(plist)])
    return crow, rows
###################################### pymol
def init_pymol():
    global CMD
    pymol.finish_launching(["pymol", "-qc"])
    CMD = pymol.cmd
def pymol_pairs(path, name):
    cmd = CMD
    cmd.reinitialize()
    cmd.load(str(path), "s")
    allp = "s and polymer and (chain A or chain B)"
    A, B = f"{allp} and chain A", f"{allp} and chain B"
    atoms = {a.index: (a.chain, a.resi, a.resn, np.array(a.coord, float)) for a in cmd.get_model(allp).atom}
    groups = {
        0: cmd.find_pairs(A, B, mode=1, cutoff=params["hbond_cutoff"]),
        1: cmd.find_pairs(f"{allp} and ({POS_SEL})", f"{allp} and ({NEG_SEL})", mode=0, cutoff=params["salt_cutoff"]),
        2: cmd.find_pairs(f"{A} and elem C", f"{B} and elem C", mode=0, cutoff=params["hydro_cutoff"]),
        3: cmd.find_pairs(A, B, mode=1, cutoff=12.0),
    }
    agg = defaultdict(lambda: [0, 0, 0, np.inf])
    for slot, pairs in groups.items():
        for (_, i), (_, j) in pairs:
            ca, cb = atoms[i], atoms[j]
            if ca[0] == cb[0]:
                continue
            if ca[0] != "A":
                ca, cb = cb, ca
            key = (ca[0], ca[1], ca[2], cb[0], cb[1], cb[2])
            entry = agg[key]
            if slot < 3:
                entry[slot] += 1
            entry[3] = min(entry[3], float(np.linalg.norm(ca[3] - cb[3])))
    rows = []
    for key, (hb, sb, hy, dm) in agg.items():
        score = params["w_hbond"] * hb + params["w_salt"] * sb + params["w_hydro"] * hy
        rows.append([name, *key, hb, sb, hy, round(dm, 3), round(score, 3)])
    return rows
def process(path):
    name = Path(path).stem
    model = PDBParser(QUIET=True).get_structure("s", str(path))[0]
    if "A" not in model or "B" not in model:
        raise ValueError(f"{name}: missing chain A or B")
    crow, rows = measure(name, model["A"], model["B"])
    prows = pymol_pairs(path, name)
    return name, crow, rows, prows, f"dASA={crow[CIDX['dASA_A2']]:.1f}A2 n_contacts={crow[CIDX['n_contacts']]} contact_score={crow[CIDX['contact_score']]:.2f} interface_residues={len(rows)} pymol_pairs={len(prows)}"
###################################### run
rows_c, rows_r, rows_p = [], [], []
with multiprocessing.Pool(workers, initializer=init_pymol) as pool:
    for name, crow, rrows, prow, msg in tqdm(pool.imap_unordered(process, files, chunksize=1), total=len(files), desc="interfaces"):
        rows_c.append(crow); rows_r.extend(rrows); rows_p.extend(prow)
        print(f"[PYTHON-INFO] {name}: {msg}")
rows_c.sort(key=lambda r: r[0])
rows_r.sort(key=lambda r: (r[0], r[1], r[2]))
rows_p.sort(key=lambda r: (r[0], -r[-1], r[1], r[2], r[4], r[5]))
def rkey(r):
    return f"{r[1]}{r[2]} {r[3]}"
def plist(r):
    return [p for p in r[RIDX["partners"]].split(";") if p]
def pkey(a, b):
    return "|".join(sorted((a, b)))
kset, pset = defaultdict(set), defaultdict(set)
for r in rows_r:
    k = rkey(r)
    kset[k].add(r[0])
    for p in plist(r):
        pset[pkey(k, p)].add(r[0])
pair_counts = {k: len(v) for k, v in pset.items()}
rows_r = [r + [len(kset[rkey(r)]), max((pair_counts[pkey(rkey(r), p)] for p in plist(r)), default=0)] for r in rows_r]
out_c = os.path.join(params["output"], f"{in_name}_complex_scores.tsv")
out_r = os.path.join(params["output"], f"{in_name}_residue_scores.tsv")
out_p = os.path.join(params["output"], f"{in_name}_pymol_pairs.tsv")
with open(out_c, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(COMPLEX_COLS); w.writerows(rows_c)
with open(out_r, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(RESIDUE_COLS); w.writerows(rows_r)
with open(out_p, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(PYMOL_PAIR_COLS); w.writerows(rows_p)
print(f"[PYTHON-INFO] wrote {len(rows_c)} complex rows to {out_c}, {len(rows_r)} residue rows to {out_r}, {len(rows_p)} pymol pair rows to {out_p}")
###################################### plots
def spear(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(x[m])), np.argsort(np.argsort(y[m])))[0, 1])
def fin(a):
    a = np.asarray(a, float)
    return a[np.isfinite(a)]
def style(ax, xl, yl, title, n):
    ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8); ax.set_title(title, fontsize=9, loc="left")
    ax.grid(alpha=0.25, lw=0.5); ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(0.98, 0.97, f"n={n}", transform=ax.transAxes, fontsize=7, va="top", ha="right")
def scat(ax, x, y, xl, yl, title, c=None, cl=None):
    m = np.isfinite(x) & np.isfinite(y)
    sc = ax.scatter(x[m], y[m], s=12, alpha=0.7, edgecolor="none", c=None if c is None else c[m], cmap="viridis")
    if c is not None and m.any():
        ax.figure.colorbar(sc, ax=ax, label=cl, shrink=0.8).ax.tick_params(labelsize=7)
    style(ax, xl, yl, f"{title}  rho={spear(x, y):.2f}", int(m.sum()))
def hist(ax, v, xl, title, bins):
    v = fin(v)
    ax.hist(v, bins=bins, color="#1f77b4", edgecolor="white", lw=0.4)
    style(ax, xl, "count", title, len(v))
if params["plots"] and rows_c and rows_r:
    C = lambda n: np.array([r[CIDX[n]] for r in rows_c], float)
    R = lambda n: np.array([r[RIDX[n]] for r in rows_r], float)
    P = lambda n: np.array([r[PIDX[n]] for r in rows_p], float)
    ps = params["panel_size"]
    fig, ax = plt.subplots(3, 3, figsize=(3 * ps, 3 * ps), constrained_layout=True)
    scat(ax[0, 0], C("dASA_A2"), C("n_contacts"), "dASA (A^2)", "n_contacts (heavy-atom pairs)", "complex: buried area vs contacts")
    scat(ax[0, 1], C("dASA_A2"), C("contact_score"), "dASA (A^2)", "contact score", "complex: buried area vs contact score")
    hist(ax[0, 2], C("min_dist_A"), "min heavy-atom distance (A)", "complex: minimum distance", params["bins"])
    hist(ax[1, 0], R("rel_dSASA"), "relative dSASA", "residue: relative burial", params["bins"])
    ax[1, 0].axvline(0.25, ls="--", color="0.4", lw=0.8)
    scat(ax[1, 1], R("rel_dSASA"), R("contact_energy"), "relative dSASA", "contact energy", "residue: burial vs contact energy")
    ax[1, 1].axhline(0, ls="--", color="0.4", lw=0.8)
    hist(ax[1, 2], R("n_partners"), "partners per residue", "residue: contact partners", params["bins"])
    scat(ax[2, 0], R("pair_freq"), R("contact_energy"), "models containing pair", "contact energy", "residue: pair frequency vs energy")
    ax[2, 0].axhline(0, ls="--", color="0.4", lw=0.8)
    hist(ax[2, 1], P("interaction_score"), "interaction score", "pymol: pair interaction score", params["bins"])
    scat(ax[2, 2], P("n_hbond"), P("n_salt_bridge"), "H-bonds per pair", "salt bridges per pair", "pymol: H-bonds vs salt bridges", c=P("interaction_score"), cl="interaction score")
    fig.suptitle(f"{in_name}: interface summary", fontsize=12)
    svg_path = os.path.join(params["output"], f"{in_name}_summary.svg")
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    print(f"[PYTHON-INFO] wrote 3x3 summary panel {svg_path}")
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