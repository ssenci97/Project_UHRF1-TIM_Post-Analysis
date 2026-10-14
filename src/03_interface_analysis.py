#!/usr/bin/env python3
"""Calculate structural interface measurements and interaction types."""
###################################### dependencies
import argparse
import csv
import datetime
import multiprocessing
import os
import resource
import time
from pathlib import Path
import numpy as np
import yaml
from Bio.PDB import PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.Polypeptide import protein_letters_3to1_extended
from Bio.PDB.SASA import ShrakeRupley
from tqdm import tqdm

###################################### constants
NAME = "interface_analysis"
SEED = 42
DEFAULTS = {"input": None, "output": None, "workers": 8, "seed": SEED, "contact_cutoff": 8.0, "score_cutoff": 12.0, "block": 1000, "hbond_cutoff": 3.5, "salt_cutoff": 4.0, "hydro_cutoff": 4.5, "w_hbond": 1.0, "w_salt": 1.5, "w_hydro": 0.1, "clash_cutoff": 2.5}
COMPLEX_COLS = ["structure", "source", "clash", "len_res_a", "len_res_b", "len_atoms_a", "len_atoms_b", "dASA_A2", "initial_sasa_A2", "frac_dASA", "n_contacts", "iface_res_a", "iface_res_b", "min_dist_A", "contact_score", "haddock_score"]
RESIDUE_COLS = ["structure", "chain", "resid", "resname", "dSASA_A2", "isolated_sasa_A2", "rel_dSASA", "n_partners", "min_dist_A", "contact_energy", "energy_per_partner", "partners"]
PAIR_COLS = ["structure", "chain_a", "resid_a", "resname_a", "chain_b", "resid_b", "resname_b", "n_hbond", "n_salt_bridge", "n_hydrophobic_contacts", "min_dist_A", "interaction_score"]
HYDRO = {"A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08}
TO1 = {k.upper(): v for k, v in protein_letters_3to1_extended.items()}
TIEN = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159, "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174}
POS_SEL = "(resn LYS and name NZ) or (resn ARG and name NH1+NH2+NE)"
NEG_SEL = "(resn ASP and name OD1+OD2) or (resn GLU and name OE1+OE2)"
HYD_SEL = "(resn ALA+VAL+ILE+LEU+MET+PHE+TRP+PRO+TYR+CYS) and elem C and not name N+C+O+CA"
COLOR_RULES = (("abin", "abin"), ("model", "model"), ("pdbeval", "pdbeval"))
PARAMS = {}

###################################### helpers
def source_style(name):
    low = name.lower()
    return next((label for key, label in COLOR_RULES if key in low), "other")

def save_config(path, params):
    if not path:
        return
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {} if path.exists() else {}
    data[NAME] = params
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))

def write_tsv(path, columns, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        writer.writerows(rows)

def finite(value):
    return round(float(value), 4) if np.isfinite(value) else ""

def standard(chain):
    return [r for r in chain if r.id[0] == " "]

def heavy(residues):
    coords, owners = [], []
    for i, residue in enumerate(residues):
        for atom in residue:
            if (atom.element or atom.name[:1]).strip().upper() != "H":
                coords.append(atom.coord)
                owners.append(i)
    return np.asarray(coords, float).reshape(-1, 3), np.asarray(owners, int)

def sasa(chains, level="A"):
    model = Model(0)
    for chain in chains:
        model.add(chain.copy())
    ShrakeRupley().compute(model, level=level)
    if level == "A":
        return sum(atom.sasa for atom in model.get_atoms())
    return {(chain.id, residue.id[1]): residue.sasa for chain in model for residue in chain if residue.id[0] == " "}

def min_matrix(xa, oa, xb, ob, na, nb):
    out = np.full((na, nb), np.inf, np.float32)
    block = PARAMS["block"]
    for start in range(0, len(xa), block):
        d = np.linalg.norm(xa[start:start + block, None] - xb[None], axis=-1).astype(np.float32)
        np.minimum.at(out, (np.repeat(oa[start:start + block], len(ob)), np.tile(ob, d.shape[0])), d.ravel())
    return out

def measure(name, A, B):
    ra, rb = standard(A), standard(B)
    xa, oa = heavy(ra)
    xb, ob = heavy(rb)
    if not len(xa) or not len(xb):
        raise ValueError(f"{name}: no heavy atoms in chain A or B")
    sa, sb, sab = sasa([A]), sasa([B]), sasa([A, B])
    initial, dasa = sa + sb, sa + sb - sab
    dmin = min_matrix(xa, oa, xb, ob, len(ra), len(rb))
    contact = dmin <= PARAMS["contact_cutoff"]
    mask = (dmin <= PARAMS["score_cutoff"]) & (dmin > 0)
    score = float(np.sum(np.where(mask, 1 / np.maximum(dmin, 1e-6), 0)))
    mind = float(dmin.min())
    crow = [name, source_style(name), int(mind < PARAMS["clash_cutoff"]), len(ra), len(rb), len(xa), len(xb), round(dasa, 2), round(initial, 2), round(dasa / initial, 4) if initial else 0, int(contact.sum()), int(contact.any(axis=1).sum()), int(contact.any(axis=0).sum()), round(mind, 3), round(score, 4), ""]
    iso_a, iso_b, cpx = sasa([A], "R"), sasa([B], "R"), sasa([A, B], "R")
    ha = np.array([HYDRO.get(TO1.get(r.resname.upper(), "X"), 0.0) for r in ra])
    hb = np.array([HYDRO.get(TO1.get(r.resname.upper(), "X"), 0.0) for r in rb])
    energy = np.where(mask, -0.5 * (ha[:, None] + hb[None, :]) * (1 - dmin / PARAMS["score_cutoff"]), 0.0)
    rows = []
    for cid, residues, isolated, ens, contacts, mins, partner_matrix in (("A", ra, iso_a, energy.sum(axis=1), contact.sum(axis=1), dmin.min(axis=1), contact), ("B", rb, iso_b, energy.sum(axis=0), contact.sum(axis=0), dmin.min(axis=0), contact.T)):
        for i, residue in enumerate(residues):
            if not contacts[i]:
                continue
            key = (cid, residue.id[1])
            dsasa = isolated[key] - cpx[key]
            rel = dsasa / TIEN[residue.resname] if residue.resname in TIEN else float("nan")
            partners = [f"{'B' if cid == 'A' else 'A'}{residues2[j].id[1]} {residues2[j].resname}" for j in np.where(partner_matrix[i])[0] for residues2 in ([rb] if cid == "A" else [ra])]
            rows.append([name, cid, residue.id[1], residue.resname, round(float(dsasa), 2), round(float(isolated[key]), 2), finite(rel), int(contacts[i]), round(float(mins[i]), 3), round(float(ens[i]), 4), round(float(ens[i] / contacts[i]), 4), ";".join(partners)])
    return crow, rows

def pymol_pairs(path, name):
    import pymol
    pymol.finish_launching(["pymol", "-cq"])
    cmd = pymol.cmd
    cmd.reinitialize()
    cmd.load(str(path), "s")
    all_atoms = "s and polymer and (chain A or chain B)"
    chain_a, chain_b = f"{all_atoms} and chain A", f"{all_atoms} and chain B"
    atoms = {atom.index: (atom.chain, atom.resi, atom.resn, np.array(atom.coord, float)) for atom in cmd.get_model(all_atoms).atom}
    groups = [
        cmd.find_pairs(chain_a, chain_b, mode=1, cutoff=PARAMS["hbond_cutoff"]),
        cmd.find_pairs(f"{all_atoms} and ({POS_SEL})", f"{all_atoms} and ({NEG_SEL})", mode=0, cutoff=PARAMS["salt_cutoff"]),
        cmd.find_pairs(f"{chain_a} and ({HYD_SEL})", f"{chain_b} and ({HYD_SEL})", mode=0, cutoff=PARAMS["hydro_cutoff"]),
    ]
    aggregate = {}
    for slot, pairs in enumerate(groups):
        for (_, i), (_, j) in pairs:
            a, b = atoms[i], atoms[j]
            if a[0] == b[0]:
                continue
            if a[0] != "A":
                a, b = b, a
            key = (a[0], a[1], a[2], b[0], b[1], b[2])
            item = aggregate.setdefault(key, [0, 0, 0, np.inf])
            item[slot] += 1
            item[3] = min(item[3], float(np.linalg.norm(a[3] - b[3])))
    return [[name, *key, *counts[:3], round(counts[3], 3), round(PARAMS["w_hbond"] * counts[0] + PARAMS["w_salt"] * counts[1] + PARAMS["w_hydro"] * counts[2], 3)] for key, counts in aggregate.items()]

def process(path):
    name = path.stem
    model = PDBParser(QUIET=True).get_structure(name, str(path))[0]
    if "A" not in model or "B" not in model:
        raise ValueError(f"{name}: missing chain A or B")
    crow, residues = measure(name, model["A"], model["B"])
    pairs = pymol_pairs(path, name)
    return crow, residues, pairs

###################################### main
def main():
    global PARAMS
    start = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--workers", type=int)
    p.add_argument("--yaml-in")
    p.add_argument("--yaml-out", default="configs/param_configs.yaml")
    args = p.parse_args()
    PARAMS = {**DEFAULTS}
    if args.yaml_in:
        with open(args.yaml_in) as handle:
            PARAMS.update((yaml.safe_load(handle) or {}).get(NAME, {}))
    for key in ("input", "output", "workers"):
        if getattr(args, key) is not None:
            PARAMS[key] = getattr(args, key)
    if not PARAMS["input"]:
        raise SystemExit("[PYTHON-INFO] --input is required")
    np.random.seed(PARAMS["seed"])
    folder = Path(PARAMS["input"])
    files = sorted(folder.glob("*.pdb"))
    if not files:
        raise SystemExit(f"[PYTHON-INFO] no PDB files found in {folder}")
    name = folder.resolve().name
    out = Path(PARAMS["output"] or Path("data/pdbs_by_group") / name / "aligned")
    out.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(PARAMS["workers"] or (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)))
    print(f"[PYTHON-INFO] structures={len(files)} workers={workers}; all clashing models retained")
    complex_rows, residue_rows, pair_rows = [], [], []
    with multiprocessing.Pool(workers) as pool:
        for path, result in zip(tqdm(files, desc="interfaces"), pool.imap(process, files)):
            try:
                crow, rrows, prows = result
                complex_rows.append(crow)
                residue_rows.extend(rrows)
                pair_rows.extend(prows)
                print(f"[PYTHON-INFO] {path.stem}: residues={len(rrows)} PyMOL pairs={len(prows)} clash={crow[2]}")
            except Exception as exc:
                print(f"[PYTHON-INFO] warning: {path.name}: {exc}")
    complex_rows.sort(key=lambda r: r[0])
    residue_rows.sort(key=lambda r: (r[0], r[1], r[2]))
    pair_rows.sort(key=lambda r: (r[0], -r[-1], r[1], r[2], r[4], r[5]))
    write_tsv(out / f"{name}_complex_scores.tsv", COMPLEX_COLS, complex_rows)
    write_tsv(out / f"{name}_residue_scores.tsv", RESIDUE_COLS, residue_rows)
    write_tsv(out / f"{name}_pymol_pairs.tsv", PAIR_COLS, pair_rows)
    print(f"[PYTHON-INFO] wrote {len(complex_rows)} complex, {len(residue_rows)} residue and {len(pair_rows)} interaction rows")
    save_config(args.yaml_out, PARAMS)
    print(f"[PYTHON-INFO] {datetime.datetime.now():%Y-%m-%d %H:%M:%S} exec={time.time()-start:.2f}s maxmem={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}MB")

if __name__ == "__main__":
    main()


