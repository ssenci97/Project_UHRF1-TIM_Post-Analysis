#!/usr/bin/env python3
"""Align PDB ensembles, export structural viewers, quantify interfaces, and integrate HADDOCK scores."""

###################################### dependencies
import argparse
import csv
import datetime
import io
import json
import multiprocessing
import os
import re
import resource
import sys
import time
import urllib.request
from collections import defaultdict
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import yaml
from Bio.PDB import MMCIFIO, PDBParser, Superimposer
from Bio.PDB.Model import Model
from Bio.PDB.Polypeptide import protein_letters_3to1_extended
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Structure import Structure
from tqdm import tqdm

###################################### constants
NAME = "align_and_export"
EXTS = (".pdb",)
DEFAULTS = {
    "input": None, "output": None, "ref": None, "align_chain": "A",
    "workers": None, "seed": 42, "html": True, "interfaces": True,
    "plots": True, "haddock_plots": True, "capri_tsv": "data/capri_ss_all.tsv",
    "contact_cutoff": 8.0, "score_cutoff": 12.0, "block": 1000,
    "hbond_cutoff": 3.5, "salt_cutoff": 4.0, "hydro_cutoff": 4.5,
    "w_hbond": 1.0, "w_salt": 1.5, "w_hydro": 0.1,
    "clash_cutoff": 2.5, "exclude_clashes": False,
    "occ_cutoff": 0.5, "burial_cutoff": 0.25,
    "top_residues": 15, "top_pairs": 12,
}
THREEDMOL_URLS = (
    "https://3dmol.org/build/3Dmol-min.js",
    "https://3dmol.csb.pitt.edu/build/3Dmol-min.js",
)
COLOR_RULES = (
    ("abin", "salmon", "teal", "abin"),
    ("model", "grey", "black", "model"),
    ("pdbeval", "red", "blue", "pdbeval"),
)
DEFAULT_STYLE = ("orange", "purple", "other")
HYDRO = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}
TO1 = {k.upper(): v for k, v in protein_letters_3to1_extended.items()}
TIEN = {
    "ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167,
    "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197,
    "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159,
    "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174,
}
TYPES = ("hbond", "salt", "hydro")
TYPE_LABEL = {"hbond": "H-bond", "salt": "Salt bridge", "hydro": "Hydrophobic"}
TYPE_COLOR = {"hbond": "#1f5fa8", "salt": "#c0392b", "hydro": "#d4a017"}
TYPE_MARKER = {"hbond": "o", "salt": "^", "hydro": "s"}
POS_SEL = "(resn LYS and name NZ) or (resn ARG and name NH1+NH2+NE)"
NEG_SEL = "(resn ASP and name OD1+OD2) or (resn GLU and name OE1+OE2)"
HYD_SEL = "(resn ALA+VAL+ILE+LEU+MET+PHE+TRP+PRO+TYR+CYS) and elem C and not name N+C+O+CA"
MANIFEST_COLS = [
    "order", "structure", "source", "color_A", "color_B",
    "n_CA_matched", "CA_rmsd_A", "haddock_score",
]
COMPLEX_COLS = [
    "structure", "source", "clash", "len_res_a", "len_res_b",
    "len_atoms_a", "len_atoms_b", "dASA_A2", "initial_sasa_A2",
    "frac_dASA", "n_contacts", "iface_res_a", "iface_res_b",
    "min_dist_A", "contact_score", "haddock_score",
]
RESIDUE_COLS = [
    "structure", "chain", "resid", "resname", "dSASA_A2",
    "isolated_sasa_A2", "rel_dSASA", "n_partners", "min_dist_A",
    "contact_energy", "energy_per_partner", "partners",
]
PYMOL_PAIR_COLS = [
    "structure", "chain_a", "resid_a", "resname_a", "chain_b",
    "resid_b", "resname_b", "n_hbond", "n_salt_bridge",
    "n_hydrophobic_contacts", "min_dist_A", "interaction_score",
]
RES_FREQ_BASE = [
    "chain", "resid", "resname", "n_models", "occ_hbond",
    "occ_salt", "occ_hydro", "occ_any", "mean_rel_dsasa", "dominant",
]
PAIR_FREQ_BASE = [
    "chain_a", "resid_a", "resname_a", "chain_b", "resid_b",
    "resname_b", "n_models", "occ_hbond", "occ_salt", "occ_hydro",
    "occ_any", "mean_score", "mean_rel_dsasa", "dominant", "stabilizing",
]
CIDX = {c: i for i, c in enumerate(COMPLEX_COLS)}
RIDX = {c: i for i, c in enumerate(RESIDUE_COLS)}
CMD = None
params = {}

###################################### HTML template
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>__TITLE__</title><script>__LIBRARY__</script>
<style>
html,body{margin:0;height:100%;font-family:sans-serif}body{display:flex}
#panel{width:300px;min-width:260px;overflow-y:auto;padding:12px;box-sizing:border-box;border-right:1px solid #ccc;font-size:13px}
#panel h3{margin:0 0 8px;font-size:15px;word-break:break-all}#viewer{flex:1;position:relative}
label{display:block;margin:4px 0;cursor:pointer}.sw{display:inline-block;width:10px;height:10px;margin-left:4px;border:1px solid #666;vertical-align:middle}
button{margin:2px 4px 6px 0;padding:4px 8px}.note{color:#555;font-size:12px;margin-top:12px}
</style></head><body>
<div id="panel"><h3>__TITLE__</h3>
<button id="showAll">Show all</button><button id="hideAll">Hide all</button><button id="resetView">Reset view</button>
<div id="list"></div><div class="note">Swatches: chain A, chain B. Click an atom to label it. Models are aligned on chain A C-alpha atoms.</div>
</div><div id="viewer"></div><script>
const MODELS=__MODELS__;
const viewer=$3Dmol.createViewer(document.getElementById("viewer"),{backgroundColor:"white"});
MODELS.forEach(m=>viewer.addModel(m.cif,"cif"));
const shown=MODELS.map(()=>true),boxes=[];
function applyModel(i){const m=MODELS[i];if(shown[i]){viewer.setStyle({model:i,chain:"A"},{cartoon:{color:m.colorA}});viewer.setStyle({model:i,chain:"B"},{cartoon:{color:m.colorB}})}else{viewer.setStyle({model:i},{})}}
function setAll(on){MODELS.forEach((m,i)=>{shown[i]=on;boxes[i].checked=on;applyModel(i)});viewer.render()}
function swatch(c){const s=document.createElement("span");s.className="sw";s.style.background=c;return s}
const list=document.getElementById("list");
MODELS.forEach((m,i)=>{const label=document.createElement("label"),box=document.createElement("input");box.type="checkbox";box.checked=true;box.addEventListener("change",()=>{shown[i]=box.checked;applyModel(i);viewer.render()});boxes.push(box);label.appendChild(box);label.appendChild(document.createTextNode(" "+i+": "+m.name));label.appendChild(swatch(m.colorA));label.appendChild(swatch(m.colorB));list.appendChild(label)});
viewer.setClickable({},true,function(atom,v){v.removeAllLabels();const m=MODELS[atom.model];v.addLabel("Model "+atom.model+" ("+m.name+") | chain "+atom.chain+" | "+atom.resn+" "+atom.resi,{position:atom,backgroundColor:"black",fontColor:"white",fontSize:12});v.render()});
document.getElementById("showAll").onclick=()=>setAll(true);
document.getElementById("hideAll").onclick=()=>setAll(false);
document.getElementById("resetView").onclick=()=>{viewer.removeAllLabels();viewer.zoomTo();viewer.render()};
MODELS.forEach((m,i)=>applyModel(i));viewer.zoomTo();viewer.render();
</script></body></html>
"""

###################################### helpers
def source_style(stem):
    low = stem.lower()
    for key, col_a, col_b, label in COLOR_RULES:
        if key in low:
            return col_a, col_b, label
    return DEFAULT_STYLE


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--ref")
    p.add_argument("--align-chain", dest="align_chain")
    p.add_argument("--capri-tsv", dest="capri_tsv")
    p.add_argument("--workers", type=int)
    p.add_argument("--3dmol-js", dest="threedmol_js")
    p.add_argument("--no-html", action="store_true")
    p.add_argument("--no-interfaces", action="store_true")
    p.add_argument("--yaml-in")
    p.add_argument("--yaml-out", default="configs/param_configs.yaml")
    return p.parse_known_args(argv)[0]


def ca_map(model, chain_id):
    if chain_id not in model:
        return {}
    return {
        res.id[1:]: res["CA"]
        for res in model[chain_id]
        if res.id[0] == " " and "CA" in res
    }


def align(models, chain_id, ref_idx):
    ref_ca = ca_map(models[ref_idx], chain_id)
    rmsd, nmatch = [None] * len(models), [0] * len(models)
    for i, model in enumerate(models):
        if i == ref_idx:
            rmsd[i], nmatch[i] = 0.0, len(ref_ca)
            continue
        mob = ca_map(model, chain_id)
        keys = sorted(ref_ca.keys() & mob.keys())
        nmatch[i] = len(keys)
        if len(keys) < 3:
            print(f"[PYTHON-INFO] warning: model {i} has {len(keys)} matched CA atoms on chain {chain_id}; not aligned")
            continue
        sup = Superimposer()
        sup.set_atoms([ref_ca[k] for k in keys], [mob[k] for k in keys])
        sup.apply(list(model.get_atoms()))
        rmsd[i] = float(sup.rms)
    return rmsd, nmatch


def write_cif(structure, handle_or_path):
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(handle_or_path)


def ensemble_structure(models):
    structure = Structure("ensemble")
    for i, model in enumerate(models):
        copied = model.copy()
        copied.id = i
        structure.add(copied)
    return structure


def model_cif_text(model):
    structure = Structure("m")
    structure.add(model.copy())
    buf = io.StringIO()
    write_cif(structure, buf)
    return buf.getvalue()


def load_3dmol(local_path=None):
    if local_path:
        return Path(local_path).read_text(encoding="utf-8")
    cache = Path.home() / ".cache" / "aligned_export" / "3Dmol-min.js"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    last_err = None
    for url in THREEDMOL_URLS:
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                text = response.read().decode("utf-8")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
            print(f"[PYTHON-INFO] cached 3Dmol.js from {url}")
            return text
        except Exception as exc:
            last_err = exc
    raise SystemExit(f"[PYTHON-INFO] could not get 3Dmol-min.js ({last_err}); pass --3dmol-js PATH")


def inline_js(js):
    return re.sub(r"</(script)", lambda m: "<\\/" + m.group(1), js, flags=re.IGNORECASE)


def write_html(entries, title, path, library_js):
    models_json = json.dumps(entries).replace("</", "<\\/")
    html = (
        HTML_TEMPLATE.replace("__TITLE__", escape(title))
        .replace("__MODELS__", models_json)
        .replace("__LIBRARY__", inline_js(library_js))
    )
    Path(path).write_text(html, encoding="utf-8")


def write_tsv(path, columns, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        writer.writerows(rows)


def fin4(value):
    return round(float(value), 4) if np.isfinite(value) else ""


def _nz(value):
    return value if np.isfinite(value) else -1.0

###################################### HADDOCK filename matching & plotting
###################################### HADDOCK filename matching & plotting

def normalize_complex(value):
    return re.sub(r"\s+", "", str(value or ""))


def load_haddock(path):
    rows = []
    with open(path, encoding="utf-8-sig") as handle:
        header_line = handle.readline()
        if not header_line.strip():
            raise SystemExit(f"[PYTHON-INFO] Empty CAPRI table: {path}")
        header = [c.lstrip("\ufeff") for c in header_line.split()]
        required = ("score", "complex", "pdb_path")
        missing = [c for c in required if c not in header]
        if missing:
            raise SystemExit(f"[PYTHON-INFO] {path}: missing CAPRI columns {missing}; found {header}")
        
        score_idx = header.index("score")
        complex_idx = header.index("complex")
        pdb_path_idx = header.index("pdb_path")
        max_idx = max(score_idx, complex_idx, pdb_path_idx)
        
        for line_number, line in enumerate(handle, 2):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) <= max_idx:
                continue
                
            complex_name = fields[complex_idx]
            pdb_path = fields[pdb_path_idx]
            
            try:
                score = float(fields[score_idx])
            except (ValueError, IndexError):
                score = float("nan")
                
            if not complex_name or not pdb_path or pdb_path == "-":
                continue
                
            filename_after_slash = Path(pdb_path.replace("\\", "/")).name
            pdb_stem = Path(filename_after_slash).stem
            
            rows.append({
                "complex": complex_name,
                "complex_norm": normalize_complex(complex_name),
                "pdb_path": pdb_path,
                "pdb_filename": filename_after_slash,
                "stem": pdb_stem,
                "score": score,
            })
            
    print(f"[PYTHON-INFO] loaded {len(rows)} HADDOCK rows from {path}")
    return rows


def match_haddock(files, rows, complex_name):
    target = normalize_complex(complex_name)
    scores = {}
    
    # 1. List the PDB files being used at this iteration (raw stems, no normalization)
    local_stems = {p.stem: p.name for p in files}
    print(f"[PYTHON-INFO] 1. Iteration PDB files: {list(local_stems.values())}")
    
    # 2. Grep rows matching any iteration PDB stem using pdb_path (string after last /)
    matched_count = 0
    for row in rows:
        if target and target not in row["complex_norm"] and row["complex_norm"] not in target:
            continue
            
        pdb_path = row["pdb_path"]
        if not pdb_path or pdb_path == "-":
            continue
            
        filename_after_slash = Path(pdb_path.replace("\\", "/")).name
        table_stem = Path(filename_after_slash).stem
        
        for l_stem, original_name in local_stems.items():
            if table_stem == l_stem or l_stem.endswith(f"_{table_stem}") or table_stem.endswith(f"_{l_stem}"):
                try:
                    score = float(row["score"])
                    if np.isfinite(score):
                        scores[Path(original_name).stem] = score
                        matched_count += 1
                except (ValueError, TypeError):
                    pass
                break
                
    print(f"[PYTHON-INFO] 2. HADDOCK matching: {matched_count}/{len(files)} matched")
    return scores


def plot_haddock_scores(rows, out_dir, files_by_complex=None):
    groups = defaultdict(list)
    for row in rows:
        if np.isfinite(row["score"]):
            groups[row["complex"]].append(row)
            
    if not groups:
        print("[PYTHON-INFO] no finite HADDOCK scores; skipping score plots")
        return
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for complex_name, group in sorted(groups.items()):
        if files_by_complex is not None and complex_name in files_by_complex:
            allowed = {str(val) for val in files_by_complex[complex_name]}
            allowed_stems = {Path(val).stem for val in allowed}
            group = [
                row for row in group
                if row["pdb_filename"] in allowed
                or row["stem"] in allowed_stems
            ]
            
        if not group:
            continue
            
        group = sorted(group, key=lambda row: row["score"])
        names = [row["stem"] for row in group]
        vals = [row["score"] for row in group]
        
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(4, 4))
        
        colors = [source_style(name)[0] for name in names] if 'source_style' in globals() else ['steelblue'] * len(names)
        ax.bar(x, vals, color=colors, width=0.8, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90, fontsize=5)
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_ylabel("HADDOCK score (lower is better)", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        if 'source_style' in globals():
            legend_colors = {}
            for name in names:
                color, _, label = source_style(name)
                legend_colors.setdefault(label, color)
            ax.legend(handles=[Patch(color=color, label=label) for label, color in legend_colors.items()], fontsize=6, frameon=False, loc="best")
            
        ax.set_title(f"{complex_name}: {len(names)} models", fontsize=8, loc="left")
        fig.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.35)
        
        plot_path = out_dir / f"{complex_name}_haddock_scores.svg"
        fig.savefig(plot_path, format="svg")
        plt.close(fig)
        print(f"[PYTHON-INFO] wrote HADDOCK score plot {plot_path}")
###################################### interface measurements
sr = ShrakeRupley()


def std(chain):
    return [residue for residue in chain if residue.id[0] == " "]


def heavy(residues):
    coords, owners = [], []
    for i, residue in enumerate(residues):
        for atom in residue:
            if (atom.element or atom.get_name()[:1]).strip().upper() != "H":
                coords.append(atom.coord)
                owners.append(i)
    return np.array(coords, float).reshape(-1, 3), np.array(owners, int)


def sasa(chains):
    model = Model(0)
    for chain in chains:
        model.add(chain.copy())
    sr.compute(model, level="A")
    return sum(atom.sasa for atom in model.get_atoms())


def residue_sasa(chains):
    model = Model(0)
    for chain in chains:
        model.add(chain.copy())
    sr.compute(model, level="R")
    return {
        (chain.id, residue.id[1]): residue.sasa
        for chain in model for residue in chain if residue.id[0] == " "
    }


def complex_stats(xa, xb):
    n, score, dmin = 0, 0.0, np.inf
    ha, hb = np.zeros(len(xa), bool), np.zeros(len(xb), bool)
    cc, sc, block = params["contact_cutoff"], params["score_cutoff"], params["block"]
    for start in range(0, len(xa), block):
        distances = np.linalg.norm(xa[start:start + block, None] - xb[None], axis=-1)
        contacts = distances <= cc
        n += int(contacts.sum())
        mask = (distances <= sc) & (distances > 0)
        score += float((1.0 / distances[mask]).sum())
        dmin = min(dmin, float(distances.min()))
        ha[start:start + block] |= contacts.any(axis=1)
        hb |= contacts.any(axis=0)
    return n, score, dmin, ha, hb


def residue_min(xa, oa, xb, ob, na, nb):
    out = np.full((na, nb), np.inf, np.float32)
    block = params["block"]
    for start in range(0, len(xa), block):
        distances = np.linalg.norm(xa[start:start + block, None] - xb[None], axis=-1).astype(np.float32)
        np.minimum.at(
            out,
            (np.repeat(oa[start:start + block], len(ob)), np.tile(ob, distances.shape[0])),
            distances.ravel(),
        )
    return out


def measure(name, source, A, B):
    ra, rb = std(A), std(B)
    xa, oa = heavy(ra)
    xb, ob = heavy(rb)
    if not len(xa) or not len(xb):
        raise ValueError(f"{name}: no heavy atoms in chain A or B")
    sa, sb, sab = sasa([A]), sasa([B]), sasa([A, B])
    dasa, initial = sa + sb - sab, sa + sb
    frac = dasa / initial if initial > 0 else 0.0
    n, score, dmin_complex, ha, hb = complex_stats(xa, xb)
    clash = int(dmin_complex < params["clash_cutoff"])
    crow = [
        name, source, clash, len(ra), len(rb), len(xa), len(xb),
        round(dasa, 2), round(initial, 2), round(frac, 4), n,
        len({int(oa[k]) for k in np.where(ha)[0]}),
        len({int(ob[k]) for k in np.where(hb)[0]}),
        round(dmin_complex, 3), round(score, 4),
    ]
    dmin = residue_min(xa, oa, xb, ob, len(ra), len(rb))
    contact, mask = dmin <= params["contact_cutoff"], dmin <= params["score_cutoff"]
    ha_h = np.array([HYDRO.get(TO1.get(r.get_resname().upper(), "X"), 0.0) for r in ra])
    hb_h = np.array([HYDRO.get(TO1.get(r.get_resname().upper(), "X"), 0.0) for r in rb])
    with np.errstate(invalid="ignore", divide="ignore"):
        energy = np.where(
            mask,
            -0.5 * (ha_h[:, None] + hb_h[None]) * (1 - dmin / params["score_cutoff"]),
            0.0,
        )
    iso_a, iso_b, cpx = residue_sasa([A]), residue_sasa([B]), residue_sasa([A, B])
    views = (
        ("A", ra, iso_a, energy.sum(axis=1), contact.sum(axis=1), dmin.min(axis=1)),
        ("B", rb, iso_b, energy.sum(axis=0), contact.sum(axis=0), dmin.min(axis=0)),
    )
    rows = []
    for chain_id, residues, isolated, energy_sum, n_partners, min_dist in views:
        for k, residue in enumerate(residues):
            if n_partners[k] == 0:
                continue
            key = (chain_id, residue.id[1])
            dsasa = isolated[key] - cpx[key]
            max_area = TIEN.get(residue.get_resname().upper())
            rel = dsasa / max_area if max_area else float("nan")
            en = float(energy_sum[k])
            if chain_id == "A":
                partners = [f"B{rb[j].id[1]} {rb[j].get_resname()}" for j in np.where(contact[k])[0]]
            else:
                partners = [f"A{ra[i].id[1]} {ra[i].get_resname()}" for i in np.where(contact[:, k])[0]]
            rows.append([
                name, chain_id, residue.id[1], residue.get_resname(),
                round(float(dsasa), 2), round(float(isolated[key]), 2),
                round(float(rel), 4), int(n_partners[k]),
                round(float(min_dist[k]), 3), round(en, 4),
                round(en / n_partners[k], 4), ";".join(partners),
            ])
    return crow, rows


###################################### PyMOL interaction typing
def init_pymol():
    global CMD
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    CMD = pymol.cmd


def pymol_pairs(path, name):
    cmd = CMD
    cmd.reinitialize()
    cmd.load(str(path), "s")
    all_atoms = "s and polymer and (chain A or chain B)"
    chain_a, chain_b = f"{all_atoms} and chain A", f"{all_atoms} and chain B"
    atoms = {
        atom.index: (atom.chain, atom.resi, atom.resn, np.array(atom.coord, float))
        for atom in cmd.get_model(all_atoms).atom
    }
    groups = {
        0: cmd.find_pairs(chain_a, chain_b, mode=1, cutoff=params["hbond_cutoff"]),
        1: cmd.find_pairs(
            f"{all_atoms} and ({POS_SEL})",
            f"{all_atoms} and ({NEG_SEL})",
            mode=0, cutoff=params["salt_cutoff"],
        ),
        2: cmd.find_pairs(
            f"{chain_a} and ({HYD_SEL})",
            f"{chain_b} and ({HYD_SEL})",
            mode=0, cutoff=params["hydro_cutoff"],
        ),
    }
    aggregate_pairs = defaultdict(lambda: [0, 0, 0, np.inf])
    for slot, pairs in groups.items():
        for (_, i), (_, j) in pairs:
            ca, cb = atoms[i], atoms[j]
            if ca[0] == cb[0]:
                continue
            if ca[0] != "A":
                ca, cb = cb, ca
            key = (ca[0], ca[1], ca[2], cb[0], cb[1], cb[2])
            entry = aggregate_pairs[key]
            entry[slot] += 1
            entry[3] = min(entry[3], float(np.linalg.norm(ca[3] - cb[3])))
    rows = []
    for key, (hb, salt, hydro, distance) in aggregate_pairs.items():
        score = params["w_hbond"] * hb + params["w_salt"] * salt + params["w_hydro"] * hydro
        rows.append([name, *key, hb, salt, hydro, round(distance, 3), round(score, 3)])
    return rows


def process(path):
    name = Path(path).stem
    source = source_style(name)[2]
    model = PDBParser(QUIET=True).get_structure("s", str(path))[0]
    if "A" not in model or "B" not in model:
        raise ValueError(f"{name}: missing chain A or B")
    crow, rows = measure(name, source, model["A"], model["B"])
    pair_rows = pymol_pairs(path, name)
    message = (
        f"dASA={crow[CIDX['dASA_A2']]:.1f}A2 "
        f"n_contacts={crow[CIDX['n_contacts']]} "
        f"min_dist={crow[CIDX['min_dist_A']]:.2f}A "
        f"clash={crow[CIDX['clash']]} "
        f"interface_residues={len(rows)} pymol_pairs={len(pair_rows)}"
    )
    return name, crow, rows, pair_rows, message


###################################### ensemble aggregation
def occ_by_source(names, meta, sources, source_counts):
    return {
        source: sum(1 for name in names if meta[name][0] == source) / source_counts[source]
        for source in sources
    }


def aggregate(rows_c, rows_r, rows_p):
    meta = {row[0]: (row[CIDX["source"]], row[CIDX["clash"]]) for row in rows_c}
    included = set(meta)
    n_models = len(included)
    n_excluded = 0
    if n_models == 0:
        return [], [], [], 0, 0
    sources = sorted({meta[name][0] for name in included})
    source_counts = {
        source: sum(1 for name in included if meta[name][0] == source)
        for source in sources
    }
    rows_r = [row for row in rows_r if row[0] in included]
    rows_p = [row for row in rows_p if row[0] in included]
    relative_burial = defaultdict(list)
    for row in rows_r:
        value = row[RIDX["rel_dSASA"]]
        if np.isfinite(value):
            relative_burial[(row[1], str(row[2]), row[3])].append(float(value))

    def burial(key):
        values = relative_burial.get(key)
        return float(np.mean(values)) if values else float("nan")

    residue_models = defaultdict(lambda: defaultdict(set))
    pair_models = defaultdict(lambda: defaultdict(set))
    pair_scores = defaultdict(list)
    for name, ca, ria, rna, cb, rib, rnb, hb, salt, hydro, distance, score in rows_p:
        ka, kb = (ca, ria, rna), (cb, rib, rnb)
        kp = (ca, ria, rna, cb, rib, rnb)
        for interaction, count in zip(TYPES, (hb, salt, hydro)):
            if count:
                residue_models[ka][interaction].add(name)
                residue_models[kb][interaction].add(name)
                pair_models[kp][interaction].add(name)
        residue_models[ka]["any"].add(name)
        residue_models[kb]["any"].add(name)
        pair_models[kp]["any"].add(name)
        pair_scores[kp].append(score)

    residue_records = []
    for key, model_sets in residue_models.items():
        occupancy = {kind: len(model_sets[kind]) / n_models for kind in TYPES}
        dominant = max(TYPES, key=lambda kind: occupancy[kind])
        residue_records.append({
            "key": key, "occ": occupancy,
            "any": len(model_sets["any"]) / n_models,
            "n": len(model_sets["any"]),
            "by_src": occ_by_source(model_sets["any"], meta, sources, source_counts),
            "burial": burial(key),
            "dominant": dominant if occupancy[dominant] > 0 else "none",
        })
    residue_records.sort(key=lambda record: (-record["any"], -_nz(record["burial"])))
    pair_records = []
    occ_cutoff, burial_cutoff = params["occ_cutoff"], params["burial_cutoff"]
    for key, model_sets in pair_models.items():
        ka, kb = key[:3], key[3:]
        occupancy = {kind: len(model_sets[kind]) / n_models for kind in TYPES}
        dominant = max(TYPES, key=lambda kind: occupancy[kind])
        values = [burial(k) for k in (ka, kb) if np.isfinite(burial(k))]
        mean_burial = float(np.mean(values)) if values else float("nan")
        any_occ = len(model_sets["any"]) / n_models
        pair_records.append({
            "key": key, "occ": occupancy, "any": any_occ,
            "n": len(model_sets["any"]),
            "by_src": occ_by_source(model_sets["any"], meta, sources, source_counts),
            "score": float(np.mean(pair_scores[key])), "burial": mean_burial,
            "dominant": dominant if occupancy[dominant] > 0 else "none",
            "stable": bool(
                any_occ >= occ_cutoff and np.isfinite(mean_burial)
                and mean_burial >= burial_cutoff
            ),
        })
    pair_records.sort(key=lambda record: (-record["any"], -record["score"]))
    return residue_records, pair_records, sources, n_models, n_excluded


###################################### interface pipeline
def run_interfaces(files, in_name, out_dir, hscores):
    nproc = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    workers = params["workers"] or nproc
    print(
        f"[PYTHON-INFO] interfaces: workers={workers} "
        f"contact_cutoff={params['contact_cutoff']}A "
        f"hbond_cutoff={params['hbond_cutoff']}A "
        f"salt_cutoff={params['salt_cutoff']}A "
        f"hydro_cutoff={params['hydro_cutoff']}A "
        f"clash_cutoff={params['clash_cutoff']}A; all clashes retained"
    )
    rows_c, rows_r, rows_p = [], [], []
    with multiprocessing.Pool(workers, initializer=init_pymol) as pool:
        for name, crow, residue_rows, pair_rows, message in tqdm(
            pool.imap_unordered(process, files, chunksize=1),
            total=len(files), desc="interfaces",
        ):
            rows_c.append(crow)
            rows_r.extend(residue_rows)
            rows_p.extend(pair_rows)
            print(f"[PYTHON-INFO] {name}: {message}")
    for row in rows_c:
        row.append(fin4(hscores.get(row[0], float("nan"))))
    rows_c.sort(key=lambda row: row[0])
    rows_r.sort(key=lambda row: (row[0], row[1], row[2]))
    rows_p.sort(key=lambda row: (row[0], -row[-1], row[1], row[2], row[4], row[5]))
    write_tsv(out_dir / f"{in_name}_complex_scores.tsv", COMPLEX_COLS, rows_c)
    write_tsv(out_dir / f"{in_name}_residue_scores.tsv", RESIDUE_COLS, rows_r)
    write_tsv(out_dir / f"{in_name}_pymol_pairs.tsv", PYMOL_PAIR_COLS, rows_p)
    print(
        f"[PYTHON-INFO] wrote {len(rows_c)} complex rows, "
        f"{len(rows_r)} residue rows, {len(rows_p)} pymol pair rows"
    )
    residue_records, pair_records, sources, n_models, n_excluded = aggregate(rows_c, rows_r, rows_p)
    if n_models == 0:
        print("[PYTHON-INFO] no successfully processed models; skipping frequencies and summary")
        return
    residue_cols = RES_FREQ_BASE + [f"occ_any_{source}" for source in sources] + ["of_models"]
    pair_cols = PAIR_FREQ_BASE + [f"occ_any_{source}" for source in sources] + ["of_models"]
    write_tsv(
        out_dir / f"{in_name}_residue_frequencies.tsv", residue_cols,
        [[
            *record["key"], record["n"], fin4(record["occ"]["hbond"]),
            fin4(record["occ"]["salt"]), fin4(record["occ"]["hydro"]),
            fin4(record["any"]), fin4(record["burial"]), record["dominant"],
            *[fin4(record["by_src"][source]) for source in sources], n_models,
        ] for record in residue_records],
    )
    write_tsv(
        out_dir / f"{in_name}_pair_frequencies.tsv", pair_cols,
        [[
            *record["key"], record["n"], fin4(record["occ"]["hbond"]),
            fin4(record["occ"]["salt"]), fin4(record["occ"]["hydro"]),
            fin4(record["any"]), round(record["score"], 3),
            fin4(record["burial"]), record["dominant"], int(record["stable"]),
            *[fin4(record["by_src"][source]) for source in sources], n_models,
        ] for record in pair_records],
    )
    n_clashing = sum(int(row[CIDX["clash"]]) for row in rows_c)
    print(
        f"[PYTHON-INFO] {len(residue_records)} interface residues, "
        f"{len(pair_records)} residue pairs, "
        f"{sum(record['stable'] for record in pair_records)} stabilizing pairs, "
        f"{n_models} models used, {n_clashing} clashing models retained"
    )
    if params["plots"]:
        plot_summary(residue_records, pair_records, n_models, 0, in_name, out_dir)


###################################### summary figures
def _empty(ax, message):
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=7, color="0.4")
    ax.set_xticks([])
    ax.set_yticks([])


def _type_handles(extra_any=False):
    handles = [
        Line2D([], [], marker=TYPE_MARKER[kind], color=TYPE_COLOR[kind],
               ls="", markersize=5, label=TYPE_LABEL[kind])
        for kind in TYPES
    ]
    if extra_any:
        handles.append(Line2D([], [], color="0.85", lw=5, label="any type"))
    return handles


def lab(key):
    return f"{key[0]}{key[1]} {key[2]}"


def pair_label(key):
    return f"{key[0]}{key[1]} {key[2]} – {key[3]}{key[4]} {key[5]}"


def plot_residues(records, n_included, out_path):
    selected = records[:params["top_residues"]]
    fig = plt.figure(figsize=(4, 4))
    grid = fig.add_gridspec(1, 2, width_ratios=[3, 1], left=0.30, right=0.97, top=0.85, bottom=0.07, wspace=0.06)
    ax_heat = fig.add_subplot(grid[0])
    ax_burial = fig.add_subplot(grid[1], sharey=ax_heat)
    if selected:
        matrix = np.array([[record["occ"][kind] for kind in TYPES] for record in selected])
        ax_heat.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax_heat.text(j, i, f"{value:.0%}" if value > 0 else "·", ha="center", va="center", fontsize=5, color="white" if value > 0.6 else "0.15")
        ax_heat.set_yticks(range(len(selected)))
        ax_heat.set_yticklabels([lab(record["key"]) for record in selected], fontsize=5.5)
        for tick, record in zip(ax_heat.get_yticklabels(), selected):
            tick.set_color("#222222" if record["key"][0] == "A" else "#7d3c98")
        ax_heat.set_xticks(range(len(TYPES)))
        ax_heat.set_xticklabels([TYPE_LABEL[kind] for kind in TYPES], fontsize=5.5, fontweight="bold")
        for kind, tick in zip(TYPES, ax_heat.get_xticklabels()):
            tick.set_color(TYPE_COLOR[kind])
        ax_heat.xaxis.tick_top()
        burial = [record["burial"] if np.isfinite(record["burial"]) else 0.0 for record in selected]
        ax_burial.barh(range(len(selected)), burial, height=0.7, color="0.6")
        ax_burial.axvline(params["burial_cutoff"], ls="--", color="0.2", lw=0.7)
        ax_burial.set_xlim(0, 1)
        ax_burial.set_xticks([0, 0.5, 1])
        ax_burial.tick_params(axis="x", labelsize=5, labeltop=True, labelbottom=False, top=True, bottom=False)
        ax_burial.tick_params(axis="y", left=False, labelleft=False)
        ax_burial.set_xlabel("rel. burial", fontsize=5.5, labelpad=2)
        ax_burial.xaxis.set_label_position("top")
    else:
        _empty(ax_heat, "no inter-chain interactions")
        ax_burial.set_axis_off()
    fig.suptitle(f"A. Residue occupancy ({n_included} models)", fontsize=7, x=0.02, ha="left", y=0.975)
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"[PYTHON-INFO] wrote {out_path}")


def plot_pairs(records, n_included, out_path):
    selected = records[:params["top_pairs"]]
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(left=0.44, right=0.95, top=0.80, bottom=0.12)
    if selected:
        ys = np.arange(len(selected))[::-1]
        offsets = {"hbond": 0.19, "salt": 0.0, "hydro": -0.19}
        for record, y in zip(selected, ys):
            ax.barh(y, record["any"], height=0.62, color="0.88", zorder=1)
            for kind in TYPES:
                if record["occ"][kind] > 0:
                    ax.barh(y + offsets[kind], record["occ"][kind], height=0.15, color=TYPE_COLOR[kind], zorder=2)
            ax.text(record["any"] + 0.015, y, f"{record['any']:.0%}", va="center", fontsize=5)
        ax.set_yticks(ys)
        ax.set_yticklabels([pair_label(record["key"]) for record in selected], fontsize=5)
        for tick, record in zip(ax.get_yticklabels(), selected):
            if record["stable"]:
                tick.set_fontweight("bold")
        ax.set_ylim(-0.6, len(selected) - 0.4)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.tick_params(labelsize=5)
        ax.set_xlabel("occupancy (fraction of models)", fontsize=6)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        ax.legend(handles=_type_handles(extra_any=True), fontsize=5, ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01))
    else:
        _empty(ax, "no residue pairs detected")
    fig.suptitle(f"B. Residue pairs ({n_included} models)", fontsize=7, x=0.02, ha="left", y=0.975)
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"[PYTHON-INFO] wrote {out_path}")


def plot_scatter(records, n_included, out_path):
    occ_cutoff, burial_cutoff = params["occ_cutoff"], params["burial_cutoff"]
    points = [record for record in records if np.isfinite(record["burial"])]
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(left=0.17, right=0.96, top=0.85, bottom=0.14)
    if points:
        rng = np.random.default_rng(params["seed"])
        ymax = max(0.5, max(record["burial"] for record in points)) + 0.08
        xmax = 1.3
        ax.add_patch(Rectangle((occ_cutoff, burial_cutoff), xmax - occ_cutoff, ymax - burial_cutoff, color="#2e7d32", alpha=0.08, lw=0, zorder=0))
        ax.axvline(occ_cutoff, ls="--", color="0.4", lw=0.7)
        ax.axhline(burial_cutoff, ls="--", color="0.4", lw=0.7)
        for record in points:
            x = record["any"] + rng.uniform(-0.012, 0.012)
            y = record["burial"] + rng.uniform(-0.004, 0.004)
            ax.scatter(x, y, s=22, marker=TYPE_MARKER.get(record["dominant"], "x"), color=TYPE_COLOR.get(record["dominant"], "0.5"), edgecolor="white", lw=0.3, alpha=0.9, zorder=3)
        stable = sorted((record for record in points if record["any"] >= occ_cutoff and record["burial"] >= burial_cutoff), key=lambda record: (-record["any"], -record["burial"]))
        for record in stable[:8]:
            ax.annotate(lab(record["key"]), (record["any"], record["burial"]), xytext=(4, 2), textcoords="offset points", fontsize=5)
        ax.set_xlim(-0.05, xmax)
        ax.set_ylim(0, ymax)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel("occupancy of any interaction", fontsize=6)
        ax.set_ylabel("mean rel. dSASA (burial)", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=_type_handles(), fontsize=5, frameon=False, loc="upper left")
    else:
        _empty(ax, "no burial data")
    fig.suptitle(f"C. Residues: frequent and buried ({n_included} models)", fontsize=7, x=0.02, ha="left", y=0.975)
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"[PYTHON-INFO] wrote {out_path}")


def plot_summary(residue_records, pair_records, n_included, n_excluded, name, out_dir):
    label = str(n_included)
    plot_residues(residue_records, label, out_dir / f"{name}_A_residue_occupancy.svg")
    plot_pairs(pair_records, label, out_dir / f"{name}_B_pair_occupancy.svg")
    plot_scatter(residue_records, label, out_dir / f"{name}_C_burial_vs_occupancy.svg")


###################################### notebook viewer
def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def show_notebook(entries):
    import py3Dmol
    import ipywidgets as widgets
    from IPython.display import display

    view = py3Dmol.view(width=800, height=600)
    view.setBackgroundColor("white")
    for entry in entries:
        view.addModel(entry["cif"], "cif")
    state, checkboxes = {"sync": False}, []
    master = None

    def apply(index):
        entry = entries[index]
        if checkboxes[index].value:
            view.setStyle({"model": index, "chain": "A"}, {"cartoon": {"color": entry["colorA"]}})
            view.setStyle({"model": index, "chain": "B"}, {"cartoon": {"color": entry["colorB"]}})
        else:
            view.setStyle({"model": index}, {})

    def set_all(on):
        state["sync"] = True
        for index, checkbox in enumerate(checkboxes):
            checkbox.value = on
            apply(index)
        master.value = on
        state["sync"] = False
        view.update()

    def sync_master():
        state["sync"] = True
        master.value = all(checkbox.value for checkbox in checkboxes)
        state["sync"] = False

    def on_model(index):
        def handler(change):
            if state["sync"]:
                return
            apply(index)
            sync_master()
            view.update()
        return handler

    def on_master(change):
        if not state["sync"]:
            set_all(change["new"])

    checkboxes = [
        widgets.Checkbox(value=True, description=f"{i}: {entry['name']}", indent=False, layout=widgets.Layout(width="auto"))
        for i, entry in enumerate(entries)
    ]
    master = widgets.Checkbox(value=True, description="Select all / none", indent=False)
    for index, checkbox in enumerate(checkboxes):
        checkbox.observe(on_model(index), names="value")
    master.observe(on_master, names="value")
    for index in range(len(entries)):
        apply(index)

    reset_button = widgets.Button(description="Reset view & labels", button_style="info", layout=widgets.Layout(width="48%"))
    reset_button.on_click(lambda _: (view.removeAllLabels(), view.zoomTo(), view.update()))
    show_button = widgets.Button(description="Show all models", button_style="success", layout=widgets.Layout(width="48%"))
    show_button.on_click(lambda _: set_all(True))
    display(widgets.VBox([widgets.HBox([reset_button, show_button]), master] + checkboxes))
    view.setClickable({}, True, """
        function(atom, viewer) {
            viewer.removeAllLabels();
            viewer.addLabel('chain ' + atom.chain + ' | ' + atom.resn + ' ' + atom.resi,
            {position: atom, backgroundColor: 'black', fontColor: 'white', fontSize: 12});
            viewer.render();
        }
    """)
    view.zoomTo()
    view.show()


###################################### main
def main(argv):
    global params
    start = time.time()
    args = parse_args(argv)
    config = {}
    if args.yaml_in:
        with open(args.yaml_in) as handle:
            config = (yaml.safe_load(handle) or {}).get(NAME, {})
    params = {**DEFAULTS, **config}
    params["exclude_clashes"] = False
    for key in ("input", "output", "ref", "align_chain", "workers", "capri_tsv"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    if args.no_html:
        params["html"] = False
    if args.no_interfaces:
        params["interfaces"] = False
    if not params["input"]:
        raise SystemExit("[PYTHON-INFO] no input provided")
    np.random.seed(params["seed"])
    input_dir = Path(params["input"])
    files = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"[PYTHON-INFO] no .pdb files found in {input_dir}")
    name = input_dir.resolve().name
    out_dir = Path(params["output"] or Path("data/pdbs_by_group") / name / "aligned")
    out_dir.mkdir(parents=True, exist_ok=True)

    haddock_rows = load_haddock(params["capri_tsv"]) if params["capri_tsv"] else []
    scores = match_haddock(files, haddock_rows, name) if haddock_rows else {}
    if haddock_rows and params["haddock_plots"]:
        plot_haddock_scores(haddock_rows, out_dir / "haddock_scores", {name: [path.name for path in files]})

    stems = [path.stem for path in files]
    if params["ref"] and params["ref"] not in stems:
        raise SystemExit(f"[PYTHON-INFO] reference '{params['ref']}' not found in input")
    ref_idx = stems.index(params["ref"]) if params["ref"] else 0
    print(
        f"[PYTHON-INFO] structures={len(files)} reference={stems[ref_idx]} "
        f"align_chain={params['align_chain']} haddock_matched={sum(stem in scores for stem in stems)} "
        f"exclude_clashes=False"
    )

    parser = PDBParser(QUIET=True)
    models = [parser.get_structure(path.stem, str(path))[0] for path in files]
    rmsd, nmatch = align(models, params["align_chain"], ref_idx)
    cif_path = out_dir / f"{name}_aligned.cif"
    write_cif(ensemble_structure(models), str(cif_path))
    manifest_path = out_dir / f"{name}_manifest.tsv"
    entries = []
    with open(manifest_path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(MANIFEST_COLS)
        for index, (path, model) in enumerate(zip(files, models)):
            color_a, color_b, label = source_style(path.stem)
            score = fin4(scores.get(path.stem, float("nan")))
            writer.writerow([
                index, path.stem, label, color_a, color_b, nmatch[index],
                "" if rmsd[index] is None else f"{rmsd[index]:.3f}", score,
            ])
            entries.append({
                "name": path.stem, "colorA": color_a, "colorB": color_b,
                "cif": model_cif_text(model),
            })
    print(f"[PYTHON-INFO] wrote multi-model mmCIF {cif_path} ({len(models)} models)")
    print(f"[PYTHON-INFO] wrote manifest {manifest_path}")

    if params["html"]:
        library_js = load_3dmol(args.threedmol_js)
        html_path = out_dir / f"{name}_viewer.html"
        write_html(entries, name, html_path, library_js)
        print(f"[PYTHON-INFO] wrote standalone viewer {html_path} ({html_path.stat().st_size / 1e6:.1f} MB)")

    if params["interfaces"]:
        run_interfaces(files, name, out_dir, scores)

    if args.yaml_out:
        yaml_path = Path(args.yaml_out)
        existing = {}
        if yaml_path.exists():
            with open(yaml_path) as handle:
                existing = yaml.safe_load(handle) or {}
        existing[NAME] = params
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as handle:
            yaml.safe_dump(existing, handle, sort_keys=False)

    if in_notebook():
        show_notebook(entries)

    elapsed = time.time() - start
    maxmem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[PYTHON-INFO] {datetime.datetime.now():%Y-%m-%d %H:%M:%S} exec={elapsed:.2f}s maxmem={maxmem:.1f}MB")


if __name__ == "__main__":
    main([] if in_notebook() else sys.argv[1:])


