#!/usr/bin/env python3
"""Align an ensemble of .pdb models on chain-A C-alpha atoms, export them for 3D viewing,
and quantify inter-chain interfaces (H-bonds, salt bridges, hydrophobic contacts) with
per-model and ensemble-level occupancy.

Outputs (in --output):
  <name>_aligned.cif              multi-model mmCIF (PyMOL, ChimeraX, Mol*). Model order = manifest "order".
  <name>_viewer.html              self-contained viewer: structures and 3Dmol.js are embedded.
  <name>_manifest.tsv             order, source file, source class, colors, matched CA count, CA RMSD.
  <name>_complex_scores.tsv       per model: burial, contacts, min distance, source, clash flag.
  <name>_residue_scores.tsv       per model and interface residue: burial, contacts, energy, partners.
  <name>_pymol_pairs.tsv          per model and residue pair: H-bond, salt-bridge, hydrophobic counts.
  <name>_residue_frequencies.tsv  ensemble occupancy per residue, by type and by source class.
  <name>_pair_frequencies.tsv     ensemble occupancy per residue pair, by type and by source class.
  <name>_summary.svg              3-panel summary of persistent interactions.

Models flagged as clashing (min heavy-atom inter-chain distance < clash_cutoff) are written to
the per-model tables but excluded from the frequencies unless exclude_clashes is false.
Interface analysis reads the original .pdb files, so PyMOL must be installed.

Args:
    --input:        directory containing .pdb models with chains A and B.
    --output:       output directory (default data/pdbs_by_group/<name>/aligned).
    --ref:          stem of the reference model (default: first file in sorted order).
    --align-chain:  chain used for alignment (default A).
    --workers:      processes for the interface analysis (default: all CPUs).
    --3dmol-js:     path to a local 3Dmol-min.js (skips the download).
    --no-html:      skip the standalone HTML viewer.
    --no-interfaces skip the PyMOL interface analysis.
    --yaml-in:      optional yaml with parameters nested under align_and_export.
    --yaml-out:     yaml file where the parameters are stored (default configs/param_configs.yaml).
"""
###################################### dependencies
import argparse, csv, datetime, io, json, multiprocessing, os, re, resource, sys, time, urllib.request
from collections import defaultdict
from html import escape
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
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
    "input": None, "output": None, "ref": None, "align_chain": "A", "workers": None, "seed": 42,
    "html": True, "interfaces": True, "plots": True,
    "contact_cutoff": 8, "score_cutoff": 12.0, "block": 1000,
    "hbond_cutoff": 3.5, "salt_cutoff": 4.0, "hydro_cutoff": 4.5,
    "w_hbond": 1.0, "w_salt": 1.5, "w_hydro": 0.1,
    "clash_cutoff": 2.5, "exclude_clashes": True,
    "occ_cutoff": 0.5, "burial_cutoff": 0.25,
    "top_residues": 15, "top_pairs": 12, "panel_size": 4,
}
THREEDMOL_URLS = ("https://3dmol.org/build/3Dmol-min.js",
                  "https://3dmol.csb.pitt.edu/build/3Dmol-min.js")
# (filename keyword, chain A color, chain B color, source class)
COLOR_RULES = (
    ("abin", "salmon", "teal", "abin"),
    ("model", "grey", "black", "model"),
    ("pdbeval", "red", "blue", "pdbeval"),
)
DEFAULT_STYLE = ("orange", "purple", "other")
HYDRO = {"A":0.62,"R":-2.53,"N":-0.78,"D":-0.90,"C":0.29,"Q":-0.85,"E":-0.74,"G":0.48,"H":-0.40,"I":1.38,"L":1.06,"K":-1.50,"M":0.64,"F":1.19,"P":0.12,"S":-0.18,"T":-0.05,"W":0.81,"Y":0.26,"V":1.08}
TO1 = {k.upper(): v for k, v in protein_letters_3to1_extended.items()}
TIEN = {"ALA":129,"ARG":274,"ASN":195,"ASP":193,"CYS":167,"GLN":225,"GLU":223,"GLY":104,"HIS":224,"ILE":197,"LEU":201,"LYS":236,"MET":224,"PHE":240,"PRO":159,"SER":155,"THR":172,"TRP":285,"TYR":263,"VAL":174}
TYPES = ("hbond", "salt", "hydro")
TYPE_LABEL = {"hbond": "H-bond", "salt": "Salt bridge", "hydro": "Hydrophobic"}
TYPE_COLOR = {"hbond": "#1f5fa8", "salt": "#c0392b", "hydro": "#d4a017"}
TYPE_MARKER = {"hbond": "o", "salt": "^", "hydro": "s"}
POS_SEL = "(resn LYS and name NZ) or (resn ARG and name NH1+NH2+NE)"
NEG_SEL = "(resn ASP and name OD1+OD2) or (resn GLU and name OE1+OE2)"
HYD_SEL = "(resn ALA+VAL+ILE+LEU+MET+PHE+TRP+PRO+TYR+CYS) and elem C and not name N+C+O+CA"
MANIFEST_COLS = ["order", "structure", "source", "color_A", "color_B", "n_CA_matched", "CA_rmsd_A"]
COMPLEX_COLS = ["structure", "source", "clash", "len_res_a", "len_res_b", "len_atoms_a", "len_atoms_b",
                "dASA_A2", "initial_sasa_A2", "frac_dASA", "n_contacts", "iface_res_a", "iface_res_b",
                "min_dist_A", "contact_score"]
RESIDUE_COLS = ["structure", "chain", "resid", "resname", "dSASA_A2", "isolated_sasa_A2", "rel_dSASA",
                "n_partners", "min_dist_A", "contact_energy", "energy_per_partner", "partners"]
PYMOL_PAIR_COLS = ["structure", "chain_a", "resid_a", "resname_a", "chain_b", "resid_b", "resname_b",
                   "n_hbond", "n_salt_bridge", "n_hydrophobic_contacts", "min_dist_A", "interaction_score"]
RES_FREQ_BASE = ["chain", "resid", "resname", "n_models", "occ_hbond", "occ_salt", "occ_hydro", "occ_any",
                 "mean_rel_dsasa", "dominant"]
PAIR_FREQ_BASE = ["chain_a", "resid_a", "resname_a", "chain_b", "resid_b", "resname_b", "n_models",
                  "occ_hbond", "occ_salt", "occ_hydro", "occ_any", "mean_score", "mean_rel_dsasa",
                  "dominant", "stabilizing"]
CIDX = {c: i for i, c in enumerate(COMPLEX_COLS)}
RIDX = {c: i for i, c in enumerate(RESIDUE_COLS)}
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>__TITLE__</title>
<script>__LIBRARY__</script>
<style>
  html, body { margin: 0; height: 100%; font-family: sans-serif; }
  body { display: flex; }
  #panel { width: 300px; min-width: 260px; overflow-y: auto; padding: 12px; box-sizing: border-box; border-right: 1px solid #ccc; font-size: 13px; }
  #panel h3 { margin: 0 0 8px 0; font-size: 15px; word-break: break-all; }
  #viewer { flex: 1; position: relative; }
  label { display: block; margin: 4px 0; cursor: pointer; }
  .sw { display: inline-block; width: 10px; height: 10px; margin-left: 4px; border: 1px solid #666; vertical-align: middle; }
  button { margin: 2px 4px 6px 0; padding: 4px 8px; }
  .note { color: #555; font-size: 12px; margin-top: 12px; }
</style></head>
<body>
<div id="panel">
  <h3>__TITLE__</h3>
  <button id="showAll">Show all</button><button id="hideAll">Hide all</button><button id="resetView">Reset view</button>
  <div id="list"></div>
  <div class="note">Swatches: chain A, chain B. Click an atom to label it. Models are aligned on chain A C-alpha atoms.</div>
</div>
<div id="viewer"></div>
<script>
const MODELS = __MODELS__;
const viewer = $3Dmol.createViewer(document.getElementById("viewer"), { backgroundColor: "white" });
MODELS.forEach(m => viewer.addModel(m.cif, "cif"));

const shown = MODELS.map(() => true);
const boxes = [];

function applyModel(i) {
  const m = MODELS[i];
  if (shown[i]) {
    viewer.setStyle({ model: i, chain: "A" }, { cartoon: { color: m.colorA } });
    viewer.setStyle({ model: i, chain: "B" }, { cartoon: { color: m.colorB } });
  } else {
    viewer.setStyle({ model: i }, {});
  }
}

function setAll(on) {
  MODELS.forEach((m, i) => { shown[i] = on; boxes[i].checked = on; applyModel(i); });
  viewer.render();
}

function swatch(c) {
  const s = document.createElement("span");
  s.className = "sw";
  s.style.background = c;
  return s;
}

const list = document.getElementById("list");
MODELS.forEach((m, i) => {
  const label = document.createElement("label");
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = true;
  box.addEventListener("change", () => { shown[i] = box.checked; applyModel(i); viewer.render(); });
  boxes.push(box);
  label.appendChild(box);
  label.appendChild(document.createTextNode(" " + i + ": " + m.name));
  label.appendChild(swatch(m.colorA));
  label.appendChild(swatch(m.colorB));
  list.appendChild(label);
});

viewer.setClickable({}, true, function (atom, v, event, container) {
  v.removeAllLabels();
  const m = MODELS[atom.model];
  v.addLabel("Model " + atom.model + " (" + m.name + ") | chain " + atom.chain + " | " + atom.resn + " " + atom.resi,
             { position: atom, backgroundColor: "black", fontColor: "white", fontSize: 12 });
  v.render();
});

document.getElementById("showAll").onclick = () => setAll(true);
document.getElementById("hideAll").onclick = () => setAll(false);
document.getElementById("resetView").onclick = () => { viewer.removeAllLabels(); viewer.zoomTo(); viewer.render(); };

MODELS.forEach((m, i) => applyModel(i));
viewer.zoomTo();
viewer.render();
</script>
</body></html>
"""
CMD = None
params = {}
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
    p.add_argument("--workers", type=int)
    p.add_argument("--3dmol-js", dest="threedmol_js", default=None,
                   help="path to a local 3Dmol-min.js (skips the download)")
    p.add_argument("--no-html", action="store_true")
    p.add_argument("--no-interfaces", action="store_true")
    p.add_argument("--yaml-in")
    p.add_argument("--yaml-out", default="configs/param_configs.yaml")
    args, _ = p.parse_known_args(argv)  # tolerate Jupyter kernel arguments
    return args

def ca_map(model, chain_id):
    """C-alpha atoms of standard residues, keyed by (resseq, insertion code)."""
    if chain_id not in model:
        return {}
    return {res.id[1:]: res["CA"] for res in model[chain_id] if res.id[0] == " " and "CA" in res}

def align(models, chain_id, ref_idx):
    """Superimpose every model onto the reference using matched CA atoms on chain_id."""
    ref_ca = ca_map(models[ref_idx], chain_id)
    rmsd = [None] * len(models)
    nmatch = [0] * len(models)
    for i, m in enumerate(models):
        if i == ref_idx:
            rmsd[i], nmatch[i] = 0.0, len(ref_ca)
            continue
        mob = ca_map(m, chain_id)
        keys = sorted(ref_ca.keys() & mob.keys())
        nmatch[i] = len(keys)
        if len(keys) < 3:
            print(f"[PYTHON-INFO] warning: model {i} has {len(keys)} matched CA atoms on chain {chain_id}; not aligned")
            continue
        sup = Superimposer()
        sup.set_atoms([ref_ca[k] for k in keys], [mob[k] for k in keys])
        sup.apply(list(m.get_atoms()))
        rmsd[i] = float(sup.rms)
    return rmsd, nmatch

def write_cif(structure, handle_or_path):
    w = MMCIFIO()
    w.set_structure(structure)
    w.save(handle_or_path)

def ensemble_structure(models):
    """One Structure holding each model as its own MODEL, numbered by input order."""
    s = Structure("ensemble")
    for i, m in enumerate(models):
        mc = m.copy()
        mc.id = i
        s.add(mc)
    return s

def model_cif_text(model):
    s = Structure("m")
    s.add(model.copy())
    buf = io.StringIO()
    write_cif(s, buf)
    return buf.getvalue()

def load_3dmol(local_path=None):
    """Return the text of 3Dmol-min.js: from --3dmol-js, the cache, or a one-time download."""
    if local_path:
        return Path(local_path).read_text(encoding="utf-8")
    cache = Path.home() / ".cache" / "aligned_export" / "3Dmol-min.js"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    last_err = None
    for url in THREEDMOL_URLS:
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                text = r.read().decode("utf-8")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
            print(f"[PYTHON-INFO] cached 3Dmol.js from {url}")
            return text
        except Exception as e:
            last_err = e
    raise SystemExit(f"[PYTHON-INFO] could not get 3Dmol-min.js ({last_err}). "
                     "Download it manually and pass --3dmol-js PATH")

def inline_js(js):
    """Stop the library text from closing the surrounding <script> tag."""
    return re.sub(r"</(script)", r"<\\/\1", js, flags=re.IGNORECASE)

def write_html(entries, title, path, library_js):
    models_json = json.dumps(entries).replace("</", "<\\/")
    text = (HTML_TEMPLATE
            .replace("__TITLE__", escape(title))
            .replace("__MODELS__", models_json)
            .replace("__LIBRARY__", inline_js(library_js)))
    Path(path).write_text(text, encoding="utf-8")

def write_tsv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t"); w.writerow(cols); w.writerows(rows)

def fin4(x):
    return round(float(x), 4) if np.isfinite(x) else ""

def _nz(x):
    return x if np.isfinite(x) else -1.0

###################################### per-model interface measurements
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

def measure(name, source, A, B):
    ra, rb = std(A), std(B)
    xa, oa = heavy(ra); xb, ob = heavy(rb)
    if not len(xa) or not len(xb):
        raise ValueError(f"{name}: no heavy atoms in chain A or B")
    sa, sb, sab = sasa([A]), sasa([B]), sasa([A, B])
    dasa, init = sa + sb - sab, sa + sb
    frac = dasa / init if init > 0 else 0.0
    n, score, dmin_c, ha, hb = complex_stats(xa, xb)
    clash = int(dmin_c < params["clash_cutoff"])
    crow = [name, source, clash, len(ra), len(rb), len(xa), len(xb), round(dasa, 2), round(init, 2),
            round(frac, 4), n, len({int(oa[k]) for k in np.where(ha)[0]}),
            len({int(ob[k]) for k in np.where(hb)[0]}), round(dmin_c, 3), round(score, 4)]
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
            rows.append([name, cid, r.id[1], r.get_resname(), round(float(dsasa), 2), round(float(iso[key]), 2),
                         round(float(rel), 4), int(npart[k]), round(float(dm[k]), 3), round(en, 4),
                         round(en / npart[k], 4), ";".join(plist)])
    return crow, rows

###################################### PyMOL interaction typing
def init_pymol():
    global CMD
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    CMD = pymol.cmd

def pymol_pairs(path, name):
    """One row per inter-chain residue pair, with per-type atom-pair counts."""
    cmd = CMD
    cmd.reinitialize()
    cmd.load(str(path), "s")
    allp = "s and polymer and (chain A or chain B)"
    A, B = f"{allp} and chain A", f"{allp} and chain B"
    atoms = {a.index: (a.chain, a.resi, a.resn, np.array(a.coord, float)) for a in cmd.get_model(allp).atom}
    # slot -> type. mode=1 is PyMOL's donor/acceptor H-bond criterion; mode=0 is plain distance.
    groups = {
        0: cmd.find_pairs(A, B, mode=1, cutoff=params["hbond_cutoff"]),
        1: cmd.find_pairs(f"{allp} and ({POS_SEL})", f"{allp} and ({NEG_SEL})", mode=0, cutoff=params["salt_cutoff"]),
        2: cmd.find_pairs(f"{A} and ({HYD_SEL})", f"{B} and ({HYD_SEL})", mode=0, cutoff=params["hydro_cutoff"]),
    }
    agg = defaultdict(lambda: [0, 0, 0, np.inf])  # hb, salt, hydrophobic counts, min distance
    for slot, pairs in groups.items():
        for (_, i), (_, j) in pairs:
            ca, cb = atoms[i], atoms[j]
            if ca[0] == cb[0]:
                continue
            if ca[0] != "A":
                ca, cb = cb, ca
            key = (ca[0], ca[1], ca[2], cb[0], cb[1], cb[2])
            entry = agg[key]
            entry[slot] += 1
            entry[3] = min(entry[3], float(np.linalg.norm(ca[3] - cb[3])))
    rows = []
    for key, (hb, sb, hy, dm) in agg.items():
        score = params["w_hbond"] * hb + params["w_salt"] * sb + params["w_hydro"] * hy
        rows.append([name, *key, hb, sb, hy, round(dm, 3), round(score, 3)])
    return rows

def process(path):
    name = Path(path).stem
    source = source_style(name)[2]
    model = PDBParser(QUIET=True).get_structure("s", str(path))[0]
    if "A" not in model or "B" not in model:
        raise ValueError(f"{name}: missing chain A or B")
    crow, rows = measure(name, source, model["A"], model["B"])
    prows = pymol_pairs(path, name)
    msg = (f"dASA={crow[CIDX['dASA_A2']]:.1f}A2 n_contacts={crow[CIDX['n_contacts']]} "
           f"min_dist={crow[CIDX['min_dist_A']]:.2f}A clash={crow[CIDX['clash']]} "
           f"interface_residues={len(rows)} pymol_pairs={len(prows)}")
    return name, crow, rows, prows, msg

###################################### ensemble aggregation
def occ_by_source(names, meta, sources, src_n):
    return {s: sum(1 for n in names if meta[n][0] == s) / src_n[s] for s in sources}

def aggregate(rows_c, rows_r, rows_p):
    meta = {r[0]: (r[CIDX["source"]], r[CIDX["clash"]]) for r in rows_c}
    included = {n for n, (_, c) in meta.items() if not (params["exclude_clashes"] and c)}
    N = len(included)
    n_excl = len(meta) - N
    sources = sorted({meta[n][0] for n in included})
    src_n = {s: sum(1 for n in included if meta[n][0] == s) for s in sources}
    rows_r = [r for r in rows_r if r[0] in included]
    rows_p = [r for r in rows_p if r[0] in included]

    rel = defaultdict(list)
    for r in rows_r:
        v = r[RIDX["rel_dSASA"]]
        if np.isfinite(v):
            rel[(r[1], str(r[2]), r[3])].append(float(v))
    def burial(key):
        v = rel.get(key)
        return float(np.mean(v)) if v else float("nan")

    res_m = defaultdict(lambda: defaultdict(set))
    pair_m = defaultdict(lambda: defaultdict(set))
    pair_s = defaultdict(list)
    for name, ca, ria, rna, cb, rib, rnb, hb, sb, hy, dm, score in rows_p:
        ka, kb, kp = (ca, ria, rna), (cb, rib, rnb), (ca, ria, rna, cb, rib, rnb)
        for t, cnt in zip(TYPES, (hb, sb, hy)):
            if cnt:
                res_m[ka][t].add(name); res_m[kb][t].add(name); pair_m[kp][t].add(name)
        res_m[ka]["any"].add(name); res_m[kb]["any"].add(name); pair_m[kp]["any"].add(name)
        pair_s[kp].append(score)

    res_recs = []
    for k, m in res_m.items():
        occ = {t: len(m[t]) / N for t in TYPES}
        top = max(TYPES, key=lambda t: occ[t])
        res_recs.append({"key": k, "occ": occ, "any": len(m["any"]) / N, "n": len(m["any"]),
                         "by_src": occ_by_source(m["any"], meta, sources, src_n),
                         "burial": burial(k), "dominant": top if occ[top] > 0 else "none"})
    res_recs.sort(key=lambda d: (-d["any"], -_nz(d["burial"])))

    pair_recs = []
    occ_c, bur_c = params["occ_cutoff"], params["burial_cutoff"]
    for k, m in pair_m.items():
        ka, kb = k[:3], k[3:]
        occ = {t: len(m[t]) / N for t in TYPES}
        top = max(TYPES, key=lambda t: occ[t])
        vals = [b for b in (burial(ka), burial(kb)) if np.isfinite(b)]
        mb = float(np.mean(vals)) if vals else float("nan")
        any_occ = len(m["any"]) / N
        pair_recs.append({"key": k, "occ": occ, "any": any_occ, "n": len(m["any"]),
                          "by_src": occ_by_source(m["any"], meta, sources, src_n),
                          "score": float(np.mean(pair_s[k])), "burial": mb,
                          "dominant": top if occ[top] > 0 else "none",
                          "stable": bool(any_occ >= occ_c and np.isfinite(mb) and mb >= bur_c)})
    pair_recs.sort(key=lambda d: (-d["any"], -d["score"]))
    return res_recs, pair_recs, sources, N, n_excl

###################################### interface pipeline
def run_interfaces(files, in_name, out_dir):
    nproc = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    workers = params["workers"] or nproc
    print(f"[PYTHON-INFO] interfaces: workers={workers} contact_cutoff={params['contact_cutoff']}A "
          f"hbond_cutoff={params['hbond_cutoff']}A salt_cutoff={params['salt_cutoff']}A "
          f"hydro_cutoff={params['hydro_cutoff']}A clash_cutoff={params['clash_cutoff']}A")
    rows_c, rows_r, rows_p = [], [], []
    with multiprocessing.Pool(workers, initializer=init_pymol) as pool:
        for name, crow, rrows, prow, msg in tqdm(pool.imap_unordered(process, files, chunksize=1),
                                                 total=len(files), desc="interfaces"):
            rows_c.append(crow); rows_r.extend(rrows); rows_p.extend(prow)
            print(f"[PYTHON-INFO] {name}: {msg}")
    rows_c.sort(key=lambda r: r[0])
    rows_r.sort(key=lambda r: (r[0], r[1], r[2]))
    rows_p.sort(key=lambda r: (r[0], -r[-1], r[1], r[2], r[4], r[5]))

    write_tsv(out_dir / f"{in_name}_complex_scores.tsv", COMPLEX_COLS, rows_c)
    write_tsv(out_dir / f"{in_name}_residue_scores.tsv", RESIDUE_COLS, rows_r)
    write_tsv(out_dir / f"{in_name}_pymol_pairs.tsv", PYMOL_PAIR_COLS, rows_p)
    print(f"[PYTHON-INFO] wrote {len(rows_c)} complex rows, {len(rows_r)} residue rows, "
          f"{len(rows_p)} pymol pair rows")

    res_recs, pair_recs, sources, N, n_excl = aggregate(rows_c, rows_r, rows_p)
    if N == 0:
        print("[PYTHON-INFO] no models left after clash exclusion; skipping frequencies and summary")
        return
    res_cols = RES_FREQ_BASE + [f"occ_any_{s}" for s in sources] + ["of_models"]
    pair_cols = PAIR_FREQ_BASE + [f"occ_any_{s}" for s in sources] + ["of_models"]
    write_tsv(out_dir / f"{in_name}_residue_frequencies.tsv", res_cols,
              [[*d["key"], d["n"], fin4(d["occ"]["hbond"]), fin4(d["occ"]["salt"]), fin4(d["occ"]["hydro"]),
                fin4(d["any"]), fin4(d["burial"]), d["dominant"],
                *[fin4(d["by_src"][s]) for s in sources], N] for d in res_recs])
    write_tsv(out_dir / f"{in_name}_pair_frequencies.tsv", pair_cols,
              [[*d["key"], d["n"], fin4(d["occ"]["hbond"]), fin4(d["occ"]["salt"]), fin4(d["occ"]["hydro"]),
                fin4(d["any"]), round(d["score"], 3), fin4(d["burial"]), d["dominant"], int(d["stable"]),
                *[fin4(d["by_src"][s]) for s in sources], N] for d in pair_recs])
    print(f"[PYTHON-INFO] {len(res_recs)} interface residues, {len(pair_recs)} residue pairs, "
          f"{sum(d['stable'] for d in pair_recs)} stabilizing pairs, {N} models used, {n_excl} clash-excluded")
    if params["plots"]:
        plot_summary(res_recs, pair_recs, N, n_excl, in_name, out_dir / f"{in_name}_summary.svg")

###################################### summary figure
def _empty(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.4")
    ax.set_xticks([]); ax.set_yticks([])

def _type_handles(extra_any=False):
    h = [Line2D([], [], marker=TYPE_MARKER[t], color=TYPE_COLOR[t], ls="", markersize=6, label=TYPE_LABEL[t])
         for t in TYPES]
    if extra_any:
        h.append(Line2D([], [], color="0.85", lw=6, label="any type"))
    return h

def lab(k):
    return f"{k[0]}{k[1]} {k[2]}"

def pair_label(k):
    return f"{k[0]}{k[1]} {k[2]} – {k[3]}{k[4]} {k[5]}"

def plot_summary(res_recs, pair_recs, n_incl, n_excl, title, out_path):
    occ_c, bur_c = params["occ_cutoff"], params["burial_cutoff"]
    R = res_recs[:params["top_residues"]]
    P = pair_recs[:params["top_pairs"]]
    nrows = max(len(R), len(P), 1)

    # Size from content: width from label lengths, height from row count
    fig_w = 15.5
    fig_h = max(5.5, 0.36 * nrows + 2.4)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.35, 0.95], left=0.075, right=0.985,
                          top=0.86, bottom=0.11, wspace=0.22)

    # A: residue occupancy heatmap, with burial strip beside it
    sa = gs[0].subgridspec(1, 2, width_ratios=[3.0, 0.9], wspace=0.03)
    axh = fig.add_subplot(sa[0]); axb = fig.add_subplot(sa[1], sharey=axh)
    if R:
        M = np.array([[d["occ"][t] for t in TYPES] for d in R])
        axh.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                axh.text(j, i, f"{v:.0%}" if v > 0 else "·", ha="center", va="center", fontsize=7,
                         color="white" if v > 0.6 else "0.15")
        axh.set_yticks(range(len(R)))
        axh.set_yticklabels([lab(d["key"]) for d in R], fontsize=7.5)
        for tl, d in zip(axh.get_yticklabels(), R):
            tl.set_color("#222222" if d["key"][0] == "A" else "#7d3c98")
        axh.set_xticks(range(len(TYPES)))
        axh.set_xticklabels([TYPE_LABEL[t] for t in TYPES], fontsize=8, fontweight="bold")
        for t, tl in zip(TYPES, axh.get_xticklabels()):
            tl.set_color(TYPE_COLOR[t])
        axh.xaxis.tick_top()
        bur = [d["burial"] if np.isfinite(d["burial"]) else 0.0 for d in R]
        axb.barh(range(len(R)), bur, height=0.7, color="0.6")
        axb.axvline(bur_c, ls="--", color="0.2", lw=0.8)
        axb.set_xlim(0, 1); axb.set_xticks([0, 0.5, 1])
        axb.tick_params(axis="x", labelsize=7)
        axb.tick_params(axis="y", left=False, labelleft=False)
        axb.xaxis.tick_top()
        axb.set_xlabel("rel. burial", fontsize=7, labelpad=4)
        axb.xaxis.set_label_position("top")
    else:
        _empty(axh, "no inter-chain interactions detected")
        axb.set_axis_off()
    axh.set_title("A. Residue occupancy by type", loc="left", fontsize=9, pad=40)

    # B: pair occupancy. Grey bar = any type; thin colored bars = each type at its own occupancy.
    axp = fig.add_subplot(gs[1])
    if P:
        ys = np.arange(len(P))[::-1]
        offsets = {"hbond": 0.19, "salt": 0.0, "hydro": -0.19}
        for d, y in zip(P, ys):
            axp.barh(y, d["any"], height=0.62, color="0.88", zorder=1)
            for t in TYPES:
                if d["occ"][t] > 0:
                    axp.barh(y + offsets[t], d["occ"][t], height=0.15, color=TYPE_COLOR[t], zorder=2)
            axp.text(d["any"] + 0.015, y, f"{d['any']:.0%}", va="center", fontsize=7)
        axp.set_yticks(ys)
        axp.set_yticklabels([pair_label(d["key"]) for d in P], fontsize=7.5)
        for tl, d in zip(axp.get_yticklabels(), P):
            if d["stable"]:
                tl.set_fontweight("bold")
        axp.set_ylim(-0.6, len(P) - 0.4)
        axp.set_xlim(0, 1.12); axp.set_xticks([0, 0.25, 0.5, 0.75, 1])
        axp.set_xlabel("occupancy (fraction of models)", fontsize=8)
        axp.grid(axis="x", alpha=0.25, lw=0.5)
        axp.legend(handles=_type_handles(extra_any=True), fontsize=7, ncol=4, frameon=False,
                   loc="lower left", bbox_to_anchor=(0.0, 1.0))
    else:
        _empty(axp, "no residue pairs detected")
    axp.set_title("B. Residue pairs: which interactions persist", loc="left", fontsize=9, pad=28)

    # C: residue persistence vs burial; shaded quadrant = frequent and buried
    axs = fig.add_subplot(gs[2])
    pts = [d for d in res_recs if np.isfinite(d["burial"])]
    if pts:
        rng = np.random.default_rng(params["seed"])
        ymax = max(0.5, max(d["burial"] for d in pts)) + 0.08
        xmax = 1.32
        axs.add_patch(Rectangle((occ_c, bur_c), xmax - occ_c, ymax - bur_c,
                                color="#2e7d32", alpha=0.08, lw=0, zorder=0))
        axs.axvline(occ_c, ls="--", color="0.4", lw=0.8)
        axs.axhline(bur_c, ls="--", color="0.4", lw=0.8)
        for d in pts:
            x = d["any"] + rng.uniform(-0.012, 0.012)  # display jitter only; occupancy is discrete
            y = d["burial"] + rng.uniform(-0.004, 0.004)
            axs.scatter(x, y, s=42, marker=TYPE_MARKER.get(d["dominant"], "x"),
                        color=TYPE_COLOR.get(d["dominant"], "0.5"), edgecolor="white", lw=0.4, alpha=0.9, zorder=3)
        stab = sorted((d for d in pts if d["any"] >= occ_c and d["burial"] >= bur_c),
                      key=lambda d: (-d["any"], -d["burial"]))
        for d in stab[:8]:
            axs.annotate(lab(d["key"]), (d["any"], d["burial"]), xytext=(5, 2),
                         textcoords="offset points", fontsize=6.5)
        axs.set_xlim(-0.05, xmax); axs.set_ylim(0, ymax)
        axs.set_xticks([0, 0.25, 0.5, 0.75, 1])
        axs.set_xlabel("occupancy of any interaction", fontsize=8)
        axs.set_ylabel("mean rel. dSASA (burial)", fontsize=8)
        axs.tick_params(labelsize=7); axs.grid(alpha=0.25, lw=0.5)
        axs.spines["top"].set_visible(False); axs.spines["right"].set_visible(False)
        axs.legend(handles=_type_handles(), fontsize=7, frameon=False, loc="upper left")
    else:
        _empty(axs, "no burial data")
    axs.set_title("C. Residues: frequent and buried (shaded)", loc="left", fontsize=9, pad=28)

    excl = f", {n_excl} clash-excluded" if n_excl else ""
    fig.suptitle(f"{title}: persistent interface interactions over {n_incl} models{excl}", fontsize=12)
    fig.text(0.075, 0.025,
             "Occupancy = fraction of models in which the interaction is detected. Burial = mean relative dSASA "
             f"over models. Bold pair labels: occupancy ≥ {occ_c} and mean burial ≥ {bur_c}. "
             "Dominant type = type with the highest occupancy.", fontsize=7, color="0.3")
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"[PYTHON-INFO] wrote summary panel {out_path}")

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
    for e in entries:
        view.addModel(e["cif"], "cif")

    state = {"sync": False}
    cbs = []
    master = None

    def apply(i):
        e = entries[i]
        if cbs[i].value:
            view.setStyle({"model": i, "chain": "A"}, {"cartoon": {"color": e["colorA"]}})
            view.setStyle({"model": i, "chain": "B"}, {"cartoon": {"color": e["colorB"]}})
        else:
            view.setStyle({"model": i}, {})

    def sync_master():
        state["sync"] = True
        master.value = all(cb.value for cb in cbs)
        state["sync"] = False

    def set_all(on):
        state["sync"] = True
        for i, cb in enumerate(cbs):
            cb.value = on
            apply(i)
        master.value = on
        state["sync"] = False
        view.update()

    def on_model(i):
        def handler(change):
            if state["sync"]:
                return
            apply(i)
            sync_master()
            view.update()
        return handler

    def on_master(change):
        if state["sync"]:
            return
        set_all(change["new"])

    cbs = [widgets.Checkbox(value=True, description=f"{i}: {e['name']}", indent=False,
                            layout=widgets.Layout(width="auto")) for i, e in enumerate(entries)]
    master = widgets.Checkbox(value=True, description="Select all / none", indent=False)
    for i, cb in enumerate(cbs):
        cb.observe(on_model(i), names="value")
    master.observe(on_master, names="value")
    for i in range(len(entries)):
        apply(i)

    names = json.dumps([e["name"] for e in entries])
    click_js = (
        "function(atom, viewer, event, container) {"
        " viewer.removeAllLabels();"
        f" var names = {names};"
        " viewer.addLabel('Model ' + atom.model + ' (' + names[atom.model] + ') | chain ' + atom.chain"
        " + ' | ' + atom.resn + ' ' + atom.resi,"
        " {position: atom, backgroundColor: 'black', fontColor: 'white', fontSize: 12});"
        " viewer.render(); }"
    )
    view.setClickable({}, True, click_js)
    view.zoomTo()

    reset_btn = widgets.Button(description="Reset view & labels", button_style="info",
                               layout=widgets.Layout(width="48%"))
    def on_reset(_):
        view.removeAllLabels()
        view.zoomTo()
        view.update()
    reset_btn.on_click(on_reset)

    show_btn = widgets.Button(description="Show all models", button_style="success",
                              layout=widgets.Layout(width="48%"))
    show_btn.on_click(lambda _: set_all(True))

    display(widgets.VBox([widgets.HBox([reset_btn, show_btn]), master] + cbs))
    view.show()

###################################### main
def main(argv):
    global params
    t0 = time.time()
    args = parse_args(argv)
    cfg = {}
    if args.yaml_in:
        with open(args.yaml_in) as f:
            cfg = (yaml.safe_load(f) or {}).get(NAME, {})
    params = {**DEFAULTS, **cfg}
    for key in ("input", "output", "ref", "align_chain", "workers"):
        v = getattr(args, key)
        if v is not None:
            params[key] = v
    if args.no_html:
        params["html"] = False
    if args.no_interfaces:
        params["interfaces"] = False
    if not params["input"]:
        raise SystemExit("[PYTHON-INFO] no input provided")
    np.random.seed(params["seed"])

    in_dir = Path(params["input"])
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit("[PYTHON-INFO] no .pdb files found")
    name = in_dir.resolve().name
    out_dir = Path(params["output"] or os.path.join("data/pdbs_by_group", name, "aligned"))
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = [p.stem for p in files]
    if params["ref"] and params["ref"] not in stems:
        raise SystemExit(f"[PYTHON-INFO] reference '{params['ref']}' not found in input")
    ref_idx = stems.index(params["ref"]) if params["ref"] else 0
    print(f"[PYTHON-INFO] structures={len(files)} reference={stems[ref_idx]} align_chain={params['align_chain']}")

    # Alignment and multi-model export
    parser = PDBParser(QUIET=True)
    models = [parser.get_structure(p.stem, str(p))[0] for p in files]
    rmsd, nmatch = align(models, params["align_chain"], ref_idx)

    cif_path = out_dir / f"{name}_aligned.cif"
    write_cif(ensemble_structure(models), str(cif_path))

    man_path = out_dir / f"{name}_manifest.tsv"
    entries = []
    with open(man_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(MANIFEST_COLS)
        for i, (p, m) in enumerate(zip(files, models)):
            col_a, col_b, label = source_style(p.stem)
            w.writerow([i, p.stem, label, col_a, col_b, nmatch[i], "" if rmsd[i] is None else f"{rmsd[i]:.3f}"])
            entries.append({"name": p.stem, "colorA": col_a, "colorB": col_b, "cif": model_cif_text(m)})
    print(f"[PYTHON-INFO] wrote multi-model mmCIF {cif_path} ({len(models)} models)")
    print(f"[PYTHON-INFO] wrote manifest {man_path}")

    if params["html"]:
        library_js = load_3dmol(args.threedmol_js)
        html_path = out_dir / f"{name}_viewer.html"
        write_html(entries, name, html_path, library_js)
        print(f"[PYTHON-INFO] wrote standalone viewer {html_path} ({html_path.stat().st_size / 1e6:.1f} MB)")

    # Interface analysis (reads the original files; interface metrics are invariant to rigid alignment)
    if params["interfaces"]:
        run_interfaces(files, name, out_dir)

    if args.yaml_out:
        out = {}
        if os.path.exists(args.yaml_out):
            with open(args.yaml_out) as f:
                out = yaml.safe_load(f) or {}
        out[NAME] = params
        os.makedirs(os.path.dirname(args.yaml_out) or ".", exist_ok=True)
        with open(args.yaml_out, "w") as f:
            yaml.safe_dump(out, f)

    if in_notebook():
        show_notebook(entries)

    print(f"[PYTHON-INFO] {datetime.datetime.now():%Y-%m-%d %H:%M:%S} exec={time.time() - t0:.2f}s "
          f"maxmem={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}MB")

if __name__ == "__main__":
    main([] if in_notebook() else sys.argv[1:])

