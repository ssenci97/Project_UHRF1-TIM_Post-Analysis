#!/usr/bin/env python3
"""Chain-chain interface metrics for harmonized models in parallel: buried area via Biopython Shrake-Rupley, contacts via PyMOL, with TSV output, SVG summary plots and an interactive HTML scatter viewer.

Args:
    --input: directory containing harmonized .pdb models with chains A and B.
    --output: directory where the chain score TSV, SVG plots and HTML viewer are written.
    --yaml-in: optional yaml with parameters nested under the script name.
    --yaml-out: optional yaml to store the script name and parameters.
"""

###################################### dependencies
import argparse, csv, datetime, json, multiprocessing, os, random, resource, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from Bio.PDB import PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.SASA import ShrakeRupley
from pymol import cmd
from tqdm import tqdm

###################################### constants
NAME = os.path.splitext(os.path.basename(__file__))[0]
DEFAULTS = {"input": None, "output": "data/measurements", "contact_cutoff": 4.5, "score_cutoff": 12.0, "workers": None, "seed": 42, "plots": True}
EXTS = (".pdb",)
COLS = ["structure", "len_res_a", "len_res_b", "len_atoms_a", "len_atoms_b", "dASA_A2", "n_contacts", "iface_res_a", "iface_res_b", "min_dist_A", "contact_score"]
FIG_SIZE = (6, 6)
BINS = 30
HTML_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>body{font-family:sans-serif;margin:16px}svg{border:1px solid #ccc;width:100%;max-width:600px}circle{cursor:pointer;fill:#1f77b4}circle.dim{opacity:0.15}circle.hit{fill:#ff7f0e}circle.sel{stroke:#000;stroke-width:2;fill:#d62728}#panel{margin-top:12px;font-family:monospace;white-space:pre}</style></head><body>
<h3>__TITLE__</h3>
<input id="q" placeholder="search structure name" size="40"> <button id="clr">clear selection</button>
<div style="display:flex;flex-wrap:wrap;gap:16px"><div><svg id="p1" viewBox="0 0 600 600"></svg></div><div><svg id="p2" viewBox="0 0 600 600"></svg></div></div>
<div id="panel">selected: none</div>
<script>
const D=__DATA__;
const sel=new Set();
const M={l:70,r:20,t:20,b:60,w:600,h:600};
const plots=[{id:"p1",y:"contact_score",label:"contact_score"},{id:"p2",y:"n_contacts",label:"n_contacts"}];
const ns="http://www.w3.org/2000/svg";
const circles={};
function scale(v,lo,hi,a,b){return hi===lo?(a+b)/2:a+(v-lo)/(hi-lo)*(b-a);}
function el(svg,t,a,txt){const e=document.createElementNS(ns,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!==undefined)e.textContent=txt;svg.appendChild(e);return e;}
function draw(p){
 const svg=document.getElementById(p.id); svg.innerHTML=""; circles[p.id]=[];
 const xs=D.dasa, ys=D[p.y];
 const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 const cy=(M.t+M.h-M.b)/2;
 el(svg,"line",{x1:M.l,y1:M.h-M.b,x2:M.w-M.r,y2:M.h-M.b,stroke:"#000"});
 el(svg,"line",{x1:M.l,y1:M.t,x2:M.l,y2:M.h-M.b,stroke:"#000"});
 el(svg,"text",{x:(M.l+M.w-M.r)/2,y:M.h-15,"text-anchor":"middle","font-size":14},"dASA (A^2)");
 el(svg,"text",{x:18,y:cy,"text-anchor":"middle","font-size":14,transform:`rotate(-90 18 ${cy})`},p.label);
 el(svg,"text",{x:M.l,y:M.h-M.b+16,"font-size":11},x0.toFixed(0));
 el(svg,"text",{x:M.w-M.r,y:M.h-M.b+16,"font-size":11,"text-anchor":"end"},x1.toFixed(0));
 el(svg,"text",{x:M.l-6,y:M.h-M.b,"font-size":11,"text-anchor":"end"},y0.toFixed(1));
 el(svg,"text",{x:M.l-6,y:M.t+10,"font-size":11,"text-anchor":"end"},y1.toFixed(1));
 D.names.forEach((n,i)=>{
  const c=el(svg,"circle",{cx:scale(xs[i],x0,x1,M.l,M.w-M.r),cy:scale(ys[i],y0,y1,M.h-M.b,M.t),r:5,"data-i":i});
  el(c,"title",{},`${n}\\ndASA=${xs[i]} ${p.y}=${ys[i]}`);
  c.addEventListener("click",()=>{sel.has(i)?sel.delete(i):sel.add(i);update();});
  circles[p.id].push(c);
 });
}
function update(){
 const q=document.getElementById("q").value.trim().toLowerCase();
 for(const p of plots) circles[p.id].forEach((c,i)=>{
  const hit=q&&D.names[i].toLowerCase().includes(q);
  c.setAttribute("class",[sel.has(i)?"sel":"",hit?"hit":"",q&&!hit?"dim":""].filter(Boolean).join(" "));
 });
 const list=[...sel].map(i=>`${D.names[i]}  dASA=${D.dasa[i]}  contact_score=${D.contact_score[i]}  n_contacts=${D.n_contacts[i]}`);
 document.getElementById("panel").textContent=list.length?`selected (${list.length}):\\n`+list.join("\\n"):"selected: none";
}
document.getElementById("q").addEventListener("input",update);
document.getElementById("clr").addEventListener("click",()=>{sel.clear();update();});
plots.forEach(draw); update();
</script></body></html>"""

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
out_tsv = f"{in_name}_chain_scores.tsv"
Path(params["output"]).mkdir(parents=True, exist_ok=True)
print(f"[PYTHON-INFO] structures={len(files)} detected_cpus={nproc} workers={workers} contact_cutoff={params['contact_cutoff']}A score_cutoff={params['score_cutoff']}A")

###################################### measurements
sr = ShrakeRupley()

def sasa(chains):
    m = Model(0)
    for c in chains:
        m.add(c.copy())
    sr.compute(m, level="A")
    return sum(a.sasa for a in m.get_atoms())

def buried_area(path):
    s = PDBParser(QUIET=True).get_structure("s", path)[0]
    A, B = s["A"], s["B"]
    raw = sasa([A]) + sasa([B]) - sasa([A, B])
    if raw < 0:
        print(f"[PYTHON-INFO] {Path(path).stem}: negative dASA={raw:.1f}A2")
    return raw

def measure(obj, path):
    A = cmd.get_model(f"{obj} and chain A and not elem H").atom
    B = cmd.get_model(f"{obj} and chain B and not elem H").atom
    if not A or not B:
        return None
    d = np.linalg.norm(np.array([a.coord for a in A])[:, None] - np.array([b.coord for b in B])[None], axis=-1)
    contact = d <= params["contact_cutoff"]
    weighted = (d <= params["score_cutoff"]) & (d > 0)
    iface_a = {A[k].resi for k in np.where(contact.any(axis=1))[0]}
    iface_b = {B[k].resi for k in np.where(contact.any(axis=0))[0]}
    len_a = len({a.resi for a in cmd.get_model(f"{obj} and chain A and name CA").atom})
    len_b = len({b.resi for b in cmd.get_model(f"{obj} and chain B and name CA").atom})
    return [len_a, len_b, len(A), len(B), round(buried_area(path), 2), int(contact.sum()), len(iface_a), len(iface_b), round(float(d.min()), 3), round(float((1.0 / d[weighted]).sum()), 4)]

def init_worker():
    cmd.reinitialize(); cmd.feedback("disable", "all", "everything")

def process(path):
    name = Path(path).stem
    try:
        cmd.load(str(path), name)
        m = measure(name, str(path))
        if m is None:
            return name, None, "missing chain A or B"
        return name, [name] + m, f"dASA={m[4]:.1f}A2 contacts={m[5]} iface={m[6]}/{m[7]} min_dist={m[8]:.2f}A contact_score={m[9]:.2f}"
    except Exception as ex:
        return name, None, str(ex).replace("\n", " ")
    finally:
        if name in cmd.get_names():
            cmd.delete(name)

###################################### run
rows, skipped = [], []
with multiprocessing.Pool(workers, initializer=init_worker) as pool:
    for name, row, msg in tqdm(pool.imap_unordered(process, files, chunksize=1), total=len(files), desc="chain-chain"):
        if row is None:
            skipped.append(name)
            print(f"[PYTHON-INFO] {name}: skipped, {msg}")
        else:
            rows.append(row)
            print(f"[PYTHON-INFO] {name}: {msg}")
rows.sort(key=lambda r: r[0])
with open(os.path.join(params["output"], out_tsv), "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(COLS); w.writerows(rows)
print(f"[PYTHON-INFO] wrote {len(rows)} rows to {os.path.join(params['output'], out_tsv)} skipped={len(skipped)}")

###################################### plots
if params["plots"] and rows:
    dasa = np.array([r[5] for r in rows], float)
    ncon = np.array([r[6] for r in rows], float)
    mind = np.array([r[9] for r in rows], float)
    cscore = np.array([r[10] for r in rows], float)
    stem = Path(out_tsv).stem
    plots = [
        ("dasa_vs_contact_score", dasa, cscore, "dASA (A^2)", "contact_score", "scatter"),
        ("dasa_vs_n_contacts", dasa, ncon, "dASA (A^2)", "n_contacts", "scatter"),
        ("min_dist_hist", mind, None, "min_dist (A)", "count", "hist"),
    ]
    for tag, x, y, xl, yl, kind in plots:
        fig, ax = plt.subplots(figsize=FIG_SIZE, constrained_layout=True)
        if kind == "scatter":
            ax.scatter(x, y, s=14)
        else:
            ax.hist(x, bins=BINS)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"{in_name}: {tag.replace('_', ' ')}")
        path = os.path.join(params["output"], f"{stem}_{tag}.svg")
        fig.savefig(path, format="svg")
        plt.close(fig)
        print(f"[PYTHON-INFO] wrote plot {path}")
    data = {"names": [r[0] for r in rows], "dasa": [r[5] for r in rows], "n_contacts": [r[6] for r in rows], "contact_score": [r[10] for r in rows]}
    html = HTML_TEMPLATE.replace("__TITLE__", f"{in_name}: chain-chain scores").replace("__DATA__", json.dumps(data))
    html_path = os.path.join(params["output"], f"{stem}_viewer.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[PYTHON-INFO] wrote interactive viewer {html_path}")

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