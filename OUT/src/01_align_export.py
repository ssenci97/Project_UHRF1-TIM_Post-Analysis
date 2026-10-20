#!/usr/bin/env python3
"""Align PDB ensembles and export structural viewers."""
###################################### dependencies
import argparse
import csv
import datetime
import io
import json
import resource
import sys
import time
import urllib.request
from html import escape
from pathlib import Path
import numpy as np
import yaml
from Bio.PDB import MMCIFIO, PDBParser, Superimposer
from Bio.PDB.Structure import Structure
from tqdm import tqdm

###################################### constants
NAME = "align_export"
SEED = 42
DEFAULTS = {"input": None, "output": None, "ref": None, "align_chain": "A", "workers": 8, "seed": SEED, "html": True, "threedmol_js": None}
THREEDMOL_URLS = ("https://3dmol.org/build/3Dmol-min.js", "https://3dmol.csb.pitt.edu/build/3Dmol-min.js")
COLOR_RULES = (("abin", "salmon", "teal", "abin"), ("model", "grey", "black", "model"), ("pdbeval", "red", "blue", "pdbeval"))
DEFAULT_STYLE = ("orange", "purple", "other")
MANIFEST_COLS = ["order", "structure", "source", "color_A", "color_B", "n_CA_matched", "CA_rmsd_A"]

###################################### helpers
def source_style(stem):
    low = stem.lower()
    for key, a, b, label in COLOR_RULES:
        if key in low:
            return a, b, label
    return DEFAULT_STYLE

def load_config(args):
    params = {**DEFAULTS}
    if args.yaml_in:
        with open(args.yaml_in) as handle:
            params.update((yaml.safe_load(handle) or {}).get(NAME, {}))
    for key in ("input", "output", "ref", "align_chain", "threedmol_js"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    if args.no_html:
        params["html"] = False
    return params

def save_config(path, params):
    if not path:
        return
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {} if path.exists() else {}
    data[NAME] = params
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))

def ca_map(model, chain_id):
    if chain_id not in model:
        return {}
    return {res.id[1:]: res["CA"] for res in model[chain_id] if res.id[0] == " " and "CA" in res}

def align(models, chain_id, ref_idx):
    ref = ca_map(models[ref_idx], chain_id)
    rmsd, matched = [None] * len(models), [0] * len(models)
    for i, model in enumerate(tqdm(models, desc="aligning")):
        if i == ref_idx:
            rmsd[i], matched[i] = 0.0, len(ref)
            continue
        mob = ca_map(model, chain_id)
        keys = sorted(ref.keys() & mob.keys())
        matched[i] = len(keys)
        if len(keys) < 3:
            print(f"[PYTHON-INFO] warning: model {i} has only {len(keys)} matched CA atoms")
            continue
        sup = Superimposer()
        sup.set_atoms([ref[k] for k in keys], [mob[k] for k in keys])
        sup.apply(list(model.get_atoms()))
        rmsd[i] = float(sup.rms)
    return rmsd, matched

def write_cif(structure, path):
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(path))

def model_cif_text(model):
    structure = Structure("m")
    structure.add(model.copy())
    writer = MMCIFIO()
    writer.set_structure(structure)
    buf = io.StringIO()
    writer.save(buf)
    return buf.getvalue()

def ensemble_structure(models):
    structure = Structure("ensemble")
    for i, model in enumerate(models):
        copied = model.copy()
        copied.id = i
        structure.add(copied)
    return structure

def load_3dmol(local_path=None):
    if local_path:
        return Path(local_path).read_text(encoding="utf-8")
    cache = Path.home() / ".cache" / "aligned_export" / "3Dmol-min.js"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    error = None
    for url in THREEDMOL_URLS:
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                js = response.read().decode("utf-8")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(js, encoding="utf-8")
            print(f"[PYTHON-INFO] cached 3Dmol.js from {url}")
            return js
        except Exception as exc:
            error = exc
    raise SystemExit(f"[PYTHON-INFO] could not load 3Dmol.js: {error}; use --3dmol-js")

def write_html(entries, title, path, js):
    template = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>__TITLE__</title><script>__JS__</script><style>html,body{margin:0;height:100%;font-family:sans-serif}body{display:flex}#panel{width:300px;overflow:auto;padding:12px;border-right:1px solid #ccc;font-size:13px}#viewer{flex:1}label{display:block;margin:5px 0}.sw{display:inline-block;width:10px;height:10px;margin-left:4px;border:1px solid #666}</style></head><body><div id="panel"><h3>__TITLE__</h3><button id="show">Show all</button><button id="hide">Hide all</button><button id="reset">Reset view</button><div id="list"></div></div><div id="viewer"></div><script>const M=__MODELS__;const v=$3Dmol.createViewer(document.getElementById("viewer"),{backgroundColor:"white"});M.forEach(m=>v.addModel(m.cif,"cif"));const shown=M.map(()=>true),boxes=[];function apply(i){if(shown[i]){v.setStyle({model:i,chain:"A"},{cartoon:{color:M[i].a}});v.setStyle({model:i,chain:"B"},{cartoon:{color:M[i].b}})}else v.setStyle({model:i},{})}M.forEach((m,i)=>{let l=document.createElement("label"),b=document.createElement("input");b.type="checkbox";b.checked=true;b.onchange=()=>{shown[i]=b.checked;apply(i);v.render()};boxes.push(b);l.append(b,document.createTextNode(" "+i+": "+m.name));[m.a,m.b].forEach(c=>{let s=document.createElement("span");s.className="sw";s.style.background=c;l.append(s)});document.getElementById("list").append(l)});function all(x){M.forEach((m,i)=>{shown[i]=x;boxes[i].checked=x;apply(i)});v.render()}document.getElementById("show").onclick=()=>all(true);document.getElementById("hide").onclick=()=>all(false);document.getElementById("reset").onclick=()=>{v.removeAllLabels();v.zoomTo();v.render()};v.setClickable({},true,(a,v)=>{v.removeAllLabels();v.addLabel(M[a.model].name+" | chain "+a.chain+" | "+a.resn+" "+a.resi,{position:a,backgroundColor:"black",fontColor:"white"});v.render()});M.forEach((m,i)=>apply(i));v.zoomTo();v.render();</script></body></html>"""
    js = js.replace("</script", "<\\/script")
    models = json.dumps(entries).replace("</", "<\\/")
    html = template.replace("__TITLE__", escape(title)).replace("__JS__", js).replace("__MODELS__", models)
    Path(path).write_text(html, encoding="utf-8")

###################################### main
def main():
    start = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--ref")
    p.add_argument("--align-chain")
    p.add_argument("--3dmol-js", dest="threedmol_js")
    p.add_argument("--no-html", action="store_true")
    p.add_argument("--yaml-in")
    p.add_argument("--yaml-out", default="configs/param_configs.yaml")
    args = p.parse_args()
    params = load_config(args)
    if not params["input"]:
        raise SystemExit("[PYTHON-INFO] --input is required")
    np.random.seed(params["seed"])
    input_dir = Path(params["input"])
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".pdb")
    if not files:
        raise SystemExit(f"[PYTHON-INFO] no PDB files in {input_dir}")
    name = input_dir.resolve().name
    out = Path(params["output"] or Path("data/pdbs_by_group") / name / "aligned")
    out.mkdir(parents=True, exist_ok=True)
    stems = [p.stem for p in files]
    if params["ref"] and params["ref"] not in stems:
        raise SystemExit(f"[PYTHON-INFO] reference {params['ref']} not found")
    ref_idx = stems.index(params["ref"]) if params["ref"] else 0
    print(f"[PYTHON-INFO] structures={len(files)} reference={stems[ref_idx]} align_chain={params['align_chain']}")
    parser = PDBParser(QUIET=True)
    models = [parser.get_structure(p.stem, str(p))[0] for p in tqdm(files, desc="loading")]
    rmsd, matched = align(models, params["align_chain"], ref_idx)
    cif_path = out / f"{name}_aligned.cif"
    write_cif(ensemble_structure(models), cif_path)
    entries = []
    with open(out / f"{name}_manifest.tsv", "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(MANIFEST_COLS)
        for i, (path, model) in enumerate(zip(files, models)):
            a, b, label = source_style(path.stem)
            writer.writerow([i, path.stem, label, a, b, matched[i], "" if rmsd[i] is None else f"{rmsd[i]:.3f}"])
            entries.append({"name": path.stem, "a": a, "b": b, "cif": model_cif_text(model)})
    print(f"[PYTHON-INFO] wrote {cif_path} and manifest")
    if params["html"]:
        path = out / f"{name}_viewer.html"
        write_html(entries, name, path, load_3dmol(params["threedmol_js"]))
        print(f"[PYTHON-INFO] wrote {path}")
    save_config(args.yaml_out, params)
    elapsed = time.time() - start
    print(f"[PYTHON-INFO] {datetime.datetime.now():%Y-%m-%d %H:%M:%S} exec={elapsed:.2f}s maxmem={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}MB")

if __name__ == "__main__":
    main()

