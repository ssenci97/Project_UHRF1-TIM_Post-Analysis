==> src/01_dw_rcsb.py <==
#!/usr/bin/env python3
"""Download mmCIF files from RCSB PDB using a list of PDB IDs or an input file."""

###################################### DEPENDENCIES AND CONSTANTS
import argparse
import datetime
import json
import os
import re
import resource
import sys
import time
import urllib.request
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://files.rcsb.org/download/{}.cif"
DEFAULT_JSON_PATH = "configs/param_configs.json"
DEFAULT_OUT_DIR = "data/cifs"
ID_REGEX = re.compile(r"^[a-zA-Z0-9]{4}$")

###################################### ARGUMENT PARSING AND VALIDATION
def parse_args():
    parser = argparse.ArgumentParser(description="Download mmCIF files from RCSB PDB.")
    parser.add_argument("-i", "--input", type=str, help="Comma-separated PDB IDs or path to file.")
    parser.add_argument("-o", "--output", "--outdir", type=str, default=DEFAULT_OUT_DIR, help="Output directory path.")
    parser.add_argument("-j", "--json", type=str, nargs="?", const=DEFAULT_JSON_PATH, default=DEFAULT_JSON_PATH, help="Path to JSON config file.")
    parser.add_argument("--no-json", action="store_true")
    return parser.parse_args()

def get_valid_ids(input_source):
    raw_tokens = []
    if os.path.isfile(input_source):
        with open(input_source, "r") as f:
            raw_tokens = re.split(r"[,\s]+", f.read())
    else:
        raw_tokens = input_source.split(",")
    valid_ids = []
    for token in raw_tokens:
        cleaned = token.strip()
        if not cleaned:
            continue
        if ID_REGEX.match(cleaned):
            valid_ids.append(cleaned.upper())
        else:
            print(f"[PYTHON-INFO] Skipping invalid 4-character PDB ID: '{cleaned}'")
    return list(dict.fromkeys(valid_ids))

###################################### MAIN EXECUTION
def main():
    start_time = time.time()
    args = parse_args()
    config = {"cif_outdir": args.output}
    if args.no_json:
        config["input"] = args.input
    else:
        config["input"] = args.input
    if args.json is not None and not args.no_json:
        json_path = Path(args.json)
        if json_path.exists():
            print(f"[PYTHON-INFO] Reading configuration from existing JSON: {json_path}")
            with open(json_path, "r") as f:
                json_data = json.load(f)
            if "cif_outdir" not in json_data:
                print(f"[PYTHON-INFO] Updating JSON configuration with missing keys: {json_path}")
                json_data["cif_outdir"] = config["cif_outdir"]
                json_data.update({k: v for k, v in config.items() if k != "input" and k not in json_data and v is not None})
                json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(json_path, "w") as f:
                    json.dump(json_data, f, indent=2)
            if not config.get("input") and "input" in json_data:
                config["input"] = json_data["input"]
            if (not config.get("cif_outdir") or config["cif_outdir"] == DEFAULT_OUT_DIR) and "cif_outdir" in json_data:
                config["cif_outdir"] = json_data["cif_outdir"]
        else:
            print(f"[PYTHON-INFO] Writing configuration to JSON: {json_path}")
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump({k: v for k, v in config.items() if k != "input" and v is not None}, f, indent=2)
    outdir = Path(config.get("cif_outdir", DEFAULT_OUT_DIR))
    outdir.mkdir(parents=True, exist_ok=True)
    input_str = config.get("input")
    if not input_str:
        print("[PYTHON-INFO] Error: No input provided via command line or JSON config.")
        sys.exit(1)
    pdb_ids = get_valid_ids(str(input_str))
    print(f"[PYTHON-INFO] Process initiated for {len(pdb_ids)} unique valid IDs.")
    downloaded, skipped, failed = 0, 0, 0
    for pdb_id in tqdm(pdb_ids, desc="Downloading mmCIFs"):
        target_path = outdir / f"{pdb_id}.cif"
        if target_path.exists():
            skipped += 1
            continue
        download_url = BASE_URL.format(pdb_id)
        try:
            urllib.request.urlretrieve(download_url, target_path)
            downloaded += 1
        except Exception as err:
            failed += 1
            print(f"[PYTHON-INFO] Failed to download {pdb_id}: {err}")
    exec_time = time.time() - start_time
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] Processing complete | Downloaded: {downloaded} | Skipped: {skipped} | Failed: {failed}")
    print(f"[PYTHON-INFO] Completed on {current_date} | Execution time: {exec_time:.2f}s | Max Memory: {mem_mb:.2f} MB")

if __name__ == "__main__":
    main()



==> src/02_extract_pdb_data.py <==
#!/usr/bin/env python
"""Extract protein chains, reference sequences, and PDB sequences from mmCIF files."""

###################################### DEPENDENCIES AND CONSTANTS
import argparse
import datetime
import json
import multiprocessing as mp
import os
import resource
import time
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBIO, Select
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Polypeptide import protein_letters_3to1_extended
from tqdm import tqdm

DEFAULT_CHAIN_OUTDIR = "data/chains/pdb/"
DEFAULT_CHAIN_REFSEQS_OUTDIR = "data/chains/pdb_refseqs"
DEFAULT_CHAIN_SEQS_OUTDIR = "data/chains/pdb_seqs"
DEFAULT_CIF_OUTDIR = "data/cifs"
DEFAULT_CONFIG_PATH = "configs/param_configs.json"
SEED = 42

###################################### CONFIG AND HELPERS
def get_ncores():
    for key in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(key)
        if value and value.isdigit():
            return max(1, int(value))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)

def resolve_config(path, defaults, no_json):
    if no_json:
        return dict(defaults)
    p = Path(path)
    cfg = {}
    if p.exists():
        try:
            cfg = json.loads(p.read_text())
        except Exception:
            cfg = {}
    changed = False
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
            changed = True
    if changed or not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cfg, indent=2) + "\n")
    return {k: cfg[k] for k in defaults}

def clean_seq(s):
    return "".join(str(s).split()).replace("(", "").replace(")", "").upper()

def to_list(val):
    if val is None:
        return []
    return [val] if isinstance(val, (str, int, float)) else list(val)

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
    def accept_chain(self, chain):
        return chain.id == self.chain_id
    def accept_residue(self, residue):
        return residue.id[0] == " "

###################################### LOGGING
def finish_log(start, msg):
    elapsed = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem = usage / 1024 if usage < 10**7 else usage / 1024**2
    now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {now} | {msg} | execution time: {elapsed:.2f}s | max memory: {mem:.1f} MB")

###################################### EXTRACTION
def process_cif(task):
    cif_path, cfg = task
    cif_path = Path(cif_path)
    try:
        structure = MMCIFParser(QUIET=True).get_structure(cif_path.stem, str(cif_path))
        mm = MMCIF2Dict(str(cif_path))
    except Exception as e:
        return cif_path.name, 0, 0, 0, str(e)
    dirs = {k: Path(cfg[k]) for k in ("chain_outdir", "chain_seqs_outdir", "chain_refseqs_outdir")}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    eids = to_list(mm.get("_entity_poly.entity_id"))
    seqs_std = to_list(mm.get("_entity_poly.pdbx_seq_one_letter_code"))
    seqs_can = to_list(mm.get("_entity_poly.pdbx_seq_one_letter_code_can"))
    entity_seq = {}
    for idx, e in enumerate(eids):
        e_str = str(e)
        s = clean_seq(seqs_std[idx]) if idx < len(seqs_std) else ""
        if not s and idx < len(seqs_can):
            s = clean_seq(seqs_can[idx])
        if s:
            entity_seq[e_str] = s
    if not entity_seq and "_entity_poly_seq.entity_id" in mm:
        poly_eids = to_list(mm.get("_entity_poly_seq.entity_id"))
        poly_mons = to_list(mm.get("_entity_poly_seq.mon_id"))
        temp_seqs = {}
        for e, mon in zip(poly_eids, poly_mons):
            e_str = str(e)
            one_letter = protein_letters_3to1_extended.get(str(mon).upper(), "X")
            temp_seqs.setdefault(e_str, []).append(one_letter)
        for e_str, mon_list in temp_seqs.items():
            entity_seq[e_str] = "".join(mon_list)
    chain_entity_map = {}
    scheme_strands = to_list(mm.get("_pdbx_poly_seq_scheme.pdb_strand_id"))
    scheme_entities = to_list(mm.get("_pdbx_poly_seq_scheme.entity_id"))
    for strand, ent in zip(scheme_strands, scheme_entities):
        if strand and ent and str(strand) not in ("?", "."):
            chain_entity_map[str(strand)] = str(ent)
    if not chain_entity_map:
        asym = to_list(mm.get("_struct_asym.id"))
        ent = to_list(mm.get("_struct_asym.entity_id"))
        auth = to_list(mm.get("_atom_site.auth_asym_id"))
        label = to_list(mm.get("_atom_site.label_asym_id"))
        label_auth = {str(l): str(a) for l, a in zip(label, auth) if l and a}
        for a_id, e_id in zip(asym, ent):
            auth_id = label_auth.get(str(a_id), str(a_id))
            chain_entity_map[auth_id] = str(e_id)
    npdb = nseq = nref = 0
    for model in structure:
        for chain in model:
            cid = str(chain.id)
            stem = f"{cif_path.stem}_{cid}"
            pdb = dirs["chain_outdir"] / f"{stem}.pdb"
            seqfile = dirs["chain_seqs_outdir"] / f"{stem}.fasta"
            reffile = dirs["chain_refseqs_outdir"] / f"{stem}.fasta"
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(pdb), select=ChainSelect(cid))
            seq = "".join(
                protein_letters_3to1_extended[r.get_resname().strip().upper()]
                for r in chain
                if r.id[0] == " " and r.get_resname().strip().upper() in protein_letters_3to1_extended
            )
            if not seq:
                continue
            seqfile.write_text(f">{stem}\n{seq}\n")
            npdb += 1
            nseq += 1
            e_id = chain_entity_map.get(cid)
            r = entity_seq.get(e_id, "") if e_id else ""
            if r:
                reffile.write_text(f">{stem}|author_chain_{cid}\n{r}\n")
                nref += 1
        break
    return cif_path.name, npdb, nseq, nref, ""

###################################### MAIN EXECUTION
def main():
    start = time.perf_counter()
    p = argparse.ArgumentParser(description="Extract protein chains and reference sequences from mmCIFs")
    p.add_argument("-i", "--input", help="mmCIF file or directory (defaults to cif_outdir in config)")
    p.add_argument("-o", "--output", help="PDB chain output directory (defaults to chain_outdir in config)")
    p.add_argument("-j", "--json", nargs="?", const=DEFAULT_CONFIG_PATH, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--no-json", action="store_true")
    a = p.parse_args()
    defaults = {
        "cif_outdir": DEFAULT_CIF_OUTDIR,
        "chain_outdir": DEFAULT_CHAIN_OUTDIR,
        "chain_seqs_outdir": DEFAULT_CHAIN_SEQS_OUTDIR,
        "chain_refseqs_outdir": DEFAULT_CHAIN_REFSEQS_OUTDIR,
    }
    cfg = resolve_config(a.json, defaults, a.no_json)
    if a.output:
        cfg["chain_outdir"] = a.output
    src = Path(a.input or cfg["cif_outdir"])
    files = sorted(src.glob("*.cif")) if src.is_dir() else [src]
    n = get_ncores()
    print(f"[PYTHON-INFO] Extracting {len(files)} CIF files with {n} worker(s)")
    totals = [0, 0, 0]
    with mp.Pool(n) as pool:
        jobs = [pool.apply_async(process_cif, ((str(f), cfg),)) for f in files]
        for j in tqdm(jobs, desc="Extracting PDB data"):
            name, npdb, nseq, nref, err = j.get()
            totals[0] += npdb
            totals[1] += nseq
            totals[2] += nref
            if err:
                print(f"[PYTHON-INFO] {name}: {err}")
    summary_msg = f"Saved PDB={totals[0]} sequences={totals[1]} RefSeqs={totals[2]} from {len(files)} CIF files using {n} workers"
    finish_log(start, summary_msg)

if __name__ == "__main__":
    main()
==> src/03_align_pair_local.py <==
#!/usr/bin/env python
"""Perform pairwise local sequence alignment between query sequences and reference sequences."""

###################################### DEPENDENCIES AND CONSTANTS
import argparse
import datetime
import json
import resource
import time
from pathlib import Path
from Bio.Align import PairwiseAligner
import pandas as pd
from tqdm import tqdm

DEFAULT_ALN_OUTDIR = "data/aln"
DEFAULT_CHAIN_REFSEQS_OUTDIR = "data/chains_refseqs"
DEFAULT_CHAIN_SEQS_OUTDIR = "data/chains_seqs"
DEFAULT_CONFIG_PATH = "configs/param_configs.json"

###################################### CONFIG
def resolve_config(json_arg, defaults):
    if json_arg is None:
        return defaults
    p = Path(json_arg)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(defaults, indent=2))
        return defaults
    try:
        cfg = json.loads(p.read_text())
    except Exception:
        cfg = {}
    updated = False
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
            updated = True
    if updated:
        p.write_text(json.dumps(cfg, indent=2))
    return cfg

###################################### HELPERS AND ALIGNMENT
def parse_fasta_file(fpath):
    if not fpath.exists():
        return "", ""
    lines = fpath.read_text().splitlines()
    header, seq = "", []
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            header = line[1:]
        elif line:
            seq.append(line)
    return "".join(seq), header

def resolve_input_files(inp_arg, default_dir):
    if not inp_arg:
        inp_arg = default_dir
    p = Path(inp_arg)
    if p.is_dir():
        return sorted(list(p.glob("*.fasta")) + list(p.glob("*.fa")))
    if p.is_file() and p.suffix.lower() == ".txt":
        lines = [line.strip() for line in p.read_text().splitlines() if line.strip()]
        return [Path(f) for f in lines]
    paths = [Path(x.strip()) for x in str(inp_arg).split(",") if x.strip()]
    return [f for f in paths if f.is_file()]

def run_alignment(ref_seq, query_seq, ref_id, query_id):
    if not ref_seq or not query_seq:
        return None, []
    aligner = PairwiseAligner(mode="local", match_score=1, mismatch_score=-1, open_gap_score=-2, extend_gap_score=-0.5)
    alignments = aligner.align(ref_seq, query_seq)
    if not alignments:
        return None, []
    alignment = alignments[0]
    a, b = alignment
    ref_start, query_start = int(alignment.coordinates[0, 0]), int(alignment.coordinates[1, 0])
    ref_end, query_end = int(alignment.coordinates[0, 1]), int(alignment.coordinates[1, 1])
    matches = mismatches = insertions = deletions = 0
    gappos, q_pos, ref_pos = [], query_start, ref_start
    pos_rows = []
    for idx, (x, y) in enumerate(zip(a, b), start=1):
        if x != "-":
            ref_pos += 1
        if y != "-":
            q_pos += 1
        if x == "-" and y == "-":
            continue
        if x == "-":
            insertions += 1
        elif y == "-":
            deletions += 1
            gappos.append(str(q_pos))
        elif x == y:
            matches += 1
        else:
            mismatches += 1
        pos_rows.append({
            "alignment_position": idx, "ref_id": ref_id, "ref": x, "alt": y,
            "posref": ref_pos if x != "-" else "", "posalt": q_pos if y != "-" else "",
            "match": "1" if x == y and x != "-" else "0"
        })
    aligned_len = matches + mismatches + insertions + deletions
    q_aligned_len = len(query_seq) - deletions
    ident_pct = round(matches / aligned_len * 100, 2) if aligned_len else 0.0
    cov_pct = round(q_aligned_len / len(query_seq) * 100, 2)
    score = round(matches - 0.5 * mismatches - 1.5 * (insertions + deletions), 2)
    stats = {
        "query_id": query_id, "ref_id": ref_id, "query_length": len(query_seq), "ref_length": len(ref_seq),
        "aligned_length": aligned_len, "matches": matches, "mismatches": mismatches,
        "insertions_in_query": insertions, "deletions_in_query": deletions,
        "identity_percent": ident_pct, "query_coverage_percent": cov_pct, "alignment_score": score,
        "range": f"{query_start + 1}-{query_end}", "ref_start": ref_start, "ref_end": ref_end,
        "gappos": ",".join(gappos), "sequence": ref_seq[ref_start:ref_end]
    }
    return stats, pos_rows

def finish_log(start_time, msg):
    elapsed = time.perf_counter() - start_time
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_mb = usage / 1024 if usage < 10**7 else usage / 1024**2
    dt = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {dt} | {msg} | execution time: {elapsed:.2f}s | max memory: {mem_mb:.1f} MB")

###################################### MAIN EXECUTION
def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Pairwise local sequence alignment")
    parser.add_argument("-i", "--input", help="Query sequence file path(s), txt list, or directory")
    parser.add_argument("-o", "--output", help="Alignment output directory (defaults to aln_outdir in config)")
    parser.add_argument("--sequence", help="Manual reference sequence string")
    parser.add_argument("-j", "--json", nargs="?", const=DEFAULT_CONFIG_PATH, default=DEFAULT_CONFIG_PATH, help="Path to config JSON")
    parser.add_argument("--no-json", action="store_true")
    args = parser.parse_args()
    defaults = {
        "chain_seqs_outdir": DEFAULT_CHAIN_SEQS_OUTDIR,
        "chain_refseqs_outdir": DEFAULT_CHAIN_REFSEQS_OUTDIR,
        "aln_outdir": DEFAULT_ALN_OUTDIR
    }
    cfg = resolve_config(args.json, defaults)
    query_files = resolve_input_files(args.input, cfg["chain_seqs_outdir"])
    ref_dir = Path(cfg["chain_refseqs_outdir"])
    aln_dir = Path(args.output) if args.output else Path(cfg["aln_outdir"])
    aln_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PYTHON-INFO] Aligning {len(query_files)} query sequence file(s) into {aln_dir}")
    count = 0
    for q_file in tqdm(query_files, desc="Aligning sequences"):
        q_seq, _ = parse_fasta_file(q_file) if q_file.suffix in [".fasta", ".fa"] else (q_file.read_text().strip(), "")
        if not q_seq:
            continue
        q_id = q_file.stem
        if args.sequence:
            r_seq, r_id = args.sequence.strip(), "manual_ref"
        else:
            r_file = ref_dir / f"{q_id}.fasta"
            r_seq, r_id = parse_fasta_file(r_file)
            if not r_seq:
                continue
        stats, pos_rows = run_alignment(r_seq, q_seq, r_id, q_id)
        if stats:
            pd.DataFrame([stats]).to_csv(aln_dir / f"{q_id}_stats.tsv", sep="\t", index=False)
            pd.DataFrame(pos_rows).to_csv(aln_dir / f"{q_id}_positions.tsv", sep="\t", index=False)
            count += 1
    finish_log(start, f"Completed {count} pairwise alignments out of {len(query_files)} queries")

if __name__ == "__main__":
    main()



==> src/04_reindex_pdb.py <==
#!/usr/bin/env python
"""Batch reindex PDB chain residue sequence numbers with detailed residue logging."""

###################################### DEPENDENCIES AND CONSTANTS
import argparse
import datetime
import io
import json
import resource
import time
import warnings
from pathlib import Path

from Bio.PDB import Chain, MMCIFParser, Model, PDBIO, PDBParser, Structure
from Bio.PDB.Polypeptide import protein_letters_3to1_extended
from Bio.PDB.Residue import Residue
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

SEED = 42
DEFAULT_ALN_OUTDIR = "data/aln"
DEFAULT_CHAIN_OUTDIR = "data/chains"
DEFAULT_CLEAN_OUTDIR = "data/clean"
DEFAULT_CONFIG_PATH = "configs/param_configs.json"

###################################### CONFIG
def resolve_config(json_arg, defaults):
    if json_arg is None:
        return defaults
    p = Path(json_arg)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(defaults, indent=2))
        return defaults
    try:
        cfg = json.loads(p.read_text())
    except Exception:
        cfg = {}
    updated = False
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
            updated = True
    if updated:
        p.write_text(json.dumps(cfg, indent=2))
    return cfg

###################################### REINDEXING LOGIC
def load_mapping_table(table_path):
    df = pd.read_csv(table_path, sep="\t" if str(table_path).endswith(".tsv") else ",")
    mapping = {}
    alt_col = next((c for c in ["posalt", "old_resnum", "query_pos", "alt"] if c in df.columns), df.columns[0])
    ref_col = next((c for c in ["posref", "new_resnum", "ref_pos", "ref"] if c in df.columns), df.columns[1])
    for _, row in df.iterrows():
        try:
            alt_val = int(row[alt_col])
            ref_val = int(row[ref_col])
            mapping[alt_val] = ref_val
        except (ValueError, TypeError):
            continue
    return mapping

def reindex_pdb(pdb_path, mapping):
    parser = PDBParser(QUIET=True) if str(pdb_path).endswith(".pdb") else MMCIFParser(QUIET=True)
    struct = parser.get_structure("chain", str(pdb_path))
    chains = list(struct.get_chains())
    if not chains:
        return "", {}
    chain = chains[0]
    all_residues = list(chain)
    total_residues = len(all_residues)
    seq_pos = 0
    sequential_map = {}
    protein_residues = 0
    non_protein_removed = 0
    for residue in all_residues:
        resname = residue.get_resname().strip().upper()
        if residue.id[0] == " " and resname in protein_letters_3to1_extended:
            seq_pos += 1
            sequential_map[residue.id] = seq_pos
            protein_residues += 1
        else:
            non_protein_removed += 1
    new_struct = Structure.Structure("reindexed")
    new_model = Model.Model(0)
    new_chain = Chain.Chain(chain.id)
    new_model.add(new_chain)
    new_struct.add(new_model)
    reindexed_count = 0
    unmapped_removed = 0
    added_positions = set()
    for residue in all_residues:
        res_id = residue.id
        resname = residue.get_resname().strip().upper()
        if res_id[0] != " " or resname not in protein_letters_3to1_extended:
            continue
        qpos = sequential_map.get(res_id, res_id[1])
        target_pos = mapping.get(qpos, mapping.get(res_id[1]))
        if target_pos is not None:
            target_pos = int(target_pos)
            if target_pos in added_positions:
                unmapped_removed += 1
                continue
            res_copy = Residue((" ", target_pos, " "), residue.get_resname(), residue.get_segid())
            for atom in residue:
                res_copy.add(atom.copy())
            added_positions.add(target_pos)
            new_chain.add(res_copy)
            reindexed_count += 1
        else:
            unmapped_removed += 1
    writer = PDBIO()
    writer.set_structure(new_struct)
    buf = io.StringIO()
    writer.save(buf)
    stats = {
        "total_residues": total_residues,
        "protein_residues": protein_residues,
        "non_protein_removed": non_protein_removed,
        "reindexed_residues": reindexed_count,
        "unmapped_removed": unmapped_removed,
    }
    return buf.getvalue(), stats

def find_mapping_file(stem, aln_dir):
    candidates = [
        aln_dir / f"{stem}_positions.tsv",
        aln_dir / f"{stem}_positions.csv",
        aln_dir / f"{stem}.tsv",
        aln_dir / f"{stem}.csv",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None

def finish_log(start_time, msg):
    elapsed = time.perf_counter() - start_time
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_mb = usage / 1024 if usage < 10**7 else usage / 1024**2
    dt = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {dt} | {msg} | execution time: {elapsed:.2f}s | max memory: {mem_mb:.1f} MB")

###################################### MAIN EXECUTION
def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Batch reindex PDB chains with detailed residue statistics")
    parser.add_argument("-i", "--input", help="PDB file path or directory (defaults to chain_outdir in config)")
    parser.add_argument("-t", "--table", help="Specific mapping table file (optional for batch mode)")
    parser.add_argument("-o", "--output", help="Specific output file or directory")
    parser.add_argument("-j", "--json", nargs="?", const=DEFAULT_CONFIG_PATH, default=DEFAULT_CONFIG_PATH, help="Path to config JSON")
    parser.add_argument("--no-json", action="store_true", help="Skip JSON existence and update checks")
    args = parser.parse_args()
    defaults = {
        "chain_outdir": DEFAULT_CHAIN_OUTDIR,
        "aln_outdir": DEFAULT_ALN_OUTDIR,
        "clean_outdir": DEFAULT_CLEAN_OUTDIR,
    }
    cfg = resolve_config(args.json, defaults)
    chain_dir = Path(args.input) if args.input else Path(cfg["chain_outdir"])
    aln_dir = Path(cfg["aln_outdir"])
    clean_dir = Path(cfg["clean_outdir"])
    if args.output and not Path(args.output).suffix:
        clean_dir = Path(args.output)
    clean_dir.mkdir(parents=True, exist_ok=True)
    if chain_dir.is_file():
        pdb_files = [chain_dir]
    elif chain_dir.is_dir():
        pdb_files = sorted(list(chain_dir.glob("*.pdb")) + list(chain_dir.glob("*.ent")))
    else:
        print(f"[PYTHON-INFO] Target directory/file not found: {chain_dir}")
        return
    print(f"[PYTHON-INFO] Scanning {len(pdb_files)} PDB chain(s) in {chain_dir} for mapping tables in {aln_dir}")
    totals = {
        "processed_files": 0,
        "skipped_files": 0,
        "total_residues": 0,
        "protein_residues": 0,
        "non_protein_removed": 0,
        "reindexed_residues": 0,
        "unmapped_removed": 0,
    }
    per_chain_reports = []
    for pdb_path in tqdm(pdb_files, desc="Reindexing chains"):
        table_path = Path(args.table) if args.table else find_mapping_file(pdb_path.stem, aln_dir)
        if not table_path or not table_path.is_file():
            totals["skipped_files"] += 1
            continue
        mapping = load_mapping_table(table_path)
        if not mapping:
            totals["skipped_files"] += 1
            continue
        pdb_text, stats = reindex_pdb(pdb_path, mapping)
        if not stats:
            totals["skipped_files"] += 1
            continue
        if args.output and Path(args.output).suffix:
            out_file = Path(args.output)
        else:
            out_file = clean_dir / f"{pdb_path.stem}_reindexed.pdb"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(pdb_text)
        totals["processed_files"] += 1
        for k in ["total_residues", "protein_residues", "non_protein_removed", "reindexed_residues", "unmapped_removed"]:
            totals[k] += stats[k]
        report_str = (
            f"{pdb_path.stem}: Total={stats['total_residues']} | "
            f"Protein={stats['protein_residues']} | "
            f"Reindexed={stats['reindexed_residues']} | "
            f"Unmapped Removed={stats['unmapped_removed']} | "
            f"Non-Protein Removed={stats['non_protein_removed']}"
        )
        per_chain_reports.append(report_str)
    print("\n[PYTHON-INFO] Detailed Per-Chain Reindexing Breakdown:")
    for report in per_chain_reports:
        print(f"  - {report}")
    summary_msg = (
        f"Reindexed {totals['processed_files']}/{len(pdb_files)} chains "
        f"({totals['skipped_files']} skipped). "
        f"Residues -> Total: {totals['total_residues']}, "
        f"Protein: {totals['protein_residues']}, "
        f"Reindexed: {totals['reindexed_residues']}, "
        f"Unmapped Removed: {totals['unmapped_removed']}, "
        f"Non-Protein Removed: {totals['non_protein_removed']}"
    )
    finish_log(start, summary_msg)

if __name__ == "__main__":
    main()
==> src/05_compute_pdb_contacts.py <==
#!/usr/bin/env python
"""Compute residue-residue contact maps in nanometers for cleaned PDB files grouped by PDB ID."""
###################################### DEPENDENCIES AND CONSTANTS
import argparse
import datetime
import json
import resource
import time
import warnings
from collections import defaultdict
from pathlib import Path
import mdtraj as md
import numpy as np
import pandas as pd
from tqdm import tqdm
DEFAULT_CLEAN_OUTDIR = "data/clean"
DEFAULT_CONFIG_PATH = "configs/param_configs.json"
DEFAULT_CONTACT_TYPE = "inter"
DEFAULT_CONTACTS_OUTDIR = "data/contacts"
DEFAULT_CUTOFF = 0.8
SEED = 42
###################################### CONFIG
def resolve_config(json_arg, defaults):
    if json_arg is None:
        return defaults
    p = Path(json_arg)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(defaults, indent=2))
        return defaults
    try:
        cfg = json.loads(p.read_text())
    except Exception:
        cfg = {}
    updated = False
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
            updated = True
    if updated:
        p.write_text(json.dumps(cfg, indent=2))
    return cfg
###################################### CONTACT LOGIC
def extract_pdb_id(filename_stem):
    return filename_stem.split("_")[0].lower()
def load_chain_traj(pdb_path):
    try:
        traj = md.load(str(pdb_path))
    except Exception as e:
        print(f"[PYTHON-INFO] Failed to parse {pdb_path}: {e}")
        return None, None
    stem_parts = pdb_path.stem.split("_")
    chain_id = stem_parts[1] if len(stem_parts) > 1 else "A"
    if traj.topology.n_chains == 0:
        return chain_id, traj
    for chain in traj.topology.chains:
        cid = chain.chain_id if chain.chain_id.strip() else chain_id
        return cid, traj
    return chain_id, traj
def calculate_contacts_for_pdb(chain_data_list, cutoff=0.8, contact_type="inter"):
    contacts = []
    calc_intra = contact_type in ("intra", "all")
    calc_inter = contact_type in ("inter", "all")
    for i in range(len(chain_data_list)):
        cid1, traj1 = chain_data_list[i]
        top1 = traj1.topology
        if calc_intra and top1.n_residues > 1:
            pairs = []
            for r1 in range(top1.n_residues):
                for r2 in range(r1 + 1, top1.n_residues):
                    res1 = top1.residue(r1)
                    res2 = top1.residue(r2)
                    if abs(res1.resSeq - res2.resSeq) < 1:
                        continue
                    pairs.append([r1, r2])
            if pairs:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, message=".*No alpha carbon.*")
                    dists, res_pairs = md.compute_contacts(traj1, contacts=pairs, scheme="ca", ignore_nonprotein=False, periodic=False)
                for k, (r1_idx, r2_idx) in enumerate(res_pairs):
                    min_d = float(dists[0, k])
                    if min_d <= cutoff:
                        res1 = top1.residue(r1_idx)
                        res2 = top1.residue(r2_idx)
                        contacts.append({
                            "chain_1": cid1,
                            "resnum_1": res1.resSeq,
                            "resname_1": res1.name.strip(),
                            "chain_2": cid1,
                            "resnum_2": res2.resSeq,
                            "resname_2": res2.name.strip(),
                            "min_distance_nm": round(min_d, 4),
                            "contact_type": "intra",
                        })
        if calc_inter:
            for j in range(i + 1, len(chain_data_list)):
                cid2, traj2 = chain_data_list[j]
                top2 = traj2.topology
                combined = traj1.stack(traj2)
                n1 = top1.n_residues
                pairs = [[r1, n1 + r2] for r1 in range(n1) for r2 in range(top2.n_residues)]
                if pairs:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning, message=".*No alpha carbon.*")
                        dists, res_pairs = md.compute_contacts(combined, contacts=pairs, scheme="ca", ignore_nonprotein=False, periodic=False)
                    for k, (r1_idx, r2_idx) in enumerate(res_pairs):
                        min_d = float(dists[0, k])
                        if min_d <= cutoff:
                            res1 = top1.residue(r1_idx)
                            res2 = top2.residue(r2_idx - n1)
                            contacts.append({
                                "chain_1": cid1,
                                "resnum_1": res1.resSeq,
                                "resname_1": res1.name.strip(),
                                "chain_2": cid2,
                                "resnum_2": res2.resSeq,
                                "resname_2": res2.name.strip(),
                                "min_distance_nm": round(min_d, 4),
                                "contact_type": "inter",
                            })
    return contacts
def finish_log(start_time, msg):
    elapsed = time.perf_counter() - start_time
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_mb = usage / 1024 if usage < 10**7 else usage / 1024**2
    dt = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {dt} | {msg} | execution time: {elapsed:.2f}s | max memory: {mem_mb:.1f} MB")
###################################### MAIN EXECUTION
def main():
    np.random.seed(SEED)
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Compute contact maps across cleaned PDB chains grouped by PDB ID")
    parser.add_argument("-i", "--input", help="Clean PDB file path or directory (defaults to clean_outdir in config)")
    parser.add_argument("-o", "--output", help="Output directory for contact TSVs (defaults to contacts_outdir in config)")
    parser.add_argument("-d", "--cutoff", type=float, help="Heavy atom distance cutoff threshold in nanometers (default: 0.8 nm)")
    parser.add_argument("--type", choices=["inter", "intra", "all"], default=DEFAULT_CONTACT_TYPE, help="Type of contacts to calculate: 'inter' (default), 'intra', or 'all'")
    parser.add_argument("-j", "--json", nargs="?", const=DEFAULT_CONFIG_PATH, default=DEFAULT_CONFIG_PATH, help="Path to config JSON")
    args = parser.parse_args()
    defaults = {"clean_outdir": DEFAULT_CLEAN_OUTDIR, "contact_type": DEFAULT_CONTACT_TYPE, "contacts_outdir": DEFAULT_CONTACTS_OUTDIR, "distance_threshold": DEFAULT_CUTOFF}
    cfg = resolve_config(args.json, defaults)
    clean_dir = Path(args.input) if args.input else Path(cfg["clean_outdir"])
    contacts_dir = Path(args.output) if args.output else Path(cfg["contacts_outdir"])
    cutoff = args.cutoff if args.cutoff is not None else float(cfg.get("distance_threshold", DEFAULT_CUTOFF))
    contact_type = args.type if args.type != DEFAULT_CONTACT_TYPE else cfg.get("contact_type", DEFAULT_CONTACT_TYPE)
    contacts_dir.mkdir(parents=True, exist_ok=True)
    if clean_dir.is_file():
        pdb_files = [clean_dir]
    elif clean_dir.is_dir():
        pdb_files = sorted(list(clean_dir.glob("*.pdb")) + list(clean_dir.glob("*.ent")))
    else:
        print(f"[PYTHON-INFO] Target directory/file not found: {clean_dir}")
        return
    groups = defaultdict(list)
    for fpath in pdb_files:
        pdb_id = extract_pdb_id(fpath.stem)
        groups[pdb_id].append(fpath)
    print(f"[PYTHON-INFO] Found {len(pdb_files)} PDB chain(s) spanning {len(groups)} unique PDB ID(s) in {clean_dir}")
    print(f"[PYTHON-INFO] Computing '{contact_type}' contacts (cutoff: {cutoff} nm) into {contacts_dir}")
    total_contacts = 0
    total_inter = 0
    total_intra = 0
    for pdb_id, files in tqdm(groups.items(), desc="Calculating contacts"):
        chain_data_list = []
        for fpath in files:
            cid, traj = load_chain_traj(fpath)
            if traj is not None and traj.topology.n_residues > 0:
                chain_data_list.append((cid, traj))
        if not chain_data_list:
            continue
        contacts = calculate_contacts_for_pdb(chain_data_list, cutoff=cutoff, contact_type=contact_type)
        if contacts:
            df = pd.DataFrame(contacts)
            df.insert(0, "pdb_id", pdb_id)
            out_file = contacts_dir / f"{pdb_id}_contacts_{contact_type}.tsv"
            df.to_csv(out_file, sep="\t", index=False)
            total_contacts += len(df)
            total_inter += len(df[df["contact_type"] == "inter"])
            total_intra += len(df[df["contact_type"] == "intra"])
    finish_log(start, f"Saved contacts for {len(groups)} PDB IDs -> Total: {total_contacts} (Inter-chain: {total_inter}, Intra-chain: {total_intra}) at ≤{cutoff} nm")
if __name__ == "__main__":
    main()



==> src/06_summarize_contacts.py <==
#!/usr/bin/env python
"""Generate sequence contact string CSV for cleaned PDB chains using alignment position files."""
###################################### DEPENDENCIES AND CONSTANTS
import argparse
import datetime
import json
import resource
import time
from pathlib import Path
from Bio import SeqIO
import pandas as pd
from tqdm import tqdm
DEFAULT_ALN_DIR = "data/aln"
DEFAULT_CLEAN_OUTDIR = "data/clean"
DEFAULT_CONFIG_PATH = "configs/param_configs.json"
DEFAULT_CONTACT_TYPE = "inter"
DEFAULT_CONTACTS_OUTDIR = "data/contacts"
DEFAULT_OUTPUT_PATH = "data/contacts/contacts_sequences.csv"
DEFAULT_REFSEQS_DIR = "data/refseqs"
SEED = 42
###################################### CONFIG
def resolve_config(json_arg, defaults):
    if json_arg is None: return defaults
    p = Path(json_arg)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(defaults, indent=2))
        return defaults
    try: cfg = json.loads(p.read_text())
    except Exception: cfg = {}
    updated = False
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
            updated = True
    if updated: p.write_text(json.dumps(cfg, indent=2))
    return cfg
###################################### REFSEQ LOADER
def load_all_refseqs(*search_paths):
    refseq_map = {}
    for path_str in search_paths:
        if not path_str: continue
        p = Path(path_str)
        if not p.exists(): continue
        files = [p] if p.is_file() else list(p.rglob("*.fasta")) + list(p.rglob("*.fa")) + list(p.rglob("*.faa"))
        for f in files:
            try:
                for record in SeqIO.parse(f, "fasta"):
                    seq_str = str(record.seq).upper()
                    rec_id = record.id.strip().lower()
                    refseq_map[rec_id] = seq_str
                    stem_clean = f.stem.replace("_refseq", "").replace("_positions", "").lower()
                    refseq_map[stem_clean] = seq_str
            except Exception: continue
    return refseq_map
###################################### ALIGNMENT AND CONTACT PARSER
def parse_alignment_file(aln_path):
    try: df = pd.read_csv(aln_path, sep=r"\s+|\t|,", engine="python")
    except Exception: return None
    if not {"posref", "posalt"}.issubset(set(df.columns)): return None
    df["posref"] = pd.to_numeric(df["posref"], errors="coerce")
    df["posalt"] = pd.to_numeric(df["posalt"], errors="coerce")
    ref_df = df.dropna(subset=["posref"]).copy()
    if ref_df.empty: return None
    ref_df["posref"] = ref_df["posref"].astype(int)
    stem = aln_path.stem.replace("_positions", "")
    parts = stem.split("_")
    pdb_id = parts[0].upper()
    chain_id = parts[1] if len(parts) > 1 else "A"
    min_posref = int(ref_df["posref"].min())
    max_posref = int(ref_df["posref"].max())
    ref_seq_dict = {}
    if "ref" in ref_df.columns:
        for _, row in ref_df.iterrows():
            ref_seq_dict[int(row["posref"])] = str(row["ref"]).upper()
    pdb_df = ref_df.dropna(subset=["posalt"]).copy()
    pdb_df["posalt"] = pdb_df["posalt"].astype(int)
    pdb_ref_pos = set(pdb_df["posref"])
    return {"pdb_id": pdb_id, "chain_id": chain_id, "chain_key": f"{pdb_id}_{chain_id}".lower(), "min_posref": min_posref, "max_posref": max_posref, "aln_df": pdb_df, "pdb_ref_pos": pdb_ref_pos, "ref_seq_dict": ref_seq_dict}
def load_contacts_for_pdb(contacts_dir, pdb_id, contact_type="inter"):
    pdb_lower = pdb_id.lower()
    candidates = [contacts_dir / f"{pdb_lower}_contacts_{contact_type}.tsv", contacts_dir / f"{pdb_lower}_contacts_{contact_type}.csv", contacts_dir / f"{pdb_id}_contacts_{contact_type}.tsv", contacts_dir / f"{pdb_id}_contacts_{contact_type}.csv", contacts_dir / f"{pdb_lower}_contacts.tsv", contacts_dir / f"{pdb_lower}_contacts.csv", contacts_dir / f"{pdb_id}_contacts.tsv", contacts_dir / f"{pdb_id}_contacts.csv", contacts_dir / f"{pdb_lower}.tsv", contacts_dir / f"{pdb_lower}.csv", contacts_dir / f"{pdb_id}.tsv", contacts_dir / f"{pdb_id}.csv"]
    for cand in candidates:
        if cand.is_file():
            try: return pd.read_csv(cand, sep=r"\s+|\t|,", engine="python")
            except Exception: pass
    return None
def finish_log(start_time, msg):
    elapsed = time.perf_counter() - start_time
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_mb = usage / 1024 if usage < 10**7 else usage / 1024**2
    dt = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {dt} | {msg} | execution time: {elapsed:.2f}s | max memory: {mem_mb:.1f} MB")
###################################### DEDUPLICATION AND MERGING FORMATTING
def format_complex_and_id(pdb_chains):
    sorted_pdbs = sorted(set(p.upper() for p in pdb_chains.keys()))
    complex_str = "_".join(sorted_pdbs)
    id_parts = []
    for pdb_id in sorted_pdbs:
        chains = []
        for k, v in pdb_chains.items():
            if k.upper() == pdb_id: chains.extend(v)
        unique_chains = sorted(set(chains))
        chains_str = "".join(unique_chains)
        id_parts.append(f"{pdb_id}-{chains_str}")
    id_str = "_".join(id_parts)
    return complex_str, id_str
def merge_pdb_chains(dict1, dict2):
    merged = {}
    for d in (dict1, dict2):
        for pdb_id, chains in d.items():
            p_upper = pdb_id.upper()
            if p_upper not in merged: merged[p_upper] = []
            for c in chains:
                if c not in merged[p_upper]: merged[p_upper].append(c)
    return merged
def align_and_merge(r1, r2):
    s1, c1 = r1["sequence"], r1["contacts"]
    s2, c2 = r2["sequence"], r2["contacts"]
    l1, l2 = len(s1), len(s2)
    best_off = None
    for off in range(-20, 21):
        st1, en1 = max(0, off), min(l1, l2 + off)
        st2, en2 = max(0, -off), min(l2, l1 - off)
        if en1 <= st1: continue
        match = True
        for i, j in zip(range(st1, en1), range(st2, en2)):
            if s1[i] != "X" and s2[j] != "X" and s1[i] != s2[j]:
                match = False
                break
        if match:
            best_off = off
            break
    if best_off is None: return None
    off = best_off
    shift1 = -min(0, off)
    shift2 = off - min(0, off)
    new_len = max(l1 + shift1, l2 + shift2)
    seq_chars, con_chars = [], []
    for k in range(new_len):
        i1, i2 = k - shift1, k - shift2
        ch1 = s1[i1] if 0 <= i1 < l1 else "X"
        ch2 = s2[i2] if 0 <= i2 < l2 else "X"
        seq_chars.append(ch1 if ch1 != "X" else ch2)
        ct1 = c1[i1] if 0 <= i1 < l1 else "X"
        ct2 = c2[i2] if 0 <= i2 < l2 else "X"
        if ct1 == "1" or ct2 == "1": con_chars.append("1")
        elif ct1 == "0" or ct2 == "0": con_chars.append("0")
        else: con_chars.append("X")
    return "".join(seq_chars), "".join(con_chars)
def deduplicate_records(records, dedup_max=False):
    groups = {}
    if dedup_max: groups["ALL"] = records
    else:
        for r in records: groups.setdefault(r["complex"], []).append(r)
    final_rows = []
    for grp_recs in tqdm(groups.values(), desc="Deduplicating clusters"):
        active = grp_recs[:]
        changed = True
        while changed:
            changed = False
            new_active = []
            skip = set()
            for i in range(len(active)):
                if i in skip: continue
                curr = active[i]
                for j in range(i + 1, len(active)):
                    if j in skip: continue
                    target = active[j]
                    merged_res = align_and_merge(curr, target)
                    if merged_res is not None:
                        m_seq, m_con = merged_res
                        merged_chains = merge_pdb_chains(curr["pdb_chains"], target["pdb_chains"])
                        new_cmplx, new_id = format_complex_and_id(merged_chains)
                        curr = {"complex": new_cmplx, "id": new_id, "sequence": m_seq, "contacts": m_con, "pdb_chains": merged_chains}
                        skip.add(j)
                        changed = True
                new_active.append(curr)
            active = new_active
        final_rows.extend(active)
    return final_rows
###################################### MAIN EXECUTION
def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Build contacts-sequence 3-state CSV using alignment positions data")
    parser.add_argument("-i", "--input", "-a", "--aln", help="Alignment directory")
    parser.add_argument("-c", "--contacts", help="Contacts directory")
    parser.add_argument("-r", "--refseqs", help="Optional reference sequences directory")
    parser.add_argument("-o", "--output", help="Output CSV path")
    parser.add_argument("--type", choices=["inter", "intra", "all"], default=DEFAULT_CONTACT_TYPE)
    parser.add_argument("-j", "--json", nargs="?", const=DEFAULT_CONFIG_PATH, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--dedup-max", action="store_true")
    args = parser.parse_args()
    defaults = {"aln_outdir": DEFAULT_ALN_DIR, "clean_outdir": DEFAULT_CLEAN_OUTDIR, "contact_type": DEFAULT_CONTACT_TYPE, "contacts_outdir": DEFAULT_CONTACTS_OUTDIR, "contacts_seq_output": DEFAULT_OUTPUT_PATH, "refseqs_dir": DEFAULT_REFSEQS_DIR}
    cfg = resolve_config(args.json, defaults)
    aln_dir = Path(args.input) if args.input else Path(cfg.get("aln_outdir", DEFAULT_ALN_DIR))
    contacts_dir = Path(args.contacts) if args.contacts else Path(cfg["contacts_outdir"])
    refseq_dir = Path(args.refseqs) if args.refseqs else Path(cfg.get("refseqs_dir", DEFAULT_REFSEQS_DIR))
    contact_type = args.type if args.type != DEFAULT_CONTACT_TYPE else cfg.get("contact_type", DEFAULT_CONTACT_TYPE)
    if args.output: out_file = Path(args.output)
    elif "contacts_seq_output" in cfg and cfg["contacts_seq_output"] != DEFAULT_OUTPUT_PATH: out_file = Path(cfg["contacts_seq_output"])
    else: out_file = contacts_dir / f"contacts_sequences_{contact_type}.csv"
    tag = "_dedup_max" if args.dedup_max else ("_dedup" if args.dedup else "")
    if tag and not out_file.stem.endswith(tag): out_file = out_file.with_name(f"{out_file.stem}{tag}{out_file.suffix}")
    if out_file.suffix.lower() != ".csv": out_file = out_file.with_suffix(".csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    refseq_map = load_all_refseqs(refseq_dir, aln_dir)
    print(f"[PYTHON-INFO] Loaded {len(refseq_map)} sequence records")
    if aln_dir.is_file(): aln_files = [aln_dir]
    elif aln_dir.is_dir():
        pos_files = list(aln_dir.glob("*_positions.tsv")) + list(aln_dir.glob("*_positions.csv"))
        if not pos_files:
            pos_files = [f for f in aln_dir.glob("*.tsv") if not f.name.endswith("_stats.tsv")] + [f for f in aln_dir.glob("*.csv") if not f.name.endswith("_stats.csv")]
        aln_files = sorted(set(pos_files))
    else:
        print(f"[PYTHON-INFO] Alignment path not found: {aln_dir}")
        return
    rows = []
    processed_keys = set()
    for aln_path in tqdm(aln_files, desc="Building contact sequences"):
        info = parse_alignment_file(aln_path)
        if not info: continue
        pdb_id = info["pdb_id"]
        chain_id = info["chain_id"]
        chain_key = info["chain_key"]
        if chain_key in processed_keys: continue
        min_pos = info["min_posref"]
        max_pos = info["max_posref"]
        aln_df = info["aln_df"]
        pdb_ref_pos = info["pdb_ref_pos"]
        ref_seq_dict = info["ref_seq_dict"]
        ref_contact_pos = set()
        contacts_df = load_contacts_for_pdb(contacts_dir, pdb_id, contact_type)
        if contacts_df is not None and not contacts_df.empty:
            contacts_df.columns = [str(c).lower().strip() for c in contacts_df.columns]
            c1_col = next((c for c in ["chain_1", "chain1", "chain_a"] if c in contacts_df.columns), None)
            c2_col = next((c for c in ["chain_2", "chain2", "chain_b"] if c in contacts_df.columns), None)
            r1_col = next((c for c in ["resnum_1", "resnum1", "res_1", "resnum_a"] if c in contacts_df.columns), None)
            r2_col = next((c for c in ["resnum_2", "resnum2", "res_2", "resnum_b"] if c in contacts_df.columns), None)
            structure_contact_pos = set()
            target_chain = str(chain_id).strip()
            if c1_col and r1_col:
                mask1 = contacts_df[c1_col].astype(str).str.strip() == target_chain
                res1 = pd.to_numeric(contacts_df.loc[mask1, r1_col], errors="coerce").dropna().astype(int)
                structure_contact_pos.update(res1)
            if c2_col and r2_col:
                mask2 = contacts_df[c2_col].astype(str).str.strip() == target_chain
                res2 = pd.to_numeric(contacts_df.loc[mask2, r2_col], errors="coerce").dropna().astype(int)
                structure_contact_pos.update(res2)
            if structure_contact_pos:
                ref_contact_pos = set(aln_df[aln_df["posalt"].isin(structure_contact_pos)]["posref"].astype(int))
        full_refseq = refseq_map.get(chain_key) or refseq_map.get(pdb_id.lower()) or refseq_map.get(pdb_id)
        target_len = max_pos - min_pos + 1
        if full_refseq:
            start_idx = max(0, min_pos - 1)
            end_idx = min(len(full_refseq), max_pos)
            extracted_sequence = full_refseq[start_idx:end_idx]
        else:
            extracted_sequence = "".join(ref_seq_dict.get(pos, "X") for pos in range(min_pos, max_pos + 1))
        if len(extracted_sequence) < target_len:
            extracted_sequence = extracted_sequence.ljust(target_len, "X")
        elif len(extracted_sequence) > target_len:
            extracted_sequence = extracted_sequence[:target_len]
        contact_bits = ["1" if pos in ref_contact_pos else ("0" if pos in pdb_ref_pos else "X") for pos in range(min_pos, max_pos + 1)]
        contacts_str = "".join(contact_bits)
        if len(extracted_sequence) != len(contacts_str):
            print(f"[PYTHON-INFO] Forcing length equality for {pdb_id}-{chain_id}: seq={len(extracted_sequence)} cont={len(contacts_str)}")
            ml = min(len(extracted_sequence), len(contacts_str))
            extracted_sequence = extracted_sequence[:ml]
            contacts_str = contacts_str[:ml]
        assert len(extracted_sequence) == len(contacts_str), f"Post-force length mismatch {pdb_id}-{chain_id}"
        pdb_chains = {pdb_id: [chain_id]}
        cmplx, rec_id = format_complex_and_id(pdb_chains)
        rows.append({"complex": cmplx, "id": rec_id, "sequence": extracted_sequence, "contacts": contacts_str, "pdb_chains": pdb_chains})
        processed_keys.add(chain_key)
    if args.dedup_max: rows = deduplicate_records(rows, dedup_max=True)
    elif args.dedup: rows = deduplicate_records(rows, dedup_max=False)
    if rows:
        out_df = pd.DataFrame(rows)[["complex", "id", "sequence", "contacts"]]
        for _, r in out_df.iterrows():
            if len(r["sequence"]) != len(r["contacts"]):
                ml = min(len(r["sequence"]), len(r["contacts"]))
                out_df.loc[_, "sequence"] = r["sequence"][:ml]
                out_df.loc[_, "contacts"] = r["contacts"][:ml]
                print(f"[PYTHON-INFO] Final force equality for {r['id']}: truncated to {ml}")
        out_df = out_df[out_df.apply(lambda r: len(r["sequence"]) == len(r["contacts"]), axis=1)]
        out_df.to_csv(out_file, index=False)
        summary_msg = f"Saved {len(out_df)} chain contact sequences to CSV format at {out_file}"
    else:
        summary_msg = f"No chains found to write to {out_file}"
    finish_log(start, summary_msg)
if __name__ == "__main__":
    main()


==> src/plot_coverage.py <==
#!/usr/bin/env python
"""Generate multi-page PDF coverage plots from local-alignment statistics."""

###################################### DEPENDENCIES AND CONSTANTS
import argparse
import datetime
import json
import os
import re
import resource
import time
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

A4_WIDTH_IN = 8.27
DEFAULT_ALN_OUTDIR = "data/chains/aln"
DEFAULT_CONFIG_PATH = "configs/param_configs.json"
DEFAULT_OUTPUT = "coverage.pdf"
DEFAULT_PLOTS_OUTDIR = "data/plots"
LEGEND_HEIGHT_CM = 1.2
PAGE_SIZE = 100
ROW_HEIGHT_CM = 1.6
SEED = 42
TITLE_HEIGHT_CM = 0.6

###################################### CONFIG
def resolve_config(json_arg, defaults, no_json):
    if no_json:
        return dict(defaults)
    path = Path(json_arg)
    cfg = {}
    if path.exists():
        try:
            cfg = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            cfg = {}
    changed = False
    for key, value in defaults.items():
        if key not in cfg:
            cfg[key] = value
            changed = True
    if changed or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2) + "\n")
    return {key: cfg[key] for key in defaults}

###################################### INPUT
def load_stats(aln_dir):
    path = Path(aln_dir)
    files = [path] if path.is_file() else sorted(path.glob("*_stats.tsv"))
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f, sep="\t")
        except Exception as exc:
            print(f"[PYTHON-INFO] Could not read {f}: {exc}")
            continue
        if df.empty:
            continue
        df["stats_file"] = f.name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    if "query_id" not in df.columns:
        df["query_id"] = df["stats_file"].str.replace("_stats.tsv", "", regex=False)
    parts = df["query_id"].astype(str).str.rsplit("_", n=1, expand=True)
    df["pdb_id"] = parts[0]
    df["chain_id"] = parts[1] if parts.shape[1] > 1 else ""
    return df

def numeric_column(df, name, default=0):
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(default)

###################################### COVERAGE
def parse_gappos(gap_str):
    if pd.isna(gap_str) or not str(gap_str).strip():
        return set()
    gaps = set()
    for part in str(gap_str).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            p = part.split("-")
            try:
                gaps.update(range(int(p[0]), int(p[1]) + 1))
            except ValueError:
                pass
        else:
            try:
                gaps.add(int(part))
            except ValueError:
                pass
    return gaps

def coverage_ranges(row):
    seq = str(row.get("sequence", "")).strip() if pd.notna(row.get("sequence", "")) else ""
    intervals = []
    if seq and seq.upper() != "NAN":
        try:
            rs = int(row.get("ref_start", 1))
        except (ValueError, TypeError):
            rs = 1
        rs = max(1, rs)
        curr_pos = rs
        block_start = None
        for char in seq:
            if char != "-":
                if block_start is None:
                    block_start = curr_pos
                curr_pos += 1
            else:
                if block_start is not None:
                    intervals.append((block_start, curr_pos - 1))
                    block_start = None
        if block_start is not None:
            intervals.append((block_start, curr_pos - 1))
    if not intervals:
        range_val = row.get("range", "")
        if pd.notna(range_val) and str(range_val).strip():
            matches = re.findall(r"(\d+)\s*[\-\.\:]+\s*(\d+)", str(range_val))
            for s, e in matches:
                start, end = int(s), int(e)
                if start >= 0 and end >= start:
                    intervals.append((start, end))
    if not intervals:
        rs_val = row.get("ref_start", "")
        re_val = row.get("ref_end", "")
        rs_list = [int(x) for x in re.findall(r"\d+", str(rs_val))] if pd.notna(rs_val) else []
        re_list = [int(x) for x in re.findall(r"\d+", str(re_val))] if pd.notna(re_val) else []
        if rs_list and re_list and len(rs_list) == len(re_list):
            for s, e in zip(rs_list, re_list):
                if s >= 0 and e >= s:
                    intervals.append((s, e))
        elif len(rs_list) == 1 and len(re_list) == 1:
            s, e = rs_list[0], re_list[0]
            if s >= 0 and e >= s:
                intervals.append((s, e))
    if not intervals:
        return []
    gappos_val = row.get("gappos", "")
    gaps = parse_gappos(gappos_val)
    if not gaps:
        return intervals
    split_intervals = []
    for start, end in intervals:
        curr = start
        for pos in range(start, end + 1):
            if pos in gaps:
                if curr < pos:
                    split_intervals.append((curr, pos - 1))
                curr = pos + 1
        if curr <= end:
            split_intervals.append((curr, end))
    return split_intervals

def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged

def get_nice_step(xmax):
    target = 5
    raw = xmax / target
    mag = 10 ** np.floor(np.log10(max(raw, 1)))
    res = raw / mag
    if res < 1.5:
        factor = 1
    elif res < 3:
        factor = 2
    elif res < 7:
        factor = 5
    else:
        factor = 10
    return int(factor * mag)

###################################### PLOTTING
def generate_plot(df, output_pdf):
    if df.empty:
        print("[PYTHON-INFO] No alignment statistics found; plot skipped")
        return 0
    df = df.copy()
    df["ref_length"] = numeric_column(df, "ref_length", 0).astype(int)
    df["ref_start"] = numeric_column(df, "ref_start", 0).astype(int)
    df["ref_end"] = numeric_column(df, "ref_end", 0).astype(int)
    df["query_length"] = numeric_column(df, "query_length", 0).astype(int)
    df["identity_percent"] = numeric_column(df, "identity_percent", 0)
    df["method"] = "local alignment"

    grouped_rows = []
    for (query_id, pdb_id, chain_id), group in df.groupby(["query_id", "pdb_id", "chain_id"], sort=False):
        ref_len = max(group["ref_length"].max(), 1)
        query_len = max(group["query_length"].max(), 1)
        identity = group["identity_percent"].mean()
        all_intervals = []
        for _, r in group.iterrows():
            all_intervals.extend(coverage_ranges(r))
        merged = merge_intervals(all_intervals)
        grouped_rows.append(
            {
                "query_id": query_id,
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "ref_length": ref_len,
                "query_length": query_len,
                "identity_percent": identity,
                "method": "local alignment",
                "ranges": merged,
            }
        )
    df_plot = pd.DataFrame(grouped_rows)
    df_plot["sort_pdb"] = df_plot["pdb_id"].astype(str).str.upper()
    df_plot = df_plot.sort_values(
        by=["sort_pdb", "chain_id", "ref_length"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    total_records = len(df_plot)
    total_pages = (total_records + PAGE_SIZE - 1) // PAGE_SIZE
    methods = list(df_plot["method"].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {method: cmap(i % 10) for i, method in enumerate(methods)}
    handles = [
        patches.Patch(
            facecolor=color_map[method],
            edgecolor="black",
            linewidth=0.5,
            label=method,
        )
        for method in methods
    ]
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        for page_idx in tqdm(range(total_pages), desc="Plotting PDF pages"):
            chunk = df_plot.iloc[
                page_idx * PAGE_SIZE : (page_idx + 1) * PAGE_SIZE
            ].reset_index(drop=True)
            n = len(chunk)
            fig_h = (
                TITLE_HEIGHT_CM + LEGEND_HEIGHT_CM + n * ROW_HEIGHT_CM
            ) / 2.54
            fig = plt.figure(figsize=(A4_WIDTH_IN, fig_h))
            gs = fig.add_gridspec(
                n + 2,
                1,
                height_ratios=[TITLE_HEIGHT_CM, LEGEND_HEIGHT_CM]
                + [ROW_HEIGHT_CM] * n,
                hspace=0.85,
            )
            ax_title = fig.add_subplot(gs[0])
            ax_legend = fig.add_subplot(gs[1])
            ax_title.axis("off")
            ax_legend.axis("off")
            title = (
                f"PDB Chain Coverage (Page {page_idx + 1}/{total_pages})"
                if total_pages > 1
                else "PDB Chain Coverage"
            )
            ax_title.text(
                0.5,
                0.5,
                title,
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="center",
            )
            ax_legend.legend(
                handles=handles,
                loc="center",
                ncol=min(3, len(handles)),
                frameon=False,
                fontsize=10,
            )
            for i, (_, row) in enumerate(chunk.iterrows()):
                ax = fig.add_subplot(gs[i + 2])
                pdb_id = str(row["pdb_id"]).upper()
                chain_id = str(row["chain_id"])
                ref_len = max(int(row["ref_length"]), 1)
                query_length = max(int(row["query_length"]), 1)
                identity = float(row["identity_percent"])
                ranges = row["ranges"]

                # Calculate total aligned coverage length
                coverage_length = sum(
                    (end - start) if start == 0 else (end - start + 1)
                    for start, end in ranges
                )
                ref_cov_pct = min(100.0, coverage_length / ref_len * 100) if ref_len else 0.0
                query_cov_pct = min(100.0, coverage_length / query_length * 100) if query_length else 0.0

                # Focus X-axis scale (window size <= 2 * total aligned span)
                if ranges:
                    align_min = min(s for s, e in ranges)
                    align_max = max(e for s, e in ranges)
                    aligned_len = max(align_max - align_min + 1, 1)
                    max_window = 2 * aligned_len
                    
                    if ref_len <= max_window:
                        xmin = 0.0
                        xmax = max(ref_len, 10) * 1.05
                    else:
                        center = (align_min + align_max) / 2.0
                        half_w = max_window / 2.0
                        xmin = max(0.0, center - half_w)
                        xmax = xmin + max_window
                        if xmax > ref_len:
                            xmax = float(ref_len)
                            xmin = max(0.0, xmax - max_window)
                else:
                    xmin = 0.0
                    xmax = max(ref_len, 50) * 1.05

                ax.set_xlim(xmin, xmax)
                ax.set_ylim(-0.45, 1.45)
                ax.set_yticks([0.5])

                # All information organized inside Y tick label
                y_label = (
                    f"{pdb_id}:{chain_id}\n"
                    f"Ref: {ref_len} aa ({ref_cov_pct:.1f}% cov)\n"
                    f"Query: {query_length} aa ({query_cov_pct:.1f}% cov, {identity:.1f}% id)"
                )
                ax.set_yticklabels([y_label], fontsize=8.5)

                # Reference sequence hatched background bar
                ax.add_patch(
                    patches.Rectangle(
                        (1, 0.2),
                        max(0, ref_len - 1),
                        0.6,
                        facecolor="none",
                        edgecolor="gray",
                        hatch="\\\\",
                        linewidth=0.6,
                    )
                )
                
                # Aligned region blocks
                color = color_map["local alignment"]
                for start, end in ranges:
                    p_start = max(1, start)
                    p_end = max(p_start, end)
                    width = p_end - p_start if start == 0 else max(0, p_end - p_start)
                    ax.add_patch(
                        patches.Rectangle(
                            (p_start, 0.2),
                            width,
                            0.6,
                            facecolor=color,
                            edgecolor="black",
                            linewidth=0.5,
                            alpha=0.9,
                        )
                    )

                step = get_nice_step(xmax - xmin)
                start_tick = np.ceil(xmin / step) * step if xmin > 0 else 0
                ticks = np.arange(start_tick, xmax + step * 0.01, step)
                ticks = [t for t in ticks if xmin <= t <= xmax]
                tick_labels = [str(int(t)) if t != 0 else "" for t in ticks]
                
                ax.set_xticks(ticks)
                ax.set_xticklabels(tick_labels)
                mid_step = step / 2
                mid_ticks = np.arange(xmin + mid_step, xmax, step)
                mid_ticks = [t for t in mid_ticks if xmin <= t <= xmax]
                ax.set_xticks(mid_ticks, minor=True)

                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=True,
                )
                ax.tick_params(
                    axis="x",
                    which="major",
                    labelsize=8.5,
                    pad=2,
                )
                for x in ticks:
                    if x > xmin and x < xmax:
                        ax.axvline(
                            x=x,
                            linestyle="--",
                            alpha=0.35,
                            linewidth=0.7,
                        )
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if i == n - 1:
                    ax.set_xlabel(
                        "Reference Sequence Position",
                        fontsize=11,
                        fontweight="bold",
                    )
            fig.subplots_adjust(
                left=0.32,
                right=0.96,
                top=0.98,
                bottom=0.06,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return total_records

###################################### LOGGING
def finish_log(start, message):
    elapsed = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory_mb = usage / 1024 if usage < 10**7 else usage / 1024**2
    now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[PYTHON-INFO] {now} | {message} | "
        f"execution time: {elapsed:.2f}s | max memory: {memory_mb:.1f} MB"
    )

###################################### MAIN EXECUTION
def main():
    np.random.seed(SEED)
    start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Plot PDB chain coverage from alignment statistics"
    )
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="Alignment statistics directory or *_stats.tsv file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Coverage PDF path",
    )
    parser.add_argument(
        "-j",
        "--json",
        nargs="?",
        const=DEFAULT_CONFIG_PATH,
        default=DEFAULT_CONFIG_PATH,
        help="JSON configuration path",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Use built-in defaults without reading or updating JSON",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing coverage PDF",
    )
    args = parser.parse_args()
    defaults = {
        "aln_outdir": DEFAULT_ALN_OUTDIR,
        "plots_outdir": DEFAULT_PLOTS_OUTDIR,
        "coverage_pdf": DEFAULT_OUTPUT,
    }
    cfg = resolve_config(args.json, defaults, args.no_json)
    input_path = Path(args.input or cfg["aln_outdir"])
    output_pdf = Path(
        args.output or Path(cfg["plots_outdir"]) / cfg["coverage_pdf"]
    )
    if output_pdf.exists() and not args.force:
        print(f"[PYTHON-INFO] Coverage PDF already exists: {output_pdf}; skipped")
        finish_log(start, "Coverage plot skipped")
        return
    df = load_stats(input_path)
    if df.empty:
        raise FileNotFoundError(
            f"[PYTHON-INFO] No *_stats.tsv files found in {input_path}"
        )
    records = generate_plot(df, output_pdf)
    finish_log(
        start,
        f"Saved {output_pdf} from {records} alignment statistics",
    )

if __name__ == "__main__":
    main()
==> src/to_json.py <==
#!/usr/bin/env python3
"""Generates AlphaFold Server input JSON with pairwise combinations across complexes."""

###################################### DEPENDENCIES AND CONSTANTS
import datetime
import itertools
import json
import resource
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm

CFG_PATH = Path("configs/param_configs.json")
SEED = 42
CHAIN_IDS = ("A", "B")
DEFAULT_CFG = {
    "accession_ranges_path": "configs/accession_ranges.json",
    "alphafold_json_path": "data/inference/alphafold_input.json",
    "output_csv_path": "data/inference/all_sequences.csv",
    "uniprot_dir": "data/uniprot"
}

###################################### MAIN EXECUTION
def main():
    start_time = time.time()
    print("[PYTHON-INFO] Starting AlphaFold Server JSON input generation...")
    if not CFG_PATH.exists():
        CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CFG_PATH.write_text(json.dumps(DEFAULT_CFG, indent=4))
        print(f"[PYTHON-INFO] Created default configuration file at {CFG_PATH}.")
        params = DEFAULT_CFG.copy()
    else:
        try:
            params = {**DEFAULT_CFG, **json.loads(CFG_PATH.read_text())}
        except Exception:
            params = DEFAULT_CFG.copy()
    csv_fp = Path(params["output_csv_path"])
    if not csv_fp.exists():
        raise FileNotFoundError(f"[PYTHON-INFO] Input sequences CSV not found at: {csv_fp}")
    af_json_fp = Path(params["alphafold_json_path"])
    df = pd.read_csv(csv_fp)
    complexes = sorted(df["complex"].unique())
    complex_pairs = list(itertools.combinations(complexes, 2))
    af_jobs = []
    for c1, c2 in tqdm(complex_pairs, desc="[PYTHON-INFO] Generating combinations"):
        df_c1 = df[df["complex"] == c1]
        df_c2 = df[df["complex"] == c2]
        for row1 in df_c1.itertuples():
            for row2 in df_c2.itertuples():
                job_entry = {
                    "name": f"{CHAIN_IDS[0]}:{row1.id}_x_{CHAIN_IDS[1]}:{row2.id}",
                    "modelSeeds": [str(SEED)],
                    "sequences": [
                        {"proteinChain": {"id": CHAIN_IDS[0], "sequence": str(row1.sequence), "count": 1}},
                        {"proteinChain": {"id": CHAIN_IDS[1], "sequence": str(row2.sequence), "count": 1}}
                    ]
                }
                af_jobs.append(job_entry)
    print(f"[PYTHON-INFO] Total AlphaFold Server combinations generated: {len(af_jobs)}")
    af_json_fp.parent.mkdir(parents=True, exist_ok=True)
    af_json_fp.write_text(json.dumps(af_jobs, indent=2))
    params["alphafold_json_path"] = str(af_json_fp)
    CFG_PATH.write_text(json.dumps(params, indent=4))
    exec_time = time.time() - start_time
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    curr_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] Completed on {curr_date} | Execution Time: {exec_time:.2f}s | Max Memory: {max_mem:.2f} MB")

if __name__ == "__main__":
    main()