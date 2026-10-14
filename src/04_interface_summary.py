from pathlib import Path
from collections import defaultdict
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
    
    local_stems = {p.stem: p.name for p in files}
    print(f"[PYTHON-INFO] 1. Iteration PDB files: {list(local_stems.values())}")
    
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
        
        # Proporzioni migliorate: figura più compatta (6x5 pollici) con font grandi e leggibili
        fig, ax = plt.subplots(figsize=(6, 5))
        
        colors = [source_style(name)[0] for name in names] if 'source_style' in globals() else ['#2b5c8f'] * len(names)
        ax.bar(x, vals, color=colors, width=0.7, edgecolor="none", alpha=0.9)
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10, fontweight="medium")
        ax.axhline(0, color="0.3", lw=0.8, linestyle="--")
        
        ax.set_ylabel("HADDOCK score (lower is better)", fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.3, lw=0.6)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        if 'source_style' in globals():
            legend_colors = {}
            for name in names:
                color, _, label = source_style(name)
                legend_colors.setdefault(label, color)
            ax.legend(
                handles=[Patch(color=color, label=label) for label, color in legend_colors.items()], 
                fontsize=10, frameon=False, loc="upper left"
            )
            
        ax.set_title(f"{complex_name} ({len(names)} models)", fontsize=12, fontweight="bold", pad=12, loc="left")
        
        # Margini calibrati per evitare tagli sulle etichette ruotate
        fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.25)
        
        plot_path = out_dir / f"{complex_name}_haddock_scores.svg"
        fig.savefig(plot_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"[PYTHON-INFO] wrote HADDOCK score plot {plot_path}")


