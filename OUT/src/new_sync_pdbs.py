import datetime
import io
import resource
import time
import urllib.request
from collections import OrderedDict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from Bio.Align import PairwiseAligner
from Bio.PDB import Chain, MMCIFParser, PDBIO, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Polypeptide import protein_letters_3to1_extended
from Bio.PDB.Residue import Residue
import pandas as pd

from bioutils import SyncedStructure

def _load_fasta(fp):
    with open(fp) as f:
        return "".join(line.strip() for line in f if not line.strip().startswith('>'))

fasta_dir = Path('rawdata/uniprot')        
fastas = fasta_dir.rglob("*.fasta")
refseqs = {f.stem: _load_fasta(f) for f in fastas}
refseqs.keys()

def process_single_structure(fp, refseqs, out_dir):
    """Processes a single CIF file independently in a worker process."""
    try:
        test = SyncedStructure.from_Path(fp)
        if len(test.chain_ids) != 2:
            raise ValueError(f"File {fp.name} does not have exactly 2 chains.")
            
        tim_df = test.match_chains(refseqs['TIM'])
        uhrf_df = test.match_chains(refseqs['UHRF1'])
        
        tim_chain = tim_df.sort_values('aligned_residues', ascending=False).iloc[0]['chain_id'] if not tim_df.empty else None
        uhrf_chain = uhrf_df.sort_values('aligned_residues', ascending=False).iloc[0]['chain_id'] if not uhrf_df.empty else None
        
        if not tim_chain or not uhrf_chain or tim_chain == uhrf_chain or set([tim_chain, uhrf_chain]) != set(test.chain_ids):
            raise ValueError(f"Failed 1:1 assignment for {fp.name}: TIM -> {tim_chain}, UHRF1 -> {uhrf_chain}")
            
        test.rename_chains({tim_chain: 'A', uhrf_chain: 'B'})
        test.set_reference_sequences({'A': refseqs['TIM'], 'B': refseqs['UHRF1']})
        test.apply_reference_numbering()
        
        saved = test.save_pdb(out_dir / f"{test.structure_id}.pdb")
        return {
            "success": True, 
            "filename": fp.name, 
            "saved": saved, 
            "error": None
        }
    except Exception as e:
        return {
            "success": False, 
            "filename": fp.name, 
            "saved": None, 
            "error": str(e)
        }

if __name__ == "__main__":
    out_dir = Path("data/synced_pdbs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define or load your reference sequences dictionary here if not already defined globally
    # refseqs = {'TIM': '...', 'UHRF1': '...'}
    directory = Path("rawdata/candidate_complexes")
    file_paths = list(directory.glob("*.cif")) + list(directory.glob("*.pdb"))
    
    # Process files in parallel using all available CPU cores
    with ProcessPoolExecutor() as executor:
        # Submit all tasks to the executor
        futures = {executor.submit(process_single_structure, fp, refseqs, out_dir): fp for fp in file_paths}
        
        for future in as_completed(futures):
            res = future.result()
            if res["success"]:
                print(f"[PYTHON-INFO] Saved remapped structure: {res['saved']}")
                print(f"Processed: {res['filename']}")
            else:
                raise ValueError(res['error'])