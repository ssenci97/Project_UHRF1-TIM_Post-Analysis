#!/usr/bin/env python
"""Load, edit, synchronize, save, and analyze protein structures, sequence contacts, and interface metrics."""

###################################### IMPORTS AND CONSTANTS

import copy
import datetime
import io
import itertools
import json
import multiprocessing
import os
import re
import resource
import time
import urllib.request
import warnings
from collections import defaultdict
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser, PDBIO, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Model import Model
from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1_extended
from Bio.PDB.SASA import ShrakeRupley
from tqdm import tqdm

SEED = 42
RCSB_URL = "https://files.rcsb.org/download/{}.cif"
DEFAULT_CUTOFF = 0.8
DEFAULT_CONTACT_TYPE = "inter"
DEFAULT_CONFIG_PATH = "configs/param_configs.json"
DEFAULT_CONFIG = {
    "chain_seqs_outdir": "data/chains_seqs",
    "chain_refseqs_outdir": "data/chains_refseqs",
    "aln_outdir": "data/aln",
    "save_outdir": "data/saved",
    "aln_match_score": 1,
    "aln_mismatch_score": -1,
    "aln_open_gap_score": -2,
    "aln_extend_gap_score": -0.5,
    "interface_contact_cutoff_a": 4.5,
    "interface_score_cutoff_a": 12.0,
    "interface_block_size": 1000,
}
SUPPORTED_SUFFIXES = {".pdb", ".cif", ".mmcif"}
NUMBERINGS = ("original", "sequential", "reference")
GLOBAL_MATCH_SCORE = 2.0
GLOBAL_MISMATCH_SCORE = -1.0
GLOBAL_GAP_OPEN_SCORE = -5.0
GLOBAL_GAP_EXTEND_SCORE = -0.5
CONTACT_COLUMNS = [
    "chain_1", "resnum_1", "resname_1", "sequence_index_1",
    "chain_2", "resnum_2", "resname_2", "sequence_index_2",
    "min_distance_nm", "contact_type",
]
CONTACT_SEQUENCE_COLUMNS = ["structure_id", "chain_id", "sequence", "contacts", "contact_type", "cutoff_nm"]
RESIDUE_COLUMNS = [
    "structure_id", "chain_id", "pdb_resnum", "resname",
    "sequence_index", "reference_index",
]
MAPPING_COLUMNS = ["pdb_resnum", "sequence_index", "reference_index"]
INTERFACE_COMPLEX_COLUMNS = [
    "structure", "len_res_a", "len_res_b", "len_atoms_a", "len_atoms_b", "dASA_A2",
    "initial_sasa_A2", "frac_dASA", "n_contacts", "iface_res_a", "iface_res_b",
    "min_dist_A", "contact_score",
]
INTERFACE_RESIDUE_COLUMNS = [
    "structure", "chain", "resid", "resname", "dSASA_A2", "isolated_sasa_A2", "rel_dSASA",
    "n_partners", "min_dist_A", "contact_energy", "energy_per_partner", "partners",
]
HYDRO = {"A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08}
TO1 = {k.upper(): v for k, v in protein_letters_3to1_extended.items()}
TIEN = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159, "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174}

###################################### CONFIG AND LOGGING


def resolve_config(json_arg, defaults):
    """Load a JSON config, filling missing keys from defaults and saving the result."""
    if json_arg is None:
        return dict(defaults)

    path = Path(json_arg)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(defaults, indent=2))
        return dict(defaults)

    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        config = {}

    missing = {key: value for key, value in defaults.items() if key not in config}
    if missing:
        config.update(missing)
        path.write_text(json.dumps(config, indent=2))
    return config


def finish_log(start_time, message):
    """Print a timestamped message with elapsed time and peak memory."""
    elapsed = time.perf_counter() - start_time
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_mb = usage / 1024 if usage < 10**7 else usage / 1024**2
    timestamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[PYTHON-INFO] {timestamp} | {message} | "
        f"execution time: {elapsed:.2f}s | max memory: {mem_mb:.1f} MB"
    )


###################################### SEQUENCE AND MMCIF HELPERS


def clean_sequence(sequence):
    """Return a sequence as uppercase one-letter codes, using X for unknown symbols."""
    sequence = "".join(str(sequence).split()).upper()
    return "".join(char if char.isalpha() else "X" for char in sequence)


def cif_sequence(sequence):
    """Clean a polymer sequence from mmCIF, replacing modified residues like (SEP) with X."""
    sequence = re.sub(r"\([^)]*\)", "X", str(sequence))
    return clean_sequence(sequence)


def to_list(value):
    """Return an mmCIF field as a list of strings, whether it holds one value or many."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def residue_to_letter(residue):
    """Return the one-letter code for a residue, or X if unknown."""
    return protein_letters_3to1_extended.get(residue.resname.upper(), "X")


def is_protein_residue(residue):
    """Return True if the residue is recognized as an amino acid."""
    return is_aa(residue, standard=False)


def make_global_aligner():
    """Return the global aligner used to map observed sequences onto references."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = GLOBAL_MATCH_SCORE
    aligner.mismatch_score = GLOBAL_MISMATCH_SCORE
    aligner.open_gap_score = GLOBAL_GAP_OPEN_SCORE
    aligner.extend_gap_score = GLOBAL_GAP_EXTEND_SCORE
    return aligner


def make_local_aligner(config):
    """Return the local aligner configured from the JSON settings."""
    return PairwiseAligner(
        mode="local",
        match_score=config["aln_match_score"],
        mismatch_score=config["aln_mismatch_score"],
        open_gap_score=config["aln_open_gap_score"],
        extend_gap_score=config["aln_extend_gap_score"],
    )


def _check_contact_args(cutoff, contact_type):
    if cutoff <= 0:
        raise ValueError("cutoff must be greater than zero.")
    if contact_type not in {"inter", "intra", "all"}:
        raise ValueError("contact_type must be 'inter', 'intra', or 'all'.")


def _keep_pair(chain_a, chain_b, contact_type):
    """Return True if a residue pair matches the requested contact type."""
    if contact_type == "all":
        return True
    same_chain = chain_a == chain_b
    return (contact_type == "intra") == same_chain


###################################### SYNCHRONIZED STRUCTURE


class SyncedStructure:
    """Manage chains, reference sequences, residue mappings, C-alpha contacts, interface metrics, and saving.

    Numbering conventions:
        - sequence_index in residue tables is 1-based, counting observed residues.
        - sequence_index in contact tables is 0-based (used by the contact strings).
        - reference_index is 1-based in residue tables and 0-based in internal mappings.
    """

    def __init__(self, structure, structure_id="structure", mmcif_dict=None,
                 file_format="cif", config_path=DEFAULT_CONFIG_PATH, default_entity_id=None):
        self.structure = structure
        self.structure_id = str(structure_id)
        self.file_format = str(file_format).lower()
        self.mmcif_dict = mmcif_dict or {}
        self.default_entity_id = None if default_entity_id is None else str(default_entity_id)
        self.config = resolve_config(config_path, DEFAULT_CONFIG)
        self.model = next(structure.get_models())
        self.chains = {}

        self._global_aligner = make_global_aligner()
        self._local_aligner = make_local_aligner(self.config)
        self._alignment_cache = {}
        self._interface_cache = {}
        self._reference_by_entity = self._entity_sequences()
        self._entity_by_chain = self._chain_entity_map()
        self._build_chains()

    ###################################### CONSTRUCTORS

    @classmethod
    def from_rcsb_id(cls, pdb_id, config_path=DEFAULT_CONFIG_PATH):
        """Download a structure and its metadata from RCSB and load it."""
        start = time.perf_counter()
        pdb_id = str(pdb_id).strip().upper()

        with urllib.request.urlopen(RCSB_URL.format(pdb_id), timeout=30) as response:
            text = response.read().decode("utf-8")

        structure = MMCIFParser(QUIET=True).get_structure(pdb_id, io.StringIO(text))
        obj = cls(
            structure,
            structure_id=pdb_id,
            mmcif_dict=MMCIF2Dict(io.StringIO(text)),
            file_format="cif",
            config_path=config_path,
        )
        finish_log(start, f"Loaded RCSB structure {pdb_id}")
        return obj

    @classmethod
    def from_Path(cls, path, config_path=DEFAULT_CONFIG_PATH):
        """Load a local .pdb, .cif, or .mmcif file; the file stem is the structure ID and default entity ID."""
        start = time.perf_counter()
        path = Path(path).expanduser()

        if not path.is_file():
            raise FileNotFoundError(f"Structure file not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported structure format: {suffix}. "
                f"Supported formats: {sorted(SUPPORTED_SUFFIXES)}"
            )

        structure_id = path.stem
        if suffix == ".pdb":
            structure = PDBParser(QUIET=True).get_structure(structure_id, str(path))
            mmcif_dict, file_format = {}, "pdb"
        else:
            structure = MMCIFParser(QUIET=True).get_structure(structure_id, str(path))
            mmcif_dict, file_format = MMCIF2Dict(str(path)), "cif"

        obj = cls(
            structure,
            structure_id=structure_id,
            mmcif_dict=mmcif_dict,
            file_format=file_format,
            config_path=config_path,
            default_entity_id=structure_id,
        )
        finish_log(start, f"Loaded local structure {path}")
        return obj

    @classmethod
    def from_path(cls, path, config_path=DEFAULT_CONFIG_PATH):
        """Alias for from_Path."""
        return cls.from_Path(path, config_path=config_path)

    @classmethod
    def from_biopython(cls, structure, structure_id=None, config_path=DEFAULT_CONFIG_PATH):
        """Wrap an existing Biopython Structure object."""
        return cls(
            structure,
            structure_id=structure_id or structure.id,
            file_format="pdb",
            config_path=config_path,
        )

    ###################################### REFERENCE METADATA

    def _entity_sequences(self):
        """Return {entity_id: cleaned reference sequence} from mmCIF metadata."""
        entity_ids = to_list(self.mmcif_dict.get("_entity_poly.entity_id"))
        canonical = to_list(self.mmcif_dict.get("_entity_poly.pdbx_seq_one_letter_code_can"))
        original = to_list(self.mmcif_dict.get("_entity_poly.pdbx_seq_one_letter_code"))

        sequences = {}
        for index, entity_id in enumerate(entity_ids):
            raw = canonical[index] if index < len(canonical) else ""
            if not raw and index < len(original):
                raw = original[index]
            sequence = cif_sequence(raw) if raw else ""
            if sequence:
                sequences[entity_id] = sequence
        return sequences

    def _chain_entity_map(self):
        """Return {chain_id: entity_id} from mmCIF metadata."""
        chain_ids = to_list(self.mmcif_dict.get("_struct_asym.id"))
        entity_ids = to_list(self.mmcif_dict.get("_struct_asym.entity_id"))
        return dict(zip(chain_ids, entity_ids))

    def _reference_for_chain(self, chain_id):
        """Return the metadata reference sequence for a chain, or an empty string."""
        entity_id = self._entity_by_chain.get(str(chain_id))
        return self._reference_by_entity.get(entity_id, "")

    def _resolve_entity(self, chain_id, explicit=None):
        """Return the explicit entity ID, else the mmCIF entity, else the file stem."""
        if explicit is not None:
            return str(explicit)
        return self._entity_by_chain.get(chain_id) or self.default_entity_id

    ###################################### CHAIN MANAGEMENT

    def _require_chain(self, chain_id):
        """Return the chain ID as a string, or raise KeyError if it is not loaded."""
        chain_id = str(chain_id)
        if chain_id not in self.chains:
            raise KeyError(f"Unknown chain: {chain_id}")
        return chain_id

    @staticmethod
    def _protein_residues(chain):
        return [residue for residue in chain.get_residues() if is_protein_residue(residue)]

    def _map_to_reference(self, sequence, reference):
        """Return {observed 0-based index: reference 0-based index} from a global alignment."""
        if not sequence or not reference:
            return {}
        alignment = self._global_aligner.align(reference, sequence)[0]
        return {
            int(seq_index): int(ref_index)
            for ref_index, seq_index in alignment.indices.T
            if ref_index >= 0 and seq_index >= 0
        }

    def _store_mapping(self, data, index_mapping):
        """Save a sequence-to-reference mapping and update the mapping counts."""
        residues = data["residues"]
        data["sequence_index_to_ref_index"] = index_mapping
        data["resid_to_ref_index"] = {
            residues[seq_index].id: ref_index
            for seq_index, ref_index in index_mapping.items()
        }
        data["mapped_residues"] = len(index_mapping)

    def _make_chain_data(self, chain, reference_sequence=None, entity_id=None, metadata=None):
        chain_id = str(chain.id)
        residues = self._protein_residues(chain)

        if reference_sequence is None:
            reference_sequence = self._reference_for_chain(chain_id)

        data = {
            "chain": chain,
            "chain_id": chain_id,
            "entity_id": self._resolve_entity(chain_id, entity_id),
            "residues": residues,
            "sequence": "".join(residue_to_letter(residue) for residue in residues),
            "reference_sequence": clean_sequence(reference_sequence),
            "metadata": dict(metadata or {}),
        }
        data["sequence_length"] = len(data["sequence"])
        data["reference_length"] = len(data["reference_sequence"])
        self._store_mapping(
            data,
            self._map_to_reference(data["sequence"], data["reference_sequence"]),
        )
        return data

    def _build_chains(self):
        """Rebuild chain data from the model, keeping user edits to existing chains."""
        previous = self.chains
        rebuilt = {}

        for chain in self.model.get_chains():
            chain_id = str(chain.id)
            old = previous.get(chain_id, {})
            rebuilt[chain_id] = self._make_chain_data(
                chain,
                reference_sequence=old.get("reference_sequence"),
                entity_id=old.get("entity_id"),
                metadata=old.get("metadata"),
            )
        self.chains = rebuilt
        self._interface_cache.clear()

    @property
    def chain_ids(self):
        """List of loaded chain IDs."""
        return list(self.chains)

    def __getitem__(self, chain_id):
        return self.chains[str(chain_id)]

    def __iter__(self):
        return iter(self.chains.values())

    def __len__(self):
        return len(self.chains)

    def __contains__(self, chain_id):
        return str(chain_id) in self.chains

    def get_sequence(self, chain_id):
        """Return the observed one-letter sequence for a chain."""
        return self.chains[self._require_chain(chain_id)]["sequence"]

    def get_reference_sequence(self, chain_id):
        """Return the current reference sequence for a chain."""
        return self.chains[self._require_chain(chain_id)]["reference_sequence"]

    ###################################### SEQUENCE MATCHING AND RENAMING

    def _pair_identity(self, seq_a, seq_b):
        """Return (identity_percent, aligned_residues, coverage_percent_of_b) from a local alignment."""
        if not seq_a or not seq_b:
            return 0.0, 0, 0.0

        alignments = self._local_aligner.align(seq_a, seq_b)
        if not alignments:
            return 0.0, 0, 0.0

        top, bottom = alignments[0]
        columns = [(x, y) for x, y in zip(top, bottom) if x != "-" and y != "-"]
        matches = sum(x == y for x, y in columns)
        aligned = len(columns)

        identity = round(matches / aligned * 100, 2) if aligned else 0.0
        coverage = round(aligned / len(seq_b) * 100, 2)
        return identity, aligned, coverage

    def match_chains(self, sequence, min_identity=0.0):
        """Return the chains that align to a query sequence, best match first."""
        query = clean_sequence(sequence)
        rows = []
        for chain_id, data in self.chains.items():
            identity, aligned, coverage = self._pair_identity(data["sequence"], query)
            if aligned and identity >= min_identity:
                rows.append({
                    "chain_id": chain_id,
                    "identity_percent": identity,
                    "aligned_residues": aligned,
                    "query_coverage_percent": coverage,
                })

        columns = ["chain_id", "identity_percent", "aligned_residues", "query_coverage_percent"]
        frame = pd.DataFrame(rows, columns=columns)
        return frame.sort_values("identity_percent", ascending=False, ignore_index=True)

    def rename_chains(self, mapping):
        """Rename chains with a {old_id: new_id} dict, keeping their sequences, references, and metadata."""
        mapping = {self._require_chain(old): str(new) for old, new in mapping.items()}
        final_ids = [mapping.get(chain_id, chain_id) for chain_id in self.chains]
        if len(set(final_ids)) != len(final_ids):
            raise ValueError("Renaming would create duplicate chain IDs.")

        moved = {old: self.model[old] for old in mapping}
        for old in mapping:
            self.model.detach_child(old)
        for old, new in mapping.items():
            moved[old].id = new
            self.model.add(moved[old])

        self.chains = {mapping.get(c, c): data for c, data in self.chains.items()}
        self._alignment_cache.clear()
        self._build_chains()
        return self

    def rename_by_sequence(self, targets, min_identity=90.0):
        """Rename chains to match a {new_id: sequence} dict, one-to-one, highest identity first."""
        targets = {str(new): clean_sequence(seq) for new, seq in targets.items()}

        candidates = []
        for new_id, sequence in targets.items():
            for old_id, data in self.chains.items():
                identity, aligned, _ = self._pair_identity(data["sequence"], sequence)
                candidates.append((identity, aligned, old_id, new_id))
        candidates.sort(key=lambda row: -row[0])

        renames, used_old, used_new = {}, set(), set()
        report = {}
        for identity, aligned, old_id, new_id in candidates:
            if new_id in used_new or old_id in used_old or identity < min_identity:
                continue
            renames[old_id] = new_id
            used_old.add(old_id)
            used_new.add(new_id)
            report[new_id] = (old_id, identity, "renamed")

        rows = []
        for new_id in targets:
            if new_id in report:
                old_id, identity, status = report[new_id]
            else:
                old_id, identity, status = None, None, "unmatched_target"
            rows.append({"new_chain_id": new_id, "old_chain_id": old_id,
                         "identity_percent": identity, "status": status})
        for chain_id in self.chains:
            if chain_id not in used_old:
                rows.append({"new_chain_id": None, "old_chain_id": chain_id,
                             "identity_percent": None, "status": "unmatched_chain"})

        if renames:
            self.rename_chains(renames)
            for old_id, new_id in renames.items():
                self.set_reference_sequence(new_id, targets[new_id])

        return pd.DataFrame(rows, columns=["new_chain_id", "old_chain_id",
                                           "identity_percent", "status"])

    ###################################### LOCAL ALIGNMENT

    def _local_alignment(self, chain_id):
        """Return (stats, position_rows, sequence_to_reference) for a chain, cached per sequence pair."""
        reference = self.get_reference_sequence(chain_id)
        observed = self.get_sequence(chain_id)
        key = (reference, observed)

        cached = self._alignment_cache.get(chain_id)
        if cached and cached[0] == key:
            return cached[1]

        result = self._compute_local_alignment(chain_id, reference, observed)
        self._alignment_cache[chain_id] = (key, result)
        return result

    def _compute_local_alignment(self, chain_id, reference, observed):
        empty = (None, [], {})
        if not reference or not observed:
            return empty

        alignments = self._local_aligner.align(reference, observed)
        if not alignments:
            return empty

        best = alignments[0]
        ref_aligned, obs_aligned = best
        ref_start, obs_start = int(best.coordinates[0, 0]), int(best.coordinates[1, 0])
        ref_end, obs_end = int(best.coordinates[0, 1]), int(best.coordinates[1, 1])

        counts = {"matches": 0, "mismatches": 0, "insertions": 0, "deletions": 0}
        gap_positions, rows, seq_to_ref = [], [], {}
        ref_pos, obs_pos = ref_start, obs_start

        for column, (r, o) in enumerate(zip(ref_aligned, obs_aligned), start=1):
            if r != "-":
                ref_pos += 1
            if o != "-":
                obs_pos += 1

            if r == "-":
                counts["insertions"] += 1
            elif o == "-":
                counts["deletions"] += 1
                gap_positions.append(str(obs_pos))
            elif r == o:
                counts["matches"] += 1
            else:
                counts["mismatches"] += 1

            if r != "-" and o != "-":
                seq_to_ref[obs_pos] = ref_pos

            rows.append({
                "structure_id": self.structure_id,
                "chain_id": chain_id,
                "alignment_position": column,
                "ref": r,
                "alt": o,
                "posref": ref_pos if r != "-" else None,
                "posalt": obs_pos if o != "-" else None,
                "match": int(r == o and r != "-"),
            })

        matches = counts["matches"]
        mismatches = counts["mismatches"]
        insertions = counts["insertions"]
        deletions = counts["deletions"]
        aligned_length = sum(counts.values())

        stats = {
            "structure_id": self.structure_id,
            "query_id": f"{self.structure_id}_{chain_id}",
            "ref_id": f"{self.structure_id}_{chain_id}_reference",
            "query_length": len(observed),
            "ref_length": len(reference),
            "aligned_length": aligned_length,
            "matches": matches,
            "mismatches": mismatches,
            "insertions_in_query": insertions,
            "deletions_in_query": deletions,
            "identity_percent": round(matches / aligned_length * 100, 2) if aligned_length else 0.0,
            "query_coverage_percent": round((len(observed) - deletions) / len(observed) * 100, 2),
            "alignment_score": round(matches - 0.5 * mismatches - 1.5 * (insertions + deletions), 2),
            "range": f"{obs_start + 1}-{obs_end}",
            "ref_start": ref_start,
            "ref_end": ref_end,
            "gappos": ",".join(gap_positions),
            "sequence": reference[ref_start:ref_end],
        }
        return stats, rows, seq_to_ref

    ###################################### RESIDUE MAPPING

    def _residue_frame(self, chain_id):
        """Return one row per observed residue of a chain, with its reference position."""
        _, _, seq_to_ref = self._local_alignment(chain_id)
        rows = [
            {
                "structure_id": self.structure_id,
                "chain_id": chain_id,
                "pdb_resnum": residue.id[1],
                "resname": residue.resname.strip().upper(),
                "sequence_index": position,
                "reference_index": seq_to_ref.get(position),
            }
            for position, residue in enumerate(self.chains[chain_id]["residues"], start=1)
        ]
        frame = pd.DataFrame(rows, columns=RESIDUE_COLUMNS)
        frame["reference_index"] = frame["reference_index"].astype("Int64")
        return frame

    def get_residue_mapping(self, chain_id):
        """Return a DataFrame with pdb_resnum, sequence_index, and reference_index for one chain."""
        chain_id = self._require_chain(chain_id)
        return self._residue_frame(chain_id)[MAPPING_COLUMNS].reset_index(drop=True)

    ###################################### EXPOSED DATAFRAMES

    @property
    def alignment_stats(self):
        """One row per chain with local alignment statistics."""
        rows = [self._local_alignment(chain_id)[0] for chain_id in self.chains]
        return pd.DataFrame([row for row in rows if row])

    @property
    def alignment_positions(self):
        """Column-by-column local alignment for every chain."""
        rows = []
        for chain_id in self.chains:
            rows.extend(self._local_alignment(chain_id)[1])
        return pd.DataFrame(rows)

    @property
    def residue_map(self):
        """All observed residues across chains, with PDB number, sequence index, and reference index."""
        if not self.chains:
            return pd.DataFrame(columns=RESIDUE_COLUMNS)
        return pd.concat(
            [self._residue_frame(chain_id) for chain_id in self.chains],
            ignore_index=True,
        )

    def export_alignments(self, outdir=None):
        """Write alignment stats and positions as TSV files and return the output directory."""
        outdir = self._prepare_dir(outdir or self.config["aln_outdir"])
        self.alignment_stats.to_csv(outdir / f"{self.structure_id}_stats.tsv", sep="\t", index=False)
        self.alignment_positions.to_csv(outdir / f"{self.structure_id}_positions.tsv", sep="\t", index=False)
        return outdir

    ###################################### EDITING REFERENCES AND METADATA

    def set_reference_sequence(self, chain_id, sequence, remap=True):
        """Set a chain's reference sequence and, optionally, rebuild its mapping."""
        data = self.chains[self._require_chain(chain_id)]
        data["reference_sequence"] = clean_sequence(sequence)
        data["reference_length"] = len(data["reference_sequence"])

        index_mapping = (
            self._map_to_reference(data["sequence"], data["reference_sequence"])
            if remap else {}
        )
        self._store_mapping(data, index_mapping)
        return self

    def set_reference_sequences(self, sequences, remap=True):
        """Set reference sequences for several chains from a {chain_id: sequence} dict."""
        for chain_id, sequence in sequences.items():
            self.set_reference_sequence(chain_id, sequence, remap=remap)
        return self

    def remap_chain(self, chain_id):
        """Rebuild the sequence-to-reference mapping for one chain."""
        data = self.chains[self._require_chain(chain_id)]
        return self.set_reference_sequence(chain_id, data["reference_sequence"], remap=True)

    def remap_all_chains(self):
        """Rebuild sequence-to-reference mappings for every chain."""
        for chain_id in list(self.chains):
            self.remap_chain(chain_id)
        return self

    def set_chain_metadata(self, chain_id, key, value):
        """Store a user-defined value on a chain."""
        self.chains[self._require_chain(chain_id)]["metadata"][str(key)] = value
        return self

    def set_entity_id(self, chain_id, entity_id):
        """Set the entity ID of a chain."""
        self.chains[self._require_chain(chain_id)]["entity_id"] = str(entity_id)
        return self

    def _apply_chain_options(self, chain_id, reference_sequence, entity_id, metadata):
        if reference_sequence is not None:
            self.set_reference_sequence(chain_id, reference_sequence)
        if entity_id is not None:
            self.chains[chain_id]["entity_id"] = str(entity_id)
        if metadata:
            self.chains[chain_id]["metadata"].update(metadata)

    def add_chain(self, chain, reference_sequence=None, entity_id=None,
                  metadata=None, replace=False):
        """Add a Biopython chain to the structure."""
        chain_id = str(chain.id)
        if chain_id in self.chains:
            if not replace:
                raise ValueError(f"Chain already exists: {chain_id}")
            self.model.detach_child(chain_id)

        self.model.add(copy.deepcopy(chain))
        self._build_chains()
        self._apply_chain_options(chain_id, reference_sequence, entity_id, metadata)
        return self

    def remove_chain(self, chain_id):
        """Remove a chain from the structure."""
        chain_id = self._require_chain(chain_id)
        self.model.detach_child(chain_id)
        self.chains.pop(chain_id)
        self._alignment_cache.pop(chain_id, None)
        self._interface_cache.clear()
        return self

    def replace_chain(self, chain_id, chain, reference_sequence=None,
                      entity_id=None, metadata=None):
        """Replace an existing chain with a new Biopython chain."""
        chain_id = self._require_chain(chain_id)
        new_chain = copy.deepcopy(chain)
        new_chain.id = chain_id
        self.model.detach_child(chain_id)
        self.model.add(new_chain)
        self._build_chains()
        self._apply_chain_options(chain_id, reference_sequence, entity_id, metadata)
        return self

    ###################################### RENUMBERING, TRIMMING AND PDB EXPORT

    @staticmethod
    def _reassign(chain, pairs):
        """Set residue IDs from (residue, new_id) pairs in two phases to avoid transient collisions."""
        for offset, (residue, _) in enumerate(pairs):
            residue.id = ("#", offset, " ")
        chain.child_dict = {residue.id: residue for residue in chain.child_list}
        for residue, new_id in pairs:
            residue.id = new_id
        chain.child_dict = {residue.id: residue for residue in chain.child_list}
        
    @staticmethod
    def _renumber(chain):
        """Renumber a chain's residues 1..N in place."""
        SyncedStructure._reassign(chain, [(residue, (" ", number, " "))
                                          for number, residue in enumerate(chain.get_residues(), start=1)])

    @classmethod
    def _keep_mapped(cls, chain, index_mapping):
        """Remove residues not in the mapping and renumber the kept ones by reference position (1-based)."""
        protein = [residue for residue in chain.get_residues() if is_protein_residue(residue)]
        pairs = [(protein[index], (" ", ref_index + 1, " "))
                 for index, ref_index in index_mapping.items() if index < len(protein)]
        kept = {id(residue) for residue, _ in pairs}
        for residue in list(chain.get_residues()):
            if id(residue) not in kept:
                chain.detach_child(residue.id)
        cls._reassign(chain, pairs)

    def apply_reference_numbering(self, chain_ids=None):
        """Keep only residues mapped to the reference, renumber them by reference position, and update the structure in place."""
        selected = self.chain_ids if chain_ids is None else [self._require_chain(c) for c in chain_ids]
        trimmed = []
        for chain_id in selected:
            mapping = self.chains[chain_id]["sequence_index_to_ref_index"]
            if mapping:
                self._keep_mapped(self.chains[chain_id]["chain"], mapping)
                trimmed.append(chain_id)
        self._build_chains()
        for chain_id in trimmed:
            data = self.chains[chain_id]
            self._store_mapping(data, {index: residue.id[1] - 1 for index, residue in enumerate(data["residues"])})
        return self

    def reindex_chain(self, chain_id, in_place=False):
        """Return a chain with residues renumbered 1..N; set in_place=True to modify the structure."""
        chain_id = self._require_chain(chain_id)
        chain = (self.chains[chain_id]["chain"] if in_place
                 else copy.deepcopy(self.chains[chain_id]["chain"]))
        self._renumber(chain)
        if in_place:
            self._build_chains()
        return chain

    def to_pdb_string(self, chain_id=None, numbering="original"):
        """Return the structure, or one chain, as PDB text with original, sequential, or reference numbering.

        Reference numbering keeps only residues mapped to the reference in the exported copy, without
        modifying this object. Chains without a mapping keep their original numbering.
        """
        if numbering not in NUMBERINGS:
            raise ValueError(f"numbering must be one of {NUMBERINGS}.")
        selected = {self._require_chain(chain_id)} if chain_id is not None else set(self.chains)

        structure = copy.deepcopy(self.structure)
        model = next(structure.get_models())
        for chain in list(model.get_chains()):
            if str(chain.id) not in selected:
                model.detach_child(chain.id)

        for chain in list(model.get_chains()):
            chain_key = str(chain.id)
            if numbering == "sequential":
                self._renumber(chain)
            elif numbering == "reference" and self.chains[chain_key]["sequence_index_to_ref_index"]:
                self._keep_mapped(chain, self.chains[chain_key]["sequence_index_to_ref_index"])

        writer = PDBIO()
        writer.set_structure(structure)
        output = io.StringIO()
        writer.save(output)
        return output.getvalue()

    def save_pdb(self, output_path, chain_id=None, numbering="original"):
        """Save the structure, or one chain, as a PDB file with the chosen numbering."""
        output_path = self._prepare_path(output_path)
        output_path.write_text(self.to_pdb_string(chain_id=chain_id, numbering=numbering))
        return output_path

    def to_fasta(self, output_path, reference=False):
        """Write observed or reference sequences of all chains as FASTA; chains without a reference are skipped."""
        lines = []
        for chain_id, data in self.chains.items():
            if reference and not data["reference_sequence"]:
                continue
            sequence = data["reference_sequence"] if reference else data["sequence"]
            suffix = " reference" if reference else ""
            lines.extend([f">{self.structure_id}_{chain_id}{suffix}", sequence])
        output_path = self._prepare_path(output_path)
        output_path.write_text("\n".join(lines) + "\n")
        return output_path

    def to_biopython(self):
        """Return a deep copy of the underlying Biopython Structure."""
        return copy.deepcopy(self.structure)

    def to_dataframe(self):
        """Return the per-residue table; alias for residue_map."""
        return self.residue_map

    @staticmethod
    def _prepare_path(path):
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _prepare_dir(path):
        path = Path(path).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    ###################################### SYSTEMATIC SAVING

    def save_all(self, outdir=None, numbering=NUMBERINGS, chain_pairs=None, cutoff=DEFAULT_CUTOFF):
        """Save every structure, mapping, alignment, contact, sequence, and interface table with a manifest.

        Writes one PDB per numbering mode, the chain summary, residue map, alignments, contacts,
        contact sequences, FASTA files, interface tables for each chain pair (all pairs by default),
        and a JSON manifest listing the written files. Returns the list of written paths.
        """
        start = time.perf_counter()
        outdir = self._prepare_dir(outdir or self.config["save_outdir"])
        sid = self.structure_id
        contacts = self.compute_contacts(cutoff=cutoff, contact_type="all")

        tasks = [(f"{sid}_{mode}.pdb", lambda p, m=mode: self.save_pdb(p, numbering=m)) for mode in numbering]
        tasks += [
            (f"{sid}_chains.tsv", lambda p: self.summary().to_csv(p, sep="\t", index=False)),
            (f"{sid}_residue_map.tsv", lambda p: self.residue_map.to_csv(p, sep="\t", index=False)),
            (f"{sid}_alignment_stats.tsv", lambda p: self.alignment_stats.to_csv(p, sep="\t", index=False)),
            (f"{sid}_alignment_positions.tsv", lambda p: self.alignment_positions.to_csv(p, sep="\t", index=False)),
            (f"{sid}_contacts.tsv", lambda p: contacts.to_csv(p, sep="\t", index=False)),
            (f"{sid}_contact_summary.tsv", lambda p: self.get_contact_summary(contacts=contacts).to_csv(p, sep="\t", index=False)),
            (f"{sid}_contact_sequences.csv", lambda p: self.export_contact_sequences(p, cutoff=cutoff, contact_type="all")),
            (f"{sid}_contact_sequences_reference.csv", lambda p: self.export_contact_sequences(p, cutoff=cutoff, contact_type="all", reference=True)),
            (f"{sid}_sequences.fasta", lambda p: self.to_fasta(p)),
            (f"{sid}_reference_sequences.fasta", lambda p: self.to_fasta(p, reference=True)),
        ]

        for chain_a, chain_b in (chain_pairs or itertools.combinations(self.chain_ids, 2)):
            try:
                result = self.interface(chain_a, chain_b)
            except ValueError as ex:
                print(f"[PYTHON-INFO] {sid}: interface {chain_a}{chain_b} skipped, {ex}")
                continue
            tasks.append((f"{sid}_{chain_a}{chain_b}_interface_complex.tsv",
                          lambda p, r=result: pd.DataFrame([r["complex"]]).to_csv(p, sep="\t", index=False)))
            tasks.append((f"{sid}_{chain_a}{chain_b}_interface_residues.tsv",
                          lambda p, r=result: r["residues"].to_csv(p, sep="\t", index=False)))

        written = []
        for name, writer in tqdm(tasks, desc="saving", unit="file"):
            path = outdir / name
            writer(path)
            written.append(str(path))

        manifest_path = outdir / f"{sid}_manifest.json"
        manifest = {
            "structure_id": sid,
            "file_format": self.file_format,
            "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "config": self.config,
            "files": written,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        written.append(str(manifest_path))
        finish_log(start, f"Saved {len(written)} files for {sid} to {outdir}")
        return written

    ###################################### C-ALPHA TRAJECTORY AND CONTACTS

    def _ca_records(self, chain_ids=None):
        """Return (chain_id, sequence_index, residue) for protein residues with a C-alpha atom."""
        selected = None if chain_ids is None else {str(c) for c in chain_ids}
        return [
            (chain_id, index, residue)
            for chain_id, data in self.chains.items()
            if selected is None or chain_id in selected
            for index, residue in enumerate(data["residues"])
            if "CA" in residue
        ]

    def _make_ca_trajectory(self, chain_ids=None):
        """Build an MDTraj trajectory of C-alpha atoms (in nm) and per-residue records."""
        topology = md.Topology()
        md_chains = {}
        coordinates, records = [], []

        for chain_id, index, residue in self._ca_records(chain_ids):
            if chain_id not in md_chains:
                md_chains[chain_id] = topology.add_chain(chain_id=chain_id)

            name = residue.resname.strip().upper()
            resnum = int(residue.id[1])
            md_residue = topology.add_residue(name, md_chains[chain_id], resSeq=resnum)
            topology.add_atom("CA", md.element.carbon, md_residue)

            coordinates.append(np.asarray(residue["CA"].coord, dtype=np.float32) / 10.0)
            records.append({
                "chain_id": chain_id,
                "sequence_index": index,
                "resnum": resnum,
                "resname": name,
            })

        if not coordinates:
            raise ValueError("No protein C-alpha atoms found in the selected chains.")

        trajectory = md.Trajectory(
            xyz=np.asarray(coordinates, dtype=np.float32)[None, :, :],
            topology=topology,
        )
        return trajectory, records

    def to_ca_trajectory(self, chain_ids=None):
        """Return the C-alpha MDTraj trajectory (nm) for the selected chains."""
        if chain_ids is not None:
            chain_ids = [self._require_chain(chain_id) for chain_id in chain_ids]
        return self._make_ca_trajectory(chain_ids)[0]

    def compute_contacts(self, cutoff=DEFAULT_CUTOFF,
                         contact_type=DEFAULT_CONTACT_TYPE, chain_ids=None):
        """Return C-alpha residue pairs closer than cutoff (nm) as a DataFrame.

        contact_type: 'inter' (different chains), 'intra' (same chain), or 'all'.
        """
        _check_contact_args(cutoff, contact_type)
        if chain_ids is not None:
            chain_ids = [self._require_chain(chain_id) for chain_id in chain_ids]

        trajectory, records = self._make_ca_trajectory(chain_ids)
        pairs = [
            (i, j)
            for i in range(len(records))
            for j in range(i + 1, len(records))
            if _keep_pair(records[i]["chain_id"], records[j]["chain_id"], contact_type)
        ]
        if not pairs:
            return pd.DataFrame(columns=CONTACT_COLUMNS)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*No alpha carbon.*")
            distances, residue_pairs = md.compute_contacts(
                trajectory,
                contacts=np.asarray(pairs, dtype=int),
                scheme="ca",
                ignore_nonprotein=False,
                periodic=False,
            )

        rows = []
        for (first, second), distance in zip(residue_pairs, distances[0]):
            if distance > cutoff:
                continue
            a, b = records[int(first)], records[int(second)]
            rows.append({
                "chain_1": a["chain_id"],
                "resnum_1": a["resnum"],
                "resname_1": a["resname"],
                "sequence_index_1": a["sequence_index"],
                "chain_2": b["chain_id"],
                "resnum_2": b["resnum"],
                "resname_2": b["resname"],
                "sequence_index_2": b["sequence_index"],
                "min_distance_nm": round(float(distance), 4),
                "contact_type": "intra" if a["chain_id"] == b["chain_id"] else "inter",
            })
        return pd.DataFrame(rows, columns=CONTACT_COLUMNS)

    ###################################### CONTACT STRINGS

    def get_contact_sequence(self, chain_id, cutoff=DEFAULT_CUTOFF,
                             contact_type=DEFAULT_CONTACT_TYPE, contacts=None):
        """Return a string over observed residues: 'X' = no C-alpha, '0' = no contact, '1' = contact."""
        chain_id = self._require_chain(chain_id)
        if contacts is None:
            contacts = self.compute_contacts(cutoff=cutoff, contact_type=contact_type)

        bits = ["X"] * self.chains[chain_id]["sequence_length"]
        for _, index, _ in self._ca_records([chain_id]):
            bits[index] = "0"

        touched = set()
        for row in contacts.itertuples(index=False):
            if row.chain_1 == chain_id:
                touched.add(int(row.sequence_index_1))
            if row.chain_2 == chain_id:
                touched.add(int(row.sequence_index_2))
        for index in touched:
            if bits[index] == "0":
                bits[index] = "1"

        return "".join(bits)

    def get_reference_contact_sequence(self, chain_id, cutoff=DEFAULT_CUTOFF,
                                       contact_type=DEFAULT_CONTACT_TYPE, contacts=None):
        """Return the contact string laid out along the full reference sequence."""
        chain_id = self._require_chain(chain_id)
        data = self.chains[chain_id]
        reference = data["reference_sequence"]
        if not reference:
            raise ValueError(
                f"Chain {chain_id} has no reference sequence. "
                "Set one with set_reference_sequence() first."
            )

        observed_bits = self.get_contact_sequence(
            chain_id, cutoff=cutoff, contact_type=contact_type, contacts=contacts,
        )

        bits = ["X"] * len(reference)
        for seq_index, ref_index in data["sequence_index_to_ref_index"].items():
            if seq_index < len(observed_bits) and 0 <= ref_index < len(bits):
                bits[ref_index] = observed_bits[seq_index]
        return "".join(bits)

    ###################################### SUMMARIES AND EXPORT

    def get_contact_summary(self, cutoff=DEFAULT_CUTOFF,
                            contact_type=DEFAULT_CONTACT_TYPE, contacts=None):
        """Return one row per chain with contact counts and the fraction of residues in contact."""
        if contacts is None:
            contacts = self.compute_contacts(cutoff=cutoff, contact_type=contact_type)

        rows = []
        for chain_id, data in self.chains.items():
            involved = contacts[
                (contacts["chain_1"] == chain_id) | (contacts["chain_2"] == chain_id)
            ]
            contacted = set(involved.loc[involved["chain_1"] == chain_id, "sequence_index_1"])
            contacted |= set(involved.loc[involved["chain_2"] == chain_id, "sequence_index_2"])

            length = data["sequence_length"]
            rows.append({
                "chain_id": chain_id,
                "sequence_length": length,
                "reference_length": data["reference_length"],
                "contact_pairs": len(involved),
                "contacted_residues": len(contacted),
                "contact_fraction": len(contacted) / length if length else 0.0,
            })
        return pd.DataFrame(rows)

    def export_contact_sequences(self, output_path, cutoff=DEFAULT_CUTOFF,
                                 contact_type=DEFAULT_CONTACT_TYPE, reference=False):
        """Write each chain's sequence and contact string to a CSV file; reference mode skips chains without one."""
        contacts = self.compute_contacts(cutoff=cutoff, contact_type=contact_type)

        rows = []
        for chain_id, data in self.chains.items():
            if reference:
                if not data["reference_sequence"]:
                    continue
                sequence = data["reference_sequence"]
                contact_string = self.get_reference_contact_sequence(
                    chain_id, cutoff=cutoff, contact_type=contact_type, contacts=contacts,
                )
            else:
                sequence = data["sequence"]
                contact_string = self.get_contact_sequence(
                    chain_id, cutoff=cutoff, contact_type=contact_type, contacts=contacts,
                )
            rows.append({
                "structure_id": self.structure_id,
                "chain_id": chain_id,
                "sequence": sequence,
                "contacts": contact_string,
                "contact_type": contact_type,
                "cutoff_nm": cutoff,
            })

        output_path = self._prepare_path(output_path)
        pd.DataFrame(rows, columns=CONTACT_SEQUENCE_COLUMNS).to_csv(output_path, index=False)
        return output_path

    def export_alignments_and_contacts(self, outdir=None, cutoff=DEFAULT_CUTOFF):
        """Write alignment and contact tables; alias kept for compatibility with save_all."""
        return self.save_all(outdir=outdir, numbering=("original",), cutoff=cutoff)

    def summary(self):
        """Return one row per chain with sequence lengths and mapping coverage."""
        rows = []
        for chain_id, data in self.chains.items():
            length = data["sequence_length"]
            rows.append({
                "chain_id": chain_id,
                "entity_id": data["entity_id"],
                "sequence_length": length,
                "reference_length": data["reference_length"],
                "mapped_residues": data["mapped_residues"],
                "mapping_fraction": round(data["mapped_residues"] / length, 3) if length else 0.0,
            })
        return pd.DataFrame(rows)

    ###################################### INTERFACE METRICS

    @staticmethod
    def _standard_residues(chain):
        return [residue for residue in chain if residue.id[0] == " "]

    @staticmethod
    def _heavy_atoms(residues):
        coords, owners = [], []
        for index, residue in enumerate(residues):
            for atom in residue:
                if (atom.element or atom.get_name()[:1]).strip().upper() != "H":
                    coords.append(atom.coord)
                    owners.append(index)
        return np.array(coords, float).reshape(-1, 3), np.array(owners, int)

    @staticmethod
    def _total_sasa(chains):
        model = Model(0)
        for chain in chains:
            model.add(chain.copy())
        ShrakeRupley().compute(model, level="A")
        return float(sum(atom.sasa for atom in model.get_atoms()))

    @staticmethod
    def _residue_sasa(chains):
        model = Model(0)
        for chain in chains:
            model.add(chain.copy())
        ShrakeRupley().compute(model, level="R")
        return {(chain.id, residue.id[1]): residue.sasa
                for chain in model for residue in chain if residue.id[0] == " "}

    @staticmethod
    def _atom_contacts(xa, xb, cutoff, score_cutoff, block):
        n, score, dmin = 0, 0.0, np.inf
        ha, hb = np.zeros(len(xa), bool), np.zeros(len(xb), bool)
        for start in range(0, len(xa), block):
            d = np.linalg.norm(xa[start:start + block, None] - xb[None], axis=-1)
            m = d <= cutoff
            n += int(m.sum())
            w = (d <= score_cutoff) & (d > 0)
            score += float((1.0 / d[w]).sum())
            dmin = min(dmin, float(d.min()))
            ha[start:start + block] |= m.any(axis=1)
            hb |= m.any(axis=0)
        return n, score, dmin, ha, hb

    @staticmethod
    def _residue_min_distances(xa, oa, xb, ob, na, nb, block):
        out = np.full((na, nb), np.inf, np.float32)
        for start in range(0, len(xa), block):
            d = np.linalg.norm(xa[start:start + block, None] - xb[None], axis=-1).astype(np.float32)
            np.minimum.at(out, (np.repeat(oa[start:start + block], len(ob)), np.tile(ob, d.shape[0])), d.ravel())
        return out

    @staticmethod
    def _hydrophobicity(residue):
        return HYDRO.get(TO1.get(residue.get_resname().upper(), "X"), 0.0)

    @staticmethod
    def _split_partners(text):
        return [p for p in str(text).split(";") if p]

    def _compute_interface(self, chain_a, chain_b):
        cutoff = self.config["interface_contact_cutoff_a"]
        score_cutoff = self.config["interface_score_cutoff_a"]
        block = self.config["interface_block_size"]

        A, B = self.chains[chain_a]["chain"], self.chains[chain_b]["chain"]
        ra, rb = self._standard_residues(A), self._standard_residues(B)
        xa, oa = self._heavy_atoms(ra)
        xb, ob = self._heavy_atoms(rb)
        if not len(xa) or not len(xb):
            raise ValueError(f"No heavy atoms in chain {chain_a} or {chain_b}.")

        sa, sb, sab = self._total_sasa([A]), self._total_sasa([B]), self._total_sasa([A, B])
        dasa, initial = sa + sb - sab, sa + sb
        n, score, dmin_c, ha, hb = self._atom_contacts(xa, xb, cutoff, score_cutoff, block)
        dmin = self._residue_min_distances(xa, oa, xb, ob, len(ra), len(rb), block)
        contact = dmin <= cutoff
        mask = dmin <= score_cutoff

        ha_h = np.array([self._hydrophobicity(r) for r in ra])
        hb_h = np.array([self._hydrophobicity(r) for r in rb])
        with np.errstate(invalid="ignore", divide="ignore"):
            energy = np.where(mask, -0.5 * (ha_h[:, None] + hb_h[None]) * (1 - dmin / score_cutoff), 0.0)

        complex_row = {
            "structure": self.structure_id,
            "len_res_a": len(ra), "len_res_b": len(rb),
            "len_atoms_a": len(xa), "len_atoms_b": len(xb),
            "dASA_A2": round(dasa, 2),
            "initial_sasa_A2": round(initial, 2),
            "frac_dASA": round(dasa / initial, 4) if initial > 0 else 0.0,
            "n_contacts": n,
            "iface_res_a": int(np.unique(oa[ha]).size),
            "iface_res_b": int(np.unique(ob[hb]).size),
            "min_dist_A": round(dmin_c, 3),
            "contact_score": round(score, 4),
        }

        iso_a = self._residue_sasa([A])
        iso_b = self._residue_sasa([B])
        cpx = self._residue_sasa([A, B])
        sides = (
            (chain_a, chain_b, ra, rb, iso_a, energy.sum(axis=1), contact.sum(axis=1), dmin.min(axis=1), contact),
            (chain_b, chain_a, rb, ra, iso_b, energy.sum(axis=0), contact.sum(axis=0), dmin.min(axis=0), contact.T),
        )

        rows = []
        for cid, oid, residues, other, iso, en_sum, npart, dm, pairs in sides:
            for k, residue in enumerate(residues):
                if not npart[k]:
                    continue
                key = (cid, residue.id[1])
                dsasa = float(iso[key] - cpx[key])
                maxa = TIEN.get(residue.get_resname().upper())
                en = float(en_sum[k])
                labels = [f"{oid}{other[j].id[1]} {other[j].get_resname()}" for j in np.flatnonzero(pairs[k])]
                rows.append({
                    "structure": self.structure_id,
                    "chain": cid,
                    "resid": residue.id[1],
                    "resname": residue.get_resname(),
                    "dSASA_A2": round(dsasa, 2),
                    "isolated_sasa_A2": round(float(iso[key]), 2),
                    "rel_dSASA": round(dsasa / maxa, 4) if maxa else float("nan"),
                    "n_partners": int(npart[k]),
                    "min_dist_A": round(float(dm[k]), 3),
                    "contact_energy": round(en, 4),
                    "energy_per_partner": round(en / npart[k], 4),
                    "partners": ";".join(labels),
                })

        return {
            "complex": complex_row,
            "residues": pd.DataFrame(rows, columns=INTERFACE_RESIDUE_COLUMNS),
        }

    def interface(self, chain_a="A", chain_b="B"):
        """Return {'complex': dict, 'residues': DataFrame} with buried area, contacts, distances, energies, and partners."""
        chain_a, chain_b = self._require_chain(chain_a), self._require_chain(chain_b)
        if chain_a == chain_b:
            raise ValueError("chain_a and chain_b must be different chains.")
        key = (chain_a, chain_b, self.config["interface_contact_cutoff_a"],
               self.config["interface_score_cutoff_a"], self.config["interface_block_size"])
        if key not in self._interface_cache:
            self._interface_cache[key] = self._compute_interface(chain_a, chain_b)
        cached = self._interface_cache[key]
        return {"complex": dict(cached["complex"]), "residues": cached["residues"].copy()}

    def interface_complex(self, chain_a="A", chain_b="B"):
        """Return complex-level interface metrics as a dict."""
        return self.interface(chain_a, chain_b)["complex"]

    def interface_residues(self, chain_a="A", chain_b="B"):
        """Return residue-level interface metrics as a DataFrame."""
        return self.interface(chain_a, chain_b)["residues"]

    def buried_area(self, chain_a="A", chain_b="B"):
        """Return the buried surface area in A^2 between two chains."""
        return self.interface_complex(chain_a, chain_b)["dASA_A2"]

    def initial_sasa(self, chain_a="A", chain_b="B"):
        """Return the sum of isolated SASA of both chains in A^2."""
        return self.interface_complex(chain_a, chain_b)["initial_sasa_A2"]

    def buried_fraction(self, chain_a="A", chain_b="B"):
        """Return buried area divided by initial SASA."""
        return self.interface_complex(chain_a, chain_b)["frac_dASA"]

    def contact_count(self, chain_a="A", chain_b="B"):
        """Return the number of heavy-atom pairs within the contact cutoff."""
        return self.interface_complex(chain_a, chain_b)["n_contacts"]

    def min_distance(self, chain_a="A", chain_b="B"):
        """Return the minimum heavy-atom distance between two chains in A."""
        return self.interface_complex(chain_a, chain_b)["min_dist_A"]

    def contact_score(self, chain_a="A", chain_b="B"):
        """Return the sum of inverse heavy-atom distances within the score cutoff."""
        return self.interface_complex(chain_a, chain_b)["contact_score"]

    def partners(self, chain_a="A", chain_b="B"):
        """Return {'chain+resid resname': [partner labels]} for interface residues of both chains."""
        frame = self.interface_residues(chain_a, chain_b)
        return {
            f"{row.chain}{row.resid} {row.resname}": self._split_partners(row.partners)
            for row in frame.itertuples(index=False)
        }

    ###################################### INTERFACE ENSEMBLES

    @staticmethod
    def add_interface_frequencies(frame):
        """Add single_freq and pair_freq: number of structures containing each residue or residue pair."""
        frame = frame.copy()
        keys = (frame["chain"].astype(str) + frame["resid"].astype(str) + " " + frame["resname"].astype(str)).tolist()
        partner_lists = [SyncedStructure._split_partners(p) for p in frame["partners"]]
        singles, pairs = defaultdict(set), defaultdict(set)
        for name, key, plist in zip(frame["structure"], keys, partner_lists):
            singles[key].add(name)
            for partner in plist:
                pairs["|".join(sorted((key, partner)))].add(name)
        frame["single_freq"] = [len(singles[k]) for k in keys]
        frame["pair_freq"] = [
            max((len(pairs["|".join(sorted((k, p)))]) for p in plist), default=0)
            for k, plist in zip(keys, partner_lists)
        ]
        return frame

    @classmethod
    def interface_ensemble(cls, input_dir, chain_a="A", chain_b="B", workers=None,
                           config_path=DEFAULT_CONFIG_PATH, outdir=None):
        """Compute interfaces for every structure in a directory in parallel, with frequencies across the ensemble.

        Returns {'complex': DataFrame, 'residues': DataFrame, 'skipped': list}. If outdir is given,
        writes <dir>_complex_scores.tsv and <dir>_residue_scores.tsv there.
        """
        start = time.perf_counter()
        input_dir = Path(input_dir).expanduser()
        files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"No structure files found in {input_dir}")

        resolve_config(config_path, DEFAULT_CONFIG)
        jobs = [(str(p), chain_a, chain_b, str(config_path)) for p in files]
        complexes, frames, skipped = [], [], []
        with multiprocessing.Pool(workers or os.cpu_count()) as pool:
            for name, complex_row, frame, error in tqdm(
                pool.imap_unordered(_interface_job, jobs), total=len(jobs), desc="interfaces",
            ):
                if error:
                    skipped.append(name)
                    print(f"[PYTHON-INFO] {name}: skipped, {error}")
                    continue
                complexes.append(complex_row)
                frames.append(frame)

        complex_frame = pd.DataFrame(complexes, columns=INTERFACE_COMPLEX_COLUMNS).sort_values("structure", ignore_index=True)
        residue_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=INTERFACE_RESIDUE_COLUMNS)
        residue_frame = cls.add_interface_frequencies(residue_frame).sort_values(
            ["structure", "chain", "resid"], ignore_index=True,
        )

        if outdir:
            outdir = cls._prepare_dir(outdir)
            tag = input_dir.resolve().name
            complex_frame.to_csv(outdir / f"{tag}_complex_scores.tsv", sep="\t", index=False)
            residue_frame.to_csv(outdir / f"{tag}_residue_scores.tsv", sep="\t", index=False)

        finish_log(start, f"Interfaces {chain_a}-{chain_b} for {len(complexes)} structures, skipped {len(skipped)}")
        return {"complex": complex_frame, "residues": residue_frame, "skipped": skipped}

    ###################################### EXAMPLE USAGE


def _interface_job(job):
    """Worker: compute one structure's interface, returning (name, complex_row, residue_frame, error)."""
    path, chain_a, chain_b, config_path = job
    name = Path(path).stem
    try:
        result = SyncedStructure.from_Path(path, config_path=config_path).interface(chain_a, chain_b)
        return name, result["complex"], result["residues"], None
    except Exception as ex:
        return name, None, None, str(ex).replace("\n", " ")


if __name__ == "__main__":
    start = time.perf_counter()
    np.random.seed(SEED)

    structure = SyncedStructure.from_rcsb_id("1UBQ")
    print(structure)
    print(structure.summary())
    print(structure.alignment_stats)
    print(structure.residue_map.head())

    if structure.chain_ids:
        chain_id = structure.chain_ids[0]
        contacts = structure.compute_contacts(cutoff=DEFAULT_CUTOFF, contact_type="inter")
        print(f"[PYTHON-INFO] Chain: {chain_id}")
        print(f"[PYTHON-INFO] Observed sequence: {structure.get_sequence(chain_id)}")
        print(f"[PYTHON-INFO] Contact sequence:  {structure.get_contact_sequence(chain_id, contacts=contacts)}")
        print(structure.get_contact_summary(contacts=contacts))

    structure.save_all()
    finish_log(start, "SyncedStructure example completed")