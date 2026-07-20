"""
generate the Data objects for GAT, one per protein
this consists of
"""

from pathlib import Path
import json
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT_DIR / "data/cryptobench/cryptobench-dataset/dataset.json"
CIF_FILES_DIR = (
    ROOT_DIR / "data/cryptobench/cryptobench-dataset/auxiliary-data/cif-files"
)
JSON_PATH = ROOT_DIR / "data/chain_to_auth_seq_id_map.json"

# from kyte-doolittle
AA_PROPERTIES = {
    # [hydrophobicity, charge, polar]
    "ALA": [1.8, 0, 0],
    "CYS": [2.5, 0, 1],
    "ASP": [-3.5, -1, 1],
    "GLU": [-3.5, -1, 1],
    "PHE": [2.8, 0, 0],
    "GLY": [-0.4, 0, 0],
    "HIS": [-3.2, 0.5, 1],
    "ILE": [4.5, 0, 0],
    "LYS": [-3.9, 1, 1],
    "LEU": [3.8, 0, 0],
    "MET": [1.9, 0, 0],
    "ASN": [-3.5, 0, 1],
    "PRO": [-1.6, 0, 0],
    "GLN": [-3.5, 0, 1],
    "ARG": [-4.5, 1, 1],
    "SER": [-0.8, 0, 1],
    "THR": [-0.7, 0, 1],
    "VAL": [4.2, 0, 0],
    "TRP": [-0.9, 0, 1],
    "TYR": [-1.3, 0, 1],
}

AAs = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]
LOOKUP = {aa: i for i, aa in enumerate(AAs)}


def compute_local_curvature(coords, index):
    pass


parser = MMCIFParser(QUIET=True)
shrake_rupley = ShrakeRupley()

with open(DATASET_PATH) as f:
    dataset = json.load(f)
with open(JSON_PATH) as f1:
    m = json.load(f1)

num_missing_CA = 0
total = 0

# get all the chains per protein (from dataset.json)
# sort the chains
# look up the chains in the chain_to_auth_seq_id_map in this sorted order
# and within that chain follow the order of residues because this order is the same as the residues
# in embeddings. so orders within files will match and now within one protein's full Data object file,
# we know the chains are in sorted order. so when we vstack chain's embeddings to match to pass into mlp,
# we can just use sorted(chains) and we know they match on both outer and inner layers
pdb_to_chains = {}
for pdb_id, entries in dataset.items():

    chains = set()
    for entry in entries:
        chains.update(entry["apo_chain"].split("-"))

    chains = sorted(list(chains))
    pdb_to_chains[pdb_id] = chains

for pdb_id, chains in pdb_to_chains.items():
    structure = parser.get_structure(pdb_id, CIF_FILES_DIR / f"{pdb_id}.cif")
    model = structure[0]
    feature_matrix = []
    coords = []

    for chain in model:
        residues_to_remove = [r for r in chain if r.get_id()[0] != " "]
        for residue in residues_to_remove:
            chain.detach_child(residue.get_id())

    # annotates each residue with a sasa property in angstroms squared
    shrake_rupley.compute(structure, level="R")

    for chain_id in chains:  # these are in sorted order
        key = f"{pdb_id}_{chain_id}"
        if key not in m:
            # this just means we won't load this chain when we stack residue embeddings either
            continue
        auth_seq_ids = m[key]
        if chain_id not in model:
            # this is i think impossible because the embeddings were made from the cif
            # so everything that shows up now from iterating the json must be in the cif i think
            print(f"bad bad on pdb_id {pdb_id} and chain_id {chain_id}!!!!!!!!!")
            continue
        chain = model[chain_id]

        for seq_id in auth_seq_ids:

            # access the residue
            seq_id = str(seq_id)
            icode = seq_id[-1] if seq_id[-1].isalpha() else " "
            seq_id_int = int(seq_id.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            tupl = (" ", seq_id_int, icode)
            residue = chain[tupl]

            # calculate what we need
            if "CA" not in residue:
                # TODO: implement fallback to centroid for coords? hopefully not very common
                xyz = [0, 0, 0]
                b_factor = 0
                num_missing_CA += 1
            else:
                xyz = residue["CA"].get_vector().get_array()  # 3 dim
                b_factor = residue[
                    "CA"
                ].get_bfactor()  # high b factor means very flexible
            total += 1
            coords.append(xyz)
            sasa = residue.sasa  # how exposed it is
            AA_one_hot = [0] * 20
            idx = LOOKUP.get(
                residue.get_resname(), None
            )  # shouldn't have any Nones but just in case
            if idx is not None:
                AA_one_hot[idx] = 1
            biophysical = AA_PROPERTIES.get(residue.get_resname(), [0, 0, 0])
            row = np.concatenate(
                [AA_one_hot, [b_factor], [sasa], biophysical]  # 20  # 1  # 1  # 3
            )  # 2 more from curvature added below
            # TODO: look into adding secondary structure here
            feature_matrix.append(row)

    for idx in range(len(coords)):
        k1, k2 = compute_local_curvature(coords, idx)
        feature_matrix[idx] = np.concatenate([feature_matrix[idx], [k1, k2]])

    # implement compute_local_curvature

    # compute pairwise Ca distances

    # build y labels

    # package into Data objects and store as {pdb_id}.pt

print(
    f"{num_missing_CA} residues out of {total} are missing Ca atoms, this is {num_missing_CA/total*100:.3f}%"
)
