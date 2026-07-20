"""
generate the Data objects for GAT, one per protein
this consists of
"""

from pathlib import Path
import json
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley

ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT_DIR / "data/cryptobench/cryptobench-dataset/dataset.json"
CIF_FILES_DIR = (
    ROOT_DIR / "data/cryptobench/cryptobench-dataset/auxiliary-data/cif-files"
)
JSON_PATH = ROOT_DIR / "data/chain_to_auth_seq_id_map.json"

parser = MMCIFParser(QUIET=True)
shrake_rupley = ShrakeRupley()

with open(DATASET_PATH) as f:
    dataset = json.load(f)
with open(JSON_PATH) as f1:
    m = json.load(f1)

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
    for chain in chains:
        key = f"{pdb_id}_{chain}"
        auth_seq_ids = m[key]
