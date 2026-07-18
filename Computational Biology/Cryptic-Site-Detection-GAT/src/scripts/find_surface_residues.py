"""

finds sasa values of all apo-pocket-selected residues
goal of this script was to find out how many apo-pocket-selected residues would be lost by using 20 square angstroms as the cut off
because the original plan was to only include surface residues (>= 20) as nodes in the GAT.
decided to look at the distribution anyway, and it ended up being very informative in showing that nearly 40% of apo-pocket-selected
residues have sasa lower than 20, meaning they are buried in apo form but are still classified as part of the cryptic pocket.
this is an important finding, since cutting 40% of the positive class out of the GAT entirely would be devastating for training and esp fusion.
now we'll include all residues in the GAT

barring any miscalculations with the sasa values, but I was very careful with this and checked articles and docs so
I think it should be good

"""

import json
from pathlib import Path
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley
from collections import defaultdict
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATASET_PATH = DATA_DIR / "cryptobench/cryptobench-dataset/dataset.json"
CIF_FILES_PATH = DATA_DIR / "cryptobench/cryptobench-dataset/auxiliary-data/cif-files"

# cutoff for sasa to determine surface-exposed or not
THRESHOLD = 20

with open(DATASET_PATH) as f:
    dataset = json.load(f)

parser = MMCIFParser(QUIET=True)
shrake_rupley = ShrakeRupley()

surface_exposed_residues = defaultdict(list)
auth_seq_id_to_pos_map = {}

binding_sasa_values = []

num_surface = 0
total_residues = 0

with open(DATA_DIR / "chain_to_auth_seq_id_map.json") as f:
    m = json.load(f)

for pdb_id, entries in dataset.items():
    # entries is an array
    # first go through all entries and make a set of all chains
    chains = set()
    apo_pocket_selection = set()
    for entry in entries:
        chains.update(entry["apo_chain"].split("-"))
        apo_pocket_selection.update(
            entry["apo_pocket_selection"]
        )  # each value is like A_12 which is chain_authseqid
    # apo_pocket_selection now has all positive residues for this apo structure

    # strip off all non-standard items in the structure
    # realized we had to do this after reading https://www.biostars.org/p/9518409/ and asking claude how to handle
    # idk what the effect would've been if we skipped but glad i found it, claude not reliable enough here
    structure = parser.get_structure(pdb_id, CIF_FILES_PATH / f"{pdb_id}.cif")
    model = structure[0]  # this is the only one we use
    for chain in model:
        residues_to_remove = [r for r in chain if r.get_id()[0] != " "]
        for residue in residues_to_remove:
            chain.detach_child(residue.get_id())

    # annotates each residue with a sasa property in angstroms squared
    shrake_rupley.compute(structure, level="R")

    # mark each residue as surface exposed or not, and then we need to save this all to a dict and dump in json file or something
    for chain_id in chains:
        chain = model[chain_id]
        key = f"{pdb_id}_{chain_id}"
        if key not in m:
            continue
            # represents the 15 ones that didnt have uniprot embeddings
        for residue in chain:
            surface_exposed_residues[key].append(residue.sasa)
            # this says the residue at this position has this value for sasa

        for pos, seq_id in enumerate(m[key]):  # key is pdbid_chainid, this is correct
            auth_seq_id_to_pos_map[f"{pdb_id}_{chain_id}_{seq_id}"] = pos

    for p in apo_pocket_selection:
        chain, seq_id = p.split("_")
        try:
            idx = auth_seq_id_to_pos_map[f"{pdb_id}_{chain}_{seq_id}"]
        except KeyError:
            print(
                f"{pdb_id}_{chain}_{seq_id} not in embedding map, probably non-standard residue"
            )
            continue
        if f"{pdb_id}_{chain}" not in surface_exposed_residues:
            continue  # this missing key stuff is getting a bit confusing but it's all just the 15 missing chains propogating

        val = surface_exposed_residues[f"{pdb_id}_{chain}"][idx]
        if val >= THRESHOLD:
            num_surface += 1
        total_residues += 1

        binding_sasa_values.append(val)

with open(DATA_DIR / "chain_to_sasa_values_map.json", "w") as f:
    json.dump(surface_exposed_residues, f)

print(
    f"{num_surface} out of {total_residues} total apo-pocket selected residues are surface at 20 angstrom threshold, this is {num_surface/total_residues*100:.3f}%"
)

# thanks claude
plt.figure(figsize=(8, 5))
plt.hist(binding_sasa_values, bins=50, color="steelblue", edgecolor="white")
plt.axvline(
    x=THRESHOLD, color="red", linestyle="--", label=f"threshold = {THRESHOLD} Å²"
)
plt.xlabel("SASA (Å²)")
plt.ylabel("Count")
plt.title("SASA distribution of binding residues")
plt.legend()
plt.tight_layout()
plt.savefig("binding_sasa_distribution.png", dpi=150)
plt.show()
