# Generate the ESMC embeddings for all of the chains used in dataset.json
# Later used to generate ESM2 embeddings
# Abandoned in favor of using the whole UniProt sequences after both esmc and esm2 yielded very poor results

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.Data.IUPACData import protein_letters_3to1
from collections import defaultdict

dataset_path = (
    Path(__file__).parent.parent.parent
    / "data/cryptobench/cryptobench-dataset/dataset.json"
)
cif_files_path = (
    Path(__file__).parent.parent.parent
    / "data/cryptobench/cryptobench-dataset/auxiliary-data/cif-files"
)
# embeddings_path = Path(__file__).parent.parent.parent / "data/esmc-embeddings"
embeddings_path = Path(__file__).parent.parent.parent / "data/esm2-embeddings"
data_path = Path(__file__).parent.parent.parent / "data"

# huggingface parameter for device_map only works for cuda multi-gpu
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# model = AutoModel.from_pretrained("biohub/ESMC-600M").eval().to(device)
model = (
    AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D", torch_dtype=torch.float32)
    .eval()
    .to(device)
)
# tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")


with open(dataset_path) as f:
    dataset = json.load(f)

parser = MMCIFParser(QUIET=True)

residue_map = defaultdict(list)

for pdb_id, entries in dataset.items():
    # entries is an array
    # first go through all entries and make a set of all chains
    # make sure to split entries like A-B
    # we already validated that all chain names are valid in cif files and to just split on "-"
    chains = set()
    for entry in entries:
        chains.add(entry["apo_chain"])  # "X" or "X-Y" or "AAA" or "AAA-CCC"
    new_chains = set()
    for chain in chains:
        # need to split the hyphenated ones, split("-") has no effect on the others
        split = chain.split("-")
        new_chains.update(split)

    structure = parser.get_structure(pdb_id, cif_files_path / f"{pdb_id}.cif")
    bio_model = structure[0]

    # now we have all individual chains mentioned for this protein in new_chains
    for chain_id in new_chains:
        key = f"{pdb_id}_{chain_id}"
        chain = bio_model[chain_id]
        sequence = ""
        for residue in chain:
            hetflag, auth_seq_id, icode = residue.get_id()
            if hetflag == " ":
                code = residue.get_resname()
                letter = protein_letters_3to1.get(code, "X")
                sequence += letter

                residue_map[key].append(f"{auth_seq_id}{icode.strip()}")

        if len(sequence) > 1022:
            print(f"WARNING: {key} truncated from {len(sequence)} to 1022 residues")
            sequence = sequence[:1022]
            residue_map[key] = residue_map[key][:1022]

        inputs = tokenizer(sequence, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = model(**inputs)
        embeddings = output.last_hidden_state[0]
        embeddings = embeddings[1:-1]
        embeddings = embeddings.to(torch.float32).cpu().numpy()
        np.save(embeddings_path / f"{pdb_id}_{chain_id}.npy", embeddings)
        print(f"successfully saved {pdb_id}_{chain_id}.npy")

with open(data_path / "chain_to_auth_seq_id_map.json", "w") as f:
    json.dump(residue_map, f)
