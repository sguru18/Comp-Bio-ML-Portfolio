import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.Data.IUPACData import protein_letters_3to1
from collections import defaultdict
import requests

dataset_path = (
    Path(__file__).parent.parent.parent
    / "data/cryptobench/cryptobench-dataset/dataset.json"
)
cif_files_path = (
    Path(__file__).parent.parent.parent
    / "data/cryptobench/cryptobench-dataset/auxiliary-data/cif-files"
)
embeddings_path = Path(__file__).parent.parent.parent / "data/uniprot-esm2-embeddings"
data_path = Path(__file__).parent.parent.parent / "data"

# huggingface parameter for device_map only works for cuda multi-gpu
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D").eval().to(device)
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
    chain_to_uniprot_id = {}
    for entry in entries:
        chains = entry["apo_chain"].split("-")
        uniprotIds = entry["uniprot_id"].split("-")
        # we validated that if an entry has 1 uId, broadcast it for all chainIds, else there are N uIds for N chainIds
        if len(uniprotIds) == 1 and len(chains) != 1:
            uniprotIds = uniprotIds * len(chains)
        assert len(uniprotIds) == len(
            chains
        ), f"length assertion failed on pdb_id {pdb_id} entry/entries {uniprotIds}"

        for i in range(len(uniprotIds)):
            # might overwrite some stuff but chains : uId is 1:1 so its all good
            chain_to_uniprot_id[chains[i]] = uniprotIds[i]

    structure = parser.get_structure(pdb_id, cif_files_path / f"{pdb_id}.cif")
    bio_model = structure[0]

    # now we have all individual chains mentioned for this protein in new_chains
    # and their uniprot ids to hit the api with
    for chain_id in chain_to_uniprot_id.keys():
        uId = chain_to_uniprot_id[chain_id]
        url = f"https://rest.uniprot.org/uniprotkb/{uId}.fasta"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  WARNING: UniProt FASTA error for {uId}: {exc}")
        # response is a header line that starts with > and then lines of 60 char chunks of AA codes
        lines = resp.text.strip().split("\n")
        if not lines or not lines[0].startswith(">"):
            print(f"  WARNING: Unexpected FASTA format for {uId}")
        seq = "".join(lines[1:])

        key = f"{pdb_id}_{chain_id}"
        chain = bio_model[chain_id]
        sequence = ""
        for residue in chain:
            hetflag, auth_seq_id, icode = residue.get_id()
            if hetflag == " ":
                code = residue.get_resname()
                letter = protein_letters_3to1.get(code, "X")
                sequence += letter
                # maps the pdb_id_chain_id to a list of auth_seq_ids where each auth_seq_id is in the
                # position of its residue. ie. residue 0 of this chain maps to the auth_seq_id in index 0
                residue_map[key].append(f"{auth_seq_id}{icode.strip()}")

        # TODO: continue from step 4 here, align pdb fragment to uniprot sequence using pairwisealigner
        # requires implementing a sliding window approach

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
