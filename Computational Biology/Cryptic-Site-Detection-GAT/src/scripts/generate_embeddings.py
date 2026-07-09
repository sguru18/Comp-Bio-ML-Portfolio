import token
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.Align import PairwiseAligner
from collections import defaultdict
import requests
import math

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

# claude told me how to use these
parser = MMCIFParser(QUIET=True)
_aligner = PairwiseAligner()
_aligner.mode = "global"
_aligner.match_score = 2
_aligner.mismatch_score = -1
_aligner.open_gap_score = -2
_aligner.extend_gap_score = -0.5

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
            # we saw that this does happen, 14 chains will not have uniprot sequences
            # TODO: fallback to embedding the pdb sequence itself without uniprot
        uniprot_sequence = "".join(lines[1:])

        key = f"{pdb_id}_{chain_id}"
        chain = bio_model[chain_id]
        pdb_sequence = ""
        for residue in chain:
            hetflag, auth_seq_id, icode = residue.get_id()
            if hetflag == " ":
                code = residue.get_resname()
                letter = protein_letters_3to1.get(code, "X")
                pdb_sequence += letter
                # maps the pdb_id_chain_id to a list of auth_seq_ids where each auth_seq_id is in the
                # position of its residue. ie. residue 0 of this chain maps to the auth_seq_id in index 0
                residue_map[key].append(f"{auth_seq_id}{icode.strip()}")

        # TODO: implement step 4 here, align pdb fragment to uniprot sequence using pairwisealigner
        # in one case the pdb chain is actually longer than the uniprot so account for that
        # this means some residues will also map to None in the sequence, can fallback to esm2 embedding again if we want to
        # but these residues don't exist in the canonical form so esm2 embedding might not be very informative
        # should be a minor case, we can count how many times it happens

        # step1: split the uniprot sequence into N overlapping groups
        # step2: embed them all
        # step3: average rows together that represent the same residue based on overlaps. this gives one
        # matrix of size [len_uniprot_seq, 2560]
        # step4: map each residue to its embedding in this matrix using the indices given by the aligner
        if len(uniprot_sequence) > 1022:
            # assuming ~50% overlap, this means each split after the first one contains 511 new residues
            n = len(uniprot_sequence) - 511
            # now this means each split chain including the first one needs to contribute 511 out of n residues
            # so the number of split chains is just ceil(n/511)
            # and the last 511 residues in one split chain are the first 511 in the next one
            # maybe we should weight the average by distance from the middle?
            # if a residue is near the end in one, then it's near the middle in the next one and latter is a richer representation?
            # TODO: research weighted average and implement if it represents something significant
            # im guessing not because of bidirectional attention for esm2
            num_chains = math.ceil(n / 511)
            split_chains = []
            # i = 0 -> [0:1022] which is [0:511] + [511:1022]
            # i = 1 -> [511:1533] which is [511:1022] + [1022:1533]
            # i = 2 -> [1022:2044] which is [1022:1533] + [1533:2044]
            # l is always 511 * i, r is 511 * (i+1)
            # beautiful
            # step1
            for i in range(num_chains):
                l = 511 * i
                if i == num_chains - 1: # go until the end, no need to manually set r
                    split_chains.append(uniprot_sequence[l:])
                else:
                    r = l + 511 
                    split_chains.append(uniprot_sequence[l:r])
            uniprot_embeddings = []
            # step2
            for chain in split_chains:
                inputs = tokenizer(chain, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    output = model(**inputs)
                embeddings = output.last_hidden_state[0]
                embeddings = embeddings[1:-1]
                embeddings = embeddings.to(torch.float32).cpu().numpy()
                uniprot_embeddings.append(embeddings)
            # step3
            for i in range(num_chains-1):
                this_chain = embeddings[i]
                next_chain = embeddings[i+1]
                # second half of this chain and first half of next chain represent the same residues
                # have to be careful at the end though

        else:
            # lol

        np.save(embeddings_path / f"{pdb_id}_{chain_id}.npy", embeddings)
        print(f"successfully saved {pdb_id}_{chain_id}.npy")

with open(data_path / "chain_to_auth_seq_id_map.json", "w") as f:
    json.dump(residue_map, f)
