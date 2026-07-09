import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import numpy as np


class ResidueDataset(Dataset):

    def __init__(self, fold_paths):
        # load all npy files into ram upon init, small enough to where this does not hinder anything
        # options to enhance if needed are numpy memory mapped arrays or HDF5 but overkill
        # avoids loading files inefficiently in __getitem__
        self.count = 0  # the total number of residues in this dataset
        self.index = []  # maps an int index in [0,count) to a specific residue
        self.embeddings = {}
        self.auth_seq_id_to_pos_map = {}
        self.positive_set = set()
        DATA_DIR = Path(__file__).parent.parent / "data"
        EMBEDDINGS_DIR = Path(__file__).parent.parent / "data/esm2-embeddings"
        with open(DATA_DIR / "chain_to_auth_seq_id_map.json") as f:
            m = json.load(f)

        for fold_path in fold_paths:
            with open(fold_path) as f:
                dataset = json.load(f)
            for pdb_id, entries in dataset.items():
                chains = set()
                apo_pocket_selection = set()
                for entry in entries:
                    chains.update(entry["apo_chain"].split("-"))
                    # NOTE: using the union of all apo_pocket_selections instead of selection
                    # from the is_main_holo_structure pair only because this is what authors did
                    # also many non-main pairings have pRMSD very close to the main, ie.
                    # 2.39 to 2.17 so many other residues did actually move, using them
                    # as negatives would hurt model training. alternative might be to introduce
                    # a min_pRMSD parameter into dataset class
                    apo_pocket_selection.update(entry["apo_pocket_selection"])
                for chain in chains:
                    key = (pdb_id, chain)
                    self.embeddings[key] = np.load(
                        EMBEDDINGS_DIR / f"{pdb_id}_{chain}.npy"
                    )
                    num_residues = self.embeddings[key].shape[0]
                    for pos in range(num_residues):
                        self.index.append((pdb_id, chain, pos))
                        # pos here is the index of the residue in this chain's embedding file

                    # count the total number of residues so that we can add it to self.count
                    self.count += num_residues

                    # m (from the json file) maps protein_chain to auth_seq_ids in the same
                    # order their embeddings appear in the embedding file
                    # here we are reversing it, mapping the seq_id to their position in m
                    # and therefore the index in the embedding file
                    # now we can lookup easily for apo_pocket_selection, retrieving the
                    # index in the embedding file that this auth_seq_id is at
                    for pos, seq_id in enumerate(m[f"{pdb_id}_{chain}"]):
                        self.auth_seq_id_to_pos_map[f"{pdb_id}_{chain}_{seq_id}"] = pos

                # add the ones in apo_pocket_selection to self.positive_set
                for p in apo_pocket_selection:
                    chain, seq_id = p.split("_")
                    try:
                        pos = self.auth_seq_id_to_pos_map[f"{pdb_id}_{chain}_{seq_id}"]
                    except KeyError:
                        print(
                            f"non-standard amino acid OR truncated residue on 1 of 5 chains found in apo_pocket_selection and skipped"
                        )
                        continue
                    self.positive_set.add((pdb_id, chain, pos))

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        pdb_id, chain, pos = self.index[idx]
        key = (pdb_id, chain)
        embedding = self.embeddings[key][pos]

        tupl = (pdb_id, chain, pos)
        label = int(tupl in self.positive_set)

        sample = {
            "embedding_vector": torch.tensor(embedding, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
        }

        return sample
