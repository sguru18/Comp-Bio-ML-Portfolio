# Check whether we just have to generate embeddings for the first apo_chain per protein
# in dataset.json, or whether one protein's array can have multiple apo_chains of interest

import json
from pathlib import Path

dataset_path = (
    Path(__file__).parent.parent.parent
    / "data/cryptobench/cryptobench-dataset/dataset.json"
)

with open(dataset_path) as f:
    dataset = json.load(f)

conflicts = {}
# entries is an array
for pdb_id, entries in dataset.items():
    chains = set(e["apo_chain"] for e in entries)
    if len(chains) > 1:
        conflicts[pdb_id] = chains

if conflicts:
    print(f"Found {len(conflicts)} PDB IDs with multiple apo_chains:")
    for pdb_id, chains in conflicts.items():
        print(f"  {pdb_id}: {chains}")
else:
    print("All PDB IDs have a consistent apo_chain across entries.")

# Output confirms that not all holo entries per protein are on the
# same chain, need to iterate across all of the entries so that we find all chains referenced
# Found 23 PDB IDs with multiple apo_chains:
#   1g1m: {'A-B', 'B'}
#   1g1o: {'A', 'A-D', 'D'}
#   1lx7: {'A-B', 'A'}
#   1o24: {'A-C-D', 'D'}
#   1vr6: {'A-D', 'D'}
#   2b3s: {'A-B', 'A', 'B'}
#   2nt1: {'A-B', 'B'}
#   2zcg: {'A-B', 'A'}
#   3lnz: {'C', 'C-O', 'O'}
#   3pfp: {'A-B', 'A', 'B'}
#   4hr7: {'A', 'D'}
#   4yaj: {'B-D', 'B'}
#   4yt8: {'A-B', 'B'}
#   5dem: {'A', 'A-C'}
#   5lgr: {'B', 'B-F'}
#   5u9m: {'A', 'C', 'A-C'}
#   6ial: {'F-G', 'F'}
#   6iz9: {'A-B', 'A'}
#   6z2h: {'B-C', 'B'}
#   7nbc: {'CCC', 'AAA-CCC'}
#   8dfn: {'A-B', 'A', 'B'}
#   8gxj: {'C', 'B-C'}
#   8h6p: {'A-B', 'A'}
