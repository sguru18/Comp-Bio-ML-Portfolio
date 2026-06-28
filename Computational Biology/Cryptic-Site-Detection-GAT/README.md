download cryptobench dataset from https://osf.io/pz4a9/overview and drop unzipped folder into the data folder
create virtual environment in project root, install dependencies, and run generate-esmc-embeddings.py script

---

## Implementation Notes

### Dataset structure

- `dataset.json` keys are apo PDB IDs. Values are arrays of holo pairings (one apo can pair with multiple holos).
- `apo_chain` is consistent across all entries for a given apo PDB ID (verified).
- `apo_chain` can be a single chain (`"A"`) or multiple chains delimited by `-` (`"A-B"`). Split on `-` to get individual chain IDs. `-` is a delimiter, not a range.
- Chain IDs can be multi-character (e.g. `"AAA"`) for newer mmCIF structures — verified that BioPython reads these correctly.
- For embedding generation, collect all unique individual chains per PDB ID and embed each one separately. Save as `{pdb_id}_{chain}.npy`.

### Pocket residue format

- `apo_pocket_selection` entries like `"B_12"` mean `auth_asym_id=B`, `auth_seq_id=12` (author numbering, per README).
- BioPython's `MMCIFParser` exposes author fields by default: `chain.id` = `auth_asym_id`, `residue.get_id()[1]` = `auth_seq_id` (integer), `residue.get_id()[2]` = insertion code.
- When mapping pocket labels to residues, match on both `seq_id` AND `icode == ' '` to handle insertion codes (e.g. residues 12 and 12A both have seq_id=12 in BioPython).
- Use `is_main_holo_structure: true` entry as the canonical label source for each apo protein (highest pRMSD pairing).

### ESMC embeddings

- Use `AutoModel` (not `AutoModelForMaskedLM`) — no LM head, outputs raw per-residue representations.
- Input: full chain sequence as a string of 1-letter amino acid codes.
- Output: `output.last_hidden_state[0, 1:-1, :]` → shape `[seq_len, 1152]`. Strip index 0 (BOS) and -1 (EOS).
- Feed the full sequence even if you only care about surface residues — transformer needs full context. SASA filtering happens later at graph construction time.
- Context window limit: 2048 tokens.

### Graph construction (future step)

- Nodes: surface-exposed residues (filter by SASA threshold computed separately via BioPython/FreeSASA).
- Node features: ESMC embedding vector for that residue from the precomputed `.npy` file.
- Edges: spatial proximity in 3D (distance between residues in CIF coordinates).
- For multi-chain pockets (e.g. `"A-B"`): add cross-chain edges between residues of different chains based on 3D proximity. Inter-chain context is handled here, not at the ESMC embedding stage.
