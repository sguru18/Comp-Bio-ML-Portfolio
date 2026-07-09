"""
Validation: Full UniProt sequence ESM2-3B embeddings vs PDB-fragment embeddings.

Hypothesis: Embedding the full UniProt sequence (instead of the shorter
PDB-observed fragment) gives richer context to ESM2, improving AUPRC for
cryptic binding-site detection.

Runs on the first 50 PDB IDs in dataset.json for validation speed, then
fits a sklearn LogisticRegression and prints AUPRC / AUROC.

If AUPRC > 0.15 → scale to all 1107 proteins.
"""

import sys
import requests  # only used for UniProt FASTA fetch
import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from Bio.Align import PairwiseAligner
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB.MMCIFParser import MMCIFParser
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "cryptobench/cryptobench-dataset/dataset.json"
CIF_DIR = DATA_DIR / "cryptobench/cryptobench-dataset/auxiliary-data/cif-files"
UNIPROT_EMB_DIR = DATA_DIR / "esm2-uniprot-embeddings"
UNIPROT_EMB_DIR.mkdir(parents=True, exist_ok=True)
MAP_PATH = DATA_DIR / "chain_to_uniprot_auth_seq_id_map.json"

# ── Dataset — first 50 PDB IDs ───────────────────────────────────────────────
with open(DATASET_PATH) as f:
    full_dataset = json.load(f)

pdb_ids_50 = list(full_dataset.keys())[:50]
dataset_50 = {pid: full_dataset[pid] for pid in pdb_ids_50}

parser = MMCIFParser(QUIET=True)
residue_map: dict = defaultdict(list)  # key → [auth_seq_id, ...]

# ── Aligner (re-used across all chains) ──────────────────────────────────────
_aligner = PairwiseAligner()
_aligner.mode = "global"
_aligner.match_score = 2
_aligner.mismatch_score = -1
_aligner.open_gap_score = -2
_aligner.extend_gap_score = -0.5


# ── Helper functions ─────────────────────────────────────────────────────────


def fetch_uniprot_fasta(accession: str):
    """Fetch the canonical sequence for *accession* from UniProt FASTA."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  WARNING: UniProt FASTA error for {accession}: {exc}")
        return None
    lines = resp.text.strip().split("\n")
    if not lines or not lines[0].startswith(">"):
        print(f"  WARNING: Unexpected FASTA format for {accession}")
        return None
    return "".join(lines[1:])


def build_uniprot_positions(alignment, pdb_len: int) -> list:
    """
    Map each PDB residue index to its 0-based UniProt position.

    Uses alignment.aligned coordinate blocks (numpy arrays of shape [n, 2]).
    Residues that fall in gap regions are assigned None.
    """
    result = [None] * pdb_len
    for (p_start, p_end), (u_start, u_end) in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        for offset in range(int(p_end) - int(p_start)):
            result[int(p_start) + offset] = int(u_start) + offset
    return result


def embed_sequence(seq: str) -> np.ndarray:
    """
    Embed *seq* with ESM2-3B and strip CLS / EOS tokens.

    ESM2 max context is 1024 tokens (incl. CLS+EOS) → max residues 1022.
    Longer sequences are truncated with a printed notice; the alignment guard
    (upos < emb.shape[0]) then assigns zero vectors to out-of-range residues.

    Returns float32 numpy array of shape [min(len(seq), 1022), 2560].
    """
    if len(seq) > 1022:
        print(
            f"    NOTE: UniProt seq len {len(seq)} > 1022; truncating to 1022 for ESM2."
        )
        seq = seq[:1022]
    inputs = tokenizer(seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs)
    emb = output.last_hidden_state[0, 1:-1, :]  # strip CLS / EOS → [seq_len, 2560]
    return emb.to(torch.float32).cpu().numpy()


# ── Main embedding loop ───────────────────────────────────────────────────────

for pdb_id, entries in dataset_50.items():
    # Collect unique chain IDs and their UniProt IDs directly from dataset.json.
    # Every entry carries a "uniprot_id" field — no external API call needed.
    chains: set = set()
    chain_to_uniprot: dict = {}  # chain_id → UniProt accession
    for entry in entries:
        uid = entry.get("uniprot_id")
        for chain in entry["apo_chain"].split("-"):
            chains.add(chain)
            # Keep the first accession seen for this chain (all entries for the
            # same apo chain reference the same UniProt entry).
            if uid and chain not in chain_to_uniprot:
                chain_to_uniprot[chain] = uid

    try:
        structure = parser.get_structure(pdb_id, CIF_DIR / f"{pdb_id}.cif")
    except Exception as exc:
        print(f"WARNING: Could not parse CIF for {pdb_id}: {exc}")
        continue
    bio_model = structure[0]

    for chain_id in sorted(chains):
        key = f"{pdb_id}_{chain_id}"
        print(f"\n[{key}]")

        # ── 1. UniProt accession — read directly from dataset.json ───────────
        accession = chain_to_uniprot.get(chain_id)
        if accession is None:
            print(
                f"  WARNING: No uniprot_id in dataset for {pdb_id} chain {chain_id} — skipping."
            )
            continue
        print(f"  UniProt accession : {accession}")

        # ── 2. Full canonical UniProt sequence via UniProt REST ──────────────
        uniprot_seq = fetch_uniprot_fasta(accession)
        if uniprot_seq is None:
            print(f"  WARNING: Could not fetch FASTA for {accession} — skipping.")
            continue
        print(f"  UniProt seq len   : {len(uniprot_seq)}")

        # ── 3. PDB-observed sequence & auth_seq_ids (same filter as existing) ─
        try:
            chain_obj = bio_model[chain_id]
        except KeyError:
            print(
                f"  WARNING: Chain {chain_id} not found in CIF for {pdb_id} — skipping."
            )
            continue

        pdb_seq = ""
        auth_seq_ids: list = []
        for residue in chain_obj:
            hetflag, auth_seq_id, icode = residue.get_id()
            if hetflag == " ":
                letter = protein_letters_3to1.get(residue.get_resname(), "X")
                pdb_seq += letter
                auth_seq_ids.append(f"{auth_seq_id}{icode.strip()}")

        if not pdb_seq:
            print(f"  WARNING: Empty sequence for {key} — skipping.")
            continue

        # ── 4. Truncation guard (matches existing generate_embeddings.py) ────
        if len(pdb_seq) > 1022:
            print(f"  WARNING: {key} PDB seq truncated {len(pdb_seq)} → 1022 residues")
            pdb_seq = pdb_seq[:1022]
            auth_seq_ids = auth_seq_ids[:1022]

        print(f"  PDB seq len       : {len(pdb_seq)}")

        # ── 5. Global alignment: PDB observed → full UniProt ─────────────────
        alignments_gen = _aligner.align(pdb_seq, uniprot_seq)
        best_aln = next(iter(alignments_gen))
        uniprot_positions = build_uniprot_positions(best_aln, len(pdb_seq))

        # ── 6. Embed the full UniProt sequence (truncated if > 1022) ─────────
        print(f"  Embedding UniProt sequence …")
        uniprot_emb = embed_sequence(uniprot_seq)  # [≤1022, 2560]
        emb_dim = uniprot_emb.shape[1]  # 2560

        # ── 7. Gather per-residue embeddings via alignment positions ──────────
        final_emb = np.zeros((len(pdb_seq), emb_dim), dtype=np.float32)
        for i, upos in enumerate(uniprot_positions):
            # upos may be None (gap) or ≥ emb rows (truncated UniProt) → zero vector
            if upos is not None and upos < uniprot_emb.shape[0]:
                final_emb[i] = uniprot_emb[upos]

        # ── 8. Save embeddings ────────────────────────────────────────────────
        np.save(UNIPROT_EMB_DIR / f"{key}.npy", final_emb)
        print(f"  Saved  {key}.npy   shape={final_emb.shape}")

        # ── 9. Update residue map ─────────────────────────────────────────────
        residue_map[key].extend(auth_seq_ids)


# ── Persist residue map ───────────────────────────────────────────────────────
with open(MAP_PATH, "w") as f:
    json.dump(residue_map, f)
print(f"\nSaved chain_to_uniprot_auth_seq_id_map.json  ({len(residue_map)} chains)")


# ── Inline sklearn validation ─────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 60)
    print("VALIDATION: LogisticRegression on 50-protein subset")
    print("=" * 60)

    # Build auth_seq_id → embedding-row-index lookup from residue_map
    auth_seq_id_to_pos: dict = {}
    for key, seq_ids in residue_map.items():
        for pos, seq_id in enumerate(seq_ids):
            auth_seq_id_to_pos[f"{key}_{seq_id}"] = pos

    # Collect positive (pdb_id, chain, pos) triples from dataset_50
    positive_set: set = set()
    for pdb_id, entries in dataset_50.items():
        pocket: set = set()
        for entry in entries:
            pocket.update(entry["apo_pocket_selection"])
        for p in pocket:
            # apo_pocket_selection entries are formatted as "{chain}_{auth_seq_id}"
            chain, seq_id = p.split("_", 1)
            pos = auth_seq_id_to_pos.get(f"{pdb_id}_{chain}_{seq_id}")
            if pos is not None:
                positive_set.add((pdb_id, chain, pos))

    # Split 50 proteins: first 40 train, last 10 val
    all_keys = list(residue_map.keys())
    # group keys by pdb_id to split at protein level
    pdb_ids_list = pdb_ids_50  # ordered list of 50 pdb_ids
    train_pdb_ids = set(pdb_ids_list[:40])
    val_pdb_ids = set(pdb_ids_list[40:])

    def build_Xy(pdb_id_set):
        X_rows, y_rows = [], []
        for key, seq_ids in residue_map.items():
            pdb_id, chain_id = key.split("_", 1)
            if pdb_id not in pdb_id_set:
                continue
            emb_file = UNIPROT_EMB_DIR / f"{key}.npy"
            if not emb_file.exists():
                continue
            emb = np.load(emb_file)
            for pos in range(emb.shape[0]):
                X_rows.append(emb[pos])
                y_rows.append(int((pdb_id, chain_id, pos) in positive_set))
        return np.stack(X_rows), np.array(y_rows, dtype=np.int32)

    X_train, y_train = build_Xy(train_pdb_ids)
    X_val, y_val = build_Xy(val_pdb_ids)

    print(
        f"Train: {X_train.shape[0]:,} residues, {y_train.sum():,} positives ({100.0*y_train.mean():.2f}%)"
    )
    print(
        f"Val  : {X_val.shape[0]:,} residues, {y_val.sum():,} positives ({100.0*y_val.mean():.2f}%)"
    )

    if X_val.shape[0] == 0:
        print("ERROR: No val embeddings found.")
        sys.exit(1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("\nFitting LogisticRegression(class_weight='balanced') …")
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
    clf.fit(X_train_scaled, y_train)
    probs = clf.predict_proba(X_val_scaled)[:, 1]

    auprc = average_precision_score(y_val, probs)
    auroc = roc_auc_score(y_val, probs)

    baseline = float(y_val.sum()) / len(y_val)
    print(f"\n{'─' * 40}")
    print(f"AUPRC          : {auprc:.4f}  (random baseline ≈ {baseline:.4f})")
    print(f"AUROC          : {auroc:.4f}  (random baseline = 0.5000)")
    print(f"{'─' * 40}")
    if auprc > 0.15:
        print("RESULT: AUPRC > 0.15 — UniProt embeddings show meaningful signal.")
        print("        Proceed with full-scale embedding of all 1107 proteins.")
    else:
        print("RESULT: AUPRC ≤ 0.15 — investigate before scaling to all proteins.")
