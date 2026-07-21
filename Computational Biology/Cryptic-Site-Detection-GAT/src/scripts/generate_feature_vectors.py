"""
generate the Data objects for GAT, one per protein
this consists of iterating each residue in the protein and calculating the desired features
we also compute the edge attribute matrix for residues that are less than 10 squared angstroms away from each other
curvature calculation is pretty long but cool, uses a lot of math concepts and kd tree!
^ me designed but AI-implemented
features are AA_one_hot, [b_factor], [sasa], biophysical, k1, k2, burial depth
"""

from pathlib import Path
import json
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley
import numpy as np
from numpy.linalg import eigh
from scipy.spatial import cKDTree
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data
import torch

ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT_DIR / "data/cryptobench/cryptobench-dataset/dataset.json"
CIF_FILES_DIR = (
    ROOT_DIR / "data/cryptobench/cryptobench-dataset/auxiliary-data/cif-files"
)
JSON_PATH = ROOT_DIR / "data/chain_to_auth_seq_id_map.json"
OUTPUT_DIR = ROOT_DIR / "data/protein-feature-vectors"

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

# curvature calculation ended up being a lot more complicated than i thought
# spent a long time going through the calculation process with different models
# very cool applications of the math but did hand it off to AI to implement
# else too long to implement and error-prone in my writing of it, big time input basically
# kD trees are cool I remember these from 225
SASA_EXPOSED_THRESHOLD = 1.0  # Å²
CURVATURE_RADIUS = 9.0  # Å

RADIUS_VALUE_FOR_EDGES = 10.0


def residue_surface_query_point(residue):
    """SASA-weighted centroid of exposed atoms, else Cα (or atom centroid).
    this biases the lookup point toward the more exposed atoms of this residue"""
    exposed = [
        (atom.get_coord(), float(atom.sasa))
        for atom in residue.get_atoms()
        if getattr(atom, "sasa", 0.0) is not None
        and float(atom.sasa) > SASA_EXPOSED_THRESHOLD
    ]
    if exposed:
        coords = np.asarray([c for c, _ in exposed], dtype=np.float64)
        weights = np.asarray([w for _, w in exposed], dtype=np.float64)
        return np.average(coords, axis=0, weights=weights)

    if "CA" in residue:
        return np.asarray(residue["CA"].get_coord(), dtype=np.float64)

    atoms = list(residue.get_atoms())
    if atoms:
        return np.mean([a.get_coord() for a in atoms], axis=0).astype(np.float64)
    return np.zeros(3, dtype=np.float64)


def build_exposed_atom_tree(model):
    """
    Preprocessing (once per protein):
    - all-atom coords + per-atom SASA (caller runs ShrakeRupley level='A')
    - keep exposed atoms with SASA > threshold
    - build cKDTree over those coords
    """
    exposed_coords = []
    exposed_residue_keys = []
    all_coords = []

    for chain in model:
        for residue in chain.get_residues():
            res_key = (chain.id, residue.get_id())
            for atom in residue.get_atoms():
                coord = np.asarray(atom.get_coord(), dtype=np.float64)
                all_coords.append(coord)
                sasa = getattr(atom, "sasa", None)
                if sasa is not None and float(sasa) > SASA_EXPOSED_THRESHOLD:
                    exposed_coords.append(coord)
                    exposed_residue_keys.append(res_key)

    if not exposed_coords:
        exposed_coords = np.zeros((0, 3), dtype=np.float64)
        tree = None
    else:
        exposed_coords = np.asarray(exposed_coords, dtype=np.float64)
        tree = cKDTree(exposed_coords)

    protein_centroid = (
        np.mean(all_coords, axis=0) if all_coords else np.zeros(3, dtype=np.float64)
    )
    return tree, exposed_coords, exposed_residue_keys, protein_centroid


def compute_local_curvature(
    query_point,
    center_residue_key,
    tree,
    exposed_coords,
    exposed_residue_keys,
    protein_centroid,
    radius=CURVATURE_RADIUS,
):
    """
    Local surface curvature + burial depth at a residue.

    Preprocessing (once per protein):
    1. Get all-atom coordinates (not just Cα).
    2. Run SASA → per-atom SASA values.
    3. Filter to exposed_atoms where SASA > threshold (~1 Å²).
    4. Build tree = cKDTree(exposed_atoms.coords).
    5. For each residue, precompute a surface query point: SASA-weighted centroid of
       that residue's own exposed atoms if it has any; otherwise fall back to its Cα
       (buried — this point is just for the depth query, since curvature will bail out).

    Per-residue (this function):
    1. query_point = that residue's surface query point.
    2. depth = distance to nearest exposed atom (always computed / meaningful).
    3. Radius ball (~8–10 Å) around query_point → candidate neighbor exposed atoms.
    4. Exclude atoms belonging to this residue (not neighboring surface).
    5. If < 5 neighbors: return (0, 0, depth) — sparse patch or buried.
    6. Center neighbors at query_point.
    7. PCA → smallest-eigenvalue eigenvector = candidate normal.
    8. Orient normal outward: flip if it points toward the protein centroid.
    9. Express neighbors in the PCA tangent basis (not lab-frame x,y after
       subtracting the normal): z = n·r, x = e1·r, y = e2·r, where e1,e2 are the
       other two eigenvectors. This is the actual projection onto the tangent plane
       with in-plane coordinates.
    10. Least-squares fit z = dx + ey + ax² + bxy + cy².
    11. H = [[2a, b], [b, 2c]]; eigenvalues of H = κ1, κ2.
    12. Return (κ1, κ2, depth).
    """
    query_point = np.asarray(query_point, dtype=np.float64)

    if tree is None or len(exposed_coords) == 0:
        return 0.0, 0.0, 0.0

    depth = float(tree.query(query_point, k=1)[0])

    candidate_idx = tree.query_ball_point(query_point, r=radius)
    neighbor_idx = [
        i for i in candidate_idx if exposed_residue_keys[i] != center_residue_key
    ]

    if len(neighbor_idx) < 5:
        return 0.0, 0.0, depth

    neighbors = exposed_coords[neighbor_idx] - query_point

    # PCA: normal = direction of least variance
    _, eigvecs = eigh(neighbors.T @ neighbors)
    normal = eigvecs[:, 0]
    e1 = eigvecs[:, 1]
    e2 = eigvecs[:, 2]

    # Orient normal away from protein centroid
    if np.dot(normal, protein_centroid - query_point) > 0:
        normal = -normal

    # Coordinates in the PCA tangent frame
    x = neighbors @ e1
    y = neighbors @ e2
    z = neighbors @ normal

    # z = d x + e y + a x² + b x y + c y²
    try:
        A = np.column_stack([x, y, x**2, x * y, y**2])
        params, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        _, _, a, b, c = params
        H = np.array([[2 * a, b], [b, 2 * c]], dtype=np.float64)
        curvatures = np.linalg.eigvalsh(H)
        return float(curvatures[0]), float(curvatures[1]), depth
    except np.linalg.LinAlgError:
        return 0.0, 0.0, depth


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
pdb_to_positives = {}
for pdb_id, entries in dataset.items():
    positives = set()
    chains = set()
    for entry in entries:
        chains.update(entry["apo_chain"].split("-"))
        positives.update(entry["apo_pocket_selection"])

    chains = sorted(list(chains))
    pdb_to_chains[pdb_id] = chains
    pdb_to_positives[pdb_id] = positives

for pdb_id, chains in pdb_to_chains.items():
    structure = parser.get_structure(pdb_id, CIF_FILES_DIR / f"{pdb_id}.cif")
    model = structure[0]
    feature_matrix = []
    coords = []
    query_points = []
    residue_keys = []

    for chain in model:
        residues_to_remove = [r for r in chain if r.get_id()[0] != " "]
        for residue in residues_to_remove:
            chain.detach_child(residue.get_id())

    # per-atom SASA (Å²); residue SASA = sum of its atom SASAs
    shrake_rupley.compute(structure, level="A")
    tree, exposed_coords, exposed_residue_keys, protein_centroid = (
        build_exposed_atom_tree(model)
    )
    y = []
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
            res_key = (chain_id, residue.get_id())

            y.append(1.0 if f"{chain_id}_{seq_id}" in pdb_to_positives[pdb_id] else 0.0)

            # calculate what we need
            if "CA" not in residue:
                # use centroid instead, little bit weirder but 0 vector distorts curvature value
                # for all neighboring residues
                xyz = (
                    np.mean([a.get_coord() for a in residue.get_atoms()], axis=0)
                    if list(residue.get_atoms())
                    else [
                        0,
                        0,
                        0,
                    ]  # this should not happen at all. 0 atoms in a residue??
                )
                b_factor = 0
                num_missing_CA += 1
            else:
                xyz = residue["CA"].get_vector().get_array()  # 3 dim
                b_factor = residue[
                    "CA"
                ].get_bfactor()  # high b factor means very flexible
            total += 1
            coords.append(xyz)
            query_points.append(residue_surface_query_point(residue))
            residue_keys.append(res_key)
            sasa = float(
                sum(getattr(atom, "sasa", 0.0) or 0.0 for atom in residue.get_atoms())
            )
            AA_one_hot = [0] * 20
            idx = LOOKUP.get(
                residue.get_resname(), None
            )  # shouldn't have any Nones but just in case
            if idx is not None:
                AA_one_hot[idx] = 1
            biophysical = AA_PROPERTIES.get(residue.get_resname(), [0, 0, 0])
            row = np.concatenate(
                [AA_one_hot, [b_factor], [sasa], biophysical]  # 20  # 1  # 1  # 3
            )  # 3 more from curvature added below
            # TODO: look into adding secondary structure here
            feature_matrix.append(row)

    coords = np.asarray(coords, dtype=np.float32)
    for idx in range(len(feature_matrix)):
        k1, k2, depth = compute_local_curvature(
            query_points[idx],
            residue_keys[idx],
            tree,
            exposed_coords,
            exposed_residue_keys,
            protein_centroid,
        )
        feature_matrix[idx] = np.concatenate([feature_matrix[idx], [k1, k2, depth]])

    # compute pairwise Ca distances
    pos = torch.tensor(coords, dtype=torch.float32)
    edge_index = radius_graph(pos, r=RADIUS_VALUE_FOR_EDGES, loop=False)  # [2, E]
    src, dst = edge_index
    edge_attr = (pos[src] - pos[dst]).norm(dim=-1, keepdim=True)  # [E, 1]

    x = np.asarray(feature_matrix, dtype=np.float32)
    y = torch.tensor(y, dtype=torch.float32)
    # package into Data objects and store as {pdb_id}.pt
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)
    torch.save(data, OUTPUT_DIR / f"{pdb_id}.pt")

print(
    f"{num_missing_CA} residues out of {total} are missing Ca atoms, this is {num_missing_CA/total*100:.3f}%"
)
