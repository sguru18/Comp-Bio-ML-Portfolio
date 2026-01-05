# Split data/train_val_list.txt into separate train and val lists

import numpy as np
from collections import defaultdict

SEED = 42
RATIO = 0.2

generator = np.random.default_rng(SEED)
ids = defaultdict(list)

# 00000008_000.png
# 00000008_001.png
# 00000008_002.png
f = open("../data/train_val_list.txt")

for line in f:
    filename = line.strip()
    iD = filename.split("_")[0]
    ids[iD].append(filename)

unique_ids = list(ids.keys())
generator.shuffle(unique_ids)

# Put RATIO ids in val and the rest in train
num_val = int(len(unique_ids) * RATIO)
val_ids = set(unique_ids[:num_val])
train_ids = set(unique_ids[num_val:])

assert val_ids.isdisjoint(train_ids)

# Write val list
o = open("../data/val_list.txt", "w")
for iD in val_ids:
    for filename in ids[iD]:
        o.write(f'{filename}\n')
o.close()

# Write train list
o = open("../data/train_list.txt", "w")
for iD in train_ids:
    for filename in ids[iD]:
        o.write(f'{filename}\n')
o.close()

f.close()

print(f"Total patients in train_val_list.txt: {len(unique_ids)}")
print(f"Num patients in val: {len(val_ids)} ({len(val_ids) / len(unique_ids)})")
print(f"Num patients in train: {len(train_ids)}")

# TODO: stratification by disease class might be good


