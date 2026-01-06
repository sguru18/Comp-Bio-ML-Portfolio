# Split data/train_val_list.txt into separate train and val lists

import numpy as np
import pandas as pd
from collections import defaultdict

SEED = 42
RATIO = 0.2

generator = np.random.default_rng(SEED)
ids = defaultdict(list)
df = pd.read_csv("../../data/Data_Entry_2017.csv")

# 00000008_000.png
# 00000008_001.png
# 00000008_002.png
f = open("../../data/train_val_list.txt")

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

# Write val list and csv
o = open("../../data/val_list.txt", "w")
val_filenames = set()
for iD in val_ids:
    for filename in ids[iD]:
        o.write(f'{filename}\n')
        val_filenames.add(filename)
o.close()

bools = df["Image Index"].isin(val_filenames)
val_df = df[bools]
val_df.to_csv("../../data/val.csv", index = False)

# Write train list and csv
o = open("../../data/train_list.txt", "w")
train_filenames = set()
for iD in train_ids:
    for filename in ids[iD]:
        o.write(f'{filename}\n')
        train_filenames.add(filename)
o.close()

bools = df["Image Index"].isin(train_filenames)
train_df = df[bools]
train_df.to_csv("../../data/train.csv", index = False)

f.close()

# print(f"Total patients in train_val_list.txt: {len(unique_ids)}")
# print(f"Num patients in val: {len(val_ids)} ({len(val_ids) / len(unique_ids)})")
# print(f"Num patients in train: {len(train_ids)}")

# Write test csv
o = open("../../data/test_list.txt")
test_filenames = set()
for line in o:
    test_filenames.add(line.strip())
bools = df["Image Index"].isin(test_filenames)
test_df = df[bools]
test_df.to_csv("../../data/test.csv", index = False)
o.close()

# TODO: stratification by disease class might be good


