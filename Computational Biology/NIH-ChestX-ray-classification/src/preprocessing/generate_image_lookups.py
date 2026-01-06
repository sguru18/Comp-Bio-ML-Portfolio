import os
from pathlib import Path
import json

# go through folders images_001 through images_012 and store filename : path mapping
data = {}
BASE_DIR = Path(__file__).resolve().parent.parent.parent # root dir
DATA_DIR = BASE_DIR / "data"

for i in range(1, 13):
    num = i if i > 9 else "0" + str(i)
    path = DATA_DIR / f"images/images_0{num}/images"
    for f in os.listdir(path):
        data[f] = f"data/images/images_0{num}/images/" + f

with open(DATA_DIR / 'lookup.json', "w", encoding='utf-8') as output:
    json.dump(data, output, ensure_ascii=False, indent=4)
