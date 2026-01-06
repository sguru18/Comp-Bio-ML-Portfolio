import os
import json

# go through folders images_001 through images_012 and store filename : path mapping
data = {}

for i in range(1, 13):
    num = i if i > 9 else "0" + str(i)
    path = f"../../data/images/images_0{num}/images"
    for f in os.listdir(path):
        data[f] = f"data/images/images_0{num}/images/" + f

with open('../../data/lookup.json', "w", encoding='utf-8') as output:
    json.dump(data, output, ensure_ascii=False, indent=4)
