import json
import os

import requests
from tqdm import tqdm

with open("./metadata.json", "r") as f:
    metadata = json.load(f)
    print("metadata:", len(metadata))

existed = set(os.listdir("./detail_pages"))
print("existed:", len(existed))

for i in tqdm(range(len(metadata))):
    data = metadata[i]
    href = data["href"]
    name = href.replace("/", "-")

    if name in existed:
        continue

    response = requests.get(f"https://imgflip.com{href}")
    with open(f"./detail_pages/{name}", "w") as f:
        f.write(response.text)
