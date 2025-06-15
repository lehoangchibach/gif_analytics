import json

from thefuzz import fuzz
from tqdm import tqdm

with open("./outputs/metadata_w_examples.json", "r") as f:
    metadata = json.load(f)

titles = []
res = []
for i in tqdm(range(len(metadata))):
    data = metadata[i]

    title = data["title"]

    for index, t in titles:
        simple_r = fuzz.ratio(title.lower(), t.lower())
        token_sort_r = fuzz.token_sort_ratio(title.lower(), t.lower())

        ratios = {
            "simple_r": simple_r,
            "token_sort_r": token_sort_r,
        }

        r = max(list(ratios.values()))
        res.append(((title, i), (t, index), r, ratios))

    titles.append((i, title))

with open("./outputs/comparing.json", "w") as f:
    json.dump(res, f)

print(len(res), res[0])
