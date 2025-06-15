import json

from bs4 import BeautifulSoup
from bs4.element import Tag
from tqdm import tqdm

with open("./outputs/metadata.json", "r") as f:
    metadata = json.load(f)


for i in tqdm(range(len(metadata))):
    data = metadata[i]
    data["html_file"] = data["href"].replace("/", "-")

    html_file = data["html_file"]

    with open(f"./outputs/detail_pages/{html_file}", "r") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    source = soup.find("source")
    if not isinstance(source, Tag):
        continue

    data["src"] = source["src"][2:]
    data["src_type"] = source["type"]
    p_tags = soup.find_all("p")

    keywords = ["Template ID", "Format", "Dimensions", "Filesize"]
    for tag in p_tags:
        text = tag.get_text()
        if not any(keyword in text for keyword in keywords):
            continue
        key, value = text.split(":")
        data[key.strip()] = value.strip()

with open("./outputs/metadata.json", "w") as f:
    json.dump(metadata, f)
