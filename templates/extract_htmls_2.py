import json
import os

from bs4 import BeautifulSoup
from tqdm import tqdm

htmls = os.listdir("./outputs/gif_pages/")
res = []


for html in tqdm(htmls):
    with open(f"./outputs/gif_pages/{html}", "r") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    h3 = soup.find_all("h3")

    for template in h3:
        data = {}
        data["href"] = template.a["href"]
        data["title"] = template.a.text
        res.append(data)


with open("./outputs/metadata.json", "w") as f:
    json.dump(res, f)
