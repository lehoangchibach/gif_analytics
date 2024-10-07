import json
import os

from bs4 import BeautifulSoup

htmls = os.listdir("./gif_pages/")
res = []


for html in htmls:
    with open(f"./gif_pages/{html}", "r") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    h3 = soup.find_all("h3")

    for template in h3:
        data = {}
        data["href"] = template.a["href"]
        data["title"] = template.a.text
        res.append(data)


with open("metadata.json", "w") as f:
    json.dump(res, f)
