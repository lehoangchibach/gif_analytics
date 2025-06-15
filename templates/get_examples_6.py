import json
import multiprocessing
import random

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def divide_metadata(n, metadatas) -> list[list[dict]]:
    random.shuffle(metadatas)
    chunk_size = len(metadatas) // n
    chunk_end = n * chunk_size
    remainder = len(metadatas) % n
    res = [metadatas[i : i + chunk_size] for i in range(0, chunk_end, chunk_size)]
    for i in range(remainder):
        res[i].append(metadatas[chunk_end + i])

    return res


def get_example(metadata: dict) -> None:
    template_id = metadata["Template ID"]
    sorts = ["", "latest", "top-7d", "top-30d", "top-365d"]

    examples = []
    added = set()
    for sort in sorts:
        response = requests.get(f"https://imgflip.com/meme/{template_id}?sort={sort}")
        soup = BeautifulSoup(response.text, "html.parser")

        titles = soup.find_all("h2", class_="base-unit-title")
        links = soup.find_all("a", class_="base-img-link")

        for title, link in zip(titles, links):
            example_id = link.video["data-src"].rsplit("/", 1)[1].split(".")[0]
            if example_id in added:
                continue
            else:
                added.add(example_id)

            exp_data = {}
            exp_data["title"] = title.a.text
            exp_data["url"] = "https:" + link.video["data-src"]
            exp_data["id"] = example_id
            examples.append(exp_data)

    metadata["examples"] = examples


def get_examples(data: list[dict]) -> list[dict]:
    for metadata in tqdm(data):
        get_example(metadata)

    return data


if __name__ == "__main__":
    with open("./outputs/metadata.json", "r") as f:
        data = json.load(f)

    NUM_PROCESS = 15

    splitted_metadata = divide_metadata(NUM_PROCESS, data)

    with multiprocessing.Pool(processes=NUM_PROCESS) as pool:
        results = pool.map(get_examples, splitted_metadata)
        merged_result = [item for sublist in results for item in sublist]
        print(merged_result)

    with open("./outputs/metadata_w_examples.json", "w") as f:
        json.dump(merged_result, f)
