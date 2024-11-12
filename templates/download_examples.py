import hashlib
import json
import multiprocessing
import random

import requests
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
    examples = metadata["examples"]
    template_id = metadata["Template ID"]

    for example in examples:
        byte_string = example["title"].encode()
        md5_hash = hashlib.md5(byte_string).hexdigest()
        name = f"{md5_hash}_{template_id}"
        ext = example["url"].split("/")[-1].split(".")[-1]
        filename = f"{name}.{ext}"

        response = requests.get(example["url"])
        with open(f"./template_examples/{filename}", "wb") as f:
            f.write(response.content)


def get_examples(data: list[dict]) -> None:
    for i in tqdm(range(len(data))):
        get_example(data[i])


if __name__ == "__main__":
    with open("./metadata_w_examples.json", "r") as f:
        data = json.load(f)

    NUM_PROCESS = 15

    splitted_metadata = divide_metadata(NUM_PROCESS, data)

    with multiprocessing.Pool(processes=NUM_PROCESS) as pool:
        results = pool.map(get_examples, splitted_metadata)
        print(results)
