import json
import multiprocessing
import os
import pickle as pk
import random
import sys

import requests


def divide_url_filename_pairs(n, metadata) -> list[list[tuple[str, str]]]:
    """
    Return a list of n sublists.
    Each sublists is a list of 2 elems tuple contains (url, filename)
    """
    url_filename_pairs = [
        (data["src"], data["html_file"] + "." + data["Format"]) for data in metadata
    ]
    random.shuffle(url_filename_pairs)
    chunk_size = len(url_filename_pairs) // n
    chunk_end = n * chunk_size
    remainder = len(url_filename_pairs) % n
    divided_url_filename_pairs = [
        url_filename_pairs[i : i + chunk_size] for i in range(0, chunk_end, chunk_size)
    ]
    for i in range(remainder):
        divided_url_filename_pairs[i].append(url_filename_pairs[chunk_end + i])

    return divided_url_filename_pairs


def pull_gif(url, filename) -> bool:
    try:
        response = requests.get(f"https://{url}")
        if response.status_code != 200 or not response.content:
            return False

        with open(f"./outputs/template_data/{filename}", "wb") as f:
            f.write(response.content)

        return True

    except Exception as e:
        print(e)
        sys.stdout.flush()
    finally:
        return False


def pull_gifs(data: list[tuple[str, str]]) -> list[str]:
    pulled = []
    with open("./outputs/existed.pickle", "rb") as f:
        existed: set = pk.load(f)
    for i, (url, filename) in enumerate(data):
        if i % 100 == 0:
            print(os.getpid(), f"{i}/{len(data)}")
            sys.stdout.flush()

        if filename not in existed and pull_gif(url, filename):
            pulled.append(filename)

    return pulled


def get_existed():
    files = os.listdir(
        "/home/lehoangchibach/Documents/Code/gif_analytics/templates/outputs/template_data"
    )

    gen = (file.split(".")[0] for file in files)
    existed = set(gen)
    print("existed gifs:", len(existed))
    with open("./outputs/existed.pickle", "wb") as f:
        pk.dump(existed, f)


if __name__ == "__main__":
    get_existed()
    with open("./outputs/metadata.json", "r") as f:
        data = json.load(f)

    NUM_PROCESS = 15

    divided_url_filename_pairs = divide_url_filename_pairs(NUM_PROCESS, data)

    with multiprocessing.Pool(processes=NUM_PROCESS) as pool:
        results = pool.map(pull_gifs, divided_url_filename_pairs)
        merged_result = [item for sublist in results for item in sublist]
        print(merged_result)
