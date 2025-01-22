import multiprocessing
import os
import pickle as pk
import random
import sys

import requests


def divide_urls_keys(n, objs) -> list[list[tuple[str, str]]]:
    """
    Return a list of n sublists.
    Each sublists is a list of 2 elems tuple contains (url, filename)
    """
    keys = list(objs.keys())
    random.shuffle(keys)
    chunk_size = len(keys) // n
    chunk_end = n * chunk_size
    remainder = len(keys) % n
    divided_keys = [keys[i : i + chunk_size] for i in range(0, chunk_end, chunk_size)]
    for i in range(remainder):
        divided_keys[i].append(keys[chunk_end + i])

    res = []
    for keys in divided_keys:
        url_k_pairs = []
        for k in keys:
            url_k_pairs.append((objs[k]["url"], k))
        res.append(url_k_pairs)

    return res


def pull_gif(url, filename) -> bool:
    try:
        response = requests.get(url)
        if response.status_code != 200 or not response.content:
            return False

        extension = url.split(".")[-1]
        with open(f"/d/gifs/{filename}.{extension}", "wb") as f:
            f.write(response.content)

        return True

    except Exception as e:
        print(e)
        sys.stdout.flush()
    finally:
        return False


def pull_gifs(data: list[tuple[str, str]]) -> list[str]:
    pulled = []
    with open("data/existed.pickle", "rb") as f:
        existed: set = pk.load(f)
    for i, (url, filename) in enumerate(data):
        if i % 100 == 0:
            print(os.getpid(), f"{i}/{len(data)}")
            sys.stdout.flush()

        if filename not in existed and pull_gif(url, filename):
            pulled.append(filename)

    return pulled


def get_existed():
    files = os.listdir("/home/lehoangchibach/Documents/Code/gif_analytics/gifs")
    files.extend(os.listdir("/d/gifs"))

    gen = (file.split(".")[0] for file in files)
    existed = set(gen)
    print("existed gifs:", len(existed))
    with open("data/existed.pickle", "wb") as f:
        pk.dump(existed, f)


if __name__ == "__main__":
    get_existed()

    with open("outputs/objs.pickle", "rb") as f:
        objs = pk.load(f)

    NUM_PROCESS = 15

    divided_url_k_pairs = divide_urls_keys(NUM_PROCESS, objs)

    with multiprocessing.Pool(processes=NUM_PROCESS) as pool:
        results = pool.map(pull_gifs, divided_url_k_pairs)
        merged_result = [item for sublist in results for item in sublist]
        print(merged_result)
