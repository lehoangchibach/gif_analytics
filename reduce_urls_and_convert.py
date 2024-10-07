import json

if __name__ == "__main__":
    with open("outputs/urls.txt", "r") as f:
        urls = f.readlines()

    with open("outputs/objs.txt", "r") as f:
        objs = f.readlines()

    urls = [url.strip() for url in urls]
    objs = [json.loads(obj) for obj in objs]

    urls_set = set()
    result = {}

    for i, url in enumerate(urls):
        if url in urls_set:
            continue

        urls_set.add(url)
        obj = objs[i]
        result[obj["id"]] = obj

    with open("outputs/gif_objs.txt", "w") as f:
        f.writelines(json.dumps(result))
