import json


def filter_url(obj: dict) -> bool:
    url = obj["url"]
    if not url:
        return False

    if url.split(".")[-1] in {"jpeg", "jpg", "png"}:
        return False
    if url.split(".")[-1] in {"gif", "gifv"}:
        return True
    return False


def filter_objs(objs: list[str]) -> list:
    # return [obj["url"] for obj in filter(filter_url, objs)]
    return [json.dumps(obj) for obj in filter(filter_url, objs)]


def main():
    line_cnt = 0
    objs = []
    with open("./data/memes_submissions", "r") as f:
        while True:
            line = f.readline()
            if not line:
                filtered_objs = filter_objs(objs)
                with open("outputs/objs.txt", "a") as obj_f:
                    obj_f.writelines("\n".join(filtered_objs))
                break

            objs.append(json.loads(line))
            line_cnt += 1
            if line_cnt % 1000000 == 0:
                print(line_cnt)
                filtered_objs = filter_objs(objs)
                with open("outputs/objs.txt", "a") as obj_f:
                    obj_f.writelines("\n".join(filtered_objs) + "\n")
                objs = []


if __name__ == "__main__":
    main()
