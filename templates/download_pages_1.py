import requests
from tqdm import tqdm

if __name__ == "__main__":
    page_cnt = 1
    base_url = "https://imgflip.com/gif-templates?page="

    for i in tqdm(range(161)):
        response = requests.get(f"{base_url}{page_cnt}")

        with open(f"./gif_pages/{page_cnt}", "w") as f:
            f.write(response.text)

        if page_cnt == 161:
            break
        page_cnt += 1
