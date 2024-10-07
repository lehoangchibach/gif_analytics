import requests

if __name__ == "__main__":
    page_cnt = 2
    base_url = "https://imgflip.com/gif-templates?page="

    while True:
        response = requests.get(f"{base_url}{page_cnt}")

        with open(f"./gif_pages/{page_cnt}", "w") as f:
            f.write(response.text)

        if page_cnt == 144:
            break
        page_cnt += 1
