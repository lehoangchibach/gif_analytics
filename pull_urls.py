from auth import get_access_token, USER_AGENT
import requests

def filter_url(url) -> bool:
    if url[-1] == "/":
        return False
    if url.split(".")[-1] in {"jpeg", "jpg", "png"}:
        return False
    if url.split(".")[-1] in {"gif", "gifv"} or url.split("/")[-1]:
        return True
    return False

if __name__ == "__main__":
    access_token = get_access_token()
    
    headers = {"Authorization": f"bearer {access_token}", "User-Agent": USER_AGENT}
    params = {"t": "year", "limit": 100}
    urls = []
    afters = []
    
    try:
        for i in range(4500):
            print(f"{i}", end=" - ")
            
            response = requests.get(
                "https://oauth.reddit.com/r/memes/top.json", params=params, headers=headers
            )
            response.raise_for_status()
            data = response.json()

            children = data["data"]["children"]
            for i in range(len(children)):
                urls.append(children[i]["data"]["url"])
            
            params["after"] = data["data"]["after"]
            if params["after"] is None:
                afters.append("EOF")
                break
            afters.append(params["after"] + "\n")
            

    except Exception as e:
        print(str(data).encode("utf-8"), e)
        
    finally:
        with open("outputs/afters.txt", "w") as f:
            f.writelines(afters)
        filter_urls = [url + "\n" for url in urls if filter_url(url)]
        with open("outputs/urls.txt", "w") as f:
            f.writelines(filter_urls)