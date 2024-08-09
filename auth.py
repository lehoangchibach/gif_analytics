import os

import requests
import requests.auth
from dotenv import load_dotenv

load_dotenv()
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")
USER_AGENT = f"PersonalScript/0.1 by {REDDIT_USERNAME}"


def get_access_token() -> str:
    client_auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
    post_data = {
        "grant_type": "password",
        "username": REDDIT_USERNAME,
        "password": REDDIT_PASSWORD,
    }
    headers = {"User-Agent": USER_AGENT}
    response = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=client_auth,
        data=post_data,
        headers=headers,
    )
    return response.json()["access_token"]
