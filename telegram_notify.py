"""
Telegram notification sender for the Smart Room System.

Sends a text message (and optionally a photo) to your phone
via a Telegram bot. Requires a .env file with:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
"""

import os
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

"""
added this portion to deal with the inconsistency of the telegram prompt. force it to attempt
three times. the factor is the delay and the forcelist are the http status codes that arise
"""
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

def send_message(text):
    """Sends a plain text message to your Telegram chat."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    response = session.post(url, data=payload, timeout=10)
    return response.json()


def send_photo(image_path, caption=""):
    """Sends a photo with an optional caption to your Telegram chat."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(image_path, "rb") as photo:
        files = {"photo": photo}
        payload = {"chat_id": CHAT_ID, "caption": caption}
        response = session.post(url, data=payload, files=files, timeout=15)
    return response.json()


if __name__ == "__main__":
    # Quick test
    result = send_message("Smart Room System: test notification working!")
    print(result)
