# making a controller to run main.py via telegram on phone and bot commands
# sudo systemctl restart smart-room.service
# journalctl -u smart-room.service -
#


import time
import os 
import signal
import subprocess
import requests
import sys
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_URL = f"https://api.telegram.org/bot{TOKEN}"

room = None

def send(text):
    requests.post(f"{API_URL}/sendMessage", data={"chat_id": CHAT_ID, "text": text}, timeout= 10)

def start_room():
    global room
    if room and room.poll() is None:
        send("Room is already running.")
        return
    
    log = open("room.log", "a")
    # solves the issue of the main.py outputs flooding the terminal and not making the bot run comands
    room = subprocess.Popen(["nice", "-n", "10", sys.executable, "main.py"], stdout=log, stderr=log)
    send("Room started.")


def stop_room():
    global room
    if not (room and room.poll() is None):
        send("Room is not running.")
        return

    send("Stopping room...")
    room.send_signal(signal.SIGINT)
    try:
        room.wait(timeout=8)
    except subprocess.TimeoutExpired:
        send("Room did not stop in time, killing...")
        room.kill()
        room.wait()
    room = None
    send("Room stopped.")


def status_room():
    send("Room is running." if room and room.poll() is None else "Room is not running.")

# deal with old commands that were lingering prior to start up
inital = requests.get(f"{API_URL}/getUpdates").json().get("result", [])
offset = inital[-1]["update_id"] + 1 if inital else None
try:
    while True:
        try:
            # adding timeout delay to deal with the bot not responding to commands
            response = requests.get(f"{API_URL}/getUpdates", 
                                params={"offset": offset, "timeout": 30}, timeout= 25).json()

            for update in response.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message", {})
        
                if str(message.get("chat", {}).get("id")) != str(CHAT_ID):
                    continue 
                text = message.get("text", "").strip().lower()
                if text == "/start":
                    start_room()
                elif text == "/stop":
                    stop_room()
                elif text == "/status":
                    status_room()
        except Exception as e:
            print("controller error:", e)
            time.sleep(5)
except KeyboardInterrupt:
    print("controller shutting down...")
    if room and room.poll() is None:
        room.send_signal(signal.SIGINT)
        room.wait()

            