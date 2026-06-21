import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-modify-playback-state user-read-playback-state"
))


def get_device_id(device_type=None):
    devices = sp.devices()
    if not devices['devices']:
        return None

    if device_type:
        for d in devices['devices']:
            if d['type'].lower() == device_type.lower():
                return d['id']

    return devices['devices'][0]['id']


def play_for_person(person_name, uri, device_type=None):
    
    #plays either a playlist/album (context_uri) or a single track (uris list)
    #switch spotify uri format tells us which: spotify:track:... vs spotify:playlist:...
    
    device_id = get_device_id(device_type)
    if not device_id:
        print("no matching spotify device found")
        return

    if "track" in uri:
        sp.start_playback(device_id=device_id, uris=[uri])
    else:
        sp.start_playback(device_id=device_id, context_uri=uri)

    print(f"playing for {person_name}")

def next_track(device_type=None):
    device_id = get_device_id(device_type)
    if not device_id:
        print("no matching spotify device found")
        return
    
    sp.next_track(device_id=device_id)
    print("skipped to next track")

def toggle_play_pause(device_type=None):
    device_id = get_device_id(device_type)
    if not device_id:
        print("no matching spotify device found")
        return
    
    playback = sp.current_playback()
    if playback and playback['is_playing']:
        sp.pause_playback(device_id=device_id)
        print("playback paused")
    else:
        sp.start_playback(device_id=device_id)
        print("playback started")


if __name__ == "__main__":
    test_playlist = os.getenv("DEFAULT_PLAYLIST_URI")
    test_device = os.getenv("DEFAULT_DEVICE_TYPE")

    if test_playlist:
        play_for_person("test", test_playlist, device_type=test_device)
    else:
        print("set DEFAULT_PLAYLIST_URI in .env to test playback")
