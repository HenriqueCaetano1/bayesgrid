import pooch
import os

# --- 1. Define the base URL to your data ---
# This is the URL to your GitHub release (or other host)
BASE_URL = "https://github.com/HenriqueCaetano1/bayesgrid_data/releases/download/start/"

# --- 2. Define the local cache path ---
# This will be a user-specific cache folder, e.g.:
# C:\Users\YourUser\AppData\Local\bayesgrid\cache
CACHE_PATH = pooch.os_cache("bayesgrid")

# --- 3. Define the registry of all your files ---
TRACE_FILES = {
    "power": {
        "filename": "trace_power_and_phase.nc",
        "known_hash": "sha256:aca05e41296b11506d20df62efe6b073b8aac40c99f84bca35d1fa507db1ff13",
    },
    "frequency": {
        "filename": "trace_fic.nc",
        "known_hash": "sha256:e6be78cd3d1c587254d56e9b6054673d9a65b82dcc703b2538639b53e8d5b4e0",
    },
    "duration": {
        "filename": "trace_dic.nc",
        "known_hash": "sha256:f533cbc4e45f59db0848fa87b894feab1c8f5d78ead9c8a302241feb1103650d",
    },
    "impedance_r": {
        "filename": "trace_r.nc",
        "known_hash": "sha256:48a26553f88a763dfb3250d86004fd6ebbe153fddf838c1f6bcc4fce8003796c",
    },
    "impedance_x": {
        "filename": "trace_x.nc",
        "known_hash": "sha256:4f36194d361e25ac469b646df4d300fd8a7a5c9af91f7beb21b610243f66f4e1",
    }
}

def fetch_trace(trace_name):
    """
    Fetches a pre-trained trace file using pooch.retrieve.
    
    If the file is not in the local cache, it will be downloaded from
    the repository and saved for future use.
    """
    if trace_name not in TRACE_FILES:
        raise ValueError(f"Unknown trace file: {trace_name}")
    
    file_info = TRACE_FILES[trace_name]
    
    # This is the simplified one-shot function
    # It handles downloading, caching, and hash-checking all in one.
    file_path = pooch.retrieve(
        url=f"{BASE_URL}{file_info['filename']}",
        known_hash=file_info["known_hash"],
        fname=file_info["filename"], # The name to save it as
        path=CACHE_PATH,             # The local directory to cache it
        progressbar=True,
    )
    
    return file_path

