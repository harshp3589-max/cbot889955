import time, requests
from typing import Optional, Dict

class FNGCache:
    def __init__(self):
        self.data = None
        self.ts = 0.0

_fng_cache = FNGCache()

def get_fng(cache_minutes: int = 15) -> Optional[Dict]:
    now = time.time()
    if _fng_cache.data and (now - _fng_cache.ts) < cache_minutes*60:
        return _fng_cache.data
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        if r.status_code != 200:
            return None
        j = r.json()
        d0 = j.get("data", [{}])[0]
        data = {
            "value": int(d0.get("value", "0")),
            "classification": d0.get("value_classification","Unknown")
        }
        _fng_cache.data = data
        _fng_cache.ts = now
        return data
    except Exception:
        return None
