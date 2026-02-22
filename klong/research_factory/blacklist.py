from __future__ import annotations
import json
from urllib.parse import urlparse

class Blacklist:
    def __init__(self):
        self._urls: set[str] = set()

    def _normalize(self, url: str) -> str:
        url = url.strip().rstrip("/")
        if not url.startswith("http"):
            url = "https://" + url
        parsed = urlparse(url)
        return f"{parsed.netloc}{parsed.path}".lower().rstrip("/")

    def add(self, url: str):
        self._urls.add(self._normalize(url))

    def is_blocked(self, url: str) -> bool:
        normalized = self._normalize(url)
        return any(normalized.startswith(blocked) for blocked in self._urls)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(list(self._urls), f)

    @classmethod
    def load(cls, path: str) -> Blacklist:
        bl = cls()
        with open(path) as f:
            bl._urls = set(json.load(f))
        return bl
