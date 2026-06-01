import json
import yaml
import requests
from typing import List, Dict, Any

_SESSION = requests.Session()
_SESSION.headers.update({"Accept": "application/vnd.github+json"})


def _raw_url(repo: str, path: str, branch: str = "main") -> str:
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"


def _api_url(repo: str, path: str) -> str:
    return f"https://api.github.com/repos/{repo}/contents/{path}"


def read_jsonl(repo: str, path: str, branch: str = "main") -> List[dict]:
    url = _raw_url(repo, path, branch)
    try:
        resp = _SESSION.get(url, timeout=15)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        records = []
        for line in resp.text.splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records
    except Exception:
        return []


def read_yaml(repo: str, path: str, branch: str = "main") -> dict:
    url = _raw_url(repo, path, branch)
    try:
        resp = _SESSION.get(url, timeout=15)
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return yaml.safe_load(resp.text) or {}
    except Exception:
        return {}


def list_version_yamls(repo: str, branch: str = "main") -> List[dict]:
    api_url = _api_url(repo, "history/strategy_versions")
    try:
        resp = _SESSION.get(api_url, timeout=15)
        if resp.status_code != 200:
            return []
        files = resp.json()
        versions = []
        for f in sorted(files, key=lambda x: x.get("name", "")):
            name = f.get("name", "")
            if name.endswith(".yaml") and name.startswith("v"):
                dl_url = f.get("download_url")
                if dl_url:
                    try:
                        yr = _SESSION.get(dl_url, timeout=15)
                        data = yaml.safe_load(yr.text) or {}
                        data["_file"] = name
                        versions.append(data)
                    except Exception:
                        pass
        return versions
    except Exception:
        return []
