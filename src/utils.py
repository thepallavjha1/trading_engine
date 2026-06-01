import os
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

BASE_DIR = Path(__file__).parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data"
HISTORY_DIR = BASE_DIR / "history" / "strategy_versions"
REPORTS_DIR = BASE_DIR / "reports"
MARKET_CACHE_DIR = DATA_DIR / "market_cache"
TRADES_FILE = DATA_DIR / "trades.jsonl"
REFLECTIONS_FILE = DATA_DIR / "reflections.jsonl"

for d in [DATA_DIR, HISTORY_DIR, REPORTS_DIR, MARKET_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str) -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_goal() -> dict:
    return load_yaml(CONFIGS_DIR / "goal.yaml")


def load_strategy() -> dict:
    return load_yaml(CONFIGS_DIR / "strategy.yaml")


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator
