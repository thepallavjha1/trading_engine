import json
from pathlib import Path
from utils import DATA_DIR, setup_logger

logger = setup_logger("live_state")
LIVE_STATE_FILE = DATA_DIR / "live_state.json"

_DEFAULT: dict = {"in_position": False, "balance": None}


def load_live_state() -> dict:
    if not LIVE_STATE_FILE.exists():
        return dict(_DEFAULT)
    try:
        return json.loads(LIVE_STATE_FILE.read_text())
    except Exception:
        return dict(_DEFAULT)


def save_live_state(state: dict) -> None:
    LIVE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LIVE_STATE_FILE.write_text(json.dumps(state, indent=2))
    logger.debug(f"State saved: in_position={state.get('in_position')} balance={state.get('balance')}")
