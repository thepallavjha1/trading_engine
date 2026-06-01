import re
from pathlib import Path
from typing import Any, Optional, Tuple
from utils import (
    load_yaml, save_yaml, HISTORY_DIR, CONFIGS_DIR,
    append_jsonl, REFLECTIONS_FILE, now_iso, setup_logger
)

logger = setup_logger("version_manager")

STRATEGY_FILE = CONFIGS_DIR / "strategy.yaml"


def _get_next_version() -> Tuple[str, str]:
    existing = sorted(HISTORY_DIR.glob("v*.yaml"))
    if not existing:
        return "01", "02"
    last = existing[-1].stem
    match = re.match(r"v(\d+)", last)
    if match:
        num = int(match.group(1))
        return f"{num:02d}", f"{num + 1:02d}"
    return "01", "02"


def get_current_version() -> str:
    try:
        strat = load_yaml(STRATEGY_FILE)
        return strat.get("version", "01")
    except Exception:
        return "01"


def save_version_snapshot(strategy: dict) -> Path:
    version = strategy.get("version", "01")
    out_path = HISTORY_DIR / f"v{version.zfill(4)}.yaml"
    save_yaml(strategy, out_path)
    logger.info(f"Saved strategy snapshot: {out_path.name}")
    return out_path


def bump_version(strategy: dict) -> Tuple[dict, str]:
    current_version = strategy.get("version", "01")
    try:
        next_num = int(current_version) + 1
        next_version = f"{next_num:02d}"
    except ValueError:
        next_version = "02"
    new_strategy = dict(strategy)
    new_strategy["version"] = next_version
    return new_strategy, next_version


def apply_modification(
    changed_variable: str,
    old_value: Any,
    new_value: Any,
    reason: str,
    expected_effect: str,
) -> dict:
    strategy = load_yaml(STRATEGY_FILE)
    save_version_snapshot(strategy)

    strategy, new_version = bump_version(strategy)

    if changed_variable == "rsi_threshold":
        strategy["entry"]["threshold"] = new_value
    elif changed_variable in ("stop_loss_pct", "take_profit_pct", "position_size_pct"):
        strategy[changed_variable] = new_value
    else:
        logger.warning(f"Unknown variable to change: {changed_variable}")
        return strategy

    save_yaml(strategy, STRATEGY_FILE)

    reflection_record = {
        "timestamp": now_iso(),
        "version": new_version,
        "changed_variable": changed_variable,
        "old_value": old_value,
        "new_value": new_value,
        "reason": reason,
        "expected_effect": expected_effect,
    }
    append_jsonl(REFLECTIONS_FILE, reflection_record)

    logger.info(
        f"Applied modification: {changed_variable} {old_value} -> {new_value} "
        f"(v{strategy.get('version', '??')})"
    )
    return strategy


def load_all_versions() -> list:
    versions = []
    for path in sorted(HISTORY_DIR.glob("v*.yaml")):
        try:
            data = load_yaml(path)
            data["_file"] = path.name
            versions.append(data)
        except Exception:
            pass
    return versions
