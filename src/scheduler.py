import time
import threading
from typing import Callable, Optional
from utils import setup_logger

logger = setup_logger("scheduler")


class Scheduler:
    def __init__(self):
        self._jobs: list = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def every(self, interval_seconds: int, func: Callable, name: str = ""):
        self._jobs.append({
            "interval": interval_seconds,
            "func": func,
            "name": name or func.__name__,
            "last_run": 0.0,
        })
        return self

    def _run_loop(self):
        while self._running:
            now = time.time()
            for job in self._jobs:
                if now - job["last_run"] >= job["interval"]:
                    try:
                        logger.debug(f"Running job: {job['name']}")
                        job["func"]()
                        job["last_run"] = now
                    except Exception as e:
                        logger.error(f"Job {job['name']} failed: {e}")
            time.sleep(1)

    def start_background(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started in background")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def run_once_all(self):
        for job in self._jobs:
            try:
                logger.info(f"Running: {job['name']}")
                job["func"]()
            except Exception as e:
                logger.error(f"Job {job['name']} failed: {e}")
