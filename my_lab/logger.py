import json
import os
import time
from pathlib import Path

class Logger:
    def __init__(self, run_name: str, log_dir: str = "./scratch/logs"):
        Path(log_dir).mkdir(parent=True, exist_ok=True)
        self.path = Path(log_dir) / f"{run_name}.json"
        self.start_time = time.time()
        self.path.write_text("") #overwriting when re-running

    def log(self, metrics: dict, step: int | None = None):
        entry = {"_time": round(time.time() - self.start_time, 2)}
        if step is not None:
            entry["step"] = step
        entry.update(metrics)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")


