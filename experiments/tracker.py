import json
import os
import logging
import time

logger = logging.getLogger('experiment')

class ExperimentTracker:
    def __init__(self, name, args, base_dir="../results/", commit_changes=False, rank=0, seed=1):
        self.name = name
        self.args = args
        self.base_dir = base_dir
        self.rank = rank
        self.seed = seed
        self.path = self._setup_path()
        self.results = {"all_args": args}

    def _setup_path(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{self.name}_rank{self.rank}_seed{self.seed}_{timestamp}"
        full_path = os.path.join(self.base_dir, run_name)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def add_result(self, key, value):
        self.results[key] = value

    def store_json(self):
        file_path = os.path.join(self.path, "results.json")
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        logger.info(f"Results stored at {file_path}")