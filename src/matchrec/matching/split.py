"""Candidate-level data assignment.

- The unit is the candidate: all of a candidate's applications live on one side of every split.
- The sealed test set is a hashed candidate hold-out (`test_share`). Because the hash depends only
  on the candidate id and a salt, a candidate keeps its role across snapshots without any state.
- Inside dev, folds are a second hash of the candidate id.
- `Registry` records the snapshot where each id was first seen, to report overlap between exports.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROLES = ("dev", "test")


def _bucket(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16) / 16 ** 15


def role_of(candidate_id, test_share: float = 0.2, salt: str = "matchrec-split-v1") -> str:
    return "test" if _bucket(f"{salt}:role:{candidate_id}") < test_share else "dev"


def fold_of(candidate_id, k: int = 5, salt: str = "matchrec-split-v1") -> int:
    return int(hashlib.sha256(f"{salt}:fold:{candidate_id}".encode()).hexdigest(), 16) % k


class Registry:
    KINDS = ("candidates", "jobs", "orgs", "applications")

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = (json.loads(self.path.read_text()) if self.path.exists()
                     else {"snapshots": {}, **{k: {} for k in self.KINDS}})

    def register(self, snapshot_id: str, **ids) -> dict:
        self.data["snapshots"].setdefault(snapshot_id, {"registered_at": datetime.now(timezone.utc).isoformat()})
        report = {}
        for kind in self.KINDS:
            new = seen = 0
            for i in map(str, ids.get(kind, ())):
                if i in self.data[kind]:
                    seen += self.data[kind][i] != snapshot_id
                else:
                    self.data[kind][i] = snapshot_id
                    new += 1
            report[kind] = {"new": new, "seen_in_earlier_snapshot": seen}
        return report

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True))
