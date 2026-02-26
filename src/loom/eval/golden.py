"""Golden store — JSONL-backed ground truth for eval.

Each golden record holds one reviewed item with its expected output,
actual output, and human judgment. Files live in eval/golden/{stage}.jsonl.

Records are append-only (new judgments appended). The latest judgment for
a given ID wins when computing stats.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Stage name constants
STAGE_INGEST = "0_ingest"
STAGE_RESOLVE = "1_resolve"
STAGE_LINK = "2_link"
STAGE_CLAIMS = "3_claims"
STAGE_TRIAGE = "4_triage"
STAGE_PROPAGATE = "5_propagate"
STAGE_EXTRACT = "6_extract"
STAGE_SYNTHESIZE = "7_synthesize"

ALL_STAGES = [
    STAGE_INGEST, STAGE_RESOLVE, STAGE_LINK, STAGE_CLAIMS,
    STAGE_TRIAGE, STAGE_PROPAGATE, STAGE_EXTRACT, STAGE_SYNTHESIZE,
]


@dataclass
class GoldenRecord:
    """One reviewed item in the golden store."""
    id: str
    stage: str
    item_id: str
    input_data: dict[str, Any]
    expected_output: dict[str, Any] | None = None
    actual_output: dict[str, Any] | None = None
    judgment: str = "pending"  # approve | correct | reject | flag | pending
    notes: str = ""
    reviewer: str = ""
    timestamp: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenRecord:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class GoldenStore:
    """CRUD for JSONL-backed golden store.

    Files: {base_dir}/{stage}.jsonl
    Each line is a JSON object representing one GoldenRecord.
    """

    def __init__(self, base_dir: str | Path | None = None):
        if base_dir is None:
            # Default: eval/golden/ relative to project root
            base_dir = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "golden"
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def _path_for_stage(self, stage: str) -> Path:
        return self._base_dir / f"{stage}.jsonl"

    def add(self, record: GoldenRecord) -> str:
        """Append a record. Returns the record ID."""
        if not record.id:
            record.id = str(uuid.uuid4())[:8]
        path = self._path_for_stage(record.stage)
        with open(path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        return record.id

    def get(self, stage: str, item_id: str) -> GoldenRecord | None:
        """Get the latest record for an item (last write wins)."""
        path = self._path_for_stage(stage)
        if not path.exists():
            return None
        result = None
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("item_id") == item_id:
                result = GoldenRecord.from_dict(data)
        return result

    def list_records(self, stage: str, judgment: str | None = None) -> list[GoldenRecord]:
        """List all records for a stage, optionally filtered by judgment.

        Returns deduplicated by item_id (last write wins).
        """
        path = self._path_for_stage(stage)
        if not path.exists():
            return []
        by_item: dict[str, GoldenRecord] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            rec = GoldenRecord.from_dict(data)
            by_item[rec.item_id] = rec

        records = list(by_item.values())
        if judgment:
            records = [r for r in records if r.judgment == judgment]
        return records

    def list_all(self) -> list[GoldenRecord]:
        """List all records across all stages."""
        all_records: list[GoldenRecord] = []
        for stage in ALL_STAGES:
            all_records.extend(self.list_records(stage))
        return all_records

    def stats(self, stage: str | None = None) -> dict[str, Any]:
        """Count records by judgment for a stage (or all stages)."""
        stages = [stage] if stage else ALL_STAGES
        counts: dict[str, int] = {
            "total": 0, "approve": 0, "correct": 0,
            "reject": 0, "flag": 0, "pending": 0,
        }
        for s in stages:
            for rec in self.list_records(s):
                counts["total"] += 1
                j = rec.judgment if rec.judgment in counts else "pending"
                counts[j] += 1
        return counts

    def update_judgment(
        self, stage: str, item_id: str,
        judgment: str, notes: str = "", reviewer: str = "",
    ) -> GoldenRecord | None:
        """Update judgment for an existing item by appending a new record."""
        existing = self.get(stage, item_id)
        if existing is None:
            return None
        updated = GoldenRecord(
            id=existing.id,
            stage=existing.stage,
            item_id=existing.item_id,
            input_data=existing.input_data,
            expected_output=existing.expected_output,
            actual_output=existing.actual_output,
            judgment=judgment,
            notes=notes,
            reviewer=reviewer or existing.reviewer,
        )
        self.add(updated)
        return updated
