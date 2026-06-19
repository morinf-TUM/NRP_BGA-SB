"""Replay: load persisted TrialLog records and iterate their event streams."""

from collections.abc import Iterator
from pathlib import Path

from nrp_bga_sb.schemas import TaskEvent, TrialLog


def load_trials(path: Path) -> list[TrialLog]:
    """Load all TrialLog records from a JSONL file.

    Each non-empty line is parsed as a TrialLog. Empty lines are skipped.

    Raises FileNotFoundError if path does not exist.
    Raises pydantic.ValidationError if a line is not a valid TrialLog.
    """
    trials: list[TrialLog] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                trials.append(TrialLog.model_validate_json(line))
    return trials


def replay_events(log: TrialLog) -> Iterator[TaskEvent]:
    """Yield a trial's events in ascending sim_time order."""
    yield from sorted(log.events, key=lambda e: e.sim_time)
