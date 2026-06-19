"""Trial logger: in-memory TrialLog assembly and JSONL persistence."""

from pathlib import Path

from nrp_bga_sb.schemas import EventType, TaskEvent, TrialLog


class TrialLogger:
    """Assembles and persists trial logs to a JSONL file.

    Each call to save_trial appends one JSON line to the output file.
    The caller is responsible for populating TrialLog fields (BG timing,
    motor commands, endpoint data, etc.) between open_trial and save_trial.
    """

    def __init__(self, output_path: Path) -> None:
        """
        output_path: path to the .jsonl file where completed trials are appended.
        The file is created on first save if it does not exist.
        """
        self._output_path = output_path

    @property
    def output_path(self) -> Path:
        return self._output_path

    def open_trial(
        self,
        trial_id: int,
        seed: int,
        task_type: str,
        cue_identity: str,
        cue_onset_time: float,
    ) -> TrialLog:
        """Create and return a new TrialLog with the required trial-start fields.

        Does NOT write to disk. The caller mutates the returned object
        (adds events, BG timings, motor commands, etc.) then calls save_trial.
        """
        return TrialLog(
            trial_id=trial_id,
            seed=seed,
            task_type=task_type,
            cue_identity=cue_identity,
            cue_onset_time=cue_onset_time,
        )

    def record_event(
        self,
        log: TrialLog,
        event_type: EventType,
        sim_time: float,
        real_time: float,
        payload: dict | None = None,
    ) -> TaskEvent:
        """Append a TaskEvent to log.events and return it."""
        event = TaskEvent(
            event_type=event_type,
            sim_time=sim_time,
            real_time=real_time,
            trial_id=log.trial_id,
            payload=payload or {},
        )
        log.events.append(event)
        return event

    def save_trial(self, log: TrialLog) -> None:
        """Append the completed TrialLog as one JSON line to the output file.

        Creates the output file (and parent directories) if they do not exist.
        """
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("a", encoding="utf-8") as fh:
            fh.write(log.model_dump_json())
            fh.write("\n")
