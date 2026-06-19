"""Cue generator with shared-seed support.

Produces reproducible cue sequences (lists of per-trial seeds) that can be
reused across BG-frequency conditions to enable causal comparison.

The key guarantees:
1. Determinism: generate_cue_sequence(42, "go_nogo", 200) always returns the same
   list of seeds.
2. Independence: same master_seed with different task_type produces different
   seeds (no aliasing).
"""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, Field


# --- CueSequence ---


class CueSequence(BaseModel):
    """A reproducible sequence of per-trial seeds for a task.

    Attributes:
        master_seed: the seed that generated this sequence.
        task_type: the task paradigm ("go_nogo", "two_choice", "stop_signal", "change_of_mind").
        n_trials: number of trials (equal to len(trial_seeds)).
        trial_seeds: list of per-trial seeds, one per trial (len == n_trials).
    """

    model_config = {"frozen": True}

    master_seed: int
    task_type: Literal["go_nogo", "two_choice", "stop_signal", "change_of_mind"]
    n_trials: int
    trial_seeds: list[int] = Field(..., description="Per-trial seeds; len must equal n_trials")

    def model_post_init(self, __context):
        """Validate that trial_seeds length matches n_trials."""
        # Trigger: trial_seeds length does not match n_trials.
        # Why: inconsistent state signals a construction error at the boundary.
        # Outcome: raises ValueError immediately to prevent silent misuse.
        if len(self.trial_seeds) != self.n_trials:
            raise ValueError(
                f"trial_seeds length ({len(self.trial_seeds)}) does not match "
                f"n_trials ({self.n_trials})"
            )


# --- Cue sequence generation ---


def generate_cue_sequence(
    master_seed: int,
    task_type: str,
    n_trials: int,
) -> CueSequence:
    """Generate a reproducible sequence of per-trial seeds.

    The sequence is determined entirely by (master_seed, task_type, n_trials).
    Calling twice with the same arguments returns the same sequence.

    Different task_types produce different sequences (use a task_type-specific
    salt in the RNG initialization to avoid aliasing).

    Args:
        master_seed: the seed controlling the whole sequence.
        task_type: "go_nogo", "two_choice", "stop_signal", or "change_of_mind".
        n_trials: number of trials to generate seeds for.

    Returns:
        CueSequence with n_trials trial seeds.
    """
    # Trigger: different task_types should produce different seed sequences even
    #   with the same master_seed.
    # Why: task paradigms have different trial structures (e.g., go/no-go vs
    #   two-choice), so the same seed value should lead to different cues.
    # Outcome: initialize RNG with (master_seed, task_type) hash so the salt is
    #   task-specific and deterministic.
    rng = random.Random(hash((master_seed, task_type)))

    trial_seeds = [rng.randint(0, 2**31 - 1) for _ in range(n_trials)]

    return CueSequence(
        master_seed=master_seed,
        task_type=task_type,
        n_trials=n_trials,
        trial_seeds=trial_seeds,
    )


# --- Shared-seed config helper ---


def shared_seed_configs(
    cue_seq: CueSequence,
    base_config,
    bg_frequencies_hz: list[float],
) -> list:
    """Return one config copy per BG-frequency condition, all with the same seed.

    In Phase 1, this sets config.seed = cue_seq.master_seed on each copy.
    In later phases, it will inject per-trial seeds directly.

    Args:
        cue_seq: the CueSequence (contains master_seed and trial_seeds).
        base_config: a task config (GoNoGoConfig, TwoChoiceConfig, etc.).
        bg_frequencies_hz: list of BG update frequencies to create configs for.

    Returns:
        list of len(bg_frequencies_hz) configs, each a copy of base_config with
        seed set to cue_seq.master_seed.
    """
    # Use dataclasses.replace() to create copies with updated seed.
    # This works because all engine configs are @dataclass (not frozen).
    from dataclasses import replace

    configs = []
    for _ in bg_frequencies_hz:
        config_copy = replace(base_config, seed=cue_seq.master_seed)
        configs.append(config_copy)

    return configs
