"""Timing perturbation wrappers for BG policy callables (Task 3.2).

In the abstract Python single-call model, ``selection_latency`` in ``BGDecision``
is the proxy for timing effects.  All four wrappers modify ``selection_latency``
on the returned ``BGDecision`` to simulate timing shifts; ``DropoutWrapper`` is
the exception — it can suppress a call entirely and return a cached previous
decision.

Phase-offset perturbation is modelled as an additive latency shift because the
nrp-core phase-offset realization (engine timestep offset at FTILoop level) is
unverified for Phase 3 (see PROJECT_MEMORY §15.4).

Design constraints:
  - All wrappers are ``@dataclass`` (not Pydantic) because they hold callables.
  - All randomness is seeded from ``trial_log.seed`` via ``_derive_seed()`` so
    that the same trial always produces the same perturbation.
  - ``BGDecision`` is Pydantic v2 — copies are made with ``model_copy(update={})``;
    the original object is never mutated.
  - Constructor arguments are validated in ``__post_init__``; invalid values raise
    ``ValueError`` immediately (fail-fast contract).
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from nrp_bga_sb.schemas import ActionEvidence, BGDecision, TrialLog

# Type alias for the standard BG policy callable.
PolicyCallable = Callable[[TrialLog, ActionEvidence], BGDecision]


# --- Seed derivation ---


def _derive_seed(trial_seed: int, tag: str) -> int:
    """Derive a deterministic, process-stable integer seed from a trial seed and tag.

    Uses SHA-256 so the result is independent of PYTHONHASHSEED (unlike hash()).
    The tag differentiates between perturbation types for the same trial.
    """
    h = hashlib.sha256(f"{trial_seed}:{tag}".encode()).digest()
    return int.from_bytes(h[:8], "big")


# --- LatencyWrapper ---


@dataclass
class LatencyWrapper:
    """Adds a fixed latency offset to every BG decision.

    Models a constant pipeline delay between BG output emission and thalamic
    receipt.  In nrp-core this would be implemented as a TF that holds a
    DataPack in a queue for a fixed number of FTILoop steps.

    ``latency_ms`` must be ≥ 0; negative values indicate a configuration error.
    """

    base_policy: PolicyCallable
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        # Fail fast: negative latency is physically meaningless.
        if self.latency_ms < 0.0:
            raise ValueError(
                f"latency_ms must be >= 0, got {self.latency_ms}"
            )

    def __call__(
        self,
        trial_log: TrialLog,
        action_evidence: ActionEvidence,
    ) -> BGDecision:
        decision = self.base_policy(trial_log, action_evidence)
        new_latency = decision.selection_latency + self.latency_ms / 1000.0
        return decision.model_copy(update={"selection_latency": new_latency})


# --- JitterWrapper ---


@dataclass
class JitterWrapper:
    """Adds zero-mean Gaussian jitter to BG decision latency.

    Models trial-to-trial timing variability in the BG output pathway.
    The jitter delta is drawn from Normal(0, jitter_std_ms) and clipped at 0
    from below so that ``selection_latency`` never goes negative.

    Randomness is seeded from ``trial_log.seed`` via ``_derive_seed`` so the
    same trial always produces the same jitter perturbation.

    ``jitter_std_ms=0.0`` is a valid no-op configuration (returns the decision
    unchanged, without touching the RNG).
    """

    base_policy: PolicyCallable
    jitter_std_ms: float = 0.0

    def __post_init__(self) -> None:
        # Fail fast: negative std dev is not a valid distribution parameter.
        if self.jitter_std_ms < 0.0:
            raise ValueError(
                f"jitter_std_ms must be >= 0, got {self.jitter_std_ms}"
            )

    def __call__(
        self,
        trial_log: TrialLog,
        action_evidence: ActionEvidence,
    ) -> BGDecision:
        decision = self.base_policy(trial_log, action_evidence)

        # Trigger: jitter_std_ms is zero — no perturbation requested.
        # Why: avoids touching the RNG at all so the zero-jitter path is a
        #      guaranteed no-op with identical latency to the base decision.
        # Outcome: decision returned unchanged.
        if self.jitter_std_ms == 0.0:
            return decision

        rng = random.Random(_derive_seed(trial_log.seed, "jitter"))
        delta_ms = rng.gauss(0, self.jitter_std_ms)
        new_latency = max(0.0, decision.selection_latency + delta_ms / 1000.0)
        return decision.model_copy(update={"selection_latency": new_latency})


# --- DropoutWrapper ---


@dataclass
class DropoutWrapper:
    """Drops BG decisions with a configured probability, reusing the previous decision.

    Models missing output messages in the BG→thalamus pathway.  When a decision
    is dropped, the most recently committed (non-dropped) decision is returned,
    simulating a buffering effect where the thalamus holds the last valid command.

    Inter-call state:
        ``_last_decision`` is the only wrapper with inter-call state.  It records
        the most recently non-dropped BGDecision so that dropped calls have
        something to return.  This deliberate statefulness mirrors the buffering
        semantics of the dropout perturbation: the downstream system does not
        know that a message was dropped and continues acting on the last known
        BG output.

    First-call guarantee:
        If ``_last_decision`` is None (no prior call), the base policy is always
        called regardless of ``dropout_probability``.  This prevents a None return
        on the very first call.

    Seeding:
        The dropout RNG is seeded with ``_derive_seed(trial_log.seed, "dropout")``.
        For the same trial seed the drop verdict is always the same; different
        trial seeds produce different dropout patterns.
    """

    base_policy: PolicyCallable
    dropout_probability: float = 0.0
    # Inter-call state: last non-dropped decision (None before first call).
    _last_decision: BGDecision | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Fail fast: probability must be a valid value in [0, 1].
        if not (0.0 <= self.dropout_probability <= 1.0):
            raise ValueError(
                f"dropout_probability must be in [0.0, 1.0], got {self.dropout_probability}"
            )

    def __call__(
        self,
        trial_log: TrialLog,
        action_evidence: ActionEvidence,
    ) -> BGDecision:
        rng = random.Random(_derive_seed(trial_log.seed, "dropout"))

        # Trigger: random draw is below dropout_probability AND a previous
        #          decision is cached.
        # Why: simulates a dropped DataPack — the downstream thalamus never
        #      receives a new BG output for this trial and holds the last known
        #      committed decision.
        # Outcome: _last_decision is returned without invoking base_policy,
        #          preserving the cached state for subsequent calls.
        if rng.random() < self.dropout_probability and self._last_decision is not None:
            return self._last_decision

        decision = self.base_policy(trial_log, action_evidence)
        self._last_decision = decision
        return decision


# --- PhaseOffsetWrapper ---


@dataclass
class PhaseOffsetWrapper:
    """Adds a fixed phase-offset shift to BG decision latency.

    Models a timing offset between the BG update cycle and the trial clock.
    In nrp-core this would require engine timestep manipulation (starting
    two engines at different offsets within the same FTILoop), which is
    unverified for Phase 3 (PROJECT_MEMORY §15.4).  This wrapper therefore
    models phase offset as an additive latency shift — a conservative
    approximation that captures the net timing displacement without
    requiring engine-level changes.

    ``phase_offset_ms`` must be ≥ 0; negative offsets are not currently
    modelled (no mechanism for the BG to respond before the trial clock fires).
    """

    base_policy: PolicyCallable
    phase_offset_ms: float = 0.0

    def __post_init__(self) -> None:
        # Fail fast: negative phase offset is outside the supported model.
        if self.phase_offset_ms < 0.0:
            raise ValueError(
                f"phase_offset_ms must be >= 0, got {self.phase_offset_ms}"
            )

    def __call__(
        self,
        trial_log: TrialLog,
        action_evidence: ActionEvidence,
    ) -> BGDecision:
        decision = self.base_policy(trial_log, action_evidence)
        new_latency = decision.selection_latency + self.phase_offset_ms / 1000.0
        return decision.model_copy(update={"selection_latency": new_latency})
