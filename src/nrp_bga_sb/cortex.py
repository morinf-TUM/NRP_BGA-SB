"""Abstract cortical evidence generator (Task 4.1).

Produces time-varying ActionEvidence from trial state and elapsed time since
the decision-relevant cue onset. This module is the mechanism that makes
BG-frequency effects observable: at lower input_sampling_hz, the scheduler
reads earlier ticks (lower salience), producing weaker or absent selection.
In the constant-evidence Phase 3 model, tick-0 always committed regardless
of frequency; the ramp model breaks that invariant.

Evidence model (linear ramp):
  preferred_salience(t) = base + (peak - base) * min(1.0, t / rise_time_ms)
  competing_salience(t) = 1.0 - preferred_salience(t)

Direction is determined by trial_log.cue_identity:
  "go", "left", "no_switch", "switch_*"  → channel 0 preferred
  "right"                                → channel 1 preferred
  "no_go", "stop"                        → no preferred channel (BG withholds)
"""

from __future__ import annotations

import hashlib
import math

from pydantic import BaseModel, field_validator

from nrp_bga_sb.schemas import ActionEvidence, EventType, TrialLog

# --- CortexConfig ---


class CortexConfig(BaseModel):
    """Configuration for the abstract cortical evidence generator.

    Attributes:
        rise_time_ms:   Time (ms) for salience to reach peak from neutral.
                        The ramp is linear over [0, rise_time_ms]; salience
                        is clamped at peak beyond that.
        peak_salience:  Maximum salience of the preferred channel (frac = 1.0).
                        Competing channel salience = 1.0 - peak_salience.
        base_salience:  Starting (neutral) salience at frac = 0. Equal for
                        both channels — BG cannot decide at frac = 0.
        noise_std:      Standard deviation of per-tick Gaussian noise added to
                        channel saliences. 0.0 = deterministic (default).
    """

    rise_time_ms: float = 100.0
    peak_salience: float = 0.9
    base_salience: float = 0.5
    noise_std: float = 0.0

    @field_validator("rise_time_ms")
    @classmethod
    def _check_rise_time(cls, v: float) -> float:
        # Trigger: rise_time_ms is zero or negative.
        # Why: a non-positive rise time makes min(1.0, t / rise_time_ms) undefined.
        # Outcome: ValidationError raised; caller must fix the config.
        if v <= 0:
            raise ValueError(f"rise_time_ms must be > 0, got {v}")
        return v

    @field_validator("peak_salience", "base_salience")
    @classmethod
    def _check_salience_bounds(cls, v: float) -> float:
        # Trigger: salience outside [0, 1].
        # Why: ActionEvidence schema does not enforce salience bounds, but
        #      BGModel behaviour is undefined outside this range.
        # Outcome: ValidationError raised.
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"salience must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("noise_std")
    @classmethod
    def _check_noise_std(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"noise_std must be >= 0, got {v}")
        return v


# --- Preferred-channel map ---

# Maps cue_identity string → preferred channel index (0 or 1) or None (withhold).
# None means both channels stay at base_salience so BG cannot select either channel.
_PREFERRED_CHANNEL: dict[str, int | None] = {
    "go": 0,  # go/no-go: action channel is 0
    "no_go": None,  # BG should withhold; no directed cortical drive
    "left": 0,  # two-choice: left target = channel 0
    "right": 1,  # two-choice: right target = channel 1
    "stop": None,  # stop-signal stop trial: BG should withhold
    "no_switch": 0,  # change-of-mind baseline: action channel is 0
}


def _preferred_channel_from_cue(cue_identity: str) -> int | None:
    """Return the preferred channel (0, 1) or None for a given cue_identity.

    Handles switch_* cue_identities (change-of-mind switch trials) by
    mapping all switch categories to channel 0 (initial pre-switch direction).
    Post-switch redirection is handled in `CortexEvidenceGenerator.__call__`
    by scanning trial_log.events.
    """
    if cue_identity in _PREFERRED_CHANNEL:
        return _PREFERRED_CHANNEL[cue_identity]
    # switch_early, switch_medium, switch_late, etc. → initial direction = ch0
    if cue_identity.startswith("switch_"):
        return 0
    # Unknown cue_identity: treat as neutral (fail-safe withheld response).
    return None


# --- CortexEvidenceGenerator ---


class CortexEvidenceGenerator:
    """Produces time-varying ActionEvidence from trial context and elapsed time.

    Each __call__ is stateless and reproducible (no instance state mutated).
    The linear ramp is the simplest model that makes BG-frequency effects
    observable:

    - At higher input_sampling_hz the scheduler reads more ticks, including
      early low-salience samples, and may commit earlier (weaker evidence).
    - At lower input_sampling_hz the scheduler reads fewer ticks; if the period
      is longer than the rise time, it reads only the late (strong-evidence) tick
      or misses the rise entirely and cannot select.

    This asymmetry produces the frequency-dependent success/miss rate used in
    the Task 4.3 acceptance verification.
    """

    def __init__(self, config: CortexConfig) -> None:
        self._config = config

    def __call__(self, trial_log: TrialLog, elapsed_ms: float) -> ActionEvidence:
        """Generate ActionEvidence at elapsed_ms after the decision-relevant cue.

        Args:
            trial_log:  Current trial; cue_identity determines evidence direction.
            elapsed_ms: Time elapsed since cue onset in ms (≥ 0).

        Returns:
            ActionEvidence with n_channels=2 and time-varying channel_salience.
        """
        cfg = self._config
        preferred = _preferred_channel_from_cue(trial_log.cue_identity)

        # --- Post-switch direction reversal for change-of-mind switch trials ---
        # Trigger: cue_identity is a switch variant AND evidence_change is already
        #          in the log, meaning the engine has crossed the switch point and
        #          is making the second (post-switch) policy call.
        # Why: CortexEvidenceGenerator is stateless; the only reliable signal that
        #      the post-switch call is happening is the presence of evidence_change
        #      in trial_log.events (logged by the engine at switch_delay_ms).
        # Outcome: preferred channel flips 0 → 1, driving the BG toward the new
        #          target so it can produce a correct switch response.
        if preferred == 0 and trial_log.cue_identity.startswith("switch_"):
            has_switched = any(e.event_type == EventType.evidence_change for e in trial_log.events)
            if has_switched:
                preferred = 1

        # Linear ramp: frac ∈ [0.0, 1.0] rises over rise_time_ms.
        frac = min(1.0, max(0.0, elapsed_ms / cfg.rise_time_ms))

        # --- Build per-channel salience ---
        if preferred is None:
            # No preferred channel: keep both at base_salience.
            # Trigger: cue_identity is "no_go" or "stop".
            # Why: BG withholding is the correct response; feeding it
            #      directed evidence would cause false alarms / stop failures.
            # Outcome: BG sees symmetric evidence → selected_channel = -1.
            ch: list[float] = [cfg.base_salience, cfg.base_salience]
        else:
            # Preferred channel rises from base to peak; competing channel
            # falls symmetrically (sum = 1.0 at all frac values).
            preferred_sal = cfg.base_salience + (cfg.peak_salience - cfg.base_salience) * frac
            competing_sal = 1.0 - preferred_sal
            ch = (
                [preferred_sal, competing_sal] if preferred == 0 else [competing_sal, preferred_sal]
            )

        # --- Optional reproducible noise ---
        if cfg.noise_std > 0.0:
            noise = _sample_noise(trial_log.seed, elapsed_ms, cfg.noise_std)
            ch = [max(0.0, min(1.0, s + n)) for s, n in zip(ch, noise)]

        # --- Stop-signal detection ---
        # Trigger: a stop_signal event is present in trial_log.events before
        #          this evidence is generated.
        # Why: the stop-signal flag in ActionEvidence triggers a policy-level
        #      veto in BGAdapter without requiring evidence manipulation.
        # Outcome: stop_signal_present=True propagates to BGAdapter → -1 return.
        stop_signal_present = any(e.event_type == EventType.stop_signal for e in trial_log.events)

        return ActionEvidence(
            sim_time=trial_log.cue_onset_time + elapsed_ms / 1000.0,
            trial_id=trial_log.trial_id,
            n_channels=2,
            channel_salience=ch,
            stop_signal_present=stop_signal_present,
        )


# --- Reproducible per-tick noise ---


def _sample_noise(seed: int, elapsed_ms: float, std: float) -> list[float]:
    """Return two reproducible Gaussian noise samples using hashlib + Box-Muller.

    Uses hashlib.sha256 (not Python's hash()) to guarantee cross-process
    determinism (see cue_generator.py: hash() is PYTHONHASHSEED-dependent).
    """
    # Encode elapsed_ms at microsecond resolution to avoid floating-point
    # collision between nearby ticks.
    key = f"{seed}:{int(elapsed_ms * 1000)}"
    digest = hashlib.sha256(key.encode()).digest()

    # Two independent uniform samples from the first 8 bytes.
    # Adding 1 and dividing by (2^32 + 1) maps [0, 2^32-1] → (0, 1) strictly,
    # which is required by Box-Muller (log(0) is undefined).
    u1 = (int.from_bytes(digest[0:4], "big") + 1) / (2**32 + 1)
    u2 = (int.from_bytes(digest[4:8], "big") + 1) / (2**32 + 1)

    mag = std * math.sqrt(-2.0 * math.log(u1))
    angle = 2.0 * math.pi * u2
    return [mag * math.cos(angle), mag * math.sin(angle)]
