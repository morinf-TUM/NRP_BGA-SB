"""Thalamic gate adapter: converts BGDecision to MotorCommand (Task 4.2).

The thalamus is modelled as a threshold gate between BG output and the
descending motor command. In the abstract closed-loop model:

  - Gate stays closed when BG has no committed selection or the decision
    margin is below threshold — no motor command is released.
  - Gate opens partially when margin is between margin_threshold and
    full_open_threshold — motor command is scaled proportionally.
  - Gate is fully open when margin is at or above full_open_threshold —
    motor command is released at full gain.

This captures the role of the thalamus as a relay that requires sufficient
BG confidence (decision margin) before releasing a motor command.  The gain
modulation is a first-order model of the thalamocortical signal strength.

The MotorCommand.command vector encodes the descending signal:
  command[selected_channel] = gate_gain   (channel carrying the motor output)
  command[other channels]   = 0.0
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from nrp_bga_sb.schemas import BGDecision, MotorCommand

# --- ThalamusConfig ---


class ThalamusConfig(BaseModel):
    """Configuration for the thalamic gate adapter.

    Attributes:
        margin_threshold:    Minimum decision margin required to open the gate
                             at all. Below this value the gate remains closed.
        full_open_threshold: Decision margin at which the gate is fully open
                             (gate_gain = 1.0). Must be ≥ margin_threshold.
        n_channels:          Number of motor output channels in the command
                             vector. Should match the BG action-channel count.
    """

    margin_threshold: float = 0.05
    full_open_threshold: float = 0.30
    n_channels: int = 2

    @field_validator("margin_threshold", "full_open_threshold")
    @classmethod
    def _check_margin_bounds(cls, v: float) -> float:
        # Trigger: margin value is negative.
        # Why: negative margin thresholds are physically meaningless — the BG
        #      decision margin is always ≥ 0.
        # Outcome: ValidationError raised; caller must fix the config.
        if v < 0.0:
            raise ValueError(f"margin threshold must be >= 0, got {v}")
        return v

    @field_validator("n_channels")
    @classmethod
    def _check_n_channels(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"n_channels must be >= 1, got {v}")
        return v

    def model_post_init(self, __context: object) -> None:
        # Trigger: full_open_threshold is below margin_threshold.
        # Why: the two thresholds define a monotone gain ramp; if
        #      full_open < margin the ramp would be negative, which is undefined.
        # Outcome: ValidationError raised.
        if self.full_open_threshold < self.margin_threshold:
            raise ValueError(
                f"full_open_threshold ({self.full_open_threshold}) must be "
                f">= margin_threshold ({self.margin_threshold})"
            )


# --- ThalamusGate ---


class ThalamusGate:
    """Converts BGDecision to MotorCommand via margin-based threshold gating.

    Each __call__ is stateless and produces a fresh MotorCommand.

    Gate logic (in priority order):
      1. selected_channel == -1  → gate "closed", gain = 0.0
      2. margin < margin_threshold  → gate "closed", gain = 0.0
      3. margin_threshold ≤ margin < full_open_threshold
                                 → gate "partial", gain linearly interpolated
      4. margin ≥ full_open_threshold  → gate "open",  gain = 1.0
    """

    def __init__(self, config: ThalamusConfig) -> None:
        self._config = config

    def __call__(self, bg_decision: BGDecision) -> MotorCommand:
        """Convert a BG decision into a descending motor command.

        Args:
            bg_decision: The committed BG decision from the scheduler.

        Returns:
            MotorCommand with command vector, gate_state, and gate_gain.
        """
        cfg = self._config
        command: list[float] = [0.0] * cfg.n_channels

        # --- Gate 1: No channel selected (BG could not commit) ---
        # Trigger: selected_channel == -1.
        # Why: without a committed BG selection, the thalamus has no signal
        #      to relay; releasing a motor command would be a hallucination.
        # Outcome: gate remains closed; motor command vector is all zeros.
        if bg_decision.selected_channel < 0:
            return MotorCommand(
                sim_time=bg_decision.sim_time,
                trial_id=bg_decision.trial_id,
                command=command,
                gate_state="closed",
                gate_gain=0.0,
            )

        margin = bg_decision.decision_margin

        # --- Gate 2 / 3 / 4: Threshold-based gain calculation ---
        # Trigger: margin determines which gate regime applies.
        # Why: a weak BG decision (small margin) does not warrant full motor
        #      release; the gain ramp models gradual thalamocortical recruitment.
        # Outcome: gate_state and gate_gain set according to the regime.
        if margin <= cfg.margin_threshold:
            gate_state = "closed"
            gain = 0.0
        elif margin < cfg.full_open_threshold:
            gate_state = "partial"
            # Linear interpolation from 0.0 to 1.0 over the partial range.
            span = cfg.full_open_threshold - cfg.margin_threshold
            gain = (margin - cfg.margin_threshold) / span
        else:
            gate_state = "open"
            gain = 1.0

        # Encode the selected channel in the command vector.
        if gain > 0.0 and bg_decision.selected_channel < cfg.n_channels:
            command[bg_decision.selected_channel] = gain

        return MotorCommand(
            sim_time=bg_decision.sim_time,
            trial_id=bg_decision.trial_id,
            command=command,
            gate_state=gate_state,  # type: ignore[arg-type]
            gate_gain=gain,
        )
