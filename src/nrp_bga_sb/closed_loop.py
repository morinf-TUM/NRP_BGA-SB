"""Full abstract closed-loop policy: Cortex → BG → Thalamus → MotorCommand (Task 4.3).

ClosedLoopPolicy implements the standard (TrialLog, ActionEvidence) → BGDecision
interface so it is a drop-in replacement for any existing policy.  Internally it
chains:

  1. CortexEvidenceGenerator   — time-varying salience ramp injected into the
                                  scheduler via cortex_generator hook (Phase 4+)
  2. ScheduledBGAdapter         — four-gate frequency scheduler (Phase 3)
  3. ThalamusGate               — margin-based gate releasing MotorCommand (Phase 4)

Side effects on trial_log:
  - motor_command_series is appended with the resulting MotorCommand.
  - thalamic_relay_time is set to bg_decision.sim_time (BG→thalamus arrival).
  - thalamic_release_time is set to motor_cmd.sim_time (motor command issued).

Observable frequency effect (acceptance criterion, Task 4.3):
  At very low BG input_sampling_hz (e.g. 5 Hz, period = 200 ticks with 1 ms base
  step), the scheduler's Gate 1 fires only at tick 0 where cortical evidence is
  still neutral [0.5, 0.5].  BGModel cannot select → committed_decision has
  selected_channel = -1 → go trials become misses and no MotorCommand is released.
  At higher frequencies (≥ 10 Hz, period ≤ 100 ticks), Gate 1 fires again at tick
  100 where evidence has risen to peak → BG selects → motor command released.
"""

from __future__ import annotations

from nrp_bga_sb.bg_model import BGAdapter, BGModelConfig
from nrp_bga_sb.cortex import CortexConfig, CortexEvidenceGenerator
from nrp_bga_sb.scheduler import FrequencyConfig, ScheduledBGAdapter
from nrp_bga_sb.schemas import ActionEvidence, BGDecision, TrialLog
from nrp_bga_sb.thalamus import ThalamusConfig, ThalamusGate

# --- ClosedLoopPolicy ---


class ClosedLoopPolicy:
    """Drop-in policy implementing the full Cortex → BG → Thalamus → Motor chain.

    Each __call__ updates trial_log with the resulting MotorCommand and thalamic
    timing fields, then returns the BGDecision so existing task engines can
    classify the trial outcome using the same logic as Phase 2.
    """

    def __init__(
        self,
        scheduled_bg: ScheduledBGAdapter,
        thalamus: ThalamusGate,
    ) -> None:
        self._scheduled_bg = scheduled_bg
        self._thalamus = thalamus

    def __call__(self, trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision:
        """Run the closed-loop chain and return the committed BG decision.

        Args:
            trial_log:       Current trial; updated in-place with motor command
                             and thalamic timing.
            action_evidence: Passed to the scheduler's fallback path only;
                             when a cortex_generator is installed in the
                             ScheduledBGAdapter, it is overridden at each
                             sampling tick with time-varying evidence.

        Returns:
            The committed BGDecision from the scheduler.
        """
        bg_decision = self._scheduled_bg(trial_log, action_evidence)
        motor_cmd = self._thalamus(bg_decision)

        # --- Populate thalamic fields in trial_log ---
        # Trigger: closed-loop chain produces a BG decision and motor command.
        # Why: thalamic timing fields (§8) are required for Phase 5+ metrics and
        #      the logging contract; the abstract model has zero thalamic delay.
        # Outcome: trial_log carries observable intermediate state for diagnosis.
        trial_log.motor_command_series.append(motor_cmd)
        trial_log.thalamic_relay_time = bg_decision.sim_time
        trial_log.thalamic_release_time = motor_cmd.sim_time

        return bg_decision


# --- Factory ---


def make_closed_loop_policy(
    bg_model_config: BGModelConfig | None = None,
    cortex_config: CortexConfig | None = None,
    thalamus_config: ThalamusConfig | None = None,
    frequency_config: FrequencyConfig | None = None,
    accumulation_ms: float = 200.0,
) -> ClosedLoopPolicy:
    """Build a fully-configured ClosedLoopPolicy from component configs.

    All configs are optional; defaults give a 160 Hz high-frequency loop with
    a 100 ms cortical evidence rise time and a standard thalamic gate.

    Args:
        bg_model_config:  GPR BG model parameters.  Defaults to BGModelConfig().
        cortex_config:    Cortical evidence ramp parameters.  Defaults to
                          CortexConfig() (rise_time_ms=100, peak=0.9, base=0.5).
        thalamus_config:  Gate thresholds.  Defaults to ThalamusConfig()
                          (margin_threshold=0.05, full_open=0.30).
        frequency_config: Four-knob frequency parameters.  Defaults to
                          FrequencyConfig() (all knobs at 160 Hz).
        accumulation_ms:  Pre-decision integration window length (ms).

    Returns:
        A ClosedLoopPolicy ready to use as a task-engine policy.
    """
    bg_cfg = bg_model_config if bg_model_config is not None else BGModelConfig()
    cx_cfg = cortex_config if cortex_config is not None else CortexConfig()
    th_cfg = thalamus_config if thalamus_config is not None else ThalamusConfig()
    fr_cfg = frequency_config if frequency_config is not None else FrequencyConfig()

    cortex = CortexEvidenceGenerator(cx_cfg)
    bg_adapter = BGAdapter(bg_cfg)
    scheduled_bg = ScheduledBGAdapter(
        base_policy=bg_adapter,
        config=fr_cfg,
        accumulation_ms=accumulation_ms,
        cortex_generator=cortex,
    )
    thalamus = ThalamusGate(th_cfg)

    return ClosedLoopPolicy(scheduled_bg=scheduled_bg, thalamus=thalamus)
