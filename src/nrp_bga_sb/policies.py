"""Three reference policies for basal ganglia decision-making.

All policies implement the same interface:
    (trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision

Each policy fills in selected_channel (-1, 0, or 1), decision_margin (salience gap),
and channel_activations (pass-through from evidence).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from nrp_bga_sb.schemas import ActionEvidence, BGDecision, TrialLog

# --- Oracle Policy ---


def oracle_policy(trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision:
    """Always select the channel with the highest salience, or inhibit if stop signal present.

    Behavior:
    - If stop_signal_present=True: return channel -1 (inhibit).
    - Otherwise: if max(channel_salience) < 0.3: return channel -1 (no action).
    - Otherwise: return argmax(channel_salience).

    This policy represents an ideal Bayesian decision-maker that always picks the most
    salient action, except when a stop signal explicitly demands inhibition.
    """
    # Trigger: stop signal has arrived; inhibition is required.
    # Why: the oracle must comply with the stop signal to represent ideal cancellation.
    # Outcome: selected_channel = -1 (no action).
    if action_evidence.stop_signal_present:
        selected_channel = -1
    else:
        # Trigger: no stop signal present; examine salience.
        # Why: select the channel with the strongest evidence.
        # Outcome: selected_channel = argmax(salience) or -1 if all are weak.
        max_salience = max(action_evidence.channel_salience)
        if max_salience < 0.3:
            # Salience is too weak; inhibit by default.
            selected_channel = -1
        else:
            # Select the channel with the highest salience.
            selected_channel = action_evidence.channel_salience.index(max_salience)

    # Compute decision margin: gap between top-1 and top-2 saliences.
    # decision_margin is semantically a gap between two channels; when there is only one
    # channel, there is no gap, so margin should be 0.0 rather than the single salience.
    if len(action_evidence.channel_salience) >= 2:
        saliences_sorted = sorted(action_evidence.channel_salience, reverse=True)
        decision_margin = saliences_sorted[0] - saliences_sorted[1]
    else:
        decision_margin = 0.0

    return BGDecision(
        sim_time=action_evidence.sim_time,
        trial_id=action_evidence.trial_id,
        selected_channel=selected_channel,
        decision_margin=decision_margin,
        suppression_vector=[0.0] * len(action_evidence.channel_salience),
        channel_activations=action_evidence.channel_salience,
        selection_latency=0.0,
    )


# --- Random Policy ---


@dataclass
class RandomPolicy:
    """Randomly select channel 0, channel 1, or -1 (no action) with equal probability.

    The policy is seeded from trial_log.seed to ensure deterministic reproducibility:
    same seed → same decision every time.
    """

    def __call__(
        self, trial_log: TrialLog, action_evidence: ActionEvidence
    ) -> BGDecision:
        """Select a channel uniformly at random (0, 1, or -1) using trial_log.seed."""
        # Trigger: each trial needs a deterministic but pseudo-random decision.
        # Why: seed-based reproducibility allows reproducible policy behavior
        #      while exercising all three decision modes (select 0, select 1, inhibit).
        # Outcome: rng seeded from trial_log.seed; three-way choice equally likely.
        rng = random.Random(trial_log.seed)
        choices = [0, 1, -1]
        selected_channel = rng.choice(choices)

        # Compute decision margin: gap between top-1 and top-2 saliences.
        # decision_margin is semantically a gap between two channels; when there is only one
        # channel, there is no gap, so margin should be 0.0 rather than the single salience.
        if len(action_evidence.channel_salience) >= 2:
            saliences_sorted = sorted(action_evidence.channel_salience, reverse=True)
            decision_margin = saliences_sorted[0] - saliences_sorted[1]
        else:
            decision_margin = 0.0

        return BGDecision(
            sim_time=action_evidence.sim_time,
            trial_id=action_evidence.trial_id,
            selected_channel=selected_channel,
            decision_margin=decision_margin,
            suppression_vector=[0.0] * len(action_evidence.channel_salience),
            channel_activations=action_evidence.channel_salience,
            selection_latency=0.0,
        )


# --- Evidence-Threshold Policy ---


@dataclass
class ThresholdPolicy:
    """Select the highest-salience channel if its evidence exceeds a threshold.

    Behavior:
    - If stop_signal_present=True: return channel -1 (inhibit), regardless of salience.
    - If max(channel_salience) >= threshold: return argmax(channel_salience).
    - If max(channel_salience) < threshold: return channel -1 (no action).

    Attributes:
        threshold: configurable salience threshold (default 0.6).
    """

    threshold: float = 0.6

    def __call__(
        self, trial_log: TrialLog, action_evidence: ActionEvidence
    ) -> BGDecision:
        """Apply threshold to channel salience; override to -1 if stop signal present."""
        # Trigger: stop signal has arrived; inhibition is mandatory.
        # Why: stop signal takes priority over any salience-based decision.
        # Outcome: selected_channel = -1.
        if action_evidence.stop_signal_present:
            selected_channel = -1
        else:
            # Trigger: no stop signal; compare max salience to threshold.
            # Why: threshold gate implements confidence-gating on decision commitment.
            # Outcome: select if evidence strong enough; inhibit if weak.
            max_salience = max(action_evidence.channel_salience)
            if max_salience >= self.threshold:
                selected_channel = action_evidence.channel_salience.index(max_salience)
            else:
                selected_channel = -1

        # Compute decision margin: gap between top-1 and top-2 saliences.
        # decision_margin is semantically a gap between two channels; when there is only one
        # channel, there is no gap, so margin should be 0.0 rather than the single salience.
        if len(action_evidence.channel_salience) >= 2:
            saliences_sorted = sorted(action_evidence.channel_salience, reverse=True)
            decision_margin = saliences_sorted[0] - saliences_sorted[1]
        else:
            decision_margin = 0.0

        return BGDecision(
            sim_time=action_evidence.sim_time,
            trial_id=action_evidence.trial_id,
            selected_channel=selected_channel,
            decision_margin=decision_margin,
            suppression_vector=[0.0] * len(action_evidence.channel_salience),
            channel_activations=action_evidence.channel_salience,
            selection_latency=0.0,
        )
