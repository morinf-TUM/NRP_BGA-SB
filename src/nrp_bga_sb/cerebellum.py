"""Cerebellar adaptive-filter model (Phase 11, Task 11.1).

Two separable, independently-ablatable layers:

  AdaptiveFilter         — trial-by-trial Widrow-Hoff/LMS learning of a
                           feedforward counter-rotation (Fujita 1982;
                           Dean, Porrill & Stone 2010).
  ForwardModelController — within-trial proportional feedback steering the
                           trajectory toward the desired path (Miall et al.
                           1993 internal-forward-model / Smith-predictor line).

Cerebellum composes both behind one interface, with independent enable flags.
"""
from __future__ import annotations

import numpy as np

from nrp_bga_sb.perturbation_plant import rotate_xy

# --- AdaptiveFilter ---


class AdaptiveFilter:
    """Scalar LMS filter learning a feedforward counter-rotation across trials."""

    def __init__(self, learning_rate: float = 0.1) -> None:
        # Trigger: learning_rate outside (0, 1].
        # Why: <=0 never learns; >1 overshoots and can diverge the LMS update.
        # Outcome: fail fast so a mis-tuned experiment cannot silently produce noise.
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        self.learning_rate = learning_rate
        self.theta_hat: float = 0.0

    def precompensate(self, endpoint_xy: list[float]) -> list[float]:
        """Apply the learned feedforward counter-rotation (-theta_hat)."""
        return rotate_xy(endpoint_xy, -self.theta_hat)

    def update(self, angular_error_rad: float) -> None:
        """Widrow-Hoff/LMS step toward cancelling the observed angular error."""
        # The observed error equals the residual rotation (theta - theta_hat);
        # adding learning_rate * error drives theta_hat -> theta over trials.
        self.theta_hat += self.learning_rate * angular_error_rad

    def reset(self) -> None:
        """Clear learned state (between blocks / seeds)."""
        self.theta_hat = 0.0


# --- ForwardModelController ---


class ForwardModelController:
    """Within-trial proportional feedback toward the desired (intended) path."""

    def __init__(self, gain: float = 0.5) -> None:
        # Trigger: gain outside [0, 1].
        # Why: gain<0 pushes away from target; gain>1 over-corrects and can ring.
        # Outcome: fail fast on a mis-specified controller.
        if not 0.0 <= gain <= 1.0:
            raise ValueError(f"gain must be in [0, 1], got {gain}")
        self.gain = gain

    def integrate(
        self,
        desired_xy: list[float],
        openloop_xy: list[float],
        s_values: list[float],
    ) -> list[list[float]]:
        """Integrate the corrected trajectory step by step.

        At each step the open-loop perturbed motion increment (toward P) is
        applied, then a proportional feedback term pulls the running position
        toward the reference D*s (where the hand should be by now). gain=0 leaves
        the open-loop straight line to P; gain in (0,1) curves the path toward D.
        The (1-gain) contraction on the running position keeps the loop stable.
        """
        D = np.asarray(desired_xy, dtype=float)
        P = np.asarray(openloop_xy, dtype=float)
        pos = np.zeros(2, dtype=float)
        prev_s = 0.0
        out: list[list[float]] = []
        for s in s_values:
            ds = s - prev_s
            openloop_increment = P * ds          # perturbed feedforward motion this step
            ref = D * s                          # desired position by this point
            feedback = self.gain * (ref - pos)   # proportional pull toward desired path
            pos = pos + openloop_increment + feedback
            out.append([float(pos[0]), float(pos[1])])
            prev_s = s
        return out
