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
