"""Gurney-Prescott-Redgrave (2001) rate-coded basal ganglia model.

Reference: Gurney K, Prescott TJ, Redgrave P. A computational model of action selection
in the basal ganglia. I and II. Biological Cybernetics, 84(3), 2001. (ModelDB 83560)

This module implements the steady-state rate-coded version of the GPR model, with
Jacobi iteration for fixed-point settling. Three anatomical pathways are modelled:

  Direct pathway    D1 striatum → GPi      channel-specific inhibition of GPi (selection)
  Indirect pathway  D2 striatum → GPe      modulates GPe, which feeds back onto STN
  Hyperdirect       cortex → STN           global blanket suppression via STN→GPi

Key implementation choices:
  - STN→GPi uses the *mean* STN activation across channels so that the blanket
    suppression is scale-invariant with the number of action channels (N).
  - Stop-signal is handled as a policy-level override (selected_channel = -1) rather
    than via the hyperdirect circuit; hyperdirect modelling is deferred to Phase 3.
  - Selection latency is derived analytically from the thalamus winner output
    (inverse proportionality) to guarantee a monotone conflict→latency relationship
    required by the M2 acceptance criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nrp_bga_sb.schemas import ActionEvidence, BGDecision, TrialLog

# --- GPR Model Configuration ---


@dataclass
class BGModelConfig:
    """Configuration for the GPR rate-coded BG model.

    Default values are calibrated so that:
      - salience gap ≥ 0.3  reliably produces selection of the dominant channel;
      - equal saliences      produce no selection (conflict case);
      - zero saliences       produce no selection (tonic-inhibition baseline).

    Derivation of default w_stn_gpi = 0.7: with the remaining defaults, the
    threshold for selection is D1_winner > w_stn_gpi * mean_STN.  The low-conflict
    canonical case [0.8, 0.2] has D1_winner = 0.6 and mean_STN ≈ 0.58, giving
    0.6 > 0.7 × 0.58 = 0.406 ✓.  The medium-conflict case [0.65, 0.35] gives
    D1_winner = 0.45 > 0.7 × 0.58 = 0.406 ✓ (marginal).  The high-conflict case
    [0.55, 0.45] gives D1_winner = 0.35 < 0.406 → no selection ✓.
    """

    n_channels: int = 2

    # --- Striatum ---
    # D1 and D2 only fire when salience exceeds this threshold.
    theta_d: float = 0.2

    # --- Pathway weights ---
    # GPe → STN  (recurrent inhibition suppressing STN runaway)
    w_gpe_stn: float = 0.5
    # STN → GPe  (excitatory projection from STN to GPe)
    w_stn_gpe: float = 0.5
    # D2 → GPe   (indirect pathway; D2 striatum inhibits GPe)
    w_d2_gpe: float = 0.5
    # STN_mean → GPi  (blanket suppression via mean STN across channels)
    w_stn_gpi: float = 0.7
    # D1 → GPi   (direct pathway; channel-specific disinhibition of thalamus)
    w_d1_gpi: float = 1.0

    # --- Tonic offsets (negative threshold convention: value subtracted from input) ---
    # Positive stn_offset makes STN tonically active even with zero cortical input.
    stn_offset: float = 0.25
    # gpe_offset sustains tonic GPe activity at rest.
    gpe_offset: float = 0.2
    # gpi_offset sustains tonic GPi activity at rest (default suppression of thalamus).
    gpi_offset: float = 0.2
    # Thalamus fires when GPi drops below this level (disinhibition).
    thal_threshold: float = 0.2

    # --- Fixed-point settling ---
    max_iters: int = 1000
    tol: float = 1e-6

    # --- Optional noise ---
    # Additive Gaussian noise on input saliences; 0.0 = deterministic.
    noise_std: float = 0.0

    # --- Latency mapping parameters ---
    # Converts thalamus winner output to selection latency (ms):
    #   latency_ms = latency_min_ms + latency_scale_ms / (T_winner + latency_eps)
    # This gives a monotone inverse relationship: more conflict (smaller T_winner)
    # → longer latency, satisfying the M2 acceptance criterion.
    latency_min_ms: float = 5.0
    latency_scale_ms: float = 2.0
    latency_eps: float = 0.05   # regularizer preventing division by zero near T_winner = 0
    latency_max_ms: float = 100.0  # cap applied when no selection is made


# --- GPR Model Core ---


class BGModel:
    """Gurney-Prescott-Redgrave (2001) rate-coded BG model (steady-state variant).

    Anatomical nuclei modelled per channel i (N channels total):

      Striatum D1_i = max(0, u_i - theta_d)          [direct pathway]
      Striatum D2_i = max(0, u_i - theta_d)          [indirect pathway]
      STN_i  = max(0, u_i - w_gpe_stn*GPe_i  + stn_offset)
      GPe_i  = max(0, w_stn_gpe*STN_i - w_d2_gpe*D2_i + gpe_offset)
      GPi_i  = max(0, w_stn_gpi*mean(STN) - w_d1_gpi*D1_i + gpi_offset)
      T_i    = max(0, thal_threshold - GPi_i)        [thalamus output; > 0 = selected]

    Selection: argmax(T); -1 if max(T) = 0.
    """

    def __init__(self, config: BGModelConfig) -> None:
        self.config = config

    def compute(
        self,
        saliences: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Find the steady-state BG activations for the given salience vector.

        Args:
            saliences: per-channel cortical salience, shape (n_channels,).
            rng:       optional RNG for additive noise (seeded externally for
                       reproducibility); ignored when config.noise_std == 0.

        Returns a dict with:
            selected_channel    int     index of winner; -1 if no channel selected
            decision_margin     float   T_winner - T_runner_up (0.0 if no selection)
            suppression_vector  list    GPi values per channel
            channel_activations list    T (thalamus) values per channel
            n_iters             int     Jacobi iterations until convergence
            T_winner            float   thalamus output of the winning channel
        """
        cfg = self.config
        n = len(saliences)

        # --- Optional noise injection ---
        # Trigger: noise_std > 0 and rng supplied.
        # Why: biological trial-to-trial variability; noise tests selection stability.
        # Outcome: saliences perturbed by Gaussian noise, clipped to ≥ 0.
        if rng is not None and cfg.noise_std > 0.0:
            saliences = saliences + rng.normal(0.0, cfg.noise_std, size=n)
            saliences = np.maximum(0.0, saliences)

        # --- Striatum: D1 and D2 share the same threshold ---
        D1 = np.maximum(0.0, saliences - cfg.theta_d)
        D2 = np.maximum(0.0, saliences - cfg.theta_d)

        # --- Jacobi fixed-point iteration ---
        STN = np.zeros(n)
        GPe = np.zeros(n)
        GPi = np.zeros(n)

        n_iters = 0
        for iteration in range(cfg.max_iters):
            STN_prev = STN.copy()
            GPe_prev = GPe.copy()
            GPi_prev = GPi.copy()

            # STN: cortical salience excites, GPe inhibits (feedback).
            # stn_offset keeps STN tonically active even at zero cortical input.
            STN = np.maximum(0.0, saliences - cfg.w_gpe_stn * GPe_prev + cfg.stn_offset)

            # GPe: STN excites, D2 striatum inhibits.
            GPe = np.maximum(0.0, cfg.w_stn_gpe * STN - cfg.w_d2_gpe * D2 + cfg.gpe_offset)

            # GPi: mean STN provides global blanket suppression; D1 provides channel-
            # specific inhibition. Using mean (not sum) makes selection scale-invariant
            # with N so that adding more channels does not shrink salience requirements.
            GPi = np.maximum(
                0.0,
                cfg.w_stn_gpi * float(np.mean(STN)) - cfg.w_d1_gpi * D1 + cfg.gpi_offset,
            )

            n_iters = iteration + 1

            # Convergence: all nuclei activations stable within tolerance.
            if (
                float(np.max(np.abs(STN - STN_prev))) < cfg.tol
                and float(np.max(np.abs(GPe - GPe_prev))) < cfg.tol
                and float(np.max(np.abs(GPi - GPi_prev))) < cfg.tol
            ):
                break

        # --- Thalamus output ---
        # Trigger: GPi suppresses thalamus below thal_threshold.
        # Why: selection is implemented by disinhibition — the channel whose GPi is
        #      driven below threshold gains positive thalamic output.
        # Outcome: T_i > 0 identifies selected channels; argmax picks the winner.
        T = np.maximum(0.0, cfg.thal_threshold - GPi)

        # --- Winner selection ---
        if float(np.max(T)) > 0.0:
            selected = int(np.argmax(T))
            T_sorted = np.sort(T)[::-1]
            margin = float(T_sorted[0] - T_sorted[1]) if n >= 2 else float(T_sorted[0])
            T_winner = float(T_sorted[0])
        else:
            selected = -1
            margin = 0.0
            T_winner = 0.0

        return {
            "selected_channel": selected,
            "decision_margin": margin,
            "suppression_vector": GPi.tolist(),
            "channel_activations": T.tolist(),
            "n_iters": n_iters,
            "T_winner": T_winner,
        }


# --- BG Adapter (Policy Interface) ---


@dataclass
class BGAdapter:
    """GPR BG model wrapped as a policy callable for the Phase 1 task engines.

    Implements the shared policy interface used by all four engines:
        (trial_log: TrialLog, action_evidence: ActionEvidence) -> BGDecision

    Stop signal handling: when stop_signal_present=True the adapter immediately
    returns selected_channel=-1 without invoking the BG model.  The hyperdirect
    pathway (STN-mediated rapid suppression) is deferred to Phase 3.

    Selection latency is derived from the thalamus winner output using an inverse
    proportionality formula, ensuring latency increases monotonically with conflict.
    """

    config: BGModelConfig = field(default_factory=BGModelConfig)

    def __post_init__(self) -> None:
        self._model = BGModel(self.config)

    def __call__(
        self,
        trial_log: TrialLog,
        action_evidence: ActionEvidence,
    ) -> BGDecision:
        # --- Stop signal override ---
        # Trigger: stop signal arrived before the decision point.
        # Why: hyperdirect pathway produces rapid global GPi excitation; in Phase 2
        #      this is modelled as an immediate policy-level inhibition (-1) rather
        #      than a circuit mechanism.  Phase 3 will add an explicit STN boost.
        # Outcome: BGDecision with selected_channel=-1, full suppression, zero latency.
        if action_evidence.stop_signal_present:
            n = action_evidence.n_channels
            return BGDecision(
                sim_time=action_evidence.sim_time,
                trial_id=action_evidence.trial_id,
                selected_channel=-1,
                decision_margin=0.0,
                suppression_vector=[1.0] * n,
                channel_activations=[0.0] * n,
                selection_latency=0.0,
            )

        # Build RNG from trial seed only when noise is enabled.
        rng: np.random.Generator | None = None
        if self.config.noise_std > 0.0:
            rng = np.random.default_rng(trial_log.seed)

        saliences = np.array(action_evidence.channel_salience, dtype=float)
        result = self._model.compute(saliences, rng=rng)

        # --- Selection latency mapping ---
        # Trigger: convert T_winner to selection latency in seconds.
        # Why: T_winner ∈ (0, thal_threshold]; larger T_winner → less GPi suppression
        #      → easier decision → shorter BG settling time.  Inverse proportionality
        #      guarantees the M2 acceptance criterion (latency monotone with conflict).
        # Outcome: latency_s in [latency_min_ms, latency_max_ms] / 1000.
        T_winner = result["T_winner"]
        cfg = self.config
        if T_winner > 0.0:
            latency_ms = cfg.latency_min_ms + cfg.latency_scale_ms / (T_winner + cfg.latency_eps)
        else:
            # Trigger: no channel reached thalamic threshold.
            # Why: BG failed to decide (high conflict or weak salience); latency is
            #      capped to indicate the system did not commit to an action.
            # Outcome: latency_ms = latency_max_ms.
            latency_ms = cfg.latency_max_ms

        return BGDecision(
            sim_time=action_evidence.sim_time,
            trial_id=action_evidence.trial_id,
            selected_channel=result["selected_channel"],
            decision_margin=result["decision_margin"],
            suppression_vector=result["suppression_vector"],
            channel_activations=result["channel_activations"],
            selection_latency=latency_ms / 1000.0,  # schema field is in seconds
        )
