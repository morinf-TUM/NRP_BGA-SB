import numpy as np

from nrp_bga_sb.bg_model import (
    BGAdapter,
    BGIntegratorState,
    BGModel,
    BGModelConfig,
    selection_latency_s,
)
from nrp_bga_sb.schemas import ActionEvidence, TrialLog


def test_initial_state_reads_out_channel0_zero_margin():
    # Zero state: GPi=0 -> T=thal_threshold for all channels -> argmax=0, margin=0.
    s = BGIntegratorState.initial(2)
    assert s.selected_channel == 0
    assert s.decision_margin == 0.0
    assert s.n_sweeps == 0


def test_step_is_non_idempotent_margin_grows():
    model = BGModel(BGModelConfig())
    sal = np.array([0.65, 0.35])  # medium conflict
    s0 = BGIntegratorState.initial(2)
    s1 = model.step(s0, sal, n_sweeps=1)
    s2 = model.step(s1, sal, n_sweeps=1)
    # One sweep on medium conflict has not cleared the gate; a second sweep has.
    assert s1.decision_margin < 0.05
    assert s2.decision_margin > 0.05
    assert s2.n_sweeps == 2


def test_step_converges_to_compute_fixed_point():
    model = BGModel(BGModelConfig())
    sal = np.array([0.8, 0.2])
    ref = model.compute(sal)
    s = BGIntegratorState.initial(2)
    s = model.step(s, sal, n_sweeps=50)  # well past convergence
    assert s.selected_channel == ref["selected_channel"]
    assert abs(s.decision_margin - ref["decision_margin"]) < 1e-6
    assert abs(s.T_winner - ref["T_winner"]) < 1e-6


def test_compute_output_unchanged_canonical_levels():
    # Lock compute() exactly (M2 anchor): values verified against source.
    model = BGModel(BGModelConfig())
    low = model.compute(np.array([0.8, 0.2]))
    assert low["selected_channel"] == 0
    assert abs(low["T_winner"] - 0.194) < 1e-3
    high = model.compute(np.array([0.55, 0.45]))
    assert high["selected_channel"] == -1


def test_selection_latency_matches_adapter_formula():
    cfg = BGModelConfig()
    # T_winner>0 path and the no-selection cap.
    expected = (cfg.latency_min_ms + cfg.latency_scale_ms / 0.25) / 1000.0
    assert abs(selection_latency_s(cfg, 0.2) - expected) < 1e-12
    assert selection_latency_s(cfg, 0.0) == cfg.latency_max_ms / 1000.0


def test_adapter_still_produces_same_decision():
    # BGAdapter must be unchanged after extracting the latency helper.
    adapter = BGAdapter(BGModelConfig())
    trial = TrialLog(trial_id=1, seed=1, task_type="go_nogo", cue_identity="go", cue_onset_time=0.0)
    ev = ActionEvidence(sim_time=0.1, trial_id=1, n_channels=2, channel_salience=[0.8, 0.2])
    dec = adapter(trial, ev)
    assert dec.selected_channel == 0
    assert abs(dec.selection_latency - 0.013) < 2e-3  # ~13 ms low-conflict latency
