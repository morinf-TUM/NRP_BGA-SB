"""Tests for the cue generator module.

Coverage:
- Determinism: same args always produce the same CueSequence
- Independence: different task_types produce different sequences
- CueSequence validation: trial_seeds length must match n_trials
- shared_seed_configs: returns correct number of configs with updated seeds
- Round-trip: shared_seed_configs configs work with actual engines
"""


import pytest

from nrp_bga_sb.cue_generator import CueSequence, generate_cue_sequence, shared_seed_configs
from nrp_bga_sb.engines.change_of_mind import ChangeOfMindConfig
from nrp_bga_sb.engines.go_nogo import GoNoGoConfig
from nrp_bga_sb.engines.stop_signal import StopSignalConfig
from nrp_bga_sb.engines.two_choice import TwoChoiceConfig

# --- Determinism tests ---


class TestDeterminism:
    """Verify that generate_cue_sequence is deterministic."""

    def test_same_args_produce_same_seeds(self):
        """Calling generate_cue_sequence twice with the same args returns identical sequences."""
        seq1 = generate_cue_sequence(master_seed=42, task_type="go_nogo", n_trials=100)
        seq2 = generate_cue_sequence(master_seed=42, task_type="go_nogo", n_trials=100)

        assert seq1.trial_seeds == seq2.trial_seeds
        assert seq1.master_seed == seq2.master_seed
        assert seq1.task_type == seq2.task_type
        assert seq1.n_trials == seq2.n_trials

    def test_different_master_seeds_produce_different_sequences(self):
        """Different master_seed values produce different trial seed lists."""
        seq_42 = generate_cue_sequence(master_seed=42, task_type="go_nogo", n_trials=100)
        seq_99 = generate_cue_sequence(master_seed=99, task_type="go_nogo", n_trials=100)

        assert seq_42.trial_seeds != seq_99.trial_seeds

    def test_different_n_trials_produce_different_lengths(self):
        """Different n_trials values produce sequences of different lengths."""
        seq_100 = generate_cue_sequence(master_seed=42, task_type="go_nogo", n_trials=100)
        seq_200 = generate_cue_sequence(master_seed=42, task_type="go_nogo", n_trials=200)

        assert len(seq_100.trial_seeds) == 100
        assert len(seq_200.trial_seeds) == 200

    def test_seeds_are_process_independent(self):
        """Verify trial_seeds are stable across processes (not PYTHONHASHSEED-dependent).

        This test catches any regression to hash() which is randomized per-process.
        The expected seed values are hardcoded from the sha256-based implementation.
        """
        seq = generate_cue_sequence(master_seed=42, task_type="go_nogo", n_trials=5)

        # These values are hardcoded after computing once with the sha256
        # implementation. If the code reverts to hash(), these will differ.
        assert len(seq.trial_seeds) == 5
        assert seq.trial_seeds[0] == 421241054
        assert seq.trial_seeds == [421241054, 1315106536, 433368447, 1162719129, 1358753899]

        # Verify re-running produces the same sequence (in-process determinism)
        seq2 = generate_cue_sequence(master_seed=42, task_type="go_nogo", n_trials=5)
        assert seq2.trial_seeds[0] == seq.trial_seeds[0]
        assert seq2.trial_seeds == seq.trial_seeds

        # Verify different task types give different first seeds (independence)
        seq_tc = generate_cue_sequence(master_seed=42, task_type="two_choice", n_trials=5)
        assert seq_tc.trial_seeds[0] != seq.trial_seeds[0]


# --- Independence tests ---


class TestTaskTypeIndependence:
    """Verify that different task_types produce different sequences."""

    def test_same_master_seed_different_task_types(self):
        """Same master_seed with different task_types produces different sequences."""
        seq_go = generate_cue_sequence(master_seed=42, task_type="go_nogo", n_trials=100)
        seq_two = generate_cue_sequence(master_seed=42, task_type="two_choice", n_trials=100)
        seq_stop = generate_cue_sequence(master_seed=42, task_type="stop_signal", n_trials=100)
        seq_com = generate_cue_sequence(master_seed=42, task_type="change_of_mind", n_trials=100)

        # All four sequences should be different from each other
        assert seq_go.trial_seeds != seq_two.trial_seeds
        assert seq_go.trial_seeds != seq_stop.trial_seeds
        assert seq_go.trial_seeds != seq_com.trial_seeds
        assert seq_two.trial_seeds != seq_stop.trial_seeds
        assert seq_two.trial_seeds != seq_com.trial_seeds
        assert seq_stop.trial_seeds != seq_com.trial_seeds

    def test_all_four_task_types_accepted(self):
        """All four canonical task types are accepted."""
        for task_type in ["go_nogo", "two_choice", "stop_signal", "change_of_mind"]:
            seq = generate_cue_sequence(master_seed=42, task_type=task_type, n_trials=50)
            assert seq.task_type == task_type


# --- CueSequence model validation tests ---


class TestCueSequenceValidation:
    """Verify CueSequence construction and validation."""

    def test_construct_valid_cue_sequence(self):
        """Constructing a valid CueSequence works."""
        trial_seeds = [1, 2, 3, 4, 5]
        seq = CueSequence(
            master_seed=42,
            task_type="go_nogo",
            n_trials=5,
            trial_seeds=trial_seeds,
        )
        assert seq.master_seed == 42
        assert seq.task_type == "go_nogo"
        assert seq.n_trials == 5
        assert seq.trial_seeds == trial_seeds

    def test_trial_seeds_length_mismatch_raises_error(self):
        """Mismatched trial_seeds length vs n_trials raises ValueError."""
        with pytest.raises(ValueError, match="trial_seeds length .* does not match"):
            CueSequence(
                master_seed=42,
                task_type="go_nogo",
                n_trials=5,
                trial_seeds=[1, 2, 3],  # only 3, not 5
            )

    def test_cue_sequence_frozen(self):
        """CueSequence is frozen and cannot be modified after construction."""
        seq = generate_cue_sequence(master_seed=42, task_type="go_nogo", n_trials=5)
        with pytest.raises((TypeError, ValueError)):  # Pydantic v2 frozen raises on assignment
            seq.master_seed = 99

    def test_invalid_task_type_raises_error(self):
        """Invalid task_type values are rejected."""
        with pytest.raises(ValueError):  # Pydantic v2 Literal validation
            CueSequence(
                master_seed=42,
                task_type="invalid_task",
                n_trials=5,
                trial_seeds=[1, 2, 3, 4, 5],
            )

    def test_cue_sequence_json_roundtrip(self):
        """CueSequence can be dumped and loaded from JSON."""
        seq = generate_cue_sequence(master_seed=42, task_type="two_choice", n_trials=10)
        json_str = seq.model_dump_json()
        seq_restored = CueSequence.model_validate_json(json_str)

        assert seq.master_seed == seq_restored.master_seed
        assert seq.task_type == seq_restored.task_type
        assert seq.n_trials == seq_restored.n_trials
        assert seq.trial_seeds == seq_restored.trial_seeds


# --- shared_seed_configs tests ---


class TestSharedSeedConfigs:
    """Verify the shared_seed_configs convenience function."""

    def test_go_nogo_config_replication(self):
        """shared_seed_configs returns one GoNoGoConfig per frequency with updated seed."""
        base_config = GoNoGoConfig(
            n_trials=200,
            go_probability=0.5,
            response_window_start_ms=100,
            response_window_duration_ms=1000,
            fixation_duration_ms=500,
            cue_onset_ms=300,
            decision_point_ms=400,
            seed=999,  # will be overwritten
        )

        cue_seq = generate_cue_sequence(master_seed=42, task_type="go_nogo", n_trials=200)
        frequencies = [10.0, 20.0, 40.0]

        configs = shared_seed_configs(cue_seq, base_config, frequencies)

        # One config per frequency
        assert len(configs) == 3

        # All configs have the cue_seq's master_seed
        for config in configs:
            assert config.seed == cue_seq.master_seed

        # All other fields are preserved
        for config in configs:
            assert config.n_trials == base_config.n_trials
            assert config.go_probability == base_config.go_probability
            assert config.response_window_start_ms == base_config.response_window_start_ms

    def test_two_choice_config_replication(self):
        """shared_seed_configs works with TwoChoiceConfig."""
        base_config = TwoChoiceConfig(
            n_trials=250,
            conflict_levels={"low": [0.8, 0.2], "high": [0.55, 0.45]},
            response_window_start_ms=100,
            response_window_duration_ms=1000,
            fixation_duration_ms=500,
            target_onset_ms=300,
            decision_point_ms=400,
            seed=999,
        )

        cue_seq = generate_cue_sequence(master_seed=99, task_type="two_choice", n_trials=250)
        frequencies = [10.0, 160.0]

        configs = shared_seed_configs(cue_seq, base_config, frequencies)

        assert len(configs) == 2
        for config in configs:
            assert config.seed == cue_seq.master_seed
            assert config.conflict_levels == base_config.conflict_levels

    def test_stop_signal_config_replication(self):
        """shared_seed_configs works with StopSignalConfig."""
        base_config = StopSignalConfig(
            n_trials=500,
            stop_proportion=0.2,
            initial_ssd_ms=100,
            ssd_step_ms=50,
            ssd_min_ms=0,
            ssd_max_ms=500,
            use_staircase=True,
            response_window_duration_ms=1000,
            fixation_duration_ms=500,
            go_cue_onset_ms=300,
            decision_point_ms=400,
            seed=999,
        )

        cue_seq = generate_cue_sequence(master_seed=77, task_type="stop_signal", n_trials=500)
        frequencies = [10.0, 40.0, 80.0]

        configs = shared_seed_configs(cue_seq, base_config, frequencies)

        assert len(configs) == 3
        for config in configs:
            assert config.seed == cue_seq.master_seed

    def test_change_of_mind_config_replication(self):
        """shared_seed_configs works with ChangeOfMindConfig."""
        base_config = ChangeOfMindConfig(
            n_trials=400,
            no_switch_proportion=0.5,
            switch_delay_categories={"short": 150, "long": 300},
            response_window_duration_ms=1000,
            fixation_duration_ms=500,
            go_cue_onset_ms=300,
            initial_decision_point_ms=100,
            post_switch_decision_point_ms=400,
            initial_salience=[0.7, 0.3],
            post_switch_salience=[0.3, 0.7],
            seed=999,
        )

        cue_seq = generate_cue_sequence(
            master_seed=55, task_type="change_of_mind", n_trials=400
        )
        frequencies = [20.0, 160.0]

        configs = shared_seed_configs(cue_seq, base_config, frequencies)

        assert len(configs) == 2
        for config in configs:
            assert config.seed == cue_seq.master_seed
            assert config.switch_delay_categories == base_config.switch_delay_categories

    def test_empty_frequency_list(self):
        """shared_seed_configs handles empty frequency list."""
        base_config = GoNoGoConfig(
            n_trials=100,
            go_probability=0.5,
            response_window_start_ms=100,
            response_window_duration_ms=1000,
            fixation_duration_ms=500,
            cue_onset_ms=300,
            decision_point_ms=400,
            seed=999,
        )

        cue_seq = generate_cue_sequence(master_seed=42, task_type="go_nogo", n_trials=100)
        configs = shared_seed_configs(cue_seq, base_config, [])

        assert len(configs) == 0

    def test_single_frequency(self):
        """shared_seed_configs works with a single frequency."""
        base_config = GoNoGoConfig(
            n_trials=100,
            go_probability=0.5,
            response_window_start_ms=100,
            response_window_duration_ms=1000,
            fixation_duration_ms=500,
            cue_onset_ms=300,
            decision_point_ms=400,
            seed=999,
        )

        cue_seq = generate_cue_sequence(master_seed=42, task_type="go_nogo", n_trials=100)
        configs = shared_seed_configs(cue_seq, base_config, [40.0])

        assert len(configs) == 1
        assert configs[0].seed == cue_seq.master_seed


# --- Round-trip tests with actual engines ---


class TestRoundTripWithEngines:
    """Verify that configs from shared_seed_configs work with actual engines."""

    def test_go_nogo_round_trip(self):
        """Configs from shared_seed_configs can be used to run go_nogo trials."""
        from nrp_bga_sb.engines.go_nogo import run_go_nogo_trials
        from nrp_bga_sb.policies import oracle_policy

        base_config = GoNoGoConfig(
            n_trials=10,
            go_probability=0.5,
            response_window_start_ms=100,
            response_window_duration_ms=1000,
            fixation_duration_ms=500,
            cue_onset_ms=300,
            decision_point_ms=400,
            seed=42,
        )

        cue_seq = generate_cue_sequence(master_seed=42, task_type="go_nogo", n_trials=10)
        configs = shared_seed_configs(cue_seq, base_config, [10.0, 20.0])

        # Both configs should work to run trials
        for config in configs:
            trials = run_go_nogo_trials(config, oracle_policy, logger=None)
            assert len(trials) == 10

    def test_two_choice_round_trip(self):
        """Configs from shared_seed_configs can be used to run two_choice trials."""
        from nrp_bga_sb.engines.two_choice import run_two_choice_trials
        from nrp_bga_sb.policies import oracle_policy

        base_config = TwoChoiceConfig(
            n_trials=10,
            conflict_levels={"medium": [0.65, 0.35]},
            response_window_start_ms=100,
            response_window_duration_ms=1000,
            fixation_duration_ms=500,
            target_onset_ms=300,
            decision_point_ms=400,
            seed=42,
        )

        cue_seq = generate_cue_sequence(master_seed=42, task_type="two_choice", n_trials=10)
        configs = shared_seed_configs(cue_seq, base_config, [10.0])

        trials = run_two_choice_trials(configs[0], oracle_policy, logger=None)
        assert len(trials) == 10

    def test_stop_signal_round_trip(self):
        """Configs from shared_seed_configs can be used to run stop_signal trials."""
        from nrp_bga_sb.engines.stop_signal import run_stop_signal_trials
        from nrp_bga_sb.policies import oracle_policy

        base_config = StopSignalConfig(
            n_trials=15,
            stop_proportion=0.2,
            initial_ssd_ms=100,
            ssd_step_ms=50,
            ssd_min_ms=0,
            ssd_max_ms=500,
            use_staircase=True,
            response_window_duration_ms=1000,
            fixation_duration_ms=500,
            go_cue_onset_ms=300,
            decision_point_ms=400,
            seed=42,
        )

        cue_seq = generate_cue_sequence(master_seed=42, task_type="stop_signal", n_trials=15)
        configs = shared_seed_configs(cue_seq, base_config, [10.0])

        trials = run_stop_signal_trials(configs[0], oracle_policy, logger=None)
        assert len(trials) == 15

    def test_change_of_mind_round_trip(self):
        """Configs from shared_seed_configs can be used to run change_of_mind trials."""
        from nrp_bga_sb.engines.change_of_mind import run_change_of_mind_trials
        from nrp_bga_sb.policies import oracle_policy

        base_config = ChangeOfMindConfig(
            n_trials=20,
            no_switch_proportion=0.5,
            switch_delay_categories={"short": 150, "long": 300},
            response_window_duration_ms=1000,
            fixation_duration_ms=500,
            go_cue_onset_ms=300,
            initial_decision_point_ms=100,
            post_switch_decision_point_ms=400,
            initial_salience=[0.7, 0.3],
            post_switch_salience=[0.3, 0.7],
            seed=42,
        )

        cue_seq = generate_cue_sequence(
            master_seed=42, task_type="change_of_mind", n_trials=20
        )
        configs = shared_seed_configs(cue_seq, base_config, [10.0, 40.0])

        for config in configs:
            trials = run_change_of_mind_trials(config, oracle_policy, logger=None)
            assert len(trials) == 20
