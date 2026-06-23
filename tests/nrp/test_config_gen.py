from nrp.config_gen import build_config


def test_build_config_sets_bg_timestep():
    cfg = build_config(40.0)
    engines = {e["EngineName"]: e for e in cfg["EngineConfigs"]}
    assert set(engines) == {"cortex", "bg", "thalamus"}
    assert engines["bg"]["EngineTimestep"] == 1.0 / 40.0
    assert engines["cortex"]["EngineTimestep"] == 0.001
    assert cfg["SimulationTimeout"] == 0.3


def test_build_config_5hz_period():
    cfg = build_config(5.0)
    engines = {e["EngineName"]: e for e in cfg["EngineConfigs"]}
    assert engines["bg"]["EngineTimestep"] == 0.2   # 5 Hz -> 200 ms period


def test_build_config_sampled_independent_knobs():
    from nrp.config_gen import build_config_sampled
    cfg = build_config_sampled(input_sampling_hz=5.0, output_emission_hz=40.0)
    engines = {e["EngineName"]: e for e in cfg["EngineConfigs"]}
    assert set(engines) == {"cortex", "sampler", "bg", "thalamus"}
    assert engines["sampler"]["EngineTimestep"] == 0.2     # 5 Hz sampling
    assert engines["bg"]["EngineTimestep"] == 1.0 / 40.0   # 40 Hz emission
    tf_names = {t["Name"] for t in cfg["DataPackProcessingFunctions"]}
    assert {"cortex_to_sampler", "sampler_to_bg"} <= tf_names
    assert "cortex_to_bg" not in tf_names


def test_build_config_committed_three_rates():
    from nrp.config_gen import build_config_committed
    cfg = build_config_committed(input_sampling_hz=5.0, output_emission_hz=40.0,
                                 commitment_hz=10.0)
    engines = {e["EngineName"]: e for e in cfg["EngineConfigs"]}
    assert set(engines) == {"cortex", "sampler", "bg", "commitment", "thalamus"}
    assert engines["commitment"]["EngineTimestep"] == 0.1   # 10 Hz
    tf_names = {t["Name"] for t in cfg["DataPackProcessingFunctions"]}
    assert {"bg_to_commitment", "commitment_to_thalamus"} <= tf_names
    assert "bg_to_thalamus" not in tf_names


def test_four_knob_maps_all_rates():
    from nrp.config_gen import build_config_four_knob
    cfg, overlay = build_config_four_knob(
        input_sampling_hz=20.0, integration_hz=80.0,
        output_emission_hz=40.0, commitment_hz=10.0)
    engines = {e["EngineName"]: e for e in cfg["EngineConfigs"]}
    assert engines["sampler"]["EngineTimestep"] == 0.05      # 20 Hz
    assert engines["bg"]["EngineTimestep"] == 1.0 / 40.0     # 40 Hz emission
    assert engines["commitment"]["EngineTimestep"] == 0.1    # 10 Hz
    assert overlay["integration_hz"] == 80.0           # rate passed through to the engine driver
