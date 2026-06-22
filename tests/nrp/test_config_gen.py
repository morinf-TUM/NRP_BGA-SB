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
