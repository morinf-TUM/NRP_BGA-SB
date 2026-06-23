"""Build NRPCoreSim simulation configs for the go/no-go binding. Phase 1 wires
three engines (cortex, bg, thalamus). The BG frequency is the single swept knob;
later phases add sampler/commitment engines and additional EngineTimesteps."""

from __future__ import annotations

CORTEX_HZ = 1000.0      # finest resolution -> sets the FTILoop base step (1 ms)
THALAMUS_HZ = 1000.0


def build_config(bg_hz: float, *, name: str = "gonogo") -> dict:
    return {
        "SimulationName": name,
        "SimulationDescription": f"go/no-go BG binding, bg={bg_hz} Hz (Phase 1).",
        "SimulationTimeout": 0.3,   # 300 ms: covers the 200 ms accumulation window
        "EngineConfigs": [
            {"EngineType": "python_json", "EngineName": "cortex",
             "EngineTimestep": 1.0 / CORTEX_HZ,
             "PythonFileName": "nrp/engines/cortex_engine.py"},
            {"EngineType": "python_json", "EngineName": "bg",
             "EngineTimestep": 1.0 / bg_hz,
             "PythonFileName": "nrp/engines/bg_engine.py"},
            {"EngineType": "python_json", "EngineName": "thalamus",
             "EngineTimestep": 1.0 / THALAMUS_HZ,
             "PythonFileName": "nrp/engines/thalamus_engine.py"},
        ],
        "DataPackProcessingFunctions": [
            {"Name": "cortex_to_bg", "FileName": "nrp/tfs/tf_cortex_to_bg.py"},
            {"Name": "bg_to_thalamus", "FileName": "nrp/tfs/tf_bg_to_thalamus.py"},
            {"Name": "log_step", "FileName": "nrp/tfs/tf_log.py"},
        ],
    }


def build_config_sampled(*, input_sampling_hz: float, output_emission_hz: float,
                         name: str = "gonogo_s") -> dict:
    return {
        "SimulationName": name,
        "SimulationDescription": (
            f"go/no-go, input_sampling={input_sampling_hz} Hz, "
            f"output_emission={output_emission_hz} Hz (Phase 3)."),
        "SimulationTimeout": 0.3,
        "EngineConfigs": [
            {"EngineType": "python_json", "EngineName": "cortex",
             "EngineTimestep": 1.0 / CORTEX_HZ,
             "PythonFileName": "nrp/engines/cortex_engine.py"},
            {"EngineType": "python_json", "EngineName": "sampler",
             "EngineTimestep": 1.0 / input_sampling_hz,
             "PythonFileName": "nrp/engines/sampler_engine.py"},
            {"EngineType": "python_json", "EngineName": "bg",
             "EngineTimestep": 1.0 / output_emission_hz,
             "PythonFileName": "nrp/engines/bg_engine.py"},
            {"EngineType": "python_json", "EngineName": "thalamus",
             "EngineTimestep": 1.0 / THALAMUS_HZ,
             "PythonFileName": "nrp/engines/thalamus_engine.py"},
        ],
        "DataPackProcessingFunctions": [
            {"Name": "cortex_to_sampler", "FileName": "nrp/tfs/tf_cortex_to_sampler.py"},
            {"Name": "sampler_to_bg", "FileName": "nrp/tfs/tf_sampler_to_bg.py"},
            {"Name": "bg_to_thalamus", "FileName": "nrp/tfs/tf_bg_to_thalamus.py"},
            {"Name": "log_step", "FileName": "nrp/tfs/tf_log.py"},
        ],
    }


def build_config_committed(*, input_sampling_hz: float, output_emission_hz: float,
                           commitment_hz: float, name: str = "gonogo_c") -> dict:
    return {
        "SimulationName": name,
        "SimulationDescription": (
            f"go/no-go, sampling={input_sampling_hz} Hz, emission={output_emission_hz} Hz, "
            f"commitment={commitment_hz} Hz (Phase 4)."),
        "SimulationTimeout": 0.3,
        "EngineConfigs": [
            {"EngineType": "python_json", "EngineName": "cortex",
             "EngineTimestep": 1.0 / CORTEX_HZ,
             "PythonFileName": "nrp/engines/cortex_engine.py"},
            {"EngineType": "python_json", "EngineName": "sampler",
             "EngineTimestep": 1.0 / input_sampling_hz,
             "PythonFileName": "nrp/engines/sampler_engine.py"},
            {"EngineType": "python_json", "EngineName": "bg",
             "EngineTimestep": 1.0 / output_emission_hz,
             "PythonFileName": "nrp/engines/bg_engine.py"},
            {"EngineType": "python_json", "EngineName": "commitment",
             "EngineTimestep": 1.0 / commitment_hz,
             "PythonFileName": "nrp/engines/commitment_engine.py"},
            {"EngineType": "python_json", "EngineName": "thalamus",
             "EngineTimestep": 1.0 / THALAMUS_HZ,
             "PythonFileName": "nrp/engines/thalamus_engine.py"},
        ],
        "DataPackProcessingFunctions": [
            {"Name": "cortex_to_sampler", "FileName": "nrp/tfs/tf_cortex_to_sampler.py"},
            {"Name": "sampler_to_bg", "FileName": "nrp/tfs/tf_sampler_to_bg.py"},
            {"Name": "bg_to_commitment", "FileName": "nrp/tfs/tf_bg_to_commitment.py"},
            {"Name": "commitment_to_thalamus", "FileName": "nrp/tfs/tf_commitment_to_thalamus.py"},
            {"Name": "log_step", "FileName": "nrp/tfs/tf_log.py"},
        ],
    }


def build_config_four_knob(*, input_sampling_hz: float, integration_hz: float,
                           output_emission_hz: float, commitment_hz: float,
                           name: str = "gonogo_4k") -> tuple[dict, dict]:
    cfg = build_config_committed(
        input_sampling_hz=input_sampling_hz,
        output_emission_hz=output_emission_hz,
        commitment_hz=commitment_hz, name=name)
    # Knob 2 rides on the BG engine via a params overlay (internal sub-steps),
    # not as an EngineTimestep. substeps = how many integration steps per emission.
    substeps = max(1, round(integration_hz / output_emission_hz))
    return cfg, {"integration_substeps": substeps}
