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
