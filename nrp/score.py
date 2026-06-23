"""Offline scoring of an NRPCoreSim trace. The runtime produces decision/motor
records; outcome classification reuses the same notion of "released" as the
pure-Python ThalamusGate (open/partial gate with a non-zero command)."""

from __future__ import annotations


def trace_to_outcome(trace: list[dict]) -> dict:
    motor_released = False
    first_release_time = None
    selected_channel = -1
    for rec in trace:
        d = rec.get("decision")
        if d and d.get("selected_channel", -1) >= 0:
            selected_channel = d["selected_channel"]
        m = rec.get("motor")
        if m and m.get("gate_state") in ("open", "partial") and any(m.get("command", [])):
            if not motor_released:
                first_release_time = m.get("sim_time")
            motor_released = True
    return {
        "motor_released": motor_released,
        "selected_channel": selected_channel,
        "first_release_time": first_release_time,
    }
