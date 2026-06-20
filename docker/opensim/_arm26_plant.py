"""In-container Arm26 plant: model build + PD-tracking torque control.

Imported by run_plant.py and validate_plant.py. Requires opensim (4.6).
Confirmed-API note (Task 10.2 Step 1, verified in-container against 4.6):
CoordinateActuator / ScalarActuator.safeDownCast / overrideActuation /
setOverrideActuation / Manager.initialize / Manager.integrate /
Coordinate.getValue / Coordinate.getSpeedValue / Coordinate.setValue /
Coordinate.setSpeedValue / Muscle.set_appliesForce /
Body.getTransformInGround. The Step-1 spike printed `moved True` (a single
CoordinateActuator drove the elbow coordinate from 0.0 to 2.50 rad under a
constant 5 Nm override torque), locking these names before this controller
was written.
"""
from __future__ import annotations

import numpy as np
import opensim as osim

MODEL_PATH = "/opt/nrp/models/arm26.osim"


def _minimum_jerk_scalar(t_ms: float, T_ms: float) -> float:
    """Normalized minimum-jerk displacement: 0 at t=0, 1 at t>=T. Matches reacher.py."""
    if T_ms <= 0.0:
        return 1.0
    tau = min(t_ms / T_ms, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def _minimum_jerk_rate(t_ms: float, T_ms: float) -> float:
    """Time derivative of the minimum-jerk scalar, per millisecond.

    Used as joint-space velocity feedforward so the PD term only corrects
    tracking error rather than driving the whole motion. Without this FF the
    PD command produces large onset torque transients that, at a coarse control
    step, make the low-inertia hand zig-zag (non-monotone cartesian progress).
    Returns 0 outside the movement window (held / settled).
    """
    if T_ms <= 0.0 or t_ms >= T_ms or t_ms < 0.0:
        return 0.0
    tau = t_ms / T_ms
    # d/dt(10 tau^3 - 15 tau^4 + 6 tau^5) with tau = t/T -> divide by T.
    return (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / T_ms


class Arm26Plant:
    """Torque-controlled Arm26. One build per process; simulate() is stateless per call."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        model = osim.Model(MODEL_PATH)

        # --- Resolve driving coordinates ---
        # Trigger: cfg["coordinate_names"] is null.
        # Why: default to every model coordinate so the controller drives all DOFs.
        # Outcome: self.coord_names fixes the ordering of q0/kp/kd/q_target.
        cs = model.getCoordinateSet()
        all_names = [cs.get(i).getName() for i in range(cs.getSize())]
        self.coord_names = cfg["coordinate_names"] or all_names

        # --- Append one CoordinateActuator per driving coordinate ---
        self._act_names = []
        for cname in self.coord_names:
            act = osim.CoordinateActuator(cname)
            act.setName(f"act_{cname}")
            act.setOptimalForce(1.0)
            model.addForce(act)
            self._act_names.append(f"act_{cname}")

        # --- Disable muscle forces (torque-level control; defer muscle excitation) ---
        # Why: this plant injects joint torques directly via the actuators; the
        # native Millard muscles would otherwise add uncontrolled passive/active
        # force. Muscle excitation is a later milestone (YAGNI here).
        fs = model.updForceSet()
        for i in range(fs.getSize()):
            mus = osim.Muscle.safeDownCast(fs.get(i))
            if mus is not None:
                mus.set_appliesForce(False)

        self.model = model
        self.state = model.initSystem()
        self.coords = [model.getCoordinateSet().get(c) for c in self.coord_names]

        # --- End-effector body ---
        # Trigger: cfg["end_effector_body"] is null.
        # Why: default to the last body in the chain (the distal/hand segment).
        # Outcome: self.ee_body is the marker source for planar hand (x, y).
        bs = model.getBodySet()
        self.ee_body = (
            bs.get(cfg["end_effector_body"]) if cfg["end_effector_body"]
            else bs.get(bs.getSize() - 1)
        )

        self.q0 = np.array(cfg["q0"], dtype=float)
        self.kp = np.array(cfg["kp"], dtype=float)
        self.kd = np.array(cfg["kd"], dtype=float)

    def _hand_xy(self, state) -> list[float]:
        """Planar hand position (x, y) in ground.

        Constraint: getTransformInGround requires the state to be realized at
        least to the Position stage. After Manager.initialize / between integrate
        steps the cached stage can be lower (Model), so realize defensively.
        """
        self.model.realizePosition(state)
        p = self.ee_body.getTransformInGround(state).p()
        return [p.get(0), p.get(1)]

    def endpoint_for(self, q_target: list[float]) -> list[float]:
        """FK hand (x,y) at a target posture (used for target_endpoints_xy)."""
        s = self.model.initSystem()
        for coord, q in zip(self.coords, q_target):
            coord.setValue(s, float(q))
        self.model.realizePosition(s)
        return self._hand_xy(s)

    def simulate(self, selected_channel: int, onset_time_ms, gate_gain: float,
                 gate_state: str) -> dict:
        """Run one reach. Returns times_ms / positions_xy / onset_time_ms / selected_channel."""
        dt = self.cfg["dt_ms"]
        T_total = self.cfg["total_duration_ms"]
        T_move = self.cfg["movement_duration_ms"]
        n_steps = int(round(T_total / dt)) + 1
        times_ms = [i * dt for i in range(n_steps)]

        # --- No-movement branch ---
        # Trigger: gate closed or no channel selected.
        # Why: a closed gate / unselected channel means the plant must hold posture.
        # Outcome: return a constant hand position at q0 over the whole window.
        if gate_state == "closed" or selected_channel < 0:
            s = self.model.initSystem()
            for coord, q in zip(self.coords, self.q0):
                coord.setValue(s, float(q))
            self.model.realizePosition(s)
            hold = self._hand_xy(s)
            return {"times_ms": times_ms, "positions_xy": [hold] * n_steps,
                    "onset_time_ms": None, "selected_channel": -1}

        # --- Reach branch ---
        q_target = np.array(self.cfg["q_target"][selected_channel], dtype=float)
        # Partial gate -> short reach: gate_gain scales the commanded amplitude.
        q_ref_end = self.q0 + gate_gain * (q_target - self.q0)
        onset = 0.0 if onset_time_ms is None else float(onset_time_ms)

        # Fresh state at q0, zero velocity.
        s = self.model.initSystem()
        for coord, q in zip(self.coords, self.q0):
            coord.setValue(s, float(q)); coord.setSpeedValue(s, 0.0)
        acts = [osim.ScalarActuator.safeDownCast(self.model.getForceSet().get(n))
                for n in self._act_names]
        for a in acts:
            a.overrideActuation(s, True)
        mgr = osim.Manager(self.model)
        s.setTime(times_ms[0] / 1000.0)
        mgr.initialize(s)

        positions_xy = [self._hand_xy(s)]
        for k in range(1, n_steps):
            t_ms = times_ms[k]
            # Reference posture/velocity from minimum-jerk profile (joint space).
            if t_ms < onset:
                q_ref, qd_ref = self.q0, np.zeros_like(self.q0)
            else:
                sca = _minimum_jerk_scalar(t_ms - onset, T_move)
                q_ref = self.q0 + sca * (q_ref_end - self.q0)
                # Velocity feedforward (rad/s): rate is per-ms, *1000 -> per-s.
                # Lets Kd act on tracking error only, suppressing onset transients.
                rate_ms = _minimum_jerk_rate(t_ms - onset, T_move)
                qd_ref = rate_ms * (q_ref_end - self.q0) * 1000.0
            # Read current q/qd; the state must be at Velocity for getSpeedValue.
            self.model.realizeVelocity(s)
            q = np.array([c.getValue(s) for c in self.coords])
            qd = np.array([c.getSpeedValue(s) for c in self.coords])
            tau = self.kp * (q_ref - q) + self.kd * (qd_ref - qd)
            for a, tval in zip(acts, tau):
                a.setOverrideActuation(s, float(tval))
            # Constraint: setting the override actuation invalidates the actuator
            # Result Measure cache. The persistent Manager.integrate refuses to
            # restart unless that measure is marked valid, which requires the
            # state realized through Acceleration first. (Confirmed in-container:
            # without this realize, the 2nd integrate throws a Simbody
            # Measure::Result::markAsValid stage error.)
            self.model.realizeAcceleration(s)
            s = mgr.integrate(t_ms / 1000.0)
            positions_xy.append(self._hand_xy(s))

        return {"times_ms": times_ms, "positions_xy": positions_xy,
                "onset_time_ms": onset, "selected_channel": selected_channel}
