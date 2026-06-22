"""Adapters between pydantic schemas (nrp_bga_sb.schemas) and the plain-dict
payloads carried by nrp-core JsonDataPacks. Kept trivial on purpose: pydantic
model_dump / construction is the single source of truth for field names."""

from nrp_bga_sb.schemas import ActionEvidence, BGDecision, MotorCommand


def evidence_to_dict(ev: ActionEvidence) -> dict:
    return ev.model_dump()


def evidence_from_dict(d: dict) -> ActionEvidence:
    return ActionEvidence(**d)


def decision_to_dict(d: BGDecision) -> dict:
    return d.model_dump()


def decision_from_dict(d: dict) -> BGDecision:
    return BGDecision(**d)


def motor_to_dict(m: MotorCommand) -> dict:
    return m.model_dump()


def motor_from_dict(d: dict) -> MotorCommand:
    return MotorCommand(**d)
