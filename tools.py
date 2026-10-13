"""Bounded mock tools and CPU dispatcher. Nothing here controls a vehicle."""
import re
from dataclasses import dataclass, field

INTENTS = ["NO_TOOL", "SET_TEMPERATURE", "SET_VOLUME", "OPEN_TRUNK", "START_NAVIGATION", "UNKNOWN"]
GUARD = ["ALLOW", "REVIEW", "BLOCK"]


@dataclass
class MockState:
    motion: str = "stopped"  # stopped, moving, unknown
    permission: str = "allowed"  # allowed, denied, unknown
    temperature_c: int = 22
    volume: int = 5
    trunk_open: bool = False
    destination: str | None = None
    calls: list[dict] = field(default_factory=list)


def extract_args(intent: str, utterance: str) -> dict:
    """Deterministic slots for the pilot; ambiguity becomes a clarification."""
    if intent == "SET_TEMPERATURE":
        matches = re.findall(r"(?<!\d)(\d{1,2})\s*度", utterance)
        if len(matches) != 1:
            raise ValueError("temperature missing or ambiguous")
        return {"temperature_c": int(matches[0])}
    if intent == "SET_VOLUME":
        matches = re.findall(r"(?<!\d)(\d{1,2})\s*(?:段階|に)", utterance)
        if len(matches) != 1:
            raise ValueError("volume missing or ambiguous")
        return {"volume": int(matches[0])}
    if intent == "OPEN_TRUNK":
        return {}
    if intent == "START_NAVIGATION":
        match = re.search(r"([一-龥ぁ-んァ-ヶA-Za-z0-9]{2,20})(?:まで|への)(?:案内|ナビ|道案内)", utterance)
        if not match:
            raise ValueError("destination missing or ambiguous")
        return {"destination": match.group(1)}
    raise ValueError("unsupported tool intent")


def validate(intent: str, args: dict, state: MockState) -> str:
    if state.permission == "denied":
        return "BLOCK"
    if state.permission != "allowed":
        return "REVIEW"
    if intent == "OPEN_TRUNK":
        if state.motion == "moving":
            return "BLOCK"
        if state.motion != "stopped":
            return "REVIEW"
        if args:
            return "BLOCK"
    elif intent == "SET_TEMPERATURE":
        if set(args) != {"temperature_c"} or type(args["temperature_c"]) is not int or not 16 <= args["temperature_c"] <= 30:
            return "BLOCK"
    elif intent == "SET_VOLUME":
        if set(args) != {"volume"} or type(args["volume"]) is not int or not 0 <= args["volume"] <= 10:
            return "BLOCK"
    elif intent == "START_NAVIGATION":
        if set(args) != {"destination"} or not isinstance(args["destination"], str) or not 2 <= len(args["destination"]) <= 20:
            return "BLOCK"
    else:
        return "BLOCK"
    return "ALLOW"


def dispatch(intent: str, utterance: str, head_guard: str, state: MockState,
             action_guard: str | None = None) -> dict:
    """Head output alone cannot trigger a call; CPU policy is the final gate."""
    if intent in {"NO_TOOL", "UNKNOWN"}:
        return {"status": "DIALOGUE" if intent == "NO_TOOL" else "REVIEW", "called": False}
    if head_guard != "ALLOW":
        return {"status": head_guard, "called": False}
    try:
        args = extract_args(intent, utterance)
    except ValueError as error:
        return {"status": "REVIEW", "called": False, "reason": str(error)}
    if action_guard is not None and action_guard != "ALLOW":
        return {"status": action_guard, "called": False, "reason": "action guard"}
    policy = validate(intent, args, state)
    if policy != "ALLOW":
        return {"status": policy, "called": False}
    # Only the four mock actions below are executable.
    if intent == "SET_TEMPERATURE":
        state.temperature_c = args["temperature_c"]
    elif intent == "SET_VOLUME":
        state.volume = args["volume"]
    elif intent == "OPEN_TRUNK":
        state.trunk_open = True
    elif intent == "START_NAVIGATION":
        state.destination = args["destination"]
    call = {"tool": intent, "args": args}
    state.calls.append(call)
    return {"status": "EXECUTED", "called": True, "call": call,
            "response": f"模擬実行しました: {call}"}
