from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

from negmas import ResponseType, SAONegotiator

from .llm_backend import LLMBackend


ISSUE_NAMES = ("destination", "travel_window", "budget")


def offer_tuple_to_dict(offer: tuple[Any, ...] | None) -> dict[str, Any] | None:
    if offer is None:
        return None
    return {issue: value for issue, value in zip(ISSUE_NAMES, offer)}


@dataclass
class EventRecorder:
    events: list[dict[str, Any]] = field(default_factory=list)
    _counter: int = 0

    def add(self, event: dict[str, Any]) -> dict[str, Any]:
        self._counter += 1
        event = dict(event)
        event["event_id"] = self._counter
        self.events.append(event)
        return event


class LLMCalendarNegotiator(SAONegotiator):
    def __init__(
        self,
        llm: LLMBackend,
        recorder: EventRecorder,
        role_name: str,
        profile: dict[str, Any],
        rng: random.Random,
        llm_max_new_tokens: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.recorder = recorder
        self.role_name = role_name
        self.profile = profile
        self.rng = rng
        self.llm_max_new_tokens = llm_max_new_tokens

        self._ranked_outcomes: list[tuple[tuple[Any, ...], float]] = []
        self._offered: set[tuple[Any, ...]] = set()
        self._reservation_value = float(profile["reservation_value"])
        self._concession_exponent = float(profile["concession_exponent"])

    def on_negotiation_start(self, state) -> None:
        outcome_space = self.nmi.outcome_space
        outcomes = list(outcome_space.enumerate())
        ranked = [(o, float(self.preferences(o))) for o in outcomes]
        ranked.sort(key=lambda x: x[1], reverse=True)

        self._ranked_outcomes = ranked
        self._offered.clear()
        super().on_negotiation_start(state)

    def _aspiration(self, relative_time: float) -> float:
        t = max(0.0, min(1.0, float(relative_time)))
        decay = t ** self._concession_exponent
        target = self._reservation_value + (1.0 - self._reservation_value) * (1.0 - decay)
        return max(self._reservation_value, min(1.0, target))

    def _best_offer_given(self, threshold: float) -> tuple[Any, ...]:
        admissible = [
            (offer, utility)
            for offer, utility in self._ranked_outcomes
            if utility >= threshold and offer not in self._offered
        ]
        if admissible:
            return admissible[0][0]

        fresh = [(offer, utility) for offer, utility in self._ranked_outcomes if offer not in self._offered]
        if fresh:
            top_band = fresh[: min(len(fresh), 4)]
            return self.rng.choice(top_band)[0]

        top_band = self._ranked_outcomes[: min(len(self._ranked_outcomes), 4)]
        return self.rng.choice(top_band)[0]

    def _compose_offer_prompt(
        self,
        state,
        offer: tuple[Any, ...],
        utility: float,
        threshold: float,
    ) -> str:
        offer_dict = offer_tuple_to_dict(offer)
        return (
            "당신은 여행 일정 협상 에이전트입니다. 한국어 한 문장으로만 말하세요.\n"
            f"내 이름: {self.role_name}\n"
            f"현재 step: {state.step}, 상대와의 상대시간: {state.relative_time:.3f}\n"
            f"내 제안: {offer_dict}\n"
            f"내 효용: {utility:.3f}, 현재 양보 기준: {threshold:.3f}\n"
            "상대를 설득할 수 있도록 간결하고 정중하게 제안 이유를 말하세요."
        )

    def _compose_response_prompt(
        self,
        state,
        offer: tuple[Any, ...],
        utility: float,
        threshold: float,
        decision: ResponseType,
    ) -> str:
        offer_dict = offer_tuple_to_dict(offer)
        decision_text = "수락" if decision == ResponseType.ACCEPT_OFFER else "거절"
        return (
            "당신은 여행 일정 협상 에이전트입니다. 한국어 한 문장으로만 말하세요.\n"
            f"내 이름: {self.role_name}\n"
            f"현재 step: {state.step}, 상대시간: {state.relative_time:.3f}\n"
            f"상대 제안: {offer_dict}\n"
            f"내 효용: {utility:.3f}, 현재 수락 기준: {threshold:.3f}\n"
            f"결정: {decision_text}\n"
            "결정 이유를 협상 맥락에 맞게 짧게 말하세요."
        )

    def propose(self, state, dest: str | None = None):
        threshold = self._aspiration(state.relative_time)
        offer = self._best_offer_given(threshold=threshold)
        self._offered.add(offer)

        utility = float(self.preferences(offer))
        prompt = self._compose_offer_prompt(state, offer, utility, threshold)
        llm_result = self.llm.generate(prompt, max_new_tokens=self.llm_max_new_tokens)

        self.recorder.add(
            {
                "speaker": self.role_name,
                "kind": "offer",
                "step": int(state.step),
                "relative_time": float(state.relative_time),
                "offer_tuple": list(offer),
                "offer": offer_tuple_to_dict(offer),
                "utility_self": utility,
                "threshold": threshold,
                "decision": None,
                "message": llm_result.text,
                "llm_backend": llm_result.backend,
                "llm_prompt": llm_result.prompt,
                "llm_raw_output": llm_result.raw_text,
            }
        )
        return offer

    def respond(self, state, source: str | None = None):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        utility = float(self.preferences(offer))
        threshold = max(self._reservation_value, self._aspiration(state.relative_time) - 0.03)
        decision = ResponseType.ACCEPT_OFFER if utility >= threshold else ResponseType.REJECT_OFFER

        prompt = self._compose_response_prompt(state, offer, utility, threshold, decision)
        llm_result = self.llm.generate(prompt, max_new_tokens=self.llm_max_new_tokens)

        self.recorder.add(
            {
                "speaker": self.role_name,
                "kind": "response",
                "step": int(state.step),
                "relative_time": float(state.relative_time),
                "offer_tuple": list(offer),
                "offer": offer_tuple_to_dict(offer),
                "utility_self": utility,
                "threshold": threshold,
                "decision": decision.name,
                "message": llm_result.text,
                "llm_backend": llm_result.backend,
                "llm_prompt": llm_result.prompt,
                "llm_raw_output": llm_result.raw_text,
            }
        )
        return decision
