from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import random
import re
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
        decision_policy: str = "llm-hybrid",
        offer_top_k: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.recorder = recorder
        self.role_name = role_name
        self.profile = profile
        self.rng = rng
        self.llm_max_new_tokens = llm_max_new_tokens
        self.decision_policy = decision_policy
        self.offer_top_k = max(1, int(offer_top_k))

        if self.decision_policy not in {"heuristic", "llm-hybrid", "llm-only"}:
            raise ValueError(f"Unsupported decision_policy: {self.decision_policy}")

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

    def _offer_candidates(self, threshold: float) -> list[tuple[tuple[Any, ...], float]]:
        fresh = [(offer, utility) for offer, utility in self._ranked_outcomes if offer not in self._offered]
        if not fresh:
            fresh = list(self._ranked_outcomes)

        strong = [(offer, utility) for offer, utility in fresh if utility >= threshold]
        if strong:
            pool = strong
        else:
            # Keep near-threshold options to allow model preference differences to matter.
            margin = 0.18
            near = [(offer, utility) for offer, utility in fresh if utility >= max(self._reservation_value, threshold - margin)]
            pool = near if near else fresh

        pool.sort(key=lambda x: x[1], reverse=True)
        return pool[: min(self.offer_top_k, len(pool))]

    @staticmethod
    def _stable_hash_int(text: str) -> int:
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:8]
        return int(digest, 16)

    @staticmethod
    def _extract_message(text: str) -> str:
        m = re.search(r"message\s*[:=]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""
        filtered = [
            ln
            for ln in lines
            if not re.match(r"^(choice|action)\s*[:=]", ln, flags=re.IGNORECASE)
        ]
        return filtered[0] if filtered else lines[0]

    @staticmethod
    def _parse_choice(text: str, n_candidates: int) -> int | None:
        m = re.search(r"choice\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
        if not m:
            return None
        idx = int(m.group(1))
        if 1 <= idx <= n_candidates:
            return idx
        return None

    @staticmethod
    def _parse_action(text: str) -> ResponseType | None:
        lowered = text.lower()

        if re.search(r"action\s*[:=]", lowered):
            m = re.search(r"action\s*[:=]\s*([a-z_가-힣]+)", lowered)
            token = m.group(1) if m else ""
        else:
            token = lowered

        if any(k in token for k in ("accept", "수락", "agree", "동의", "yes")):
            return ResponseType.ACCEPT_OFFER
        if any(k in token for k in ("reject", "거절", "decline", "no")):
            return ResponseType.REJECT_OFFER
        return None

    def _compose_offer_prompt(
        self,
        state,
        candidates: list[tuple[tuple[Any, ...], float]],
        threshold: float,
    ) -> str:
        lines = [
            "당신은 여행 일정 협상 에이전트입니다.",
            f"내 이름: {self.role_name}",
            f"현재 step: {state.step}, 상대시간: {state.relative_time:.3f}",
            f"내 현재 제안 기준(aspiration): {threshold:.3f}",
            "아래 후보 중 하나를 선택하세요.",
        ]
        for idx, (offer, utility) in enumerate(candidates, start=1):
            lines.append(f"{idx}) offer={offer_tuple_to_dict(offer)} | utility={utility:.3f}")

        lines.extend(
            [
                "반드시 아래 형식으로 답하세요:",
                "CHOICE: <번호>",
                "MESSAGE: <상대에게 보낼 한국어 한 문장>",
            ]
        )
        return "\n".join(lines)

    def _compose_response_prompt(
        self,
        state,
        offer: tuple[Any, ...],
        utility: float,
        threshold: float,
    ) -> str:
        return (
            "당신은 여행 일정 협상 에이전트입니다.\n"
            f"내 이름: {self.role_name}\n"
            f"현재 step: {state.step}, 상대시간: {state.relative_time:.3f}\n"
            f"상대 제안: {offer_tuple_to_dict(offer)}\n"
            f"내 효용: {utility:.3f}\n"
            f"현재 수락 기준: {threshold:.3f}\n"
            "반드시 아래 형식으로 답하세요:\n"
            "ACTION: ACCEPT or REJECT\n"
            "MESSAGE: <상대에게 보낼 한국어 한 문장>"
        )

    def _fallback_choice_from_text(self, text: str, n_candidates: int) -> int:
        return (self._stable_hash_int(text or self.role_name) % max(n_candidates, 1)) + 1

    def _resolve_decision(
        self,
        parsed_action: ResponseType | None,
        utility: float,
        threshold: float,
        llm_text: str,
    ) -> tuple[ResponseType, str]:
        heuristic = ResponseType.ACCEPT_OFFER if utility >= threshold else ResponseType.REJECT_OFFER

        if self.decision_policy == "heuristic":
            return heuristic, "heuristic"

        if parsed_action is None:
            if self.decision_policy == "llm-only":
                hashed = self._stable_hash_int(llm_text)
                guessed = ResponseType.ACCEPT_OFFER if hashed % 2 == 0 else ResponseType.REJECT_OFFER
                return guessed, "llm-hash-fallback"

            # llm-hybrid fallback: near boundary use LLM hash signal, otherwise heuristic.
            if abs(utility - threshold) <= 0.10:
                hashed = self._stable_hash_int(llm_text)
                guessed = ResponseType.ACCEPT_OFFER if hashed % 2 == 0 else ResponseType.REJECT_OFFER
                return guessed, "llm-hash-near-boundary"
            return heuristic, "heuristic-fallback"

        if self.decision_policy == "llm-only":
            return parsed_action, "llm-only"

        # llm-hybrid: keep minimal rational guardrails.
        if utility <= self._reservation_value - 0.08:
            return ResponseType.REJECT_OFFER, "hybrid-guardrail-low-utility"
        if utility >= threshold + 0.08:
            return ResponseType.ACCEPT_OFFER, "hybrid-guardrail-high-utility"
        return parsed_action, "llm-hybrid"

    def propose(self, state, dest: str | None = None):
        threshold = self._aspiration(state.relative_time)
        candidates = self._offer_candidates(threshold=threshold)

        prompt = self._compose_offer_prompt(state, candidates, threshold)
        llm_result = self.llm.generate(prompt, max_new_tokens=self.llm_max_new_tokens)

        parsed_choice = self._parse_choice(llm_result.text, n_candidates=len(candidates))
        if parsed_choice is None:
            parsed_choice = self._fallback_choice_from_text(llm_result.text, len(candidates))
            offer_source = "llm-hash-fallback"
        else:
            offer_source = "llm-choice"

        selected_offer, selected_utility = candidates[parsed_choice - 1]
        self._offered.add(selected_offer)

        message_text = self._extract_message(llm_result.text)
        if not message_text:
            message_text = "이번 제안이 우리 둘의 제약을 함께 만족시키는 쪽에 가깝습니다."

        self.recorder.add(
            {
                "speaker": self.role_name,
                "kind": "offer",
                "step": int(state.step),
                "relative_time": float(state.relative_time),
                "offer_tuple": list(selected_offer),
                "offer": offer_tuple_to_dict(selected_offer),
                "utility_self": float(selected_utility),
                "threshold": threshold,
                "decision": None,
                "decision_source": offer_source,
                "llm_choice": int(parsed_choice),
                "offer_candidates": [
                    {
                        "rank": i,
                        "offer": offer_tuple_to_dict(offer),
                        "utility_self": round(float(utility), 6),
                    }
                    for i, (offer, utility) in enumerate(candidates, start=1)
                ],
                "message": message_text,
                "llm_backend": llm_result.backend,
                "llm_prompt": llm_result.prompt,
                "llm_raw_output": llm_result.raw_text,
            }
        )
        return selected_offer

    def respond(self, state, source: str | None = None):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        utility = float(self.preferences(offer))
        threshold = max(self._reservation_value, self._aspiration(state.relative_time) - 0.03)

        prompt = self._compose_response_prompt(state, offer, utility, threshold)
        llm_result = self.llm.generate(prompt, max_new_tokens=self.llm_max_new_tokens)

        parsed_action = self._parse_action(llm_result.text)
        decision, decision_source = self._resolve_decision(
            parsed_action=parsed_action,
            utility=utility,
            threshold=threshold,
            llm_text=llm_result.text,
        )

        message_text = self._extract_message(llm_result.text)
        if not message_text:
            if decision == ResponseType.ACCEPT_OFFER:
                message_text = "이 조건이면 현재 기준에서 수락 가능합니다."
            else:
                message_text = "이 조건은 아직 제 기준에 맞지 않아 조정이 필요합니다."

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
                "decision_source": decision_source,
                "llm_action": parsed_action.name if parsed_action is not None else None,
                "message": message_text,
                "llm_backend": llm_result.backend,
                "llm_prompt": llm_result.prompt,
                "llm_raw_output": llm_result.raw_text,
            }
        )
        return decision
