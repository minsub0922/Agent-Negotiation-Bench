from __future__ import annotations

import math
import re
from itertools import product
from statistics import mean
from typing import Any

from .agent_utils import scenario_agent_names


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _round(v: float | None, digits: int = 6) -> float | None:
    if v is None:
        return None
    return round(float(v), digits)


def _is_number(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    return isinstance(x, (int, float))


def _safe_ratio(v: Any) -> float | None:
    if not _is_number(v):
        return None
    value = float(v)
    if value > 1.0:
        if value <= 100.0:
            value = value / 100.0
    return _clamp(value, 0.0, 1.0)


def _text(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _contains_any(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(p.lower() in lowered for p in patterns)


def _hangul_ratio(text: str) -> float:
    chars = [c for c in text if c.isalpha()]
    if not chars:
        return 0.0
    hangul = sum(1 for c in chars if "가" <= c <= "힣")
    return hangul / len(chars)


def _action_tag(event: dict[str, Any]) -> str | None:
    kind = str(event.get("kind") or "").lower()
    if kind == "offer":
        return "OFFER"
    if kind == "response":
        decision = str(event.get("decision") or "").upper()
        if "ACCEPT" in decision:
            return "ACCEPT"
        if "REJECT" in decision:
            return "REJECT"
    return None


def _normalized_offer(offer: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(offer, dict):
        return None
    keys = ("destination", "travel_window", "budget")
    if not all(k in offer for k in keys):
        return None
    return {k: offer.get(k) for k in keys}


def _offers_match(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    oa = _normalized_offer(a)
    ob = _normalized_offer(b)
    if oa is None or ob is None:
        return False
    return oa == ob


def _primary_agents(agent_names: list[str]) -> tuple[str | None, str | None]:
    if not agent_names:
        return None, None
    if "agent_a" in agent_names and "agent_b" in agent_names:
        return "agent_a", "agent_b"
    if len(agent_names) == 1:
        return agent_names[0], None
    return agent_names[0], agent_names[1]


def _metric_value_for_agent(metrics: dict[str, Any], agent_name: str, legacy: str) -> float | None:
    by_agent = metrics.get("utility_by_agent")
    if isinstance(by_agent, dict) and agent_name in by_agent and _is_number(by_agent.get(agent_name)):
        return float(by_agent[agent_name])

    key = f"utility_{agent_name}"
    if _is_number(metrics.get(key)):
        return float(metrics[key])

    if _is_number(metrics.get(legacy)):
        return float(metrics[legacy])

    return None


def _build_offer_space(scenario: dict[str, Any]) -> list[tuple[str, str, str]]:
    return list(
        product(
            scenario.get("destinations", []),
            scenario.get("travel_windows", []),
            scenario.get("budgets", []),
        )
    )


def _valid_agreement_check(
    agreement: bool,
    x_star: dict[str, Any] | None,
    agreement_itinerary: dict[str, Any] | None,
    events: list[dict[str, Any]],
) -> bool:
    if not agreement:
        return False
    if _normalized_offer(x_star) is None:
        return False

    last_accept_idx = None
    for idx, event in enumerate(events):
        tag = _action_tag(event)
        if tag == "ACCEPT":
            last_accept_idx = idx

    if last_accept_idx is None:
        return False

    for event in events[last_accept_idx + 1 :]:
        if _action_tag(event) in {"ACCEPT", "REJECT", "OFFER"}:
            return False

    accept_event = events[last_accept_idx]
    accept_offer = _normalized_offer(accept_event.get("offer"))
    if _offers_match(accept_offer, x_star):
        return True

    text = _text(accept_event.get("message"))
    if not text:
        return False

    mention_count = 0
    if str(x_star.get("destination", "")) and str(x_star.get("destination")) in text:
        mention_count += 1
    window_token = str(x_star.get("travel_window") or "")
    if window_token and window_token in text:
        mention_count += 1
    else:
        start_day = agreement_itinerary.get("start_day") if isinstance(agreement_itinerary, dict) else None
        duration = agreement_itinerary.get("duration_days") if isinstance(agreement_itinerary, dict) else None
        if (
            (start_day is not None and f"{start_day}" in text)
            or (duration is not None and f"{duration}" in text)
            or ("기간" in text)
        ):
            mention_count += 1
    budget_token = str(x_star.get("budget") or "")
    if budget_token and budget_token in text:
        mention_count += 1
    elif "예산" in text:
        mention_count += 1

    return mention_count >= 2


def _action_message_consistency(turns: list[dict[str, Any]]) -> float | None:
    accept_terms = ["수락", "accept", "동의", "합의", "yes"]
    reject_terms = ["거절", "reject", "수락 못", "수락못", "불가", "어렵", "no"]
    negation_terms = ["못", "불가", "어렵", "거절", "reject", "no"]

    tagged = [t for t in turns if t.get("action_tag") in {"ACCEPT", "REJECT"}]
    if not tagged:
        return None

    inconsistent = 0
    for t in tagged:
        txt = _text(t.get("text")).lower()
        tag = t.get("action_tag")
        if tag == "ACCEPT":
            if any(term in txt for term in reject_terms):
                inconsistent += 1
        elif tag == "REJECT":
            has_accept = any(term in txt for term in accept_terms)
            has_neg = any(term in txt for term in negation_terms)
            if has_accept and not has_neg:
                inconsistent += 1

    return _clamp(1.0 - (inconsistent / max(len(tagged), 1)))


def _turn_politeness_score(text: str) -> float:
    polite_terms = ["감사", "고맙", "부탁", "죄송", "양해", "괜찮으실까요"]
    toxic_terms = ["병신", "바보", "멍청", "닥쳐", "꺼져", "idiot", "stupid", "fuck"]
    command_terms = ["반드시", "해야", "해라", "must", "should"]

    score = 0.0
    lowered = text.lower()
    if any(term in text for term in polite_terms) or re.search(r"(합니다|하세요|주세요|드립니다)\b", text):
        score += 1.0

    toxic_hits = sum(1 for t in toxic_terms if t in lowered)
    if toxic_hits > 0:
        score -= 2.0 * toxic_hits

    command_hits = sum(1 for t in command_terms if t in lowered or t in text)
    if command_hits >= 2:
        score -= 2.0

    return score


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _specificity_score(text: str, scenario: dict[str, Any]) -> float:
    lowered = text.lower()
    categories = 0

    budgets = [str(b).lower() for b in scenario.get("budgets", [])]
    if "예산" in text or any(b in lowered for b in budgets):
        categories += 1

    destinations = [str(d) for d in scenario.get("destinations", [])]
    if any(dest in text for dest in destinations):
        categories += 1

    if re.search(r"day\d+_dur\d+", lowered) or re.search(r"\d+\s*일", text) or "기간" in text:
        categories += 1

    return categories / 3.0


def _accept_threshold_consistency(events: list[dict[str, Any]]) -> float | None:
    accepts = []
    for event in events:
        if _action_tag(event) != "ACCEPT":
            continue
        utility = event.get("utility_self")
        threshold = event.get("threshold")
        if _is_number(utility) and _is_number(threshold):
            accepts.append((float(utility), float(threshold)))

    if not accepts:
        return None

    good = sum(1 for utility, threshold in accepts if utility >= threshold)
    return good / len(accepts)


def _choice_optimality_at_k(events: list[dict[str, Any]]) -> float | None:
    total = 0
    good = 0
    for event in events:
        if str(event.get("kind")) != "offer":
            continue
        candidates = event.get("offer_candidates")
        if not isinstance(candidates, list) or not candidates:
            continue

        c_utils = [float(c["utility_self"]) for c in candidates if _is_number(c.get("utility_self"))]
        if not c_utils:
            continue

        max_u = max(c_utils)
        selected_u = event.get("utility_self")
        if not _is_number(selected_u):
            continue

        total += 1
        if float(selected_u) >= max_u - 1e-9:
            good += 1

    if total == 0:
        return None
    return good / total


def _reason_alignment_for_event(
    event: dict[str, Any],
    scenario: dict[str, Any],
    ufuns_by_agent: dict[str, Any],
) -> bool | None:
    text = _text(event.get("message"))
    if not text:
        return None

    lowered = text.lower()
    if not any(term in lowered for term in ["예산", "기간", "목적지", "때문", "기준", "효용"]):
        return None

    speaker = str(event.get("speaker") or "")
    if speaker not in ufuns_by_agent:
        return None
    ufun = ufuns_by_agent[speaker]

    offer = _normalized_offer(event.get("offer"))
    if offer is None:
        return None

    decision = _action_tag(event)
    checks: list[bool] = []

    if "예산" in lowered:
        budgets = list(scenario.get("budgets", []))
        if len(budgets) >= 2:
            pairs = []
            for b in budgets:
                pairs.append((b, float(ufun((offer["destination"], offer["travel_window"], b)))))
            mono_up = all(pairs[i + 1][1] >= pairs[i][1] - 1e-9 for i in range(len(pairs) - 1))
            if decision == "REJECT":
                best_budget = max(pairs, key=lambda x: x[1])[0]
                checks.append(mono_up and offer["budget"] != best_budget)
            else:
                checks.append(mono_up)

    if "기간" in lowered:
        windows = list(scenario.get("travel_windows", []))
        meta = scenario.get("travel_window_meta", {})
        duration_to_utils: dict[int, list[float]] = {}
        for w in windows:
            m = meta.get(w) or {}
            duration = m.get("duration_days")
            if duration is None:
                continue
            u = float(ufun((offer["destination"], w, offer["budget"])))
            duration_to_utils.setdefault(int(duration), []).append(u)
        if len(duration_to_utils) >= 2:
            items = sorted((d, mean(vs)) for d, vs in duration_to_utils.items())
            mono_up = all(items[i + 1][1] >= items[i][1] - 1e-9 for i in range(len(items) - 1))
            checks.append(mono_up)

    if "목적지" in lowered:
        destinations = list(scenario.get("destinations", []))
        if len(destinations) >= 2:
            vals = [
                float(ufun((d, offer["travel_window"], offer["budget"])))
                for d in destinations
            ]
            vals_sorted = sorted(vals)
            median_v = vals_sorted[len(vals_sorted) // 2]
            current_v = float(ufun((offer["destination"], offer["travel_window"], offer["budget"])))
            if decision == "REJECT" and any(k in lowered for k in ["바꿔", "변경", "다른"]):
                checks.append(current_v <= median_v + 1e-9)
            elif decision == "ACCEPT":
                checks.append(current_v >= median_v - 1e-9)

    if not checks:
        return None
    return all(checks)


def _empty_schema() -> dict[str, Any]:
    return {
        "agreement_rate_item": 0,
        "valid_agreement": None,
        "turns": 0,
        "turn_efficiency": None,
        "quality_efficiency": None,
        "payoff": {
            "u_a": None,
            "u_b": None,
            "u_a_norm": None,
            "u_b_norm": None,
            "welfare_norm": None,
            "regret_a_norm": None,
            "regret_b_norm": None,
        },
        "game_theory": {
            "nash_distance": None,
            "pareto_optimal_item": None,
        },
        "protocol_quality": {
            "action_message_consistency": None,
            "prompt_leakage_rate": None,
            "placeholder_rate": None,
            "non_korean_rate": None,
            "truncated_turn_rate": None,
        },
        "interaction_fairness": {
            "interpersonal_politeness_score": None,
            "toxicity_flag_rate": None,
            "informational_reason_rate": None,
            "informational_specificity_score": None,
        },
        "internal_faithfulness": {
            "accept_threshold_consistency": None,
            "choice_optimality_at_k": None,
            "reason_utility_alignment": None,
        },
    }


def _deep_merge_defaults(defaults: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, default_value in defaults.items():
        if key not in payload:
            merged[key] = default_value
            continue
        value = payload[key]
        if isinstance(default_value, dict):
            if isinstance(value, dict):
                merged[key] = _deep_merge_defaults(default_value, value)
            else:
                merged[key] = default_value
        else:
            merged[key] = value
    return merged


def default_my_metrics(agreement: bool, turns: int) -> dict[str, Any]:
    out = _empty_schema()
    out["agreement_rate_item"] = 1 if bool(agreement) else 0
    out["valid_agreement"] = bool(agreement)
    out["turns"] = int(max(turns, 0))
    return out


def finalize_my_metrics_shape(
    my_metrics: dict[str, Any] | None,
    agreement: bool,
    turns: int,
) -> dict[str, Any]:
    base = default_my_metrics(agreement=agreement, turns=turns)
    if not isinstance(my_metrics, dict):
        return base
    merged = _deep_merge_defaults(base, my_metrics)
    merged["agreement_rate_item"] = 1 if bool(agreement) else 0
    merged["turns"] = int(max(turns, 0))
    if merged.get("valid_agreement") is None:
        merged["valid_agreement"] = bool(agreement)
    return merged


def compute_my_metrics(
    scenario: dict[str, Any],
    metrics: dict[str, Any],
    events: list[dict[str, Any]],
    ufuns_by_agent: dict[str, Any],
    max_turns: int | None,
) -> dict[str, Any]:
    out = _empty_schema()
    agreement = bool(metrics.get("agreement_reached"))
    out["agreement_rate_item"] = 1 if agreement else 0

    turns = []
    for event in events:
        txt = _text(event.get("message"))
        if not txt:
            continue
        turns.append(
            {
                "speaker": event.get("speaker"),
                "text": txt,
                "action_tag": _action_tag(event),
                "event": event,
            }
        )
    total_turns = len(turns)
    if total_turns == 0 and _is_number(metrics.get("negotiation_steps")):
        total_turns = int(metrics["negotiation_steps"])
    out["turns"] = int(total_turns)

    x_star = _normalized_offer(metrics.get("agreement_offer"))
    agreement_itinerary = metrics.get("agreement_itinerary") if isinstance(metrics.get("agreement_itinerary"), dict) else None
    out["valid_agreement"] = _valid_agreement_check(
        agreement=agreement,
        x_star=x_star,
        agreement_itinerary=agreement_itinerary,
        events=events,
    )

    t_max = int(max_turns) if _is_number(max_turns) else 100
    if t_max <= 2:
        t_max = 100
    if total_turns > 0:
        turn_eff = 1.0 - ((total_turns - 2.0) / (t_max - 2.0))
        out["turn_efficiency"] = _round(_clamp(turn_eff))

    # Game theory
    nash_ratio = metrics.get("nash_ratio")
    if nash_ratio is None:
        nash_ratio = metrics.get("nash_ratio_vs_best")
    nash_ratio_v = _safe_ratio(nash_ratio)
    if nash_ratio_v is not None:
        out["game_theory"]["nash_distance"] = _round(1.0 - nash_ratio_v)

    if metrics.get("pareto_optimal") is not None:
        out["game_theory"]["pareto_optimal_item"] = 1 if bool(metrics.get("pareto_optimal")) else 0

    # Payoff & normalization
    agent_names = scenario_agent_names(scenario)
    agent_a_name, agent_b_name = _primary_agents(agent_names)
    u_a = _metric_value_for_agent(metrics, agent_a_name, "utility_agent_a") if agent_a_name else None
    u_b = _metric_value_for_agent(metrics, agent_b_name, "utility_agent_b") if agent_b_name else None

    offers = _build_offer_space(scenario)
    can_eval_norm = bool(offers and agent_a_name and agent_b_name and agent_a_name in ufuns_by_agent and agent_b_name in ufuns_by_agent)

    if agreement:
        out["payoff"]["u_a"] = _round(u_a)
        out["payoff"]["u_b"] = _round(u_b)

    welfare_norm = _safe_ratio(metrics.get("welfare_ratio_vs_best"))
    if agreement and welfare_norm is not None:
        out["payoff"]["welfare_norm"] = _round(welfare_norm)

    if can_eval_norm and agreement and x_star is not None:
        ufun_a = ufuns_by_agent[agent_a_name]
        ufun_b = ufuns_by_agent[agent_b_name]
        tuples = offers
        ua_vals = [float(ufun_a(t)) for t in tuples]
        ub_vals = [float(ufun_b(t)) for t in tuples]
        ua_min, ua_max = min(ua_vals), max(ua_vals)
        ub_min, ub_max = min(ub_vals), max(ub_vals)

        x_tuple = (x_star["destination"], x_star["travel_window"], x_star["budget"])
        if u_a is None:
            u_a = float(ufun_a(x_tuple))
            out["payoff"]["u_a"] = _round(u_a)
        if u_b is None:
            u_b = float(ufun_b(x_tuple))
            out["payoff"]["u_b"] = _round(u_b)

        if not math.isclose(ua_max, ua_min):
            ua_norm = (float(u_a) - ua_min) / (ua_max - ua_min)
            ua_norm = _clamp(ua_norm)
            out["payoff"]["u_a_norm"] = _round(ua_norm)
            out["payoff"]["regret_a_norm"] = _round(1.0 - ua_norm)

        if not math.isclose(ub_max, ub_min):
            ub_norm = (float(u_b) - ub_min) / (ub_max - ub_min)
            ub_norm = _clamp(ub_norm)
            out["payoff"]["u_b_norm"] = _round(ub_norm)
            out["payoff"]["regret_b_norm"] = _round(1.0 - ub_norm)

        if out["payoff"]["welfare_norm"] is None:
            all_names = [n for n in agent_names if n in ufuns_by_agent]
            if all_names:
                w_best = max(sum(float(ufuns_by_agent[n](t)) for n in all_names) for t in tuples)
                w_cur = sum(float(ufuns_by_agent[n](x_tuple)) for n in all_names)
                if w_best > 0:
                    out["payoff"]["welfare_norm"] = _round(_clamp(w_cur / w_best))

    if out["payoff"]["welfare_norm"] is not None and total_turns > 0:
        out["quality_efficiency"] = _round(float(out["payoff"]["welfare_norm"]) / math.log1p(total_turns))

    # Protocol quality
    if turns:
        leakage_patterns = [
            "당신은 여행 일정 협상 에이전트",
            "내 이름:",
            "현재 step:",
            "상대시간",
            "choice:",
            "action:",
            "thinking process",
            "반드시 아래 형식으로 답하세요",
            "예:",
            "```",
        ]
        placeholder_patterns = [
            "<상대에게 보낼",
            "<번호>",
            "<내용>",
            "<선택한 후보>",
            "반드시",
        ]

        leakage_count = 0
        placeholder_count = 0
        non_korean_count = 0
        truncated_count = 0
        toxicity_flags = 0
        reason_flags = 0
        action_tagged_count = 0
        specificity_scores: list[float] = []
        politeness_scores: list[float] = []

        toxic_terms = ["병신", "바보", "멍청", "닥쳐", "꺼져", "idiot", "stupid", "fuck"]
        reason_terms = ["때문", "해서", "기준", "효용", "예산", "기간", "목적지", "일정", "충돌"]

        for turn in turns:
            text = turn["text"]
            lowered = text.lower()

            if _contains_any(text, leakage_patterns):
                leakage_count += 1

            if _contains_any(text, placeholder_patterns) or re.search(r"\.{3,}|…{1,}", text):
                placeholder_count += 1

            if _hangul_ratio(text) < 0.3:
                non_korean_count += 1

            broken_end = len(text) >= 180 and not re.search(r"([.!?]|다|요|죠|니다)\s*$", text)
            if "(truncated" in lowered or "...(truncated" in lowered or broken_end:
                truncated_count += 1

            if any(term in lowered for term in toxic_terms):
                toxicity_flags += 1

            tag = turn.get("action_tag")
            if tag is not None:
                action_tagged_count += 1
                if any(term in text for term in reason_terms):
                    reason_flags += 1

            specificity_scores.append(_specificity_score(text, scenario))
            politeness_scores.append(_turn_politeness_score(text))

        out["protocol_quality"]["action_message_consistency"] = _round(_action_message_consistency(turns))
        out["protocol_quality"]["prompt_leakage_rate"] = _round(leakage_count / total_turns)
        out["protocol_quality"]["placeholder_rate"] = _round(placeholder_count / total_turns)
        out["protocol_quality"]["non_korean_rate"] = _round(non_korean_count / total_turns)
        out["protocol_quality"]["truncated_turn_rate"] = _round(truncated_count / total_turns)

        if politeness_scores:
            out["interaction_fairness"]["interpersonal_politeness_score"] = _round(
                _sigmoid(sum(politeness_scores) / len(politeness_scores))
            )
        out["interaction_fairness"]["toxicity_flag_rate"] = _round(toxicity_flags / total_turns)
        if action_tagged_count > 0:
            out["interaction_fairness"]["informational_reason_rate"] = _round(reason_flags / action_tagged_count)
        out["interaction_fairness"]["informational_specificity_score"] = _round(
            sum(specificity_scores) / len(specificity_scores)
        )

    # Internal faithfulness
    out["internal_faithfulness"]["accept_threshold_consistency"] = _round(
        _accept_threshold_consistency(events)
    )
    out["internal_faithfulness"]["choice_optimality_at_k"] = _round(
        _choice_optimality_at_k(events)
    )

    reason_checks: list[bool] = []
    for event in events:
        if str(event.get("kind")) != "response":
            continue
        check = _reason_alignment_for_event(event, scenario=scenario, ufuns_by_agent=ufuns_by_agent)
        if check is not None:
            reason_checks.append(check)
    if reason_checks:
        out["internal_faithfulness"]["reason_utility_alignment"] = _round(
            sum(1 for x in reason_checks if x) / len(reason_checks)
        )

    return out
