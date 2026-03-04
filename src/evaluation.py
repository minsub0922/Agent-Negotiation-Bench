from __future__ import annotations

from itertools import product
from typing import Any

from .negotiation import ISSUE_NAMES, offer_tuple_to_dict


def agreement_to_itinerary(offer: dict[str, Any] | None, scenario: dict[str, Any]) -> dict[str, Any] | None:
    if offer is None:
        return None

    window_id = offer["travel_window"]
    window_meta = scenario["travel_window_meta"][window_id]

    return {
        "destination": offer["destination"],
        "budget": offer["budget"],
        "travel_window": window_id,
        "start_day": window_meta["start_day"],
        "duration_days": window_meta["duration_days"],
        "travel_days": window_meta["travel_days"],
    }


def _all_outcome_tuples(scenario: dict[str, Any]) -> list[tuple[str, str, str]]:
    return list(
        product(
            scenario["destinations"],
            scenario["travel_windows"],
            scenario["budgets"],
        )
    )


def _is_dominated(point: tuple[float, float], candidates: list[tuple[float, float]]) -> bool:
    for other in candidates:
        if other[0] >= point[0] and other[1] >= point[1] and (other[0] > point[0] or other[1] > point[1]):
            return True
    return False


def _pareto_frontier(points: dict[tuple[str, str, str], tuple[float, float]]) -> set[tuple[str, str, str]]:
    frontier: set[tuple[str, str, str]] = set()
    utility_points = list(points.items())
    just_values = [v for _, v in utility_points]

    for outcome, point in utility_points:
        if not _is_dominated(point, just_values):
            frontier.add(outcome)
    return frontier


def _calendar_conflict_ratio(profile: dict[str, Any], itinerary: dict[str, Any], slots: list[str]) -> float:
    busy_slots = 0
    for day in itinerary["travel_days"]:
        busy_slots += len(profile["calendar"].get(str(day), []))
    total = max(len(itinerary["travel_days"]) * len(slots), 1)
    return busy_slots / total


def compute_metrics(
    scenario: dict[str, Any],
    state,
    agreement_tuple: tuple[str, str, str] | None,
    ufun_a,
    ufun_b,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    rv_a = float(scenario["agents"]["agent_a"]["reservation_value"])
    rv_b = float(scenario["agents"]["agent_b"]["reservation_value"])

    outcomes = _all_outcome_tuples(scenario)
    utility_map: dict[tuple[str, str, str], tuple[float, float]] = {}
    for outcome in outcomes:
        utility_map[outcome] = (float(ufun_a(outcome)), float(ufun_b(outcome)))

    pareto = _pareto_frontier(utility_map)
    best_welfare = max(ua + ub for ua, ub in utility_map.values())
    best_nash = max(max(ua - rv_a, 0.0) * max(ub - rv_b, 0.0) for ua, ub in utility_map.values())

    agreement_offer = offer_tuple_to_dict(agreement_tuple)
    itinerary = agreement_to_itinerary(agreement_offer, scenario) if agreement_offer else None

    if agreement_tuple is not None:
        ua = float(ufun_a(agreement_tuple))
        ub = float(ufun_b(agreement_tuple))
        welfare = ua + ub
        nash = max(ua - rv_a, 0.0) * max(ub - rv_b, 0.0)
        fairness_gap = abs(ua - ub)
        pareto_optimal = agreement_tuple in pareto
        cal_conflict_a = _calendar_conflict_ratio(scenario["agents"]["agent_a"], itinerary, scenario["slot_names"])
        cal_conflict_b = _calendar_conflict_ratio(scenario["agents"]["agent_b"], itinerary, scenario["slot_names"])
    else:
        ua = 0.0
        ub = 0.0
        welfare = 0.0
        nash = 0.0
        fairness_gap = None
        pareto_optimal = False
        cal_conflict_a = None
        cal_conflict_b = None

    accepted_events = [e for e in events if e["kind"] == "response" and e["decision"] == "ACCEPT_OFFER"]

    return {
        "scenario_id": scenario["scenario_id"],
        "agreement_reached": agreement_tuple is not None,
        "agreement_offer": agreement_offer,
        "agreement_itinerary": itinerary,
        "negotiation_steps": int(state.step),
        "relative_time_end": float(state.relative_time),
        "timed_out": bool(state.timedout),
        "broken": bool(state.broken),
        "utility_agent_a": round(ua, 6),
        "utility_agent_b": round(ub, 6),
        "reservation_agent_a": round(rv_a, 6),
        "reservation_agent_b": round(rv_b, 6),
        "social_welfare": round(welfare, 6),
        "welfare_ratio_vs_best": round(welfare / best_welfare, 6) if best_welfare > 0 else 0.0,
        "nash_product": round(nash, 6),
        "nash_ratio_vs_best": round(nash / best_nash, 6) if best_nash > 0 else 0.0,
        "pareto_optimal": pareto_optimal,
        "fairness_gap": round(fairness_gap, 6) if fairness_gap is not None else None,
        "calendar_conflict_ratio_agent_a": round(cal_conflict_a, 6) if cal_conflict_a is not None else None,
        "calendar_conflict_ratio_agent_b": round(cal_conflict_b, 6) if cal_conflict_b is not None else None,
        "event_count": len(events),
        "offer_count": sum(1 for e in events if e["kind"] == "offer"),
        "response_count": sum(1 for e in events if e["kind"] == "response"),
        "accept_count": len(accepted_events),
    }


def enrich_event_for_human(
    event: dict[str, Any],
    scenario: dict[str, Any],
    ufun_a,
    ufun_b,
) -> dict[str, Any]:
    copied = dict(event)
    offer_tuple = tuple(event["offer_tuple"])

    copied["utility_agent_a"] = round(float(ufun_a(offer_tuple)), 6)
    copied["utility_agent_b"] = round(float(ufun_b(offer_tuple)), 6)

    offer_dict = dict(event["offer"])
    copied["itinerary"] = agreement_to_itinerary(offer_dict, scenario)
    return copied
