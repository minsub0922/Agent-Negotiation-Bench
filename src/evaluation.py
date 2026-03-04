from __future__ import annotations

from itertools import product
from typing import Any

from .agent_utils import scenario_agent_names
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


def _is_dominated(
    point: dict[str, float],
    candidates: list[dict[str, float]],
    agent_names: list[str],
) -> bool:
    for other in candidates:
        if all(other[name] >= point[name] for name in agent_names) and any(
            other[name] > point[name] for name in agent_names
        ):
            return True
    return False


def _pareto_frontier(
    points: dict[tuple[str, str, str], dict[str, float]],
    agent_names: list[str],
) -> set[tuple[str, str, str]]:
    frontier: set[tuple[str, str, str]] = set()
    utility_points = list(points.items())
    just_values = [v for _, v in utility_points]

    for outcome, point in utility_points:
        if not _is_dominated(point, just_values, agent_names):
            frontier.add(outcome)
    return frontier


def _calendar_conflict_ratio(profile: dict[str, Any], itinerary: dict[str, Any], slots: list[str]) -> float:
    busy_slots = 0
    for day in itinerary["travel_days"]:
        busy_slots += len(profile["calendar"].get(str(day), []))
    total = max(len(itinerary["travel_days"]) * len(slots), 1)
    return busy_slots / total


def _nash_product(utilities: dict[str, float], reservations: dict[str, float], agent_names: list[str]) -> float:
    value = 1.0
    for name in agent_names:
        value *= max(utilities[name] - reservations[name], 0.0)
    return value


def _append_agent_metric_aliases(
    payload: dict[str, Any],
    prefix: str,
    by_agent: dict[str, Any],
    agent_names: list[str],
) -> None:
    for name in agent_names:
        payload[f"{prefix}_{name}"] = by_agent[name]

    # Backward compatibility for strict 2-agent readers.
    payload.setdefault(f"{prefix}_agent_a", by_agent.get("agent_a"))
    payload.setdefault(f"{prefix}_agent_b", by_agent.get("agent_b"))


def compute_metrics(
    scenario: dict[str, Any],
    state,
    agreement_tuple: tuple[str, str, str] | None,
    ufuns_by_agent: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    agent_names = scenario_agent_names(scenario)
    profiles = {name: scenario["agents"][name] for name in agent_names}
    reservations = {name: float(profiles[name]["reservation_value"]) for name in agent_names}

    outcomes = _all_outcome_tuples(scenario)
    utility_map: dict[tuple[str, str, str], dict[str, float]] = {}
    for outcome in outcomes:
        utility_map[outcome] = {name: float(ufuns_by_agent[name](outcome)) for name in agent_names}

    pareto = _pareto_frontier(utility_map, agent_names=agent_names)
    best_welfare = max(sum(v.values()) for v in utility_map.values())
    best_nash = max(_nash_product(v, reservations, agent_names) for v in utility_map.values())

    agreement_offer = offer_tuple_to_dict(agreement_tuple)
    itinerary = agreement_to_itinerary(agreement_offer, scenario) if agreement_offer else None

    if agreement_tuple is not None:
        utilities = utility_map.get(agreement_tuple) or {
            name: float(ufuns_by_agent[name](agreement_tuple)) for name in agent_names
        }
        welfare = sum(utilities.values())
        nash = _nash_product(utilities, reservations, agent_names)
        fairness_gap = max(utilities.values()) - min(utilities.values()) if utilities else None
        pareto_optimal = agreement_tuple in pareto
        calendar_conflicts = {
            name: _calendar_conflict_ratio(profiles[name], itinerary, scenario["slot_names"])
            for name in agent_names
        }
    else:
        utilities = {name: 0.0 for name in agent_names}
        welfare = 0.0
        nash = 0.0
        fairness_gap = None
        pareto_optimal = False
        calendar_conflicts = {name: None for name in agent_names}

    accepted_events = [e for e in events if e["kind"] == "response" and e["decision"] == "ACCEPT_OFFER"]

    utility_by_agent = {name: round(float(utilities[name]), 6) for name in agent_names}
    reservation_by_agent = {name: round(float(reservations[name]), 6) for name in agent_names}
    conflict_by_agent = {
        name: (round(float(calendar_conflicts[name]), 6) if calendar_conflicts[name] is not None else None)
        for name in agent_names
    }

    metrics = {
        "scenario_id": scenario["scenario_id"],
        "num_agents": len(agent_names),
        "agent_names": agent_names,
        "agreement_reached": agreement_tuple is not None,
        "agreement_offer": agreement_offer,
        "agreement_itinerary": itinerary,
        "negotiation_steps": int(state.step),
        "relative_time_end": float(state.relative_time),
        "timed_out": bool(state.timedout),
        "broken": bool(state.broken),
        "utility_by_agent": utility_by_agent,
        "reservation_by_agent": reservation_by_agent,
        "calendar_conflict_ratio_by_agent": conflict_by_agent,
        "social_welfare": round(float(welfare), 6),
        "welfare_ratio_vs_best": round(float(welfare / best_welfare), 6) if best_welfare > 0 else 0.0,
        "nash_product": round(float(nash), 6),
        "nash_ratio_vs_best": round(float(nash / best_nash), 6) if best_nash > 0 else 0.0,
        "pareto_optimal": pareto_optimal,
        "fairness_gap": round(float(fairness_gap), 6) if fairness_gap is not None else None,
        "event_count": len(events),
        "offer_count": sum(1 for e in events if e["kind"] == "offer"),
        "response_count": sum(1 for e in events if e["kind"] == "response"),
        "accept_count": len(accepted_events),
    }

    _append_agent_metric_aliases(metrics, "utility", utility_by_agent, agent_names=agent_names)
    _append_agent_metric_aliases(metrics, "reservation", reservation_by_agent, agent_names=agent_names)
    _append_agent_metric_aliases(metrics, "calendar_conflict_ratio", conflict_by_agent, agent_names=agent_names)

    return metrics


def enrich_event_for_human(
    event: dict[str, Any],
    scenario: dict[str, Any],
    ufuns_by_agent: dict[str, Any],
) -> dict[str, Any]:
    copied = dict(event)
    offer_tuple = tuple(event["offer_tuple"])
    agent_names = scenario_agent_names(scenario)

    utility_by_agent = {name: round(float(ufuns_by_agent[name](offer_tuple)), 6) for name in agent_names}
    copied["utility_by_agent"] = utility_by_agent

    _append_agent_metric_aliases(copied, "utility", utility_by_agent, agent_names=agent_names)

    offer_dict = dict(event["offer"])
    copied["itinerary"] = agreement_to_itinerary(offer_dict, scenario)
    return copied
