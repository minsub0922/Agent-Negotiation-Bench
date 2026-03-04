from __future__ import annotations

import json
import math
import random
from itertools import product
from pathlib import Path
from typing import Any

from .agent_utils import default_agent_names, scenario_agent_names

SLOTS = ("morning", "afternoon", "evening")
DESTINATION_POOL = (
    "Jeju",
    "Busan",
    "Seoul",
    "Tokyo",
    "Osaka",
    "Taipei",
    "Bangkok",
    "Singapore",
    "Da Nang",
    "Kyoto",
)
BUDGET_LEVELS = ("low", "medium", "high")
WINDOW_DURATIONS = (2, 3, 4, 5)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    values = list(scores.values())
    lo = min(values)
    hi = max(values)
    if math.isclose(hi, lo):
        return {k: 1.0 for k in scores}
    return {k: _clamp((v - lo) / (hi - lo)) for k, v in scores.items()}


def _sample_calendar(rng: random.Random, horizon_days: int) -> dict[str, list[str]]:
    calendar: dict[str, list[str]] = {}
    for day in range(1, horizon_days + 1):
        n_busy = rng.choices([0, 1, 2, 3], weights=[0.22, 0.38, 0.28, 0.12], k=1)[0]
        if n_busy == 0:
            continue
        calendar[str(day)] = sorted(rng.sample(list(SLOTS), k=n_busy))
    return calendar


def _window_id(start_day: int, duration_days: int) -> str:
    return f"day{start_day}_dur{duration_days}"


def _build_window_options(
    rng: random.Random,
    horizon_days: int,
    min_windows: int,
    max_windows: int,
) -> dict[str, dict[str, Any]]:
    all_windows: list[tuple[int, int]] = []
    for duration in WINDOW_DURATIONS:
        for start_day in range(1, horizon_days - duration + 2):
            all_windows.append((start_day, duration))

    rng.shuffle(all_windows)
    target_n = rng.randint(min_windows, max_windows)
    selected = sorted(all_windows[:target_n], key=lambda x: (x[0], x[1]))

    windows: dict[str, dict[str, Any]] = {}
    for start_day, duration in selected:
        wid = _window_id(start_day, duration)
        travel_days = list(range(start_day, start_day + duration))
        weekend_days = [d for d in travel_days if d % 7 in (6, 0)]
        windows[wid] = {
            "start_day": start_day,
            "duration_days": duration,
            "travel_days": travel_days,
            "weekend_days": weekend_days,
        }
    return windows


def _window_availability(
    calendar: dict[str, list[str]],
    window_meta: dict[str, Any],
    n_slots: int,
) -> float:
    busy_slots = 0
    for day in window_meta["travel_days"]:
        busy_slots += len(calendar.get(str(day), []))
    total_slots = max(len(window_meta["travel_days"]) * n_slots, 1)
    return _clamp(1.0 - busy_slots / total_slots)


def _profile_scores(
    rng: random.Random,
    calendar: dict[str, list[str]],
    destinations: list[str],
    windows: dict[str, dict[str, Any]],
    budgets: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    destination_scores = {
        dest: _clamp(rng.betavariate(2.0, 2.0) + rng.uniform(-0.08, 0.08))
        for dest in destinations
    }

    preferred_budget_idx = rng.randint(0, len(budgets) - 1)
    budget_scores: dict[str, float] = {}
    for i, budget in enumerate(budgets):
        dist = abs(i - preferred_budget_idx)
        raw = 1.0 - 0.42 * dist + rng.uniform(-0.08, 0.08)
        budget_scores[budget] = _clamp(raw)

    preferred_duration = rng.choice(WINDOW_DURATIONS)
    duration_sigma = rng.uniform(1.1, 1.9)
    weekend_weight = rng.uniform(0.0, 0.3)

    window_scores: dict[str, float] = {}
    for window_id, window_meta in windows.items():
        availability = _window_availability(calendar, window_meta, n_slots=len(SLOTS))
        duration = window_meta["duration_days"]
        duration_score = math.exp(-((duration - preferred_duration) ** 2) / (2 * duration_sigma**2))
        weekend_density = len(window_meta["weekend_days"]) / max(len(window_meta["travel_days"]), 1)
        noisy = rng.uniform(-0.08, 0.08)
        score = 0.68 * availability + 0.22 * duration_score + weekend_weight * weekend_density + noisy
        window_scores[window_id] = _clamp(score)

    destination_scores = _normalize_scores(destination_scores)
    budget_scores = _normalize_scores(budget_scores)
    window_scores = _normalize_scores(window_scores)

    raw_weights = [rng.uniform(0.45, 1.65) for _ in range(3)]
    denom = sum(raw_weights)
    weights = {
        "destination": raw_weights[0] / denom,
        "travel_window": raw_weights[1] / denom,
        "budget": raw_weights[2] / denom,
    }

    return destination_scores, window_scores, budget_scores, weights


def enumerate_outcomes(scenario: dict[str, Any]) -> list[dict[str, str]]:
    outcomes: list[dict[str, str]] = []
    for destination, travel_window, budget in product(
        scenario["destinations"], scenario["travel_windows"], scenario["budgets"]
    ):
        outcomes.append(
            {
                "destination": destination,
                "travel_window": travel_window,
                "budget": budget,
            }
        )
    return outcomes


def utility_of(profile: dict[str, Any], offer: dict[str, str]) -> float:
    w = profile["weights"]
    return (
        w["destination"] * profile["destination_scores"][offer["destination"]]
        + w["travel_window"] * profile["window_scores"][offer["travel_window"]]
        + w["budget"] * profile["budget_scores"][offer["budget"]]
    )


def _has_feasible_outcome(scenario: dict[str, Any], min_gain: float = 0.03) -> bool:
    agent_names = scenario_agent_names(scenario)
    profiles = {name: scenario["agents"][name] for name in agent_names}
    reservations = {name: float(profile["reservation_value"]) for name, profile in profiles.items()}

    best_joint = -1.0
    feasible_count = 0
    for offer in enumerate_outcomes(scenario):
        utilities = {name: utility_of(profile, offer) for name, profile in profiles.items()}
        best_joint = max(best_joint, sum(utilities.values()))
        if all(utilities[name] >= reservations[name] + min_gain for name in agent_names):
            feasible_count += 1

    return feasible_count > 0 and best_joint > 0


def generate_scenario(
    rng: random.Random,
    scenario_index: int,
    n_agents: int = 2,
    horizon_days: int = 14,
    min_windows: int = 14,
    max_windows: int = 22,
    n_destinations: int = 5,
) -> dict[str, Any]:
    agent_names = default_agent_names(n_agents)
    destinations = sorted(rng.sample(list(DESTINATION_POOL), k=n_destinations))
    travel_windows_meta = _build_window_options(
        rng=rng,
        horizon_days=horizon_days,
        min_windows=min_windows,
        max_windows=max_windows,
    )
    travel_windows = list(travel_windows_meta.keys())
    budgets = list(BUDGET_LEVELS)

    scenario = {
        "scenario_id": f"scenario_{scenario_index:04d}",
        "horizon_days": horizon_days,
        "slot_names": list(SLOTS),
        "destinations": destinations,
        "travel_windows": travel_windows,
        "travel_window_meta": travel_windows_meta,
        "budgets": budgets,
        "agent_order": agent_names,
        "agents": {},
    }

    for agent_name in agent_names:
        calendar = _sample_calendar(rng, horizon_days=horizon_days)
        destination_scores, window_scores, budget_scores, weights = _profile_scores(
            rng=rng,
            calendar=calendar,
            destinations=destinations,
            windows=travel_windows_meta,
            budgets=budgets,
        )

        scenario["agents"][agent_name] = {
            "name": agent_name,
            "calendar": calendar,
            "destination_scores": destination_scores,
            "window_scores": window_scores,
            "budget_scores": budget_scores,
            "weights": weights,
            "reservation_value": round(rng.uniform(0.32, 0.55), 4),
            "concession_exponent": round(rng.uniform(1.05, 2.1), 4),
        }

    return scenario


def generate_dataset(
    num_scenarios: int,
    n_agents: int = 2,
    seed: int = 7,
    horizon_days: int = 14,
    max_generation_attempts: int = 40,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    scenarios: list[dict[str, Any]] = []

    for idx in range(1, num_scenarios + 1):
        selected: dict[str, Any] | None = None
        for _ in range(max_generation_attempts):
            candidate = generate_scenario(
                rng=rng,
                scenario_index=idx,
                n_agents=n_agents,
                horizon_days=horizon_days,
            )
            if _has_feasible_outcome(candidate):
                selected = candidate
                break

        if selected is None:
            selected = generate_scenario(
                rng=rng,
                scenario_index=idx,
                n_agents=n_agents,
                horizon_days=horizon_days,
            )

        scenarios.append(selected)

    return scenarios


def save_dataset_jsonl(scenarios: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario, ensure_ascii=False) + "\n")


def load_dataset_jsonl(dataset_path: Path) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scenarios.append(json.loads(line))
    return scenarios
