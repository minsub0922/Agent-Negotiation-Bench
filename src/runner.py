from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from negmas import LinearAdditiveUtilityFunction, SAOMechanism, make_issue

from .evaluation import compute_metrics, enrich_event_for_human
from .llm_backend import LLMBackend
from .negotiation import ISSUE_NAMES, EventRecorder, LLMCalendarNegotiator


def _json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_ufun(profile: dict[str, Any], issues: list[Any]) -> LinearAdditiveUtilityFunction:
    ufun = LinearAdditiveUtilityFunction(
        values={
            "destination": profile["destination_scores"],
            "travel_window": profile["window_scores"],
            "budget": profile["budget_scores"],
        },
        weights=profile["weights"],
        issues=issues,
    )
    ufun.reserved_value = float(profile["reservation_value"])
    return ufun


def _trace_to_json(trace, scenario: dict[str, Any], ufun_a, ufun_b) -> dict[str, Any]:
    offer_tuple = trace.offer
    if offer_tuple is None:
        offer = None
        itinerary = None
        ua = None
        ub = None
    else:
        offer = {
            ISSUE_NAMES[0]: offer_tuple[0],
            ISSUE_NAMES[1]: offer_tuple[1],
            ISSUE_NAMES[2]: offer_tuple[2],
        }
        window = scenario["travel_window_meta"][offer["travel_window"]]
        itinerary = {
            "destination": offer["destination"],
            "budget": offer["budget"],
            "travel_window": offer["travel_window"],
            "start_day": window["start_day"],
            "duration_days": window["duration_days"],
            "travel_days": window["travel_days"],
        }
        ua = round(float(ufun_a(offer_tuple)), 6)
        ub = round(float(ufun_b(offer_tuple)), 6)

    responses = {
        k: (v.name if hasattr(v, "name") else str(v))
        for k, v in (trace.responses.items() if trace.responses else [])
    }

    return {
        "time": float(trace.time),
        "relative_time": float(trace.relative_time),
        "step": int(trace.step),
        "negotiator": trace.negotiator,
        "offer_tuple": list(offer_tuple) if offer_tuple is not None else None,
        "offer": offer,
        "itinerary": itinerary,
        "utility_agent_a": ua,
        "utility_agent_b": ub,
        "responses": responses,
        "state": trace.state,
    }


def run_single_scenario(
    scenario: dict[str, Any],
    llm_a: LLMBackend,
    llm_b: LLMBackend,
    output_dir: Path,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    issues = [
        make_issue(scenario["destinations"], name="destination"),
        make_issue(scenario["travel_windows"], name="travel_window"),
        make_issue(scenario["budgets"], name="budget"),
    ]

    profile_a = scenario["agents"]["agent_a"]
    profile_b = scenario["agents"]["agent_b"]

    ufun_a = _build_ufun(profile_a, issues)
    ufun_b = _build_ufun(profile_b, issues)

    recorder = EventRecorder()
    rng = random.Random(seed)

    neg_a = LLMCalendarNegotiator(
        llm=llm_a,
        recorder=recorder,
        role_name="agent_a",
        profile=profile_a,
        rng=rng,
        name="agent_a",
    )
    neg_b = LLMCalendarNegotiator(
        llm=llm_b,
        recorder=recorder,
        role_name="agent_b",
        profile=profile_b,
        rng=rng,
        name="agent_b",
    )

    mechanism = SAOMechanism(issues=issues, n_steps=max_steps)
    mechanism.add(neg_a, preferences=ufun_a)
    mechanism.add(neg_b, preferences=ufun_b)

    state = mechanism.run()
    agreement_tuple = state.agreement if state.agreement else None

    scenario_out = output_dir / scenario["scenario_id"]
    scenario_out.mkdir(parents=True, exist_ok=True)

    trace_rows = [_trace_to_json(t, scenario=scenario, ufun_a=ufun_a, ufun_b=ufun_b) for t in mechanism.full_trace]
    _jsonl_dump(scenario_out / "negmas_trace.jsonl", trace_rows)

    _jsonl_dump(scenario_out / "agent_event_log.jsonl", recorder.events)

    human_dialogue = [
        enrich_event_for_human(event, scenario=scenario, ufun_a=ufun_a, ufun_b=ufun_b)
        for event in recorder.events
    ]

    metrics = compute_metrics(
        scenario=scenario,
        state=state,
        agreement_tuple=agreement_tuple,
        ufun_a=ufun_a,
        ufun_b=ufun_b,
        events=recorder.events,
    )

    _json_dump(
        scenario_out / "dialogue_human_readable.json",
        {
            "scenario_id": scenario["scenario_id"],
            "turns": human_dialogue,
            "metrics": metrics,
        },
    )
    _json_dump(scenario_out / "metrics.json", metrics)
    _json_dump(scenario_out / "scenario.json", scenario)

    return metrics


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    reached = [row for row in rows if row.get("agreement_reached")]

    def _avg(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "run_at": datetime.now().isoformat(),
        "num_scenarios": len(rows),
        "agreement_rate": len(reached) / len(rows),
        "avg_social_welfare": _avg([float(r["social_welfare"]) for r in rows]),
        "avg_nash_product": _avg([float(r["nash_product"]) for r in rows]),
        "avg_steps": _avg([float(r["negotiation_steps"]) for r in rows]),
        "avg_welfare_ratio_vs_best": _avg([float(r["welfare_ratio_vs_best"]) for r in rows]),
        "avg_nash_ratio_vs_best": _avg([float(r["nash_ratio_vs_best"]) for r in rows]),
        "avg_utility_agent_a": _avg([float(r["utility_agent_a"]) for r in rows]),
        "avg_utility_agent_b": _avg([float(r["utility_agent_b"]) for r in rows]),
    }


def run_experiment(
    scenarios: list[dict[str, Any]],
    output_root: Path,
    llm_a: LLMBackend,
    llm_b: LLMBackend,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, Any]] = []
    for idx, scenario in enumerate(scenarios):
        metric = run_single_scenario(
            scenario=scenario,
            llm_a=llm_a,
            llm_b=llm_b,
            output_dir=output_root,
            max_steps=max_steps,
            seed=seed + idx,
        )
        all_metrics.append(metric)

    _write_summary_csv(output_root / "experiment_summary.csv", all_metrics)
    aggregate = _aggregate(all_metrics)
    _json_dump(output_root / "experiment_summary.json", aggregate)

    return {
        "metrics": all_metrics,
        "aggregate": aggregate,
    }
