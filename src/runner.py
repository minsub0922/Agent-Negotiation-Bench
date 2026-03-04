from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path
import time
from typing import Any

from negmas import LinearAdditiveUtilityFunction, SAOMechanism, make_issue

from .agent_utils import scenario_agent_names
from .evaluation import compute_metrics, enrich_event_for_human
from .llm_backend import LLMBackend
from .my_metrics import compute_my_metrics, finalize_my_metrics_shape
from .negotiation import ISSUE_NAMES, EventRecorder, LLMCalendarNegotiator


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(_json_dumps(data))


def _jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _text_dump(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def _trace_to_json(trace, scenario: dict[str, Any], ufuns_by_agent: dict[str, Any]) -> dict[str, Any]:
    agent_names = scenario_agent_names(scenario)
    offer_tuple = trace.offer
    if offer_tuple is None:
        offer = None
        itinerary = None
        utility_by_agent = {name: None for name in agent_names}
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
        utility_by_agent = {
            name: round(float(ufuns_by_agent[name](offer_tuple)), 6)
            for name in agent_names
        }

    responses = {
        k: (v.name if hasattr(v, "name") else str(v))
        for k, v in (trace.responses.items() if trace.responses else [])
    }

    payload = {
        "time": float(trace.time),
        "relative_time": float(trace.relative_time),
        "step": int(trace.step),
        "negotiator": trace.negotiator,
        "offer_tuple": list(offer_tuple) if offer_tuple is not None else None,
        "offer": offer,
        "itinerary": itinerary,
        "utility_by_agent": utility_by_agent,
        "responses": responses,
        "state": trace.state,
    }
    payload["utility_agent_a"] = utility_by_agent.get("agent_a")
    payload["utility_agent_b"] = utility_by_agent.get("agent_b")
    return payload


def _to_chat_only_turns(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chat: list[dict[str, Any]] = []
    turn = 0
    for event in events:
        speaker = str(event["speaker"])
        message = str(event.get("message") or "").strip()
        if not message:
            continue
        # Remove duplicated speaker prefix such as "agent_a: ..." for cleaner chat output.
        prefix = f"{speaker}:"
        if message.lower().startswith(prefix.lower()):
            message = message[len(prefix) :].strip()
        if event.get("kind") == "response" and event.get("decision"):
            message = f"[{event['decision']}] {message}"
        turn += 1
        chat.append(
            {
                "turn": turn,
                "speaker": speaker,
                "message": message,
            }
        )
    return chat


def _itinerary_summary(metrics: dict[str, Any]) -> str:
    if not metrics.get("agreement_reached"):
        return "합의 실패"
    itinerary = metrics.get("agreement_itinerary") or {}
    if not itinerary:
        return "합의 성공 (세부 일정 정보 없음)"
    return (
        f"합의 성공: {itinerary.get('destination')} / {itinerary.get('travel_window')} "
        f"(day {itinerary.get('start_day')}, {itinerary.get('duration_days')}일) / budget={itinerary.get('budget')}"
    )


def _fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.{digits}f}%"


def _fmt_bool(value: bool) -> str:
    return "Yes" if bool(value) else "No"


def _metric_by_agent(metrics: dict[str, Any], field: str) -> dict[str, Any]:
    data = metrics.get(field)
    if isinstance(data, dict) and data:
        return data

    # Backward-compatible fallback for legacy metrics.
    prefix = field.replace("_by_agent", "")
    fallback: dict[str, Any] = {}
    for legacy in ("agent_a", "agent_b"):
        key = f"{prefix}_{legacy}"
        if key in metrics:
            fallback[legacy] = metrics.get(key)
    return fallback


def _agent_rows(
    metrics: dict[str, Any],
    field: str,
    row_name_prefix: str,
    formatter,
) -> list[str]:
    by_agent = _metric_by_agent(metrics, field)
    if not by_agent:
        return []

    agent_names = metrics.get("agent_names")
    if not isinstance(agent_names, list) or not agent_names:
        agent_names = list(by_agent.keys())

    rows: list[str] = []
    for name in agent_names:
        if name not in by_agent:
            continue
        rows.append(f"| {row_name_prefix} ({name}) | {formatter(by_agent.get(name))} |")
    return rows


def _scenario_metrics_md(scenario_id: str, metrics: dict[str, Any]) -> str:
    lines = [
        f"# Scenario Quant Summary: {scenario_id}",
        "",
        f"- Agreement: {_fmt_bool(metrics['agreement_reached'])}",
        f"- Pareto Optimal: {_fmt_bool(metrics['pareto_optimal'])}",
        f"- Num Agents: {metrics.get('num_agents', '-')}",
        f"- Result: {_itinerary_summary(metrics)}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Negotiation Steps | {_fmt_num(metrics['negotiation_steps'], 0)} |",
        f"| Social Welfare | {_fmt_num(metrics['social_welfare'])} |",
        f"| Nash Product | {_fmt_num(metrics['nash_product'])} |",
        f"| Welfare Ratio vs Best | {_fmt_pct(metrics['welfare_ratio_vs_best'])} |",
        f"| Nash Ratio vs Best | {_fmt_pct(metrics['nash_ratio_vs_best'])} |",
        f"| Fairness Gap | {_fmt_num(metrics['fairness_gap'])} |",
    ]
    lines.extend(_agent_rows(metrics, "utility_by_agent", "Utility", _fmt_num))
    lines.extend(_agent_rows(metrics, "reservation_by_agent", "Reservation", _fmt_num))
    lines.extend(_agent_rows(metrics, "calendar_conflict_ratio_by_agent", "Calendar Conflict", _fmt_pct))
    return "\n".join(lines) + "\n"


def _write_chat_outputs(
    scenario_out: Path,
    scenario_id: str,
    chat_turns: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    _json_dump(
        scenario_out / "chat_transcript.json",
        {
            "scenario_id": scenario_id,
            "chat": chat_turns,
            "result_summary": _itinerary_summary(metrics),
        },
    )

    lines = [f"[Scenario] {scenario_id}", ""]
    for turn in chat_turns:
        lines.append(f"{turn['speaker']}: {turn['message']}")
    lines.append("")
    lines.append(f"[Result] {_itinerary_summary(metrics)}")

    _text_dump(scenario_out / "chat_transcript.txt", "\n".join(lines) + "\n")


def _build_issue_bundle(
    scenario: dict[str, Any],
    metrics: dict[str, Any],
    chat_turns: list[dict[str, Any]],
    human_dialogue: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    agent_events: list[dict[str, Any]],
) -> dict[str, Any]:
    result_line = f"[Result] {_itinerary_summary(metrics)}"
    chat_with_result = list(chat_turns)
    chat_with_result.append(
        {
            "turn": len(chat_turns) + 1,
            "speaker": "result",
            "message": result_line,
            "kind": "result",
        }
    )
    return {
        "scenario_id": scenario["scenario_id"],
        "result_summary": _itinerary_summary(metrics),
        "chat_result_line": result_line,
        "chat_transcript": chat_turns,
        "chat_transcript_with_result": chat_with_result,
        "quantitative_metrics": metrics,
        "detailed_json": {
            "scenario": scenario,
            "dialogue_human_readable": human_dialogue,
            "agent_event_log": agent_events,
            "negmas_trace": trace_rows,
        },
    }


def _build_issue_md(issue_bundle: dict[str, Any]) -> str:
    scenario_id = issue_bundle["scenario_id"]
    metrics = issue_bundle["quantitative_metrics"]
    chat_turns = issue_bundle["chat_transcript"]
    chat_result_line = issue_bundle.get("chat_result_line", f"[Result] {issue_bundle['result_summary']}")
    detailed_json = issue_bundle["detailed_json"]

    lines = [
        f"# [Negotiation Report] {scenario_id}",
        "",
        "## Summary",
        "",
        f"- Result: {issue_bundle['result_summary']}",
        f"- Agreement: {_fmt_bool(metrics['agreement_reached'])}",
        f"- Pareto Optimal: {_fmt_bool(metrics['pareto_optimal'])}",
        f"- Steps: {_fmt_num(metrics['negotiation_steps'], 0)}",
        "",
        "## Chat Transcript",
        "",
    ]

    if chat_turns:
        for turn in chat_turns:
            lines.append(f"> **{turn['turn']}. {turn['speaker']}**: {turn['message']}")
    else:
        lines.append("> 대화 내역이 비어 있습니다.")
    lines.append(f"> {chat_result_line}")

    lines.extend(
        [
            "",
            "## Quantitative Scores",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Num Agents | {_fmt_num(metrics.get('num_agents'), 0)} |",
            f"| Social Welfare | {_fmt_num(metrics['social_welfare'])} |",
            f"| Nash Product | {_fmt_num(metrics['nash_product'])} |",
            f"| Welfare Ratio vs Best | {_fmt_pct(metrics['welfare_ratio_vs_best'])} |",
            f"| Nash Ratio vs Best | {_fmt_pct(metrics['nash_ratio_vs_best'])} |",
            f"| Fairness Gap | {_fmt_num(metrics['fairness_gap'])} |",
        ]
    )
    lines.extend(_agent_rows(metrics, "utility_by_agent", "Utility", _fmt_num))
    lines.extend(_agent_rows(metrics, "reservation_by_agent", "Reservation", _fmt_num))
    lines.extend(_agent_rows(metrics, "calendar_conflict_ratio_by_agent", "Calendar Conflict", _fmt_pct))
    lines.extend(
        [
            "",
            "## Detailed JSON",
            "",
            "<details>",
            "<summary><code>quantitative_metrics</code></summary>",
            "",
            "```json",
            _json_dumps(metrics),
            "```",
            "",
            "</details>",
            "",
            "<details>",
            "<summary><code>detailed_json.scenario</code></summary>",
            "",
            "```json",
            _json_dumps(detailed_json["scenario"]),
            "```",
            "",
            "</details>",
            "",
            "<details>",
            "<summary><code>detailed_json.dialogue_human_readable</code></summary>",
            "",
            "```json",
            _json_dumps(detailed_json["dialogue_human_readable"]),
            "```",
            "",
            "</details>",
            "",
            "<details>",
            "<summary><code>detailed_json.agent_event_log</code></summary>",
            "",
            "```json",
            _json_dumps(detailed_json["agent_event_log"]),
            "```",
            "",
            "</details>",
            "",
            "<details>",
            "<summary><code>detailed_json.negmas_trace</code></summary>",
            "",
            "```json",
            _json_dumps(detailed_json["negmas_trace"]),
            "```",
            "",
            "</details>",
        ]
    )

    return "\n".join(lines) + "\n"


def _build_issue_collection_md(
    scenario_reports: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> str:
    lines = [
        "# GitHub Issue Collection",
        "",
        "## Experiment Overview",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Num Scenarios | {aggregate.get('num_scenarios', '-')} |",
        f"| Agreement Rate | {_fmt_pct(aggregate.get('agreement_rate'))} |",
        f"| Pareto Rate | {_fmt_pct(aggregate.get('pareto_rate'))} |",
        f"| Avg Steps | {_fmt_num(aggregate.get('avg_steps'), 2)} |",
        f"| Avg Social Welfare | {_fmt_num(aggregate.get('avg_social_welfare'))} |",
        f"| Avg Nash Product | {_fmt_num(aggregate.get('avg_nash_product'))} |",
        "",
        "## Scenario Issue Files",
        "",
        "| Scenario | Issue Markdown | Issue JSON | Result |",
        "|---|---|---|---|",
    ]

    for report in scenario_reports:
        md_path = f"{report['scenario_id']}/{report['issue_md_file']}"
        json_path = f"{report['scenario_id']}/{report['issue_json_file']}"
        lines.append(
            "| {sid} | `{md}` | `{js}` | {res} |".format(
                sid=report["scenario_id"],
                md=md_path,
                js=json_path,
                res=report["result_summary"],
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- 각 `issue_bundle.md` 파일은 GitHub Issue 본문으로 바로 붙여 넣을 수 있는 형식입니다.",
            "- `issue_bundle.json`은 동일 내용을 구조화해 자동 파이프라인에서 재사용할 수 있습니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_single_scenario(
    scenario: dict[str, Any],
    llms_by_agent: dict[str, LLMBackend],
    output_dir: Path,
    max_steps: int,
    seed: int,
    llm_max_new_tokens: int,
    decision_policy: str,
    require_explicit_accept: bool,
) -> dict[str, Any]:
    agent_names = scenario_agent_names(scenario)
    missing_llms = [name for name in agent_names if name not in llms_by_agent]
    if missing_llms:
        raise ValueError(f"Missing llm backend for agents: {missing_llms}")

    issues = [
        make_issue(scenario["destinations"], name="destination"),
        make_issue(scenario["travel_windows"], name="travel_window"),
        make_issue(scenario["budgets"], name="budget"),
    ]

    ufuns_by_agent = {
        name: _build_ufun(scenario["agents"][name], issues)
        for name in agent_names
    }

    recorder = EventRecorder()

    mechanism = SAOMechanism(issues=issues, n_steps=max_steps)
    for idx, name in enumerate(agent_names):
        negotiator = LLMCalendarNegotiator(
            llm=llms_by_agent[name],
            recorder=recorder,
            role_name=name,
            profile=scenario["agents"][name],
            rng=random.Random(seed + idx * 1009),
            llm_max_new_tokens=llm_max_new_tokens,
            decision_policy=decision_policy,
            require_explicit_accept=require_explicit_accept,
            name=name,
        )
        mechanism.add(negotiator, preferences=ufuns_by_agent[name])

    state = mechanism.run()
    agreement_tuple = state.agreement if state.agreement else None

    scenario_out = output_dir / scenario["scenario_id"]
    scenario_out.mkdir(parents=True, exist_ok=True)

    trace_rows = [
        _trace_to_json(t, scenario=scenario, ufuns_by_agent=ufuns_by_agent)
        for t in mechanism.full_trace
    ]
    _jsonl_dump(scenario_out / "negmas_trace.jsonl", trace_rows)

    _jsonl_dump(scenario_out / "agent_event_log.jsonl", recorder.events)

    human_dialogue = [
        enrich_event_for_human(event, scenario=scenario, ufuns_by_agent=ufuns_by_agent)
        for event in recorder.events
    ]

    metrics = compute_metrics(
        scenario=scenario,
        state=state,
        agreement_tuple=agreement_tuple,
        ufuns_by_agent=ufuns_by_agent,
        events=recorder.events,
    )
    estimated_max_turns = max_steps * max(2, len(agent_names) * 2)
    turns_for_fallback = sum(1 for e in recorder.events if str(e.get("message") or "").strip())
    try:
        computed_my = compute_my_metrics(
            scenario=scenario,
            metrics=metrics,
            events=recorder.events,
            ufuns_by_agent=ufuns_by_agent,
            max_turns=estimated_max_turns,
        )
    except Exception as exc:
        print(
            f"[WARN] failed to compute my_metrics scenario={scenario.get('scenario_id')} error={exc}",
            flush=True,
        )
        computed_my = None
    my_metrics = finalize_my_metrics_shape(
        my_metrics=computed_my,
        agreement=bool(metrics.get("agreement_reached")),
        turns=turns_for_fallback,
    )
    metrics_with_my = dict(metrics)
    metrics_with_my["my_metrics"] = my_metrics

    _json_dump(
        scenario_out / "dialogue_human_readable.json",
        {
            "scenario_id": scenario["scenario_id"],
            "turns": human_dialogue,
            "metrics": metrics_with_my,
        },
    )

    chat_turns = _to_chat_only_turns(human_dialogue)
    _write_chat_outputs(
        scenario_out=scenario_out,
        scenario_id=scenario["scenario_id"],
        chat_turns=chat_turns,
        metrics=metrics_with_my,
    )

    _json_dump(scenario_out / "metrics.json", metrics_with_my)
    _text_dump(
        scenario_out / "metrics_human_readable.md",
        _scenario_metrics_md(scenario_id=scenario["scenario_id"], metrics=metrics_with_my),
    )
    _json_dump(scenario_out / "scenario.json", scenario)

    issue_bundle = _build_issue_bundle(
        scenario=scenario,
        metrics=metrics_with_my,
        chat_turns=chat_turns,
        human_dialogue=human_dialogue,
        trace_rows=trace_rows,
        agent_events=recorder.events,
    )

    issue_json_file = "issue_bundle.json"
    issue_md_file = "issue_bundle.md"
    _json_dump(scenario_out / issue_json_file, issue_bundle)
    _text_dump(scenario_out / issue_md_file, _build_issue_md(issue_bundle))

    return {
        "scenario_id": scenario["scenario_id"],
        "metrics": metrics,
        "my_metrics": my_metrics,
        "result_summary": issue_bundle["result_summary"],
        "issue_json_file": issue_json_file,
        "issue_md_file": issue_md_file,
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_dict(value, prefix=full_key))
        else:
            out[full_key] = value
    return out


def _flatten_numeric(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_numeric(value, prefix=full_key))
        elif isinstance(value, bool):
            out[full_key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            out[full_key] = float(value)
    return out


def _write_my_metrics_outputs(
    output_root: Path,
    scenario_reports: list[dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    numeric_by_metric: dict[str, list[float]] = {}
    per_scenario_stats: dict[str, dict[str, Any]] = {}

    for report in scenario_reports:
        scenario_id = str(report.get("scenario_id"))
        my_metrics = report.get("my_metrics") or {}
        flat_all = _flatten_dict(my_metrics)
        flat_numeric = _flatten_numeric(my_metrics)

        row = {"scenario_id": scenario_id}
        row.update(flat_all)
        rows.append(row)

        for metric_name, value in flat_numeric.items():
            numeric_by_metric.setdefault(metric_name, []).append(value)

        per_scenario_stats[scenario_id] = {
            metric_name: {
                "mean": value,
                "std": 0.0,
                "count": 1,
            }
            for metric_name, value in flat_numeric.items()
        }

    _write_summary_csv(output_root / "my_metrics.csv", rows)

    metric_stats: dict[str, dict[str, Any]] = {}
    for metric_name, values in numeric_by_metric.items():
        if not values:
            continue
        mu = sum(values) / len(values)
        variance = sum((v - mu) ** 2 for v in values) / len(values)
        metric_stats[metric_name] = {
            "mean": round(mu, 6),
            "std": round(variance**0.5, 6),
            "count": len(values),
        }

    _json_dump(
        output_root / "my_metrics_summary.json",
        {
            "run_at": datetime.now().isoformat(),
            "num_scenarios": len(scenario_reports),
            "per_run": metric_stats,
            "per_scenario": per_scenario_stats,
        },
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    reached = [row for row in rows if row.get("agreement_reached")]

    def _avg(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    def _avg_non_null(values: list[Any]) -> float | None:
        clean = [float(v) for v in values if v is not None]
        if not clean:
            return None
        return sum(clean) / len(clean)

    all_agent_names: list[str] = []
    for row in rows:
        for name in row.get("agent_names", []):
            if name not in all_agent_names:
                all_agent_names.append(name)

    avg_utility_by_agent: dict[str, float | None] = {}
    avg_calendar_conflict_by_agent: dict[str, float | None] = {}
    for name in all_agent_names:
        avg_utility_by_agent[name] = _avg_non_null(
            [_metric_by_agent(r, "utility_by_agent").get(name) for r in rows]
        )
        avg_calendar_conflict_by_agent[name] = _avg_non_null(
            [_metric_by_agent(r, "calendar_conflict_ratio_by_agent").get(name) for r in rows]
        )

    aggregate = {
        "run_at": datetime.now().isoformat(),
        "num_scenarios": len(rows),
        "num_agents": max(len(r.get("agent_names", [])) for r in rows),
        "agreement_rate": len(reached) / len(rows),
        "pareto_rate": sum(1 for row in rows if row.get("pareto_optimal")) / len(rows),
        "avg_social_welfare": _avg([float(r["social_welfare"]) for r in rows]),
        "avg_nash_product": _avg([float(r["nash_product"]) for r in rows]),
        "avg_steps": _avg([float(r["negotiation_steps"]) for r in rows]),
        "avg_welfare_ratio_vs_best": _avg([float(r["welfare_ratio_vs_best"]) for r in rows]),
        "avg_nash_ratio_vs_best": _avg([float(r["nash_ratio_vs_best"]) for r in rows]),
        "agent_names": all_agent_names,
        "avg_utility_by_agent": avg_utility_by_agent,
        "avg_calendar_conflict_by_agent": avg_calendar_conflict_by_agent,
    }
    aggregate["avg_utility_agent_a"] = avg_utility_by_agent.get("agent_a")
    aggregate["avg_utility_agent_b"] = avg_utility_by_agent.get("agent_b")
    aggregate["avg_calendar_conflict_agent_a"] = avg_calendar_conflict_by_agent.get("agent_a")
    aggregate["avg_calendar_conflict_agent_b"] = avg_calendar_conflict_by_agent.get("agent_b")
    return aggregate


def _build_quant_pretty(aggregate: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    avg_utility_map = aggregate.get("avg_utility_by_agent", {})
    avg_conflict_map = aggregate.get("avg_calendar_conflict_by_agent", {})
    overview = {
        "num_scenarios": aggregate.get("num_scenarios"),
        "num_agents": aggregate.get("num_agents"),
        "agreement_rate": _fmt_pct(aggregate.get("agreement_rate")),
        "pareto_rate": _fmt_pct(aggregate.get("pareto_rate")),
        "avg_steps": _fmt_num(aggregate.get("avg_steps"), 2),
        "avg_social_welfare": _fmt_num(aggregate.get("avg_social_welfare"), 4),
        "avg_nash_product": _fmt_num(aggregate.get("avg_nash_product"), 4),
        "avg_welfare_ratio_vs_best": _fmt_pct(aggregate.get("avg_welfare_ratio_vs_best")),
        "avg_nash_ratio_vs_best": _fmt_pct(aggregate.get("avg_nash_ratio_vs_best")),
        "avg_utility_by_agent": {k: _fmt_num(v) for k, v in avg_utility_map.items()},
        "avg_calendar_conflict_by_agent": {k: _fmt_pct(v) for k, v in avg_conflict_map.items()},
    }

    scenarios = []
    for row in rows:
        utility_by_agent = _metric_by_agent(row, "utility_by_agent")
        conflict_by_agent = _metric_by_agent(row, "calendar_conflict_ratio_by_agent")
        scenarios.append(
            {
                "scenario_id": row.get("scenario_id"),
                "num_agents": row.get("num_agents"),
                "agreement": _fmt_bool(bool(row.get("agreement_reached"))),
                "pareto": _fmt_bool(bool(row.get("pareto_optimal"))),
                "steps": _fmt_num(row.get("negotiation_steps"), 0),
                "utility_by_agent": {k: _fmt_num(v) for k, v in utility_by_agent.items()},
                "calendar_conflict_ratio_by_agent": {k: _fmt_pct(v) for k, v in conflict_by_agent.items()},
                "social_welfare": _fmt_num(row.get("social_welfare")),
                "nash_product": _fmt_num(row.get("nash_product")),
                "welfare_ratio_vs_best": _fmt_pct(row.get("welfare_ratio_vs_best")),
                "nash_ratio_vs_best": _fmt_pct(row.get("nash_ratio_vs_best")),
            }
        )

    return {
        "overview": overview,
        "scenario_rows": scenarios,
    }


def _build_quant_summary_md(aggregate: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    def _agent_compact_text(row: dict[str, Any], field: str, formatter) -> str:
        by_agent = _metric_by_agent(row, field)
        if not by_agent:
            return "-"
        names = row.get("agent_names")
        if not isinstance(names, list) or not names:
            names = list(by_agent.keys())
        parts = [f"{name}:{formatter(by_agent.get(name))}" for name in names if name in by_agent]
        return ", ".join(parts) if parts else "-"

    lines = [
        "# Quantitative Summary",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Num Scenarios | {aggregate.get('num_scenarios', '-')} |",
        f"| Num Agents | {aggregate.get('num_agents', '-')} |",
        f"| Agreement Rate | {_fmt_pct(aggregate.get('agreement_rate'))} |",
        f"| Pareto Rate | {_fmt_pct(aggregate.get('pareto_rate'))} |",
        f"| Avg Steps | {_fmt_num(aggregate.get('avg_steps'), 2)} |",
        f"| Avg Social Welfare | {_fmt_num(aggregate.get('avg_social_welfare'))} |",
        f"| Avg Nash Product | {_fmt_num(aggregate.get('avg_nash_product'))} |",
        f"| Avg Welfare Ratio vs Best | {_fmt_pct(aggregate.get('avg_welfare_ratio_vs_best'))} |",
        f"| Avg Nash Ratio vs Best | {_fmt_pct(aggregate.get('avg_nash_ratio_vs_best'))} |",
    ]

    for name, value in (aggregate.get("avg_utility_by_agent") or {}).items():
        lines.append(f"| Avg Utility ({name}) | {_fmt_num(value)} |")
    for name, value in (aggregate.get("avg_calendar_conflict_by_agent") or {}).items():
        lines.append(f"| Avg Calendar Conflict ({name}) | {_fmt_pct(value)} |")

    lines.append("")
    lines.append("## Scenario Table")
    lines.append("")
    lines.append("| Scenario | Agree | Pareto | Steps | Utilities | Welfare | Nash | Welfare/Best | Nash/Best | Conflicts |")
    lines.append("|---|---|---|---:|---|---:|---:|---:|---:|---|")

    for row in rows:
        lines.append(
            "| {sid} | {agr} | {par} | {st} | {us} | {sw} | {np} | {wr} | {nr} | {cf} |".format(
                sid=row.get("scenario_id", "-"),
                agr=_fmt_bool(bool(row.get("agreement_reached"))),
                par=_fmt_bool(bool(row.get("pareto_optimal"))),
                st=_fmt_num(row.get("negotiation_steps"), 0),
                us=_agent_compact_text(row, "utility_by_agent", _fmt_num),
                sw=_fmt_num(row.get("social_welfare")),
                np=_fmt_num(row.get("nash_product")),
                wr=_fmt_pct(row.get("welfare_ratio_vs_best")),
                nr=_fmt_pct(row.get("nash_ratio_vs_best")),
                cf=_agent_compact_text(row, "calendar_conflict_ratio_by_agent", _fmt_pct),
            )
        )

    return "\n".join(lines) + "\n"


def run_experiment(
    scenarios: list[dict[str, Any]],
    output_root: Path,
    llms_by_agent: dict[str, LLMBackend],
    max_steps: int,
    seed: int,
    llm_max_new_tokens: int,
    decision_policy: str,
    require_explicit_accept: bool,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)

    scenario_reports: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    total = len(scenarios)
    for idx, scenario in enumerate(scenarios):
        scenario_id = scenario.get("scenario_id", f"scenario_{idx+1:04d}")
        t0 = time.time()
        print(f"[SCENARIO][{idx+1}/{total}] start id={scenario_id}", flush=True)
        scenario_report = run_single_scenario(
            scenario=scenario,
            llms_by_agent=llms_by_agent,
            output_dir=output_root,
            max_steps=max_steps,
            seed=seed + idx,
            llm_max_new_tokens=llm_max_new_tokens,
            decision_policy=decision_policy,
            require_explicit_accept=require_explicit_accept,
        )
        scenario_reports.append(scenario_report)
        all_metrics.append(scenario_report["metrics"])
        elapsed = time.time() - t0
        metric = scenario_report["metrics"]
        print(
            f"[SCENARIO][{idx+1}/{total}] done id={scenario_id} "
            f"agreement={metric.get('agreement_reached')} steps={metric.get('negotiation_steps')} "
            f"elapsed_sec={elapsed:.1f}",
            flush=True,
        )

    _write_summary_csv(output_root / "experiment_summary.csv", all_metrics)

    aggregate = _aggregate(all_metrics)
    _json_dump(output_root / "experiment_summary.json", aggregate)

    quant_pretty = _build_quant_pretty(aggregate=aggregate, rows=all_metrics)
    _json_dump(output_root / "quant_summary_human_readable.json", quant_pretty)
    _text_dump(output_root / "quant_summary.md", _build_quant_summary_md(aggregate=aggregate, rows=all_metrics))
    _write_my_metrics_outputs(output_root=output_root, scenario_reports=scenario_reports)
    _text_dump(
        output_root / "github_issue_collection.md",
        _build_issue_collection_md(scenario_reports=scenario_reports, aggregate=aggregate),
    )

    return {
        "metrics": all_metrics,
        "aggregate": aggregate,
        "scenario_reports": scenario_reports,
    }
