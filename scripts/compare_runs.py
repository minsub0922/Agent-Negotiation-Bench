#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


MODE_CHOICES = ("answers", "answers-scores", "scores")


METRIC_SPECS: dict[str, dict[str, str]] = {
    "agreement": {"label": "agreement", "key": "agreement_reached", "format": "bool"},
    "steps": {"label": "steps", "key": "negotiation_steps", "format": "int"},
    "u_a": {"label": "u_a", "key": "utility_agent_a", "format": "float4"},
    "u_b": {"label": "u_b", "key": "utility_agent_b", "format": "float4"},
    "welfare": {"label": "welfare", "key": "social_welfare", "format": "float4"},
    "welfare_ratio": {"label": "welfare_ratio", "key": "welfare_ratio_vs_best", "format": "pct"},
    "nash": {"label": "nash", "key": "nash_product", "format": "float4"},
    "nash_ratio": {"label": "nash_ratio", "key": "nash_ratio_vs_best", "format": "pct"},
    "pareto": {"label": "pareto", "key": "pareto_optimal", "format": "bool"},
    "fairness": {"label": "fairness", "key": "fairness_gap", "format": "float4"},
    "conflict_a": {
        "label": "conflict_a",
        "key": "calendar_conflict_ratio_agent_a",
        "format": "pct",
    },
    "conflict_b": {
        "label": "conflict_b",
        "key": "calendar_conflict_ratio_agent_b",
        "format": "pct",
    },
}

ALL_METRIC_ALIASES = tuple(METRIC_SPECS.keys())


MY_METRIC_SPECS: dict[str, dict[str, str]] = {
    "agreement_item": {"label": "agreement_item", "path": "agreement_rate_item", "format": "int"},
    "valid_agreement": {"label": "valid_agreement", "path": "valid_agreement", "format": "bool"},
    "turns": {"label": "turns", "path": "turns", "format": "int"},
    "turn_eff": {"label": "turn_eff", "path": "turn_efficiency", "format": "float4"},
    "quality_eff": {"label": "quality_eff", "path": "quality_efficiency", "format": "float4"},
    "u_a_norm": {"label": "u_a_norm", "path": "payoff.u_a_norm", "format": "float4"},
    "u_b_norm": {"label": "u_b_norm", "path": "payoff.u_b_norm", "format": "float4"},
    "welfare_norm": {"label": "welfare_norm", "path": "payoff.welfare_norm", "format": "float4"},
    "regret_a_norm": {"label": "regret_a_norm", "path": "payoff.regret_a_norm", "format": "float4"},
    "regret_b_norm": {"label": "regret_b_norm", "path": "payoff.regret_b_norm", "format": "float4"},
    "nash_distance": {"label": "nash_distance", "path": "game_theory.nash_distance", "format": "float4"},
    "pareto_item": {"label": "pareto_item", "path": "game_theory.pareto_optimal_item", "format": "int"},
    "action_consistency": {
        "label": "action_consistency",
        "path": "protocol_quality.action_message_consistency",
        "format": "float4",
    },
    "leakage": {"label": "leakage", "path": "protocol_quality.prompt_leakage_rate", "format": "pct"},
    "placeholder": {"label": "placeholder", "path": "protocol_quality.placeholder_rate", "format": "pct"},
    "non_korean": {"label": "non_korean", "path": "protocol_quality.non_korean_rate", "format": "pct"},
    "truncated": {"label": "truncated", "path": "protocol_quality.truncated_turn_rate", "format": "pct"},
    "politeness": {
        "label": "politeness",
        "path": "interaction_fairness.interpersonal_politeness_score",
        "format": "float4",
    },
    "toxicity": {"label": "toxicity", "path": "interaction_fairness.toxicity_flag_rate", "format": "pct"},
    "reason_rate": {
        "label": "reason_rate",
        "path": "interaction_fairness.informational_reason_rate",
        "format": "pct",
    },
    "specificity": {
        "label": "specificity",
        "path": "interaction_fairness.informational_specificity_score",
        "format": "float4",
    },
    "threshold_consistency": {
        "label": "threshold_consistency",
        "path": "internal_faithfulness.accept_threshold_consistency",
        "format": "float4",
    },
    "choice_opt_k": {
        "label": "choice_opt_k",
        "path": "internal_faithfulness.choice_optimality_at_k",
        "format": "float4",
    },
    "reason_align": {
        "label": "reason_align",
        "path": "internal_faithfulness.reason_utility_alignment",
        "format": "float4",
    },
}
ALL_MY_METRIC_ALIASES = tuple(MY_METRIC_SPECS.keys())


@dataclass
class RunInfo:
    run_dir: Path
    run_id: str
    model_signature: str
    model_label: str
    scenario_data: dict[str, dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _short_model_name(path_value: str | None) -> str | None:
    if not path_value:
        return None
    return Path(path_value.rstrip("/")).name or None


def _resolve_model_label(run_cfg: dict[str, Any] | None, fallback: str) -> str:
    if not run_cfg:
        return fallback

    a_model = _short_model_name(run_cfg.get("agent_a_model_path"))
    b_model = _short_model_name(run_cfg.get("agent_b_model_path"))
    single_model = _short_model_name(run_cfg.get("model_path"))

    if a_model and b_model:
        if a_model == b_model:
            return a_model
        return f"a:{a_model} | b:{b_model}"
    if single_model:
        return single_model

    signature = run_cfg.get("model_signature")
    if signature:
        return str(signature)
    return fallback


def _load_run(run_dir: Path) -> RunInfo:
    run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_cfg = _read_json(run_dir / "run_config.json")
    run_id = str(run_cfg.get("run_id")) if run_cfg and run_cfg.get("run_id") else run_dir.name
    model_signature = (
        str(run_cfg.get("model_signature")) if run_cfg and run_cfg.get("model_signature") else "unknown-model"
    )
    model_label = _resolve_model_label(run_cfg, fallback=model_signature)

    scenario_data: dict[str, dict[str, Any]] = {}
    for scenario_dir in sorted(run_dir.glob("scenario_*")):
        if not scenario_dir.is_dir():
            continue
        scenario_id = scenario_dir.name
        metrics = _read_json(scenario_dir / "metrics.json") or {}
        chat = _read_json(scenario_dir / "chat_transcript.json") or {}
        scenario_data[scenario_id] = {
            "metrics": metrics,
            "chat": chat,
        }

    return RunInfo(
        run_dir=run_dir,
        run_id=run_id,
        model_signature=model_signature,
        model_label=model_label,
        scenario_data=scenario_data,
    )


def _fmt_bool(value: Any) -> str:
    return "Yes" if bool(value) else "No"


def _fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        v = float(value) * 100.0
        return f"{v:.2f}%"
    except Exception:
        return str(value)


def _parse_metric_aliases(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text or text.lower() == "all":
        return list(ALL_METRIC_ALIASES)

    aliases: list[str] = []
    for token in text.split(","):
        alias = token.strip()
        if not alias:
            continue
        if alias not in METRIC_SPECS:
            available = ", ".join(ALL_METRIC_ALIASES)
            raise ValueError(f"Unknown metric alias '{alias}'. Available: {available}")
        if alias not in aliases:
            aliases.append(alias)

    if not aliases:
        available = ", ".join(ALL_METRIC_ALIASES)
        raise ValueError(f"No valid metric alias provided. Available: {available}")
    return aliases


def _parse_my_metric_aliases(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text or text.lower() == "all":
        return list(ALL_MY_METRIC_ALIASES)

    aliases: list[str] = []
    for token in text.split(","):
        alias = token.strip()
        if not alias:
            continue
        if alias not in MY_METRIC_SPECS:
            available = ", ".join(ALL_MY_METRIC_ALIASES)
            raise ValueError(f"Unknown my-metric alias '{alias}'. Available: {available}")
        if alias not in aliases:
            aliases.append(alias)

    if not aliases:
        available = ", ".join(ALL_MY_METRIC_ALIASES)
        raise ValueError(f"No valid my-metric alias provided. Available: {available}")
    return aliases


def _render_one_line_metrics(metrics: dict[str, Any], metric_aliases: list[str]) -> str:
    parts: list[str] = []
    for alias in metric_aliases:
        spec = METRIC_SPECS[alias]
        raw = metrics.get(spec["key"])
        fmt = spec["format"]
        if fmt == "bool":
            value = _fmt_bool(raw)
        elif fmt == "int":
            value = _fmt_num(raw, 0)
        elif fmt == "pct":
            value = _fmt_pct(raw)
        else:
            value = _fmt_num(raw, 4)
        parts.append(f"{spec['label']}={value}")
    return " | ".join(parts)


def _get_nested(payload: dict[str, Any], path: str) -> Any:
    cur: Any = payload
    for token in path.split("."):
        if not isinstance(cur, dict) or token not in cur:
            return None
        cur = cur[token]
    return cur


def _render_one_line_my_metrics(metrics: dict[str, Any], my_metric_aliases: list[str]) -> str:
    my_metrics = metrics.get("my_metrics")
    if not isinstance(my_metrics, dict):
        return "(missing)"

    parts: list[str] = []
    for alias in my_metric_aliases:
        spec = MY_METRIC_SPECS[alias]
        raw = _get_nested(my_metrics, spec["path"])
        fmt = spec["format"]
        if fmt == "bool":
            value = _fmt_bool(raw)
        elif fmt == "int":
            value = _fmt_num(raw, 0)
        elif fmt == "pct":
            value = _fmt_pct(raw)
        else:
            value = _fmt_num(raw, 4)
        parts.append(f"{spec['label']}={value}")
    return " | ".join(parts)


def _render_run_table(runs: list[RunInfo]) -> list[str]:
    lines = [
        "## Runs",
        "",
        "| Alias | Run ID | Model | Directory |",
        "|---|---|---|---|",
    ]
    for idx, run in enumerate(runs, start=1):
        alias = f"R{idx}"
        lines.append(f"| {alias} | `{run.run_id}` | `{run.model_label}` | `{run.run_dir}` |")
    lines.append("")
    return lines


def _render_scores_table(run_rows: list[tuple[str, RunInfo, dict[str, Any]]]) -> list[str]:
    lines = [
        "| Run | Model | Agreement | Steps | Welfare | Nash | u_A | u_B | Pareto |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for alias, run, metrics in run_rows:
        lines.append(
            "| {alias} | `{model}` | {agree} | {steps} | {w} | {nash} | {ua} | {ub} | {pareto} |".format(
                alias=alias,
                model=run.model_label,
                agree=_fmt_bool(metrics.get("agreement_reached")),
                steps=_fmt_num(metrics.get("negotiation_steps"), 0),
                w=_fmt_num(metrics.get("social_welfare"), 4),
                nash=_fmt_num(metrics.get("nash_product"), 4),
                ua=_fmt_num(metrics.get("utility_agent_a"), 4),
                ub=_fmt_num(metrics.get("utility_agent_b"), 4),
                pareto=_fmt_bool(metrics.get("pareto_optimal")),
            )
        )
    lines.append("")
    return lines


def _render_chat(
    alias: str,
    run: RunInfo,
    chat_payload: dict[str, Any],
    metrics: dict[str, Any],
    max_turns: int,
    one_line_metrics: bool,
    metric_aliases: list[str],
    one_line_my_metrics: bool,
    my_metric_aliases: list[str],
) -> list[str]:
    chat_turns = list(chat_payload.get("chat", []))
    result_summary = chat_payload.get("result_summary") or (
        "Agreement reached" if metrics.get("agreement_reached") else "No agreement"
    )

    lines = [
        f"#### {alias} `{run.run_id}` ({run.model_label})",
        f"- Agreement: {_fmt_bool(metrics.get('agreement_reached'))}",
        f"- Result: {result_summary}",
    ]
    if one_line_metrics:
        lines.append(f"- Metrics: {_render_one_line_metrics(metrics, metric_aliases)}")
    if one_line_my_metrics:
        lines.append(f"- MyMetrics: {_render_one_line_my_metrics(metrics, my_metric_aliases)}")

    if not chat_turns:
        lines.append("- Chat: (empty)")
        lines.append("")
        return lines

    clipped = chat_turns[:max_turns]
    lines.extend([
        "<details>",
        f"<summary>Chat Transcript ({len(chat_turns)} turns)</summary>",
        "",
    ])
    for turn in clipped:
        turn_no = turn.get("turn", "?")
        speaker = turn.get("speaker", "unknown")
        message = str(turn.get("message", "")).strip()
        lines.append(f"> **{turn_no}. {speaker}**: {message}")

    if len(chat_turns) > len(clipped):
        lines.append(f"> ...(truncated {len(chat_turns) - len(clipped)} turns)")

    lines.extend([
        "",
        "</details>",
        "",
    ])
    return lines


def build_report(
    runs: list[RunInfo],
    mode: str,
    max_turns: int,
    one_line_metrics: bool,
    metric_aliases: list[str],
    one_line_my_metrics: bool,
    my_metric_aliases: list[str],
) -> str:
    all_scenarios = sorted({sid for run in runs for sid in run.scenario_data.keys()})

    lines: list[str] = [
        "# Run Comparison Report",
        "",
        f"- Generated at: {datetime.now().isoformat()}",
        f"- Mode: `{mode}`",
        f"- Runs: {len(runs)}",
        f"- Scenarios: {len(all_scenarios)}",
        "",
    ]
    lines.extend(_render_run_table(runs))

    for scenario_id in all_scenarios:
        lines.append(f"## {scenario_id}")
        lines.append("")

        run_rows: list[tuple[str, RunInfo, dict[str, Any], dict[str, Any]]] = []
        for idx, run in enumerate(runs, start=1):
            alias = f"R{idx}"
            payload = run.scenario_data.get(scenario_id, {})
            metrics = payload.get("metrics", {})
            chat = payload.get("chat", {})
            run_rows.append((alias, run, metrics, chat))

        if mode in {"scores", "answers-scores"}:
            lines.extend(_render_scores_table([(a, r, m) for a, r, m, _ in run_rows]))

        if mode == "scores" and one_line_my_metrics:
            lines.append("### My Metrics")
            lines.append("")
            for alias, run, metrics, _ in run_rows:
                lines.append(
                    f"- {alias} `{run.run_id}` ({run.model_label}): "
                    f"{_render_one_line_my_metrics(metrics, my_metric_aliases)}"
                )
            lines.append("")

        if mode in {"answers", "answers-scores"}:
            lines.append("### Answers")
            lines.append("")
            for alias, run, metrics, chat in run_rows:
                lines.extend(
                    _render_chat(
                        alias,
                        run,
                        chat,
                        metrics,
                        max_turns=max_turns,
                        one_line_metrics=one_line_metrics,
                        metric_aliases=metric_aliases,
                        one_line_my_metrics=one_line_my_metrics,
                        my_metric_aliases=my_metric_aliases,
                    )
                )

    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple run directories by scenario (answers and/or scores).",
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more run directories (e.g. outputs/run_a outputs/run_b)",
    )
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="answers-scores",
        help="answers: chat+agreement, answers-scores: chat+agreement+scores, scores: scores only",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=80,
        help="Max turns to display per run in chat mode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown file path (if omitted, print to stdout)",
    )
    parser.add_argument(
        "--one-line-metrics",
        action="store_true",
        help="Show one-line metrics summary under each run result (answers/answers-scores modes)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help=(
            "Comma-separated metric aliases for --one-line-metrics. "
            "Default: all. "
            f"Available: {', '.join(ALL_METRIC_ALIASES)}"
        ),
    )
    parser.add_argument(
        "--one-line-my-metrics",
        action="store_true",
        help="Show one-line my_metrics summary under each run result (or in scores mode section)",
    )
    parser.add_argument(
        "--my-metrics",
        type=str,
        default="all",
        help=(
            "Comma-separated my-metric aliases for --one-line-my-metrics. "
            "Default: all. "
            f"Available: {', '.join(ALL_MY_METRIC_ALIASES)}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.max_turns <= 0:
        raise ValueError("--max-turns must be > 0")

    metric_aliases = _parse_metric_aliases(args.metrics)
    my_metric_aliases = _parse_my_metric_aliases(args.my_metrics)

    run_dirs = [Path(p).expanduser() for p in args.run_dirs]
    runs = [_load_run(p) for p in run_dirs]

    report = build_report(
        runs,
        mode=args.mode,
        max_turns=args.max_turns,
        one_line_metrics=args.one_line_metrics,
        metric_aliases=metric_aliases,
        one_line_my_metrics=args.one_line_my_metrics,
        my_metric_aliases=my_metric_aliases,
    )

    if args.output is None:
        print(report)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"[DONE] report={args.output}")


if __name__ == "__main__":
    main()
