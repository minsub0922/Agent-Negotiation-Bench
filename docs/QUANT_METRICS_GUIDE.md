# Quant Metrics Guide

This document is a guide for interpreting quantitative metrics from run outputs.

## Core Metrics

| Metric | Meaning | Interpretation Hint |
|---|---|---|
| `agreement_reached` | Whether agreement was reached in the scenario | If many are `False`, review concession strategy/reservation settings |
| `utility_agent_a`, `utility_agent_b` | Utility of each agent under the final agreement | A large gap can indicate unfairness |
| `social_welfare` | `utility_agent_a + utility_agent_b` | Overall efficiency indicator |
| `nash_product` | `max(u_a-r_a,0) * max(u_b-r_b,0)` | Tradeoff indicator between efficiency and fairness |
| `pareto_optimal` | `True` if the agreement is Pareto optimal | If `False`, there exists an agreement that can improve someone without hurting others |
| `welfare_ratio_vs_best` | Ratio against max welfare in full outcome space | Closer to 1.0 means more efficient |
| `nash_ratio_vs_best` | Ratio against max Nash product in full outcome space | Closer to 1.0 means more balanced |
| `negotiation_steps` | Number of steps until agreement/termination | If too large, tune exploration/concession speed |
| `calendar_conflict_ratio_agent_a/b` | Busy-slot ratio on agreed travel days | Lower is better (fewer calendar conflicts) |

## Aggregate File Usage

- `experiment_summary.csv`
For per-scenario raw numeric analysis (statistics, plotting, regression, etc.).

- `experiment_summary.json`
Raw run-level aggregate numbers (means/ratios).

- `quant_summary_human_readable.json`
Human-readable JSON with formatting (percentages, rounding).

- `quant_summary.md`
Markdown summary you can paste directly into reports.

- `scenario_xxxx/metrics_human_readable.md`
Detailed per-scenario summary.

## Quick Analysis Checklist

1. If `agreement_rate` is low, adjust `reservation_value`, `concession_exponent`, and `max_steps`.
2. If `welfare_ratio_vs_best` is high but `nash_ratio_vs_best` is low, check for one-sided agreements.
3. If `calendar_conflict_ratio_agent_a/b` is high, increase window-score weight or calendar-constraint weight.
4. If `pareto_rate` is low, review offer-generation policy (top-band sampling, aspiration curve).

## How to Read with Chat Logs

- `chat_transcript.txt`: Inspect actual dialogue flow
- `metrics_human_readable.md`: Inspect quantitative results for the same scenario

Reading these together helps you quickly identify which utterance patterns affect agreement efficiency and fairness.
