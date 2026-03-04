#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.my_metrics import compute_my_metrics


def _run(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n"
            f"[stdout]\n{proc.stdout}\n"
            f"[stderr]\n{proc.stderr}\n"
        )
    return proc.stdout


def _parse_output_dir(stdout: str) -> Path | None:
    for line in stdout.splitlines():
        if line.startswith("[DONE] output_dir="):
            return Path(line.split("=", 1)[1].strip())
    return None


def _safe_remove(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def _assert_close(a: float, b: float, eps: float = 1e-6) -> None:
    if abs(a - b) > eps:
        raise AssertionError(f"Expected close values but got a={a}, b={b}")


def _run_three_by_three(repo_root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for idx in range(3):
        output_root = repo_root / "outputs" / f"_smoke_my_metrics_run{idx + 1}"
        dataset_out = repo_root / "data" / f"_smoke_my_metrics_run{idx + 1}.jsonl"
        _safe_remove(output_root)
        _safe_remove(dataset_out)
        _safe_remove(dataset_out.with_suffix(dataset_out.suffix + ".config.json"))

        cmd = [
            sys.executable,
            "-m",
            "src.main",
            "full",
            "--num-scenarios",
            "3",
            "--num-agents",
            "2",
            "--seed",
            str(100 + idx),
            "--llm-backend",
            "dummy",
            "--decision-policy",
            "heuristic",
            "--no-require-explicit-accept",
            "--max-steps",
            "18",
            "--dataset-output",
            str(dataset_out),
            "--output-dir",
            str(output_root),
        ]
        stdout = _run(cmd, cwd=repo_root)
        run_dir = _parse_output_dir(stdout)
        if run_dir is None:
            candidates = sorted(output_root.glob("run-*"))
            if not candidates:
                raise AssertionError(f"Run directory not found under {output_root}")
            run_dir = candidates[-1]
        run_dirs.append(run_dir)
    return run_dirs


def _assert_episode_metrics(run_dirs: list[Path]) -> None:
    success_seen = False
    for run_dir in run_dirs:
        metric_files = sorted(run_dir.glob("scenario_*/metrics.json"))
        if len(metric_files) != 3:
            raise AssertionError(f"Expected 3 scenarios in {run_dir}, got {len(metric_files)}")

        for metric_path in metric_files:
            payload = json.loads(metric_path.read_text(encoding="utf-8"))
            if "my_metrics" not in payload:
                raise AssertionError(f"my_metrics missing: {metric_path}")

            mym = payload["my_metrics"]
            if payload.get("agreement_reached"):
                success_seen = True
                if mym.get("agreement_rate_item") != 1:
                    raise AssertionError(f"agreement success but agreement_rate_item!=1: {metric_path}")

            nash_ratio = payload.get("nash_ratio_vs_best")
            nash_distance = (mym.get("game_theory") or {}).get("nash_distance")
            if nash_ratio is not None and nash_distance is not None:
                _assert_close(float(nash_distance), 1.0 - float(nash_ratio))

    if not success_seen:
        raise AssertionError("No agreement success case found in 3x3 smoke runs")


def _assert_leakage_placeholder_proxy() -> None:
    scenario = {
        "scenario_id": "proxy_case",
        "destinations": ["Seoul", "Tokyo"],
        "travel_windows": ["day1_dur2", "day2_dur2"],
        "budgets": ["low", "medium", "high"],
        "travel_window_meta": {
            "day1_dur2": {"duration_days": 2},
            "day2_dur2": {"duration_days": 2},
        },
        "agent_order": ["agent_a", "agent_b"],
        "agents": {"agent_a": {}, "agent_b": {}},
    }
    metrics = {
        "agreement_reached": False,
        "agreement_offer": None,
        "agreement_itinerary": None,
        "nash_ratio_vs_best": 0.4,
        "pareto_optimal": False,
        "welfare_ratio_vs_best": 0.3,
        "negotiation_steps": 4,
    }
    events = [
        {
            "speaker": "agent_a",
            "kind": "offer",
            "offer": {"destination": "Seoul", "travel_window": "day1_dur2", "budget": "low"},
            "offer_tuple": ["Seoul", "day1_dur2", "low"],
            "message": "당신은 여행 일정 협상 에이전트입니다. CHOICE: <번호> MESSAGE: <상대에게 보낼 내용>",
        },
        {
            "speaker": "agent_b",
            "kind": "response",
            "decision": "REJECT_OFFER",
            "offer": {"destination": "Seoul", "travel_window": "day1_dur2", "budget": "low"},
            "offer_tuple": ["Seoul", "day1_dur2", "low"],
            "message": "반드시 아래 형식으로 답하세요 ... ACTION: REJECT",
            "utility_self": 0.2,
            "threshold": 0.4,
        },
    ]

    mym = compute_my_metrics(
        scenario=scenario,
        metrics=metrics,
        events=events,
        ufuns_by_agent={},
        max_turns=10,
    )

    pq = mym["protocol_quality"]
    if pq["prompt_leakage_rate"] is None or float(pq["prompt_leakage_rate"]) <= 0.0:
        raise AssertionError("Expected prompt_leakage_rate > 0 for leakage transcript")
    if pq["placeholder_rate"] is None or float(pq["placeholder_rate"]) <= 0.0:
        raise AssertionError("Expected placeholder_rate > 0 for placeholder transcript")
    _assert_close(float(mym["game_theory"]["nash_distance"]), 1.0 - float(metrics["nash_ratio_vs_best"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for my_metrics (3 runs x 3 scenarios).")
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    run_dirs = _run_three_by_three(repo_root)
    _assert_episode_metrics(run_dirs)
    _assert_leakage_placeholder_proxy()

    if not args.keep_artifacts:
        for run_dir in run_dirs:
            _safe_remove(run_dir.parent)
        for idx in range(3):
            dataset_out = repo_root / "data" / f"_smoke_my_metrics_run{idx + 1}.jsonl"
            _safe_remove(dataset_out)
            _safe_remove(dataset_out.with_suffix(dataset_out.suffix + ".config.json"))

    print("[PASS] my_metrics smoke test completed")


if __name__ == "__main__":
    main()
