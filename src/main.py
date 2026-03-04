from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .llm_backend import build_llm
from .runner import run_experiment
from .scenario import generate_dataset, load_dataset_jsonl, save_dataset_jsonl


def _save_run_config(output_root: Path, config: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="negmas 기반 여행 일정 협상 시뮬레이션 (2-agent + local LLM)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_gen = subparsers.add_parser("generate-dataset", help="랜덤 시나리오 데이터셋 생성")
    p_gen.add_argument("--num-scenarios", type=int, default=20)
    p_gen.add_argument("--seed", type=int, default=7)
    p_gen.add_argument("--horizon-days", type=int, default=14)
    p_gen.add_argument("--output", type=Path, default=Path("data/scenarios.jsonl"))

    p_run = subparsers.add_parser("run", help="기존 데이터셋으로 협상 실행")
    p_run.add_argument("--dataset", type=Path, required=True)
    p_run.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p_run.add_argument("--max-steps", type=int, default=40)
    p_run.add_argument("--seed", type=int, default=7)
    p_run.add_argument("--num-scenarios", type=int, default=None)

    p_run.add_argument("--llm-backend", choices=["hf", "dummy"], default="hf")
    p_run.add_argument("--model-path", type=str, default=None)
    p_run.add_argument("--agent-a-model-path", type=str, default=None)
    p_run.add_argument("--agent-b-model-path", type=str, default=None)
    p_run.add_argument("--temperature", type=float, default=0.6)
    p_run.add_argument("--top-p", type=float, default=0.9)
    p_run.add_argument("--llm-max-new-tokens", type=int, default=256)
    p_run.add_argument("--device-map", type=str, default="auto")
    p_run.add_argument("--allow-dummy-fallback", action="store_true")

    p_full = subparsers.add_parser("full", help="데이터셋 생성 + 협상 실행")
    p_full.add_argument("--num-scenarios", type=int, default=20)
    p_full.add_argument("--seed", type=int, default=7)
    p_full.add_argument("--horizon-days", type=int, default=14)
    p_full.add_argument("--dataset-output", type=Path, default=Path("data/scenarios.jsonl"))
    p_full.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p_full.add_argument("--max-steps", type=int, default=40)

    p_full.add_argument("--llm-backend", choices=["hf", "dummy"], default="hf")
    p_full.add_argument("--model-path", type=str, default=None)
    p_full.add_argument("--agent-a-model-path", type=str, default=None)
    p_full.add_argument("--agent-b-model-path", type=str, default=None)
    p_full.add_argument("--temperature", type=float, default=0.6)
    p_full.add_argument("--top-p", type=float, default=0.9)
    p_full.add_argument("--llm-max-new-tokens", type=int, default=256)
    p_full.add_argument("--device-map", type=str, default="auto")
    p_full.add_argument("--allow-dummy-fallback", action="store_true")

    return parser.parse_args()


def _resolve_model_paths(args: argparse.Namespace) -> tuple[str | None, str | None]:
    model_path = args.model_path
    a_path = args.agent_a_model_path or model_path
    b_path = args.agent_b_model_path or model_path
    return a_path, b_path


def _build_both_llms(args: argparse.Namespace):
    a_path, b_path = _resolve_model_paths(args)

    llm_a = build_llm(
        backend=args.llm_backend,
        name="agent_a",
        model_path=a_path,
        temperature=args.temperature,
        top_p=args.top_p,
        allow_dummy_fallback=args.allow_dummy_fallback,
        device_map=args.device_map,
    )
    llm_b = build_llm(
        backend=args.llm_backend,
        name="agent_b",
        model_path=b_path,
        temperature=args.temperature,
        top_p=args.top_p,
        allow_dummy_fallback=args.allow_dummy_fallback,
        device_map=args.device_map,
    )
    return llm_a, llm_b


def _run_with_existing_dataset(args: argparse.Namespace) -> None:
    scenarios = load_dataset_jsonl(args.dataset)
    if args.num_scenarios is not None:
        scenarios = scenarios[: args.num_scenarios]

    llm_a, llm_b = _build_both_llms(args)

    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_root = args.output_dir / run_tag
    result = run_experiment(
        scenarios=scenarios,
        output_root=output_root,
        llm_a=llm_a,
        llm_b=llm_b,
        max_steps=args.max_steps,
        seed=args.seed,
        llm_max_new_tokens=args.llm_max_new_tokens,
    )
    _save_run_config(
        output_root,
        {
            "mode": "run",
            "dataset": str(args.dataset),
            "num_scenarios": len(scenarios),
            "llm_backend": args.llm_backend,
            "model_path": args.model_path,
            "agent_a_model_path": args.agent_a_model_path,
            "agent_b_model_path": args.agent_b_model_path,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "llm_max_new_tokens": args.llm_max_new_tokens,
            "device_map": args.device_map,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
    )

    print(f"[DONE] output_dir={output_root}")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


def _run_full(args: argparse.Namespace) -> None:
    scenarios = generate_dataset(
        num_scenarios=args.num_scenarios,
        seed=args.seed,
        horizon_days=args.horizon_days,
    )
    save_dataset_jsonl(scenarios, args.dataset_output)
    print(f"[DATASET] saved={args.dataset_output} count={len(scenarios)}")

    llm_a, llm_b = _build_both_llms(args)

    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_root = args.output_dir / run_tag
    result = run_experiment(
        scenarios=scenarios,
        output_root=output_root,
        llm_a=llm_a,
        llm_b=llm_b,
        max_steps=args.max_steps,
        seed=args.seed,
        llm_max_new_tokens=args.llm_max_new_tokens,
    )
    _save_run_config(
        output_root,
        {
            "mode": "full",
            "dataset": str(args.dataset_output),
            "num_scenarios": len(scenarios),
            "horizon_days": args.horizon_days,
            "llm_backend": args.llm_backend,
            "model_path": args.model_path,
            "agent_a_model_path": args.agent_a_model_path,
            "agent_b_model_path": args.agent_b_model_path,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "llm_max_new_tokens": args.llm_max_new_tokens,
            "device_map": args.device_map,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
    )

    print(f"[DONE] output_dir={output_root}")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


def main() -> None:
    args = _parse_args()

    if args.command == "generate-dataset":
        scenarios = generate_dataset(
            num_scenarios=args.num_scenarios,
            seed=args.seed,
            horizon_days=args.horizon_days,
        )
        save_dataset_jsonl(scenarios, args.output)
        print(f"[DONE] dataset={args.output} scenarios={len(scenarios)}")
        return

    if args.command == "run":
        _run_with_existing_dataset(args)
        return

    if args.command == "full":
        _run_full(args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
