from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .llm_backend import build_llm
from .runner import run_experiment
from .scenario import generate_dataset, load_dataset_jsonl, save_dataset_jsonl


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _save_run_config(output_root: Path, config: dict[str, Any]) -> None:
    _json_dump(output_root / "run_config.json", config)


def _dataset_config_path(dataset_path: Path) -> Path:
    return dataset_path.with_suffix(dataset_path.suffix + ".config.json")


def _save_dataset_config(dataset_path: Path, config: dict[str, Any]) -> Path:
    path = _dataset_config_path(dataset_path)
    _json_dump(path, config)
    return path


def _load_dataset_config(dataset_path: Path) -> dict[str, Any] | None:
    path = _dataset_config_path(dataset_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _slug(text: str, max_len: int = 48) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", text.strip())
    normalized = normalized.strip("-_.").lower()
    if not normalized:
        normalized = "na"
    return normalized[:max_len]


def _short_file_hash(path: Path) -> str:
    sha = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()[:10]


def _auto_dataset_id(
    dataset_stem: str,
    seed: int,
    num_scenarios: int,
    horizon_days: int,
) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return _slug(f"ds-{dataset_stem}-s{seed}-n{num_scenarios}-h{horizon_days}-{ts}", max_len=96)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="negmas 기반 여행 일정 협상 시뮬레이션 (2-agent + local LLM)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_gen = subparsers.add_parser("generate-dataset", help="랜덤 시나리오 데이터셋 생성")
    p_gen.add_argument("--num-scenarios", type=int, default=20)
    p_gen.add_argument("--seed", type=int, default=7)
    p_gen.add_argument("--horizon-days", type=int, default=14)
    p_gen.add_argument("--dataset-id", type=str, default=None)
    p_gen.add_argument("--output", type=Path, default=Path("data/scenarios.jsonl"))

    p_run = subparsers.add_parser("run", help="기존 데이터셋으로 협상 실행")
    p_run.add_argument("--dataset", type=Path, required=True)
    p_run.add_argument("--dataset-id", type=str, default=None)
    p_run.add_argument("--run-id", type=str, default=None)
    p_run.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p_run.add_argument("--max-steps", type=int, default=40)
    p_run.add_argument("--seed", type=int, default=7)
    p_run.add_argument("--num-scenarios", type=int, default=None)

    p_run.add_argument("--llm-backend", choices=["hf", "dummy"], default="hf")
    p_run.add_argument("--model-path", type=str, default=None)
    p_run.add_argument("--agent-a-model-path", type=str, default=None)
    p_run.add_argument("--agent-b-model-path", type=str, default=None)
    p_run.add_argument("--num-gpus", type=int, default=None)
    p_run.add_argument("--temperature", type=float, default=0.6)
    p_run.add_argument("--top-p", type=float, default=0.9)
    p_run.add_argument("--llm-max-new-tokens", type=int, default=256)
    p_run.add_argument("--device-map", type=str, default="auto")
    p_run.add_argument("--allow-dummy-fallback", action="store_true")

    p_full = subparsers.add_parser("full", help="데이터셋 생성 + 협상 실행")
    p_full.add_argument("--num-scenarios", type=int, default=20)
    p_full.add_argument("--seed", type=int, default=7)
    p_full.add_argument("--horizon-days", type=int, default=14)
    p_full.add_argument("--dataset-id", type=str, default=None)
    p_full.add_argument("--run-id", type=str, default=None)
    p_full.add_argument("--dataset-output", type=Path, default=Path("data/scenarios.jsonl"))
    p_full.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p_full.add_argument("--max-steps", type=int, default=40)

    p_full.add_argument("--llm-backend", choices=["hf", "dummy"], default="hf")
    p_full.add_argument("--model-path", type=str, default=None)
    p_full.add_argument("--agent-a-model-path", type=str, default=None)
    p_full.add_argument("--agent-b-model-path", type=str, default=None)
    p_full.add_argument("--num-gpus", type=int, default=None)
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


def _model_signature(args: argparse.Namespace) -> str:
    if args.llm_backend == "dummy":
        return "dummy"

    a_path, b_path = _resolve_model_paths(args)
    a_label = _slug(Path(a_path).name if a_path else "none", max_len=24)
    b_label = _slug(Path(b_path).name if b_path else "none", max_len=24)

    if a_label == b_label:
        return _slug(f"hf-{a_label}", max_len=48)
    return _slug(f"hf-a-{a_label}-b-{b_label}", max_len=64)


def _resolve_run_id(
    explicit_run_id: str | None,
    model_signature: str,
    dataset_id: str,
) -> str:
    if explicit_run_id:
        return _slug(explicit_run_id, max_len=120)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return _slug(f"run-{ts}--{model_signature}--{dataset_id}", max_len=120)


def _build_both_llms(args: argparse.Namespace):
    a_path, b_path = _resolve_model_paths(args)
    a_device_map, b_device_map, placement = _resolve_agent_device_maps(args)

    llm_a = build_llm(
        backend=args.llm_backend,
        name="agent_a",
        model_path=a_path,
        temperature=args.temperature,
        top_p=args.top_p,
        allow_dummy_fallback=args.allow_dummy_fallback,
        device_map=a_device_map,
    )
    llm_b = build_llm(
        backend=args.llm_backend,
        name="agent_b",
        model_path=b_path,
        temperature=args.temperature,
        top_p=args.top_p,
        allow_dummy_fallback=args.allow_dummy_fallback,
        device_map=b_device_map,
    )
    return llm_a, llm_b, placement


def _resolve_agent_device_maps(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    if args.num_gpus is not None and args.num_gpus < 0:
        raise ValueError("--num-gpus must be >= 0")

    manual = args.device_map
    num_gpus = args.num_gpus

    if args.llm_backend != "hf":
        return (
            manual,
            manual,
            {
                "strategy": "non-hf-backend",
                "num_gpus_requested": num_gpus,
                "agent_a_device_map": manual,
                "agent_b_device_map": manual,
            },
        )

    if num_gpus is None:
        return (
            manual,
            manual,
            {
                "strategy": "manual-device-map",
                "num_gpus_requested": None,
                "agent_a_device_map": manual,
                "agent_b_device_map": manual,
            },
        )

    if num_gpus == 0:
        return (
            "cpu",
            "cpu",
            {
                "strategy": "num-gpus=0-cpu",
                "num_gpus_requested": 0,
                "agent_a_device_map": "cpu",
                "agent_b_device_map": "cpu",
            },
        )

    a_idx = 0
    b_idx = 0 if num_gpus == 1 else 1 % num_gpus
    a_map = f"cuda:{a_idx}"
    b_map = f"cuda:{b_idx}"
    return (
        a_map,
        b_map,
        {
            "strategy": "num-gpus-dispatch",
            "num_gpus_requested": num_gpus,
            "agent_a_device_map": a_map,
            "agent_b_device_map": b_map,
        },
    )


def _build_dataset_config(
    dataset_id: str,
    dataset_path: Path,
    scenarios: list[dict[str, Any]],
    seed: int,
    horizon_days: int,
    source_command: str,
) -> dict[str, Any]:
    dataset_hash = _short_file_hash(dataset_path)
    return {
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "dataset_file_hash_sha1_10": dataset_hash,
        "num_scenarios": len(scenarios),
        "seed": seed,
        "horizon_days": horizon_days,
        "source_command": source_command,
        "generated_at": datetime.now().isoformat(),
    }


def _save_dataset_and_config(
    scenarios: list[dict[str, Any]],
    dataset_path: Path,
    dataset_id: str,
    seed: int,
    horizon_days: int,
    source_command: str,
) -> tuple[dict[str, Any], Path]:
    save_dataset_jsonl(scenarios, dataset_path)
    config = _build_dataset_config(
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        scenarios=scenarios,
        seed=seed,
        horizon_days=horizon_days,
        source_command=source_command,
    )
    config_path = _save_dataset_config(dataset_path, config)
    return config, config_path


def _resolve_dataset_identity_for_run(
    dataset_path: Path,
    explicit_dataset_id: str | None,
    dataset_config: dict[str, Any] | None,
) -> tuple[str, str]:
    dataset_hash = _short_file_hash(dataset_path)

    if explicit_dataset_id:
        dataset_id = _slug(explicit_dataset_id, max_len=96)
    elif dataset_config and dataset_config.get("dataset_id"):
        dataset_id = _slug(str(dataset_config["dataset_id"]), max_len=96)
    else:
        dataset_id = _slug(f"{dataset_path.stem}-{dataset_hash}", max_len=96)

    return dataset_id, dataset_hash


def _run_with_existing_dataset(args: argparse.Namespace) -> None:
    scenarios = load_dataset_jsonl(args.dataset)
    total_scenarios = len(scenarios)
    if args.num_scenarios is not None:
        scenarios = scenarios[: args.num_scenarios]

    dataset_cfg = _load_dataset_config(args.dataset)
    dataset_id, dataset_hash = _resolve_dataset_identity_for_run(
        dataset_path=args.dataset,
        explicit_dataset_id=args.dataset_id,
        dataset_config=dataset_cfg,
    )

    model_sig = _model_signature(args)
    run_id = _resolve_run_id(args.run_id, model_signature=model_sig, dataset_id=dataset_id)

    llm_a, llm_b, placement = _build_both_llms(args)

    output_root = args.output_dir / run_id
    result = run_experiment(
        scenarios=scenarios,
        output_root=output_root,
        llm_a=llm_a,
        llm_b=llm_b,
        max_steps=args.max_steps,
        seed=args.seed,
        llm_max_new_tokens=args.llm_max_new_tokens,
    )

    dataset_snapshot = dataset_cfg or {
        "dataset_id": dataset_id,
        "dataset_path": str(args.dataset),
        "dataset_file_hash_sha1_10": dataset_hash,
        "num_scenarios": total_scenarios,
        "source_command": "external_or_unknown",
        "loaded_at": datetime.now().isoformat(),
    }
    _json_dump(output_root / "dataset_config_snapshot.json", dataset_snapshot)

    _save_run_config(
        output_root,
        {
            "run_id": run_id,
            "mode": "run",
            "run_at": datetime.now().isoformat(),
            "dataset_id": dataset_id,
            "dataset_path": str(args.dataset),
            "dataset_file_hash_sha1_10": dataset_hash,
            "dataset_total_scenarios": total_scenarios,
            "dataset_used_scenarios": len(scenarios),
            "model_signature": model_sig,
            "llm_backend": args.llm_backend,
            "num_gpus": args.num_gpus,
            "agent_placement": placement,
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

    print(f"[DONE] run_id={run_id}")
    print(f"[DONE] output_dir={output_root}")
    print(f"[PLACEMENT] {json.dumps(placement, ensure_ascii=False)}")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


def _run_full(args: argparse.Namespace) -> None:
    scenarios = generate_dataset(
        num_scenarios=args.num_scenarios,
        seed=args.seed,
        horizon_days=args.horizon_days,
    )

    dataset_id = _slug(args.dataset_id, max_len=96) if args.dataset_id else _auto_dataset_id(
        dataset_stem=args.dataset_output.stem,
        seed=args.seed,
        num_scenarios=args.num_scenarios,
        horizon_days=args.horizon_days,
    )

    dataset_cfg, dataset_cfg_path = _save_dataset_and_config(
        scenarios=scenarios,
        dataset_path=args.dataset_output,
        dataset_id=dataset_id,
        seed=args.seed,
        horizon_days=args.horizon_days,
        source_command="full",
    )
    dataset_hash = dataset_cfg["dataset_file_hash_sha1_10"]

    print(f"[DATASET] id={dataset_id}")
    print(f"[DATASET] saved={args.dataset_output} count={len(scenarios)}")
    print(f"[DATASET] config={dataset_cfg_path}")

    model_sig = _model_signature(args)
    run_id = _resolve_run_id(args.run_id, model_signature=model_sig, dataset_id=dataset_id)

    llm_a, llm_b, placement = _build_both_llms(args)

    output_root = args.output_dir / run_id
    result = run_experiment(
        scenarios=scenarios,
        output_root=output_root,
        llm_a=llm_a,
        llm_b=llm_b,
        max_steps=args.max_steps,
        seed=args.seed,
        llm_max_new_tokens=args.llm_max_new_tokens,
    )

    _json_dump(output_root / "dataset_config_snapshot.json", dataset_cfg)

    _save_run_config(
        output_root,
        {
            "run_id": run_id,
            "mode": "full",
            "run_at": datetime.now().isoformat(),
            "dataset_id": dataset_id,
            "dataset_path": str(args.dataset_output),
            "dataset_file_hash_sha1_10": dataset_hash,
            "dataset_used_scenarios": len(scenarios),
            "horizon_days": args.horizon_days,
            "model_signature": model_sig,
            "llm_backend": args.llm_backend,
            "num_gpus": args.num_gpus,
            "agent_placement": placement,
            "model_path": args.model_path,
            "agent_a_model_path": args.agent_a_model_path,
            "agent_b_model_path": args.agent_b_model_path,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "llm_max_new_tokens": args.llm_max_new_tokens,
            "device_map": args.device_map,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "dataset_config_path": str(dataset_cfg_path),
        },
    )

    print(f"[DONE] run_id={run_id}")
    print(f"[DONE] output_dir={output_root}")
    print(f"[PLACEMENT] {json.dumps(placement, ensure_ascii=False)}")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


def _run_generate_dataset(args: argparse.Namespace) -> None:
    scenarios = generate_dataset(
        num_scenarios=args.num_scenarios,
        seed=args.seed,
        horizon_days=args.horizon_days,
    )

    dataset_id = _slug(args.dataset_id, max_len=96) if args.dataset_id else _auto_dataset_id(
        dataset_stem=args.output.stem,
        seed=args.seed,
        num_scenarios=args.num_scenarios,
        horizon_days=args.horizon_days,
    )

    dataset_cfg, dataset_cfg_path = _save_dataset_and_config(
        scenarios=scenarios,
        dataset_path=args.output,
        dataset_id=dataset_id,
        seed=args.seed,
        horizon_days=args.horizon_days,
        source_command="generate-dataset",
    )

    print(f"[DONE] dataset_id={dataset_id}")
    print(f"[DONE] dataset={args.output} scenarios={len(scenarios)}")
    print(f"[DONE] dataset_hash={dataset_cfg['dataset_file_hash_sha1_10']}")
    print(f"[DONE] dataset_config={dataset_cfg_path}")


def main() -> None:
    args = _parse_args()

    if args.command == "generate-dataset":
        _run_generate_dataset(args)
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
