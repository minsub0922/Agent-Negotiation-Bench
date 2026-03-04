from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent_utils import collect_agent_names_from_scenarios, resolve_agent_model_paths
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
    num_agents: int,
) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return _slug(
        f"ds-{dataset_stem}-s{seed}-n{num_scenarios}-h{horizon_days}-a{num_agents}-{ts}",
        max_len=96,
    )


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-scenarios", type=int, default=None)

    parser.add_argument("--llm-backend", choices=["hf", "dummy"], default="hf")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--agent-model-paths", type=str, default=None)
    parser.add_argument("--agent-a-model-path", type=str, default=None)
    parser.add_argument("--agent-b-model-path", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--llm-max-new-tokens", type=int, default=256)
    parser.add_argument("--decision-policy", choices=["heuristic", "llm-hybrid", "llm-only"], default="llm-hybrid")
    parser.add_argument("--require-explicit-accept", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--allow-dummy-fallback", action="store_true")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="negmas 기반 여행 일정 협상 시뮬레이션 (N-agent + local LLM)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_gen = subparsers.add_parser("generate-dataset", help="랜덤 시나리오 데이터셋 생성")
    p_gen.add_argument("--num-scenarios", type=int, default=20)
    p_gen.add_argument("--num-agents", type=int, default=2)
    p_gen.add_argument("--seed", type=int, default=7)
    p_gen.add_argument("--horizon-days", type=int, default=14)
    p_gen.add_argument("--dataset-id", type=str, default=None)
    p_gen.add_argument("--output", type=Path, default=Path("data/scenarios.jsonl"))

    p_run = subparsers.add_parser("run", help="기존 데이터셋으로 협상 실행")
    p_run.add_argument("--dataset", type=Path, required=True)
    _add_common_run_args(p_run)

    p_full = subparsers.add_parser("full", help="데이터셋 생성 + 협상 실행")
    p_full.add_argument("--num-scenarios", type=int, default=20)
    p_full.add_argument("--num-agents", type=int, default=2)
    p_full.add_argument("--seed", type=int, default=7)
    p_full.add_argument("--horizon-days", type=int, default=14)
    p_full.add_argument("--dataset-id", type=str, default=None)
    p_full.add_argument("--run-id", type=str, default=None)
    p_full.add_argument("--dataset-output", type=Path, default=Path("data/scenarios.jsonl"))
    p_full.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p_full.add_argument("--max-steps", type=int, default=40)

    # full 에서는 run 관련 공통 옵션 중 dataset 경로만 제외
    p_full.add_argument("--llm-backend", choices=["hf", "dummy"], default="hf")
    p_full.add_argument("--model-path", type=str, default=None)
    p_full.add_argument("--agent-model-paths", type=str, default=None)
    p_full.add_argument("--agent-a-model-path", type=str, default=None)
    p_full.add_argument("--agent-b-model-path", type=str, default=None)
    p_full.add_argument("--num-gpus", type=int, default=None)
    p_full.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    p_full.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=True)
    p_full.add_argument("--temperature", type=float, default=0.6)
    p_full.add_argument("--top-p", type=float, default=0.9)
    p_full.add_argument("--llm-max-new-tokens", type=int, default=256)
    p_full.add_argument("--decision-policy", choices=["heuristic", "llm-hybrid", "llm-only"], default="llm-hybrid")
    p_full.add_argument("--require-explicit-accept", action=argparse.BooleanOptionalAction, default=True)
    p_full.add_argument("--device-map", type=str, default="auto")
    p_full.add_argument("--allow-dummy-fallback", action="store_true")

    return parser.parse_args()


def _resolved_model_paths(args: argparse.Namespace, agent_names: list[str]) -> dict[str, str | None]:
    return resolve_agent_model_paths(
        agent_names=agent_names,
        model_path=args.model_path,
        agent_model_paths_arg=args.agent_model_paths,
        agent_a_model_path=args.agent_a_model_path,
        agent_b_model_path=args.agent_b_model_path,
    )


def _model_signature(args: argparse.Namespace, agent_names: list[str]) -> str:
    if args.llm_backend == "dummy":
        return "dummy"

    model_paths = _resolved_model_paths(args, agent_names)
    labels = {
        name: _slug(Path(path).name if path else "none", max_len=24)
        for name, path in model_paths.items()
    }

    # Backward-compatible naming for legacy 2-agent case.
    if agent_names == ["agent_a", "agent_b"]:
        a_label = labels["agent_a"]
        b_label = labels["agent_b"]
        if a_label == b_label:
            return _slug(f"hf-{a_label}", max_len=64)
        return _slug(f"hf-a-{a_label}-b-{b_label}", max_len=96)

    uniq = sorted(set(labels.values()))
    if len(uniq) == 1:
        return _slug(f"hf-{uniq[0]}", max_len=64)

    compact = "-".join(f"{_slug(name, 12)}-{labels[name]}" for name in agent_names)
    if len(compact) > 96:
        compact = hashlib.sha1(compact.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return _slug(f"hf-multi-{compact}", max_len=120)


def _resolve_run_id(
    explicit_run_id: str | None,
    model_signature: str,
    dataset_id: str,
) -> str:
    if explicit_run_id:
        return _slug(explicit_run_id, max_len=120)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return _slug(f"run-{ts}--{model_signature}--{dataset_id}", max_len=120)


def _resolve_shared_hf_device_map(args: argparse.Namespace, agent_names: list[str]) -> tuple[str, dict[str, Any]]:
    if args.num_gpus is not None and args.num_gpus < 0:
        raise ValueError("--num-gpus must be >= 0")

    if args.num_gpus is None:
        shared = args.device_map
        strategy = "shared-model-manual-device-map"
    elif args.num_gpus == 0:
        shared = "cpu"
        strategy = "shared-model-cpu"
    elif args.num_gpus == 1:
        shared = "cuda:0"
        strategy = "shared-model-single-gpu"
    else:
        shared = "auto"
        strategy = "shared-model-auto-shard"

    placement = {
        "strategy": strategy,
        "num_gpus_requested": args.num_gpus,
        "agent_device_maps": {name: shared for name in agent_names},
    }
    placement["agent_a_device_map"] = placement["agent_device_maps"].get("agent_a")
    placement["agent_b_device_map"] = placement["agent_device_maps"].get("agent_b")
    return shared, placement


def _resolve_agent_device_maps(args: argparse.Namespace, agent_names: list[str]) -> tuple[dict[str, str], dict[str, Any]]:
    if args.num_gpus is not None and args.num_gpus < 0:
        raise ValueError("--num-gpus must be >= 0")

    manual = str(args.device_map)
    num_gpus = args.num_gpus

    if args.llm_backend != "hf":
        maps = {name: manual for name in agent_names}
        placement = {
            "strategy": "non-hf-backend",
            "num_gpus_requested": num_gpus,
            "agent_device_maps": maps,
        }
        placement["agent_a_device_map"] = maps.get("agent_a")
        placement["agent_b_device_map"] = maps.get("agent_b")
        return maps, placement

    if num_gpus is None:
        maps = {name: manual for name in agent_names}
        placement = {
            "strategy": "manual-device-map",
            "num_gpus_requested": None,
            "agent_device_maps": maps,
        }
        placement["agent_a_device_map"] = maps.get("agent_a")
        placement["agent_b_device_map"] = maps.get("agent_b")
        return maps, placement

    if num_gpus == 0:
        maps = {name: "cpu" for name in agent_names}
        placement = {
            "strategy": "num-gpus=0-cpu",
            "num_gpus_requested": 0,
            "agent_device_maps": maps,
        }
        placement["agent_a_device_map"] = maps.get("agent_a")
        placement["agent_b_device_map"] = maps.get("agent_b")
        return maps, placement

    maps = {name: f"cuda:{idx % num_gpus}" for idx, name in enumerate(agent_names)}
    placement = {
        "strategy": "num-gpus-dispatch",
        "num_gpus_requested": num_gpus,
        "agent_device_maps": maps,
    }
    placement["agent_a_device_map"] = maps.get("agent_a")
    placement["agent_b_device_map"] = maps.get("agent_b")
    return maps, placement


def _build_llms_for_agents(args: argparse.Namespace, agent_names: list[str]):
    model_paths = _resolved_model_paths(args, agent_names)

    if args.llm_backend == "hf":
        missing = [name for name in agent_names if not model_paths.get(name)]
        if missing:
            raise ValueError(
                "HF backend requires model path for every agent. "
                f"Missing: {missing}. Use --model-path or --agent-model-paths"
            )

    normalized_paths = {
        name: (str(Path(path).expanduser().resolve(strict=False)) if path else None)
        for name, path in model_paths.items()
    }

    unique_hf_paths = sorted({p for p in normalized_paths.values() if p is not None})
    all_same_model = args.llm_backend == "hf" and len(unique_hf_paths) == 1

    if all_same_model:
        shared_device_map, placement = _resolve_shared_hf_device_map(args, agent_names)
        placement["agent_model_paths"] = model_paths
        print(
            f"[INIT] placement strategy={placement.get('strategy')} "
            + " ".join(f"{name}={shared_device_map}" for name in agent_names),
            flush=True,
        )
        print(
            f"[INIT] loading shared model path={model_paths[agent_names[0]]} device={shared_device_map}",
            flush=True,
        )

        shared_llm = build_llm(
            backend=args.llm_backend,
            name="agent_shared",
            model_path=model_paths[agent_names[0]],
            temperature=args.temperature,
            top_p=args.top_p,
            allow_dummy_fallback=args.allow_dummy_fallback,
            device_map=shared_device_map,
            torch_dtype=args.torch_dtype,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            max_memory=None,
        )
        llms_by_agent = {name: shared_llm for name in agent_names}
        placement["shared_model_instance"] = True
        return llms_by_agent, placement

    device_maps, placement = _resolve_agent_device_maps(args, agent_names)
    placement["agent_model_paths"] = model_paths
    print(
        f"[INIT] placement strategy={placement.get('strategy')} "
        + " ".join(f"{name}={device_maps[name]}" for name in agent_names),
        flush=True,
    )

    llm_cache: dict[tuple[str, str], Any] = {}
    llms_by_agent: dict[str, Any] = {}

    for name in agent_names:
        model_path = model_paths.get(name)
        device_map = device_maps[name]

        cache_key: tuple[str, str] | None = None
        if args.llm_backend == "hf" and model_path is not None:
            cache_key = (str(Path(model_path).expanduser().resolve(strict=False)), device_map)
            if cache_key in llm_cache:
                llms_by_agent[name] = llm_cache[cache_key]
                continue

        print(f"[INIT] loading {name} model path={model_path} device={device_map}", flush=True)
        llm = build_llm(
            backend=args.llm_backend,
            name=name,
            model_path=model_path,
            temperature=args.temperature,
            top_p=args.top_p,
            allow_dummy_fallback=args.allow_dummy_fallback,
            device_map=device_map,
            torch_dtype=args.torch_dtype,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            max_memory=None,
        )
        llms_by_agent[name] = llm

        if cache_key is not None:
            llm_cache[cache_key] = llm

    placement["shared_model_instance"] = len({id(v) for v in llms_by_agent.values()}) < len(llms_by_agent)
    return llms_by_agent, placement


def _build_dataset_config(
    dataset_id: str,
    dataset_path: Path,
    scenarios: list[dict[str, Any]],
    seed: int,
    horizon_days: int,
    source_command: str,
) -> dict[str, Any]:
    dataset_hash = _short_file_hash(dataset_path)
    agent_names = collect_agent_names_from_scenarios(scenarios)
    return {
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "dataset_file_hash_sha1_10": dataset_hash,
        "num_scenarios": len(scenarios),
        "num_agents": len(agent_names),
        "agent_names": agent_names,
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
    if not scenarios:
        raise ValueError("No scenarios available to run")

    agent_names = collect_agent_names_from_scenarios(scenarios)

    dataset_cfg = _load_dataset_config(args.dataset)
    dataset_id, dataset_hash = _resolve_dataset_identity_for_run(
        dataset_path=args.dataset,
        explicit_dataset_id=args.dataset_id,
        dataset_config=dataset_cfg,
    )

    model_sig = _model_signature(args, agent_names=agent_names)
    run_id = _resolve_run_id(args.run_id, model_signature=model_sig, dataset_id=dataset_id)

    llms_by_agent, placement = _build_llms_for_agents(args, agent_names)

    output_root = args.output_dir / run_id
    result = run_experiment(
        scenarios=scenarios,
        output_root=output_root,
        llms_by_agent=llms_by_agent,
        max_steps=args.max_steps,
        seed=args.seed,
        llm_max_new_tokens=args.llm_max_new_tokens,
        decision_policy=args.decision_policy,
        require_explicit_accept=args.require_explicit_accept,
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

    agent_model_paths = placement.get("agent_model_paths", {})
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
            "num_agents": len(agent_names),
            "agent_names": agent_names,
            "model_signature": model_sig,
            "llm_backend": args.llm_backend,
            "num_gpus": args.num_gpus,
            "agent_placement": placement,
            "agent_model_paths": agent_model_paths,
            "torch_dtype": args.torch_dtype,
            "low_cpu_mem_usage": args.low_cpu_mem_usage,
            "model_path": args.model_path,
            "agent_model_paths_arg": args.agent_model_paths,
            "agent_a_model_path": agent_model_paths.get("agent_a") if agent_model_paths else args.agent_a_model_path,
            "agent_b_model_path": agent_model_paths.get("agent_b") if agent_model_paths else args.agent_b_model_path,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "llm_max_new_tokens": args.llm_max_new_tokens,
            "decision_policy": args.decision_policy,
            "require_explicit_accept": args.require_explicit_accept,
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
    if args.num_agents < 2:
        raise ValueError("--num-agents must be >= 2")

    scenarios = generate_dataset(
        num_scenarios=args.num_scenarios,
        n_agents=args.num_agents,
        seed=args.seed,
        horizon_days=args.horizon_days,
    )
    agent_names = collect_agent_names_from_scenarios(scenarios)

    dataset_id = _slug(args.dataset_id, max_len=96) if args.dataset_id else _auto_dataset_id(
        dataset_stem=args.dataset_output.stem,
        seed=args.seed,
        num_scenarios=args.num_scenarios,
        horizon_days=args.horizon_days,
        num_agents=args.num_agents,
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

    model_sig = _model_signature(args, agent_names=agent_names)
    run_id = _resolve_run_id(args.run_id, model_signature=model_sig, dataset_id=dataset_id)

    llms_by_agent, placement = _build_llms_for_agents(args, agent_names)

    output_root = args.output_dir / run_id
    result = run_experiment(
        scenarios=scenarios,
        output_root=output_root,
        llms_by_agent=llms_by_agent,
        max_steps=args.max_steps,
        seed=args.seed,
        llm_max_new_tokens=args.llm_max_new_tokens,
        decision_policy=args.decision_policy,
        require_explicit_accept=args.require_explicit_accept,
    )

    _json_dump(output_root / "dataset_config_snapshot.json", dataset_cfg)

    agent_model_paths = placement.get("agent_model_paths", {})
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
            "num_agents": len(agent_names),
            "agent_names": agent_names,
            "horizon_days": args.horizon_days,
            "model_signature": model_sig,
            "llm_backend": args.llm_backend,
            "num_gpus": args.num_gpus,
            "agent_placement": placement,
            "agent_model_paths": agent_model_paths,
            "torch_dtype": args.torch_dtype,
            "low_cpu_mem_usage": args.low_cpu_mem_usage,
            "model_path": args.model_path,
            "agent_model_paths_arg": args.agent_model_paths,
            "agent_a_model_path": agent_model_paths.get("agent_a") if agent_model_paths else args.agent_a_model_path,
            "agent_b_model_path": agent_model_paths.get("agent_b") if agent_model_paths else args.agent_b_model_path,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "llm_max_new_tokens": args.llm_max_new_tokens,
            "decision_policy": args.decision_policy,
            "require_explicit_accept": args.require_explicit_accept,
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
    if args.num_agents < 2:
        raise ValueError("--num-agents must be >= 2")

    scenarios = generate_dataset(
        num_scenarios=args.num_scenarios,
        n_agents=args.num_agents,
        seed=args.seed,
        horizon_days=args.horizon_days,
    )

    dataset_id = _slug(args.dataset_id, max_len=96) if args.dataset_id else _auto_dataset_id(
        dataset_stem=args.output.stem,
        seed=args.seed,
        num_scenarios=args.num_scenarios,
        horizon_days=args.horizon_days,
        num_agents=args.num_agents,
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
