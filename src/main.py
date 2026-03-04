from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent_utils import collect_agent_names_from_scenarios, resolve_agent_model_paths, scenario_agent_names
from .llm_backend import build_llm
from .runner import run_experiment
from .scenario import generate_dataset, load_dataset_jsonl, save_dataset_jsonl
from .synthetic_data import build_training_corpora
from .trl_training import train_dpo, train_sft


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


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


def _optional_attr(args: argparse.Namespace, key: str, default: Any = None) -> Any:
    return getattr(args, key, default)


def _llm_runtime_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "llm_backend": _optional_attr(args, "llm_backend"),
        "num_gpus": _optional_attr(args, "num_gpus"),
        "model_path": _optional_attr(args, "model_path"),
        "agent_model_paths_arg": _optional_attr(args, "agent_model_paths"),
        "agent_a_model_path": _optional_attr(args, "agent_a_model_path"),
        "agent_b_model_path": _optional_attr(args, "agent_b_model_path"),
        "device_map": _optional_attr(args, "device_map"),
        "torch_dtype": _optional_attr(args, "torch_dtype"),
        "low_cpu_mem_usage": _optional_attr(args, "low_cpu_mem_usage"),
        "temperature": _optional_attr(args, "temperature"),
        "top_p": _optional_attr(args, "top_p"),
        "allow_dummy_fallback": bool(_optional_attr(args, "allow_dummy_fallback", False)),
        "trust_remote_code": bool(_optional_attr(args, "trust_remote_code", True)),
        "vllm_tensor_parallel_size": _optional_attr(args, "vllm_tensor_parallel_size", 1),
        "vllm_dtype": _optional_attr(args, "vllm_dtype", "auto"),
        "vllm_gpu_memory_utilization": _optional_attr(args, "vllm_gpu_memory_utilization", 0.9),
        "vllm_max_model_len": _optional_attr(args, "vllm_max_model_len"),
        "vllm_enable_prefix_caching": bool(_optional_attr(args, "vllm_enable_prefix_caching", True)),
        "vllm_swap_space": _optional_attr(args, "vllm_swap_space", 4),
    }


def _parse_csv_tokens(raw: str | None) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def _parse_labeled_model_specs(raw: str | None) -> list[dict[str, str]]:
    tokens = _parse_csv_tokens(raw)
    specs: list[dict[str, str]] = []
    used_labels: set[str] = set()

    for idx, token in enumerate(tokens, start=1):
        if "=" in token:
            label, path = token.split("=", 1)
            label = label.strip()
            path = path.strip()
        else:
            path = token
            label = _slug(Path(path).name or f"model_{idx}", max_len=32)

        if not path:
            raise ValueError(f"Empty model path in token: '{token}'")

        base_label = _slug(label, max_len=48) or f"model_{idx}"
        final_label = base_label
        suffix = 2
        while final_label in used_labels:
            final_label = _slug(f"{base_label}-{suffix}", max_len=48)
            suffix += 1

        used_labels.add(final_label)
        specs.append({"label": final_label, "path": path})

    return specs


def _add_backend_generation_args(
    parser: argparse.ArgumentParser,
    default_backend: str = "hf",
    include_agent_model_flags: bool = True,
) -> None:
    parser.add_argument("--llm-backend", choices=["hf", "vllm", "dummy"], default=default_backend)
    parser.add_argument("--model-path", type=str, default=None)
    if include_agent_model_flags:
        parser.add_argument("--agent-model-paths", type=str, default=None)
        parser.add_argument("--agent-a-model-path", type=str, default=None)
        parser.add_argument("--agent-b-model-path", type=str, default=None)
    else:
        parser.set_defaults(agent_model_paths=None, agent_a_model_path=None, agent_b_model_path=None)

    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-dummy-fallback", action="store_true")

    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--vllm-enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-swap-space", type=int, default=4)


def _add_negotiation_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--llm-max-new-tokens", type=int, default=256)
    parser.add_argument("--decision-policy", choices=["heuristic", "llm-hybrid", "llm-only"], default="llm-hybrid")
    parser.add_argument("--require-explicit-accept", action=argparse.BooleanOptionalAction, default=True)


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-scenarios", type=int, default=None)

    _add_backend_generation_args(parser, default_backend="hf", include_agent_model_flags=True)
    _add_negotiation_policy_args(parser)


def _add_trl_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="negmas negotiation pipeline with teacher-trace generation, TRL training, and benchmarking",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_gen = subparsers.add_parser("generate-dataset", help="Generate a random scenario dataset")
    p_gen.add_argument("--num-scenarios", type=int, default=20)
    p_gen.add_argument("--num-agents", type=int, default=2)
    p_gen.add_argument("--seed", type=int, default=7)
    p_gen.add_argument("--horizon-days", type=int, default=14)
    p_gen.add_argument("--dataset-id", type=str, default=None)
    p_gen.add_argument("--output", type=Path, default=Path("data/scenarios.jsonl"))

    p_run = subparsers.add_parser("run", help="Run negotiations using an existing dataset")
    p_run.add_argument("--dataset", type=Path, required=True)
    _add_common_run_args(p_run)

    p_full = subparsers.add_parser("full", help="Generate a dataset and run negotiations")
    p_full.add_argument("--num-scenarios", type=int, default=20)
    p_full.add_argument("--num-agents", type=int, default=2)
    p_full.add_argument("--seed", type=int, default=7)
    p_full.add_argument("--horizon-days", type=int, default=14)
    p_full.add_argument("--dataset-id", type=str, default=None)
    p_full.add_argument("--run-id", type=str, default=None)
    p_full.add_argument("--dataset-output", type=Path, default=Path("data/scenarios.jsonl"))
    p_full.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p_full.add_argument("--max-steps", type=int, default=40)
    _add_backend_generation_args(p_full, default_backend="hf", include_agent_model_flags=True)
    _add_negotiation_policy_args(p_full)

    p_traces = subparsers.add_parser(
        "generate-traces",
        help="Generate diverse synthetic negotiation traces with teacher models and export SFT/DPO corpora",
    )
    p_traces.add_argument("--dataset", type=Path, required=True)
    p_traces.add_argument("--dataset-id", type=str, default=None)
    p_traces.add_argument("--trace-id", type=str, default=None)
    p_traces.add_argument("--num-scenarios", type=int, default=None)
    p_traces.add_argument("--episodes-per-teacher", type=int, default=2)
    p_traces.add_argument("--teacher-model-paths", type=str, default=None)
    p_traces.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic"))
    p_traces.add_argument("--max-steps", type=int, default=40)
    p_traces.add_argument("--seed", type=int, default=7)
    p_traces.add_argument("--shuffle-agent-order", action="store_true")
    p_traces.add_argument("--shuffle-scenario-order", action="store_true")
    p_traces.add_argument("--temperature-jitter", type=float, default=0.2)
    p_traces.add_argument("--top-p-jitter", type=float, default=0.05)
    p_traces.add_argument("--training-data-eval-ratio", type=float, default=0.05)
    p_traces.add_argument("--min-preference-margin", type=float, default=0.02)
    _add_backend_generation_args(p_traces, default_backend="vllm", include_agent_model_flags=False)
    _add_negotiation_policy_args(p_traces)

    p_build_data = subparsers.add_parser(
        "build-training-data",
        help="Build SFT and DPO training files from existing run directories",
    )
    p_build_data.add_argument("--run-dirs", type=str, required=True)
    p_build_data.add_argument("--output-dir", type=Path, required=True)
    p_build_data.add_argument("--eval-ratio", type=float, default=0.05)
    p_build_data.add_argument("--seed", type=int, default=7)
    p_build_data.add_argument("--min-preference-margin", type=float, default=0.02)

    p_train_sft = subparsers.add_parser("train-sft", help="Train target negotiator model with TRL SFTTrainer")
    _add_trl_common_args(p_train_sft)

    p_train_rl = subparsers.add_parser("train-rl", help="Train target negotiator model with TRL DPOTrainer")
    _add_trl_common_args(p_train_rl)
    p_train_rl.add_argument("--ref-model-path", type=str, default=None)
    p_train_rl.add_argument("--beta", type=float, default=0.1)
    p_train_rl.add_argument("--max-prompt-length", type=int, default=1024)

    p_eval = subparsers.add_parser(
        "evaluate-models",
        help="Evaluate one or more negotiator models on a shared scenario dataset",
    )
    p_eval.add_argument("--dataset", type=Path, required=True)
    p_eval.add_argument("--dataset-id", type=str, default=None)
    p_eval.add_argument("--eval-id", type=str, default=None)
    p_eval.add_argument("--model-paths", type=str, default=None)
    p_eval.add_argument("--num-scenarios", type=int, default=None)
    p_eval.add_argument("--output-dir", type=Path, default=Path("outputs/evaluations"))
    p_eval.add_argument("--max-steps", type=int, default=40)
    p_eval.add_argument("--seed", type=int, default=7)
    _add_backend_generation_args(p_eval, default_backend="hf", include_agent_model_flags=False)
    _add_negotiation_policy_args(p_eval)

    return parser.parse_args()


def _resolved_model_paths(args: argparse.Namespace, agent_names: list[str]) -> dict[str, str | None]:
    return resolve_agent_model_paths(
        agent_names=agent_names,
        model_path=_optional_attr(args, "model_path"),
        agent_model_paths_arg=_optional_attr(args, "agent_model_paths"),
        agent_a_model_path=_optional_attr(args, "agent_a_model_path"),
        agent_b_model_path=_optional_attr(args, "agent_b_model_path"),
    )


def _model_signature(args: argparse.Namespace, agent_names: list[str]) -> str:
    backend = str(_optional_attr(args, "llm_backend", "hf"))
    if backend == "dummy":
        return "dummy"

    model_paths = _resolved_model_paths(args, agent_names)
    labels = {
        name: _slug(Path(path).name if path else "none", max_len=24)
        for name, path in model_paths.items()
    }

    if agent_names == ["agent_a", "agent_b"]:
        a_label = labels["agent_a"]
        b_label = labels["agent_b"]
        if a_label == b_label:
            return _slug(f"{backend}-{a_label}", max_len=64)
        return _slug(f"{backend}-a-{a_label}-b-{b_label}", max_len=96)

    uniq = sorted(set(labels.values()))
    if len(uniq) == 1:
        return _slug(f"{backend}-{uniq[0]}", max_len=64)

    compact = "-".join(f"{_slug(name, 12)}-{labels[name]}" for name in agent_names)
    if len(compact) > 96:
        compact = hashlib.sha1(compact.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return _slug(f"{backend}-multi-{compact}", max_len=120)


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


def _build_llm_instance(args: argparse.Namespace, name: str, model_path: str | None, device_map: str):
    return build_llm(
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
        trust_remote_code=args.trust_remote_code,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_dtype=args.vllm_dtype,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_enable_prefix_caching=args.vllm_enable_prefix_caching,
        vllm_swap_space=args.vllm_swap_space,
    )


def _build_llms_for_agents(args: argparse.Namespace, agent_names: list[str]):
    model_paths = _resolved_model_paths(args, agent_names)

    if args.llm_backend in {"hf", "vllm"}:
        missing = [name for name in agent_names if not model_paths.get(name)]
        if missing:
            raise ValueError(
                f"{args.llm_backend.upper()} backend requires model path for every agent. "
                f"Missing: {missing}. Use --model-path or --agent-model-paths"
            )

    normalized_paths = {
        name: (str(Path(path).expanduser().resolve(strict=False)) if path else None)
        for name, path in model_paths.items()
    }
    unique_paths = sorted({p for p in normalized_paths.values() if p is not None})

    if args.llm_backend == "vllm":
        placement = {
            "strategy": "shared-vllm-single-model" if len(unique_paths) == 1 else "vllm-model-cache",
            "num_gpus_requested": args.num_gpus,
            "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            "agent_device_maps": {name: "vllm" for name in agent_names},
            "agent_model_paths": model_paths,
        }
        placement["agent_a_device_map"] = placement["agent_device_maps"].get("agent_a")
        placement["agent_b_device_map"] = placement["agent_device_maps"].get("agent_b")

        llm_cache: dict[str, Any] = {}
        llms_by_agent: dict[str, Any] = {}
        for name in agent_names:
            model_path = model_paths[name]
            assert model_path is not None
            cache_key = str(Path(model_path).expanduser().resolve(strict=False))
            if cache_key in llm_cache:
                llms_by_agent[name] = llm_cache[cache_key]
                continue

            print(
                f"[INIT] loading {name} vllm path={model_path} "
                f"tp={args.vllm_tensor_parallel_size} dtype={args.vllm_dtype}",
                flush=True,
            )
            llm = _build_llm_instance(args, name=name, model_path=model_path, device_map="vllm")
            llm_cache[cache_key] = llm
            llms_by_agent[name] = llm

        placement["shared_model_instance"] = len({id(v) for v in llms_by_agent.values()}) < len(llms_by_agent)
        return llms_by_agent, placement

    if args.llm_backend == "hf" and len(unique_paths) == 1:
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

        shared_llm = _build_llm_instance(
            args,
            name="agent_shared",
            model_path=model_paths[agent_names[0]],
            device_map=shared_device_map,
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
        llm = _build_llm_instance(
            args,
            name=name,
            model_path=model_path,
            device_map=device_map,
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


def _clone_args_for_model(
    args: argparse.Namespace,
    model_path: str | None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> argparse.Namespace:
    payload = dict(vars(args))
    payload["model_path"] = model_path
    payload["agent_model_paths"] = None
    payload["agent_a_model_path"] = None
    payload["agent_b_model_path"] = None
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if top_p is not None:
        payload["top_p"] = float(top_p)
    return argparse.Namespace(**payload)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _prepare_episode_scenarios(
    scenarios: list[dict[str, Any]],
    episode_seed: int,
    shuffle_agent_order: bool,
    shuffle_scenario_order: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rng = random.Random(episode_seed)

    for scenario in scenarios:
        copied = deepcopy(scenario) if (shuffle_agent_order or shuffle_scenario_order) else scenario
        if shuffle_agent_order:
            names = list(scenario_agent_names(copied))
            rng.shuffle(names)
            copied["agent_order"] = names
        out.append(copied)

    if shuffle_scenario_order:
        rng.shuffle(out)

    return out


def _leaderboard_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Negotiation Leaderboard",
        "",
        "| Rank | Label | Agreement | Pareto | Avg Steps | Welfare | Nash | Welfare/Best | Nash/Best | Run Dir |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {label} | {agr:.2%} | {par:.2%} | {steps:.2f} | {welfare:.4f} | {nash:.4f} | {wr:.2%} | {nr:.2%} | `{run_dir}` |".format(
                rank=idx,
                label=row.get("label", "-"),
                agr=float(row.get("agreement_rate") or 0.0),
                par=float(row.get("pareto_rate") or 0.0),
                steps=float(row.get("avg_steps") or 0.0),
                welfare=float(row.get("avg_social_welfare") or 0.0),
                nash=float(row.get("avg_nash_product") or 0.0),
                wr=float(row.get("avg_welfare_ratio_vs_best") or 0.0),
                nr=float(row.get("avg_nash_ratio_vs_best") or 0.0),
                run_dir=row.get("run_dir", "-"),
            )
        )
    return "\n".join(lines) + "\n"


def _parse_lora_target_modules(raw: str | None) -> list[str] | None:
    tokens = _parse_csv_tokens(raw)
    return tokens or None


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
    run_cfg = {
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
        "agent_placement": placement,
        "agent_model_paths": agent_model_paths,
        "llm_max_new_tokens": args.llm_max_new_tokens,
        "decision_policy": args.decision_policy,
        "require_explicit_accept": args.require_explicit_accept,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }
    run_cfg.update(_llm_runtime_payload(args))
    _save_run_config(output_root, run_cfg)

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
    run_cfg = {
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
        "agent_placement": placement,
        "agent_model_paths": agent_model_paths,
        "llm_max_new_tokens": args.llm_max_new_tokens,
        "decision_policy": args.decision_policy,
        "require_explicit_accept": args.require_explicit_accept,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "dataset_config_path": str(dataset_cfg_path),
    }
    run_cfg.update(_llm_runtime_payload(args))
    _save_run_config(output_root, run_cfg)

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


def _run_generate_traces(args: argparse.Namespace) -> None:
    scenarios = load_dataset_jsonl(args.dataset)
    total_scenarios = len(scenarios)
    if args.num_scenarios is not None:
        scenarios = scenarios[: args.num_scenarios]
    if not scenarios:
        raise ValueError("No scenarios available to generate traces")

    if args.episodes_per_teacher <= 0:
        raise ValueError("--episodes-per-teacher must be >= 1")

    agent_names = collect_agent_names_from_scenarios(scenarios)

    dataset_cfg = _load_dataset_config(args.dataset)
    dataset_id, dataset_hash = _resolve_dataset_identity_for_run(
        dataset_path=args.dataset,
        explicit_dataset_id=args.dataset_id,
        dataset_config=dataset_cfg,
    )

    if args.llm_backend == "dummy":
        teacher_specs = [{"label": "dummy", "path": None}]
    else:
        teacher_specs = _parse_labeled_model_specs(args.teacher_model_paths)
        if not teacher_specs:
            raise ValueError("--teacher-model-paths is required unless --llm-backend dummy")

    trace_id = _slug(args.trace_id, max_len=120) if args.trace_id else _slug(
        f"traces-{dataset_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        max_len=120,
    )
    trace_root = args.output_dir / trace_id
    runs_root = trace_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, Any]] = []
    run_dirs: list[Path] = []

    episode_counter = 0
    for teacher in teacher_specs:
        for _ in range(args.episodes_per_teacher):
            episode_counter += 1
            episode_seed = args.seed + episode_counter * 1009
            jitter_rng = random.Random(episode_seed)
            temp = _clip(
                float(args.temperature) + jitter_rng.uniform(-abs(args.temperature_jitter), abs(args.temperature_jitter)),
                0.01,
                2.0,
            )
            top_p = _clip(
                float(args.top_p) + jitter_rng.uniform(-abs(args.top_p_jitter), abs(args.top_p_jitter)),
                0.05,
                1.0,
            )

            model_args = _clone_args_for_model(args, model_path=teacher["path"], temperature=temp, top_p=top_p)
            model_sig = _model_signature(model_args, agent_names=agent_names)
            run_id = _slug(f"trace-{episode_counter:03d}-{teacher['label']}-{model_sig}", max_len=120)
            output_root = runs_root / run_id

            scenario_batch = _prepare_episode_scenarios(
                scenarios=scenarios,
                episode_seed=episode_seed,
                shuffle_agent_order=bool(args.shuffle_agent_order),
                shuffle_scenario_order=bool(args.shuffle_scenario_order),
            )

            llms_by_agent, placement = _build_llms_for_agents(model_args, agent_names)
            result = run_experiment(
                scenarios=scenario_batch,
                output_root=output_root,
                llms_by_agent=llms_by_agent,
                max_steps=args.max_steps,
                seed=episode_seed,
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

            run_cfg = {
                "run_id": run_id,
                "mode": "generate-traces-episode",
                "run_at": datetime.now().isoformat(),
                "trace_id": trace_id,
                "dataset_id": dataset_id,
                "dataset_path": str(args.dataset),
                "dataset_file_hash_sha1_10": dataset_hash,
                "dataset_total_scenarios": total_scenarios,
                "dataset_used_scenarios": len(scenario_batch),
                "num_agents": len(agent_names),
                "agent_names": agent_names,
                "model_signature": model_sig,
                "teacher_label": teacher["label"],
                "teacher_model_path": teacher["path"],
                "agent_placement": placement,
                "temperature": temp,
                "top_p": top_p,
                "llm_max_new_tokens": args.llm_max_new_tokens,
                "decision_policy": args.decision_policy,
                "require_explicit_accept": args.require_explicit_accept,
                "max_steps": args.max_steps,
                "seed": episode_seed,
                "shuffle_agent_order": bool(args.shuffle_agent_order),
                "shuffle_scenario_order": bool(args.shuffle_scenario_order),
            }
            run_cfg.update(_llm_runtime_payload(model_args))
            _save_run_config(output_root, run_cfg)

            aggregate = result.get("aggregate", {})
            episode_row = {
                "trace_id": trace_id,
                "episode_index": episode_counter,
                "run_id": run_id,
                "teacher_label": teacher["label"],
                "teacher_model_path": teacher["path"],
                "model_signature": model_sig,
                "temperature": temp,
                "top_p": top_p,
                "seed": episode_seed,
                "run_dir": str(output_root),
                "num_scenarios": aggregate.get("num_scenarios"),
                "agreement_rate": aggregate.get("agreement_rate"),
                "pareto_rate": aggregate.get("pareto_rate"),
                "avg_steps": aggregate.get("avg_steps"),
                "avg_social_welfare": aggregate.get("avg_social_welfare"),
                "avg_nash_product": aggregate.get("avg_nash_product"),
                "avg_welfare_ratio_vs_best": aggregate.get("avg_welfare_ratio_vs_best"),
                "avg_nash_ratio_vs_best": aggregate.get("avg_nash_ratio_vs_best"),
            }
            episode_rows.append(episode_row)
            run_dirs.append(output_root)

            print(
                f"[TRACE] episode={episode_counter} run_id={run_id} "
                f"agreement_rate={aggregate.get('agreement_rate')}",
                flush=True,
            )

    training_summary = build_training_corpora(
        run_dirs=run_dirs,
        output_dir=trace_root / "training_data",
        eval_ratio=args.training_data_eval_ratio,
        seed=args.seed,
        min_preference_margin=args.min_preference_margin,
    )

    _write_rows_csv(trace_root / "trace_generation_summary.csv", episode_rows)
    _json_dump(
        trace_root / "trace_generation_summary.json",
        {
            "generated_at": datetime.now().isoformat(),
            "trace_id": trace_id,
            "dataset_id": dataset_id,
            "dataset_path": str(args.dataset),
            "dataset_file_hash_sha1_10": dataset_hash,
            "episodes_per_teacher": args.episodes_per_teacher,
            "num_runs": len(run_dirs),
            "episodes": episode_rows,
            "training_data": training_summary,
        },
    )

    print(f"[DONE] trace_id={trace_id}")
    print(f"[DONE] trace_root={trace_root}")
    print(f"[DONE] training_data={trace_root / 'training_data'}")
    print(json.dumps(training_summary, ensure_ascii=False, indent=2))


def _run_build_training_data(args: argparse.Namespace) -> None:
    run_dirs = [Path(token).resolve() for token in _parse_csv_tokens(args.run_dirs)]
    if not run_dirs:
        raise ValueError("--run-dirs must include at least one path")

    summary = build_training_corpora(
        run_dirs=run_dirs,
        output_dir=args.output_dir,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        min_preference_margin=args.min_preference_margin,
    )
    print(f"[DONE] output_dir={args.output_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _run_train_sft(args: argparse.Namespace) -> None:
    summary = train_sft(
        model_path=args.model_path,
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=_parse_lora_target_modules(args.lora_target_modules),
    )
    print(f"[DONE] sft_output={args.output_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _run_train_rl(args: argparse.Namespace) -> None:
    summary = train_dpo(
        model_path=args.model_path,
        ref_model_path=args.ref_model_path,
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=_parse_lora_target_modules(args.lora_target_modules),
    )
    print(f"[DONE] rl_output={args.output_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _run_evaluate_models(args: argparse.Namespace) -> None:
    scenarios = load_dataset_jsonl(args.dataset)
    total_scenarios = len(scenarios)
    if args.num_scenarios is not None:
        scenarios = scenarios[: args.num_scenarios]
    if not scenarios:
        raise ValueError("No scenarios available to evaluate")

    agent_names = collect_agent_names_from_scenarios(scenarios)

    dataset_cfg = _load_dataset_config(args.dataset)
    dataset_id, dataset_hash = _resolve_dataset_identity_for_run(
        dataset_path=args.dataset,
        explicit_dataset_id=args.dataset_id,
        dataset_config=dataset_cfg,
    )

    if args.llm_backend == "dummy":
        model_specs = [{"label": "dummy", "path": None}]
    else:
        model_specs = _parse_labeled_model_specs(args.model_paths)
        if not model_specs:
            raise ValueError("--model-paths is required unless --llm-backend dummy")

    eval_id = _slug(args.eval_id, max_len=120) if args.eval_id else _slug(
        f"eval-{dataset_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        max_len=120,
    )

    eval_root = args.output_dir / eval_id
    runs_root = eval_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for index, spec in enumerate(model_specs, start=1):
        model_args = _clone_args_for_model(args, model_path=spec["path"])
        model_sig = _model_signature(model_args, agent_names=agent_names)
        run_id = _slug(f"eval-{index:03d}-{spec['label']}-{model_sig}", max_len=120)
        output_root = runs_root / run_id

        llms_by_agent, placement = _build_llms_for_agents(model_args, agent_names)
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

        run_cfg = {
            "run_id": run_id,
            "mode": "evaluate-models",
            "run_at": datetime.now().isoformat(),
            "eval_id": eval_id,
            "dataset_id": dataset_id,
            "dataset_path": str(args.dataset),
            "dataset_file_hash_sha1_10": dataset_hash,
            "dataset_total_scenarios": total_scenarios,
            "dataset_used_scenarios": len(scenarios),
            "num_agents": len(agent_names),
            "agent_names": agent_names,
            "model_signature": model_sig,
            "model_label": spec["label"],
            "model_eval_path": spec["path"],
            "agent_placement": placement,
            "llm_max_new_tokens": args.llm_max_new_tokens,
            "decision_policy": args.decision_policy,
            "require_explicit_accept": args.require_explicit_accept,
            "max_steps": args.max_steps,
            "seed": args.seed,
        }
        run_cfg.update(_llm_runtime_payload(model_args))
        _save_run_config(output_root, run_cfg)

        aggregate = result["aggregate"]
        row = {
            "label": spec["label"],
            "model_path": spec["path"],
            "run_id": run_id,
            "run_dir": str(output_root),
            "model_signature": model_sig,
            "num_scenarios": aggregate.get("num_scenarios"),
            "agreement_rate": aggregate.get("agreement_rate"),
            "pareto_rate": aggregate.get("pareto_rate"),
            "avg_steps": aggregate.get("avg_steps"),
            "avg_social_welfare": aggregate.get("avg_social_welfare"),
            "avg_nash_product": aggregate.get("avg_nash_product"),
            "avg_welfare_ratio_vs_best": aggregate.get("avg_welfare_ratio_vs_best"),
            "avg_nash_ratio_vs_best": aggregate.get("avg_nash_ratio_vs_best"),
        }
        rows.append(row)

        print(
            f"[EVAL] label={spec['label']} agreement_rate={aggregate.get('agreement_rate')} "
            f"run_dir={output_root}",
            flush=True,
        )

    rows.sort(
        key=lambda r: (
            float(r.get("agreement_rate") or 0.0),
            float(r.get("avg_nash_ratio_vs_best") or 0.0),
            float(r.get("avg_welfare_ratio_vs_best") or 0.0),
            float(r.get("avg_social_welfare") or 0.0),
            -float(r.get("avg_steps") or 1e9),
        ),
        reverse=True,
    )

    _write_rows_csv(eval_root / "leaderboard.csv", rows)
    _json_dump(
        eval_root / "leaderboard.json",
        {
            "generated_at": datetime.now().isoformat(),
            "eval_id": eval_id,
            "dataset_id": dataset_id,
            "dataset_path": str(args.dataset),
            "dataset_file_hash_sha1_10": dataset_hash,
            "rows": rows,
        },
    )
    (eval_root / "leaderboard.md").write_text(_leaderboard_markdown(rows), encoding="utf-8")

    print(f"[DONE] eval_id={eval_id}")
    print(f"[DONE] output_dir={eval_root}")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


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

    if args.command == "generate-traces":
        _run_generate_traces(args)
        return

    if args.command == "build-training-data":
        _run_build_training_data(args)
        return

    if args.command == "train-sft":
        _run_train_sft(args)
        return

    if args.command == "train-rl":
        _run_train_rl(args)
        return

    if args.command == "evaluate-models":
        _run_evaluate_models(args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
