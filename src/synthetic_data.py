from __future__ import annotations

import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    lines = [line.rstrip() for line in str(text).splitlines()]
    cleaned = "\n".join(line for line in lines if line.strip())
    return cleaned.strip()


def _choice_from_text(text: str) -> int | None:
    m = re.search(r"choice\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def _action_from_text(text: str) -> str | None:
    m = re.search(r"action\s*[:=]\s*([a-z_]+)", text, flags=re.IGNORECASE)
    if m:
        token = m.group(1).upper()
        if token.startswith("ACCEPT"):
            return "ACCEPT"
        if token.startswith("REJECT"):
            return "REJECT"
    lowered = text.lower()
    if "accept" in lowered:
        return "ACCEPT"
    if "reject" in lowered:
        return "REJECT"
    return None


def _decision_to_action(decision: str | None) -> str | None:
    if not decision:
        return None
    token = decision.upper().strip()
    if token.startswith("ACCEPT"):
        return "ACCEPT"
    if token.startswith("REJECT"):
        return "REJECT"
    return None


def _default_offer_message(candidate_offer: dict[str, Any] | None) -> str:
    if not candidate_offer:
        return "I propose this candidate offer."
    return f"I propose this offer: {json.dumps(candidate_offer, ensure_ascii=False)}"


def _default_action_message(action: str) -> str:
    if action == "ACCEPT":
        return "This offer meets my current constraints, so I can accept."
    return "This offer does not meet my current constraints, so I have to reject."


def _offer_completion(choice: int, message: str) -> str:
    return f"CHOICE: {choice}\nMESSAGE: {message.strip()}"


def _response_completion(action: str, message: str) -> str:
    return f"ACTION: {action}\nMESSAGE: {message.strip()}"


def _normalize_offer_completion(event: dict[str, Any]) -> str:
    raw = _clean_text(event.get("llm_raw_output") or event.get("llm_text"))
    choice = event.get("llm_choice")
    message = _clean_text(event.get("message"))
    try:
        choice_int = int(choice)
    except Exception:
        choice_int = 1

    if raw:
        parsed = _choice_from_text(raw)
        if parsed is not None:
            return raw

    if not message:
        candidates = event.get("offer_candidates") or []
        if candidates and 1 <= choice_int <= len(candidates):
            message = _default_offer_message(candidates[choice_int - 1].get("offer"))
        else:
            message = _default_offer_message(None)

    return _offer_completion(choice_int, message)


def _normalize_response_completion(event: dict[str, Any], required_action: str | None = None) -> str:
    raw = _clean_text(event.get("llm_raw_output") or event.get("llm_text"))
    message = _clean_text(event.get("message"))

    if raw:
        parsed = _action_from_text(raw)
        if required_action is None or parsed == required_action:
            return raw

    fallback_action = required_action or _decision_to_action(event.get("decision")) or "REJECT"
    if not message:
        message = _default_action_message(fallback_action)

    return _response_completion(fallback_action, message)


def _normalize_completion(event: dict[str, Any]) -> str:
    kind = str(event.get("kind") or "").strip().lower()
    if kind == "offer":
        return _normalize_offer_completion(event)
    if kind == "response":
        return _normalize_response_completion(event)

    raw = _clean_text(event.get("llm_raw_output") or event.get("llm_text"))
    if raw:
        return raw

    message = _clean_text(event.get("message"))
    if message:
        return f"MESSAGE: {message}"
    return ""


def _format_sft_text(prompt: str, completion: str) -> str:
    return (
        "### Negotiation Prompt\n"
        f"{prompt.strip()}\n\n"
        "### Negotiation Response\n"
        f"{completion.strip()}"
    )


def _split_records(
    records: list[dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []

    ratio = max(0.0, min(float(eval_ratio), 0.5))
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    n_eval = int(round(len(shuffled) * ratio))
    if ratio > 0.0:
        n_eval = max(1, n_eval)
    n_eval = min(n_eval, len(shuffled))

    eval_rows = shuffled[:n_eval]
    train_rows = shuffled[n_eval:]
    return train_rows, eval_rows


def _build_offer_preference(
    event: dict[str, Any],
    prompt: str,
    base_meta: dict[str, Any],
    min_preference_margin: float,
) -> dict[str, Any] | None:
    candidates = event.get("offer_candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return None

    try:
        chosen_rank = int(event.get("llm_choice"))
    except Exception:
        return None
    if chosen_rank < 1 or chosen_rank > len(candidates):
        return None

    chosen_candidate = candidates[chosen_rank - 1]
    chosen_utility = _safe_float(chosen_candidate.get("utility_self"))
    if chosen_utility is None:
        return None

    best_alt: tuple[int, dict[str, Any], float] | None = None
    for rank, candidate in enumerate(candidates, start=1):
        if rank == chosen_rank:
            continue
        alt_utility = _safe_float(candidate.get("utility_self"))
        if alt_utility is None:
            continue
        margin = chosen_utility - alt_utility
        if margin < min_preference_margin:
            continue
        if best_alt is None or margin > best_alt[2]:
            best_alt = (rank, candidate, margin)

    if best_alt is None:
        return None

    rejected_rank, rejected_candidate, margin = best_alt
    chosen = _normalize_offer_completion(event)
    rejected_message = _default_offer_message(rejected_candidate.get("offer"))
    rejected = _offer_completion(rejected_rank, rejected_message)

    row = dict(base_meta)
    row.update(
        {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": "offer-candidate-ranking",
            "preference_margin": round(float(margin), 6),
        }
    )
    return row


def _build_response_preference(
    event: dict[str, Any],
    prompt: str,
    base_meta: dict[str, Any],
    min_preference_margin: float,
) -> dict[str, Any] | None:
    utility = _safe_float(event.get("utility_self"))
    threshold = _safe_float(event.get("threshold"))
    if utility is None or threshold is None:
        return None

    margin = abs(utility - threshold)
    if margin < min_preference_margin:
        return None

    should_accept = utility >= threshold
    chosen_action = "ACCEPT" if should_accept else "REJECT"
    rejected_action = "REJECT" if should_accept else "ACCEPT"

    chosen = _normalize_response_completion(event, required_action=chosen_action)
    rejected = _response_completion(rejected_action, _default_action_message(rejected_action))

    row = dict(base_meta)
    row.update(
        {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": "response-threshold-rule",
            "preference_margin": round(float(margin), 6),
            "utility_self": round(float(utility), 6),
            "threshold": round(float(threshold), 6),
        }
    )
    return row


def collect_training_examples(
    run_dirs: list[Path],
    min_preference_margin: float = 0.02,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sft_rows: list[dict[str, Any]] = []
    dpo_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        run_cfg = _read_json(run_dir / "run_config.json") or {}
        run_id = str(run_cfg.get("run_id") or run_dir.name)
        model_signature = str(run_cfg.get("model_signature") or "unknown")

        scenario_dirs = sorted(p for p in run_dir.glob("scenario_*") if p.is_dir())
        for scenario_dir in scenario_dirs:
            scenario_id = scenario_dir.name
            events = _read_jsonl(scenario_dir / "agent_event_log.jsonl")

            for event in events:
                prompt = _clean_text(event.get("llm_prompt"))
                if not prompt:
                    continue

                event_id = int(event.get("event_id") or 0)
                kind = str(event.get("kind") or "").strip().lower()
                speaker = str(event.get("speaker") or "unknown")

                base_meta = {
                    "run_id": run_id,
                    "model_signature": model_signature,
                    "scenario_id": scenario_id,
                    "event_id": event_id,
                    "event_kind": kind,
                    "speaker": speaker,
                }

                completion = _normalize_completion(event)
                if completion:
                    sft_record = dict(base_meta)
                    sft_record.update(
                        {
                            "prompt": prompt,
                            "completion": completion,
                            "text": _format_sft_text(prompt, completion),
                        }
                    )
                    sft_rows.append(sft_record)

                if kind == "offer":
                    pref = _build_offer_preference(
                        event=event,
                        prompt=prompt,
                        base_meta=base_meta,
                        min_preference_margin=min_preference_margin,
                    )
                    if pref is not None:
                        dpo_rows.append(pref)
                elif kind == "response":
                    pref = _build_response_preference(
                        event=event,
                        prompt=prompt,
                        base_meta=base_meta,
                        min_preference_margin=min_preference_margin,
                    )
                    if pref is not None:
                        dpo_rows.append(pref)

    return sft_rows, dpo_rows


def build_training_corpora(
    run_dirs: list[Path],
    output_dir: Path,
    eval_ratio: float = 0.05,
    seed: int = 7,
    min_preference_margin: float = 0.02,
) -> dict[str, Any]:
    run_dirs = [Path(p).resolve() for p in run_dirs]
    sft_rows, dpo_rows = collect_training_examples(
        run_dirs=run_dirs,
        min_preference_margin=min_preference_margin,
    )

    sft_train, sft_eval = _split_records(sft_rows, eval_ratio=eval_ratio, seed=seed)
    dpo_train, dpo_eval = _split_records(dpo_rows, eval_ratio=eval_ratio, seed=seed + 17)

    output_dir.mkdir(parents=True, exist_ok=True)

    sft_all_path = output_dir / "sft_all.jsonl"
    sft_train_path = output_dir / "sft_train.jsonl"
    sft_eval_path = output_dir / "sft_eval.jsonl"
    dpo_all_path = output_dir / "dpo_all.jsonl"
    dpo_train_path = output_dir / "dpo_train.jsonl"
    dpo_eval_path = output_dir / "dpo_eval.jsonl"

    _write_jsonl(sft_all_path, sft_rows)
    _write_jsonl(sft_train_path, sft_train)
    _write_jsonl(sft_eval_path, sft_eval)
    _write_jsonl(dpo_all_path, dpo_rows)
    _write_jsonl(dpo_train_path, dpo_train)
    _write_jsonl(dpo_eval_path, dpo_eval)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "run_dirs": [str(p) for p in run_dirs],
        "eval_ratio": eval_ratio,
        "seed": seed,
        "min_preference_margin": min_preference_margin,
        "counts": {
            "sft_all": len(sft_rows),
            "sft_train": len(sft_train),
            "sft_eval": len(sft_eval),
            "dpo_all": len(dpo_rows),
            "dpo_train": len(dpo_train),
            "dpo_eval": len(dpo_eval),
        },
        "files": {
            "sft_all": str(sft_all_path),
            "sft_train": str(sft_train_path),
            "sft_eval": str(sft_eval_path),
            "dpo_all": str(dpo_all_path),
            "dpo_train": str(dpo_train_path),
            "dpo_eval": str(dpo_eval_path),
        },
    }
    _write_json(output_dir / "training_data_summary.json", summary)
    return summary
