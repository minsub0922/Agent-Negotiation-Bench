from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResult:
    text: str
    raw_text: str
    prompt: str
    backend: str


class LLMBackend(Protocol):
    name: str

    def generate(self, prompt: str, max_new_tokens: int = 256) -> LLMResult:
        ...


class DummyLLM:
    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str, max_new_tokens: int = 256) -> LLMResult:
        normalized = re.sub(r"\s+", " ", prompt.strip())
        if "choice:" in normalized.lower():
            text = (
                "CHOICE: 1\n"
                "MESSAGE: 제안 후보 중에서 현재 일정 충돌을 줄일 수 있는 선택지를 우선 제시하겠습니다."
            )
        elif "action: accept or reject" in normalized.lower():
            u_match = re.search(r"내 효용:\\s*([0-9]*\\.?[0-9]+)", normalized)
            t_match = re.search(r"수락 기준:\\s*([0-9]*\\.?[0-9]+)", normalized)
            utility = float(u_match.group(1)) if u_match else 0.0
            threshold = float(t_match.group(1)) if t_match else 0.5
            action = "ACCEPT" if utility >= threshold else "REJECT"
            msg = (
                "이 조건이면 수락 가능합니다."
                if action == "ACCEPT"
                else "현재 기준에는 부족해 조건 조정이 필요합니다."
            )
            text = f"ACTION: {action}\\nMESSAGE: {msg}"
        else:
            text = f"MESSAGE: {self.name}: 제약을 반영해 균형 잡힌 선택을 하겠습니다."
        return LLMResult(text=text, raw_text=text, prompt=prompt, backend="dummy")


class HFLocalLLM:
    def __init__(
        self,
        model_path: str,
        name: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        trust_remote_code: bool = True,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        low_cpu_mem_usage: bool = True,
        max_memory: dict[int | str, str] | None = None,
    ):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.name = name
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        resolved_device_map = self._resolve_device_map(device_map)
        resolved_torch_dtype = self._resolve_torch_dtype(torch_dtype, torch)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            device_map=resolved_device_map,
            torch_dtype=resolved_torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            max_memory=max_memory,
        )

        self.generator = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""
        return " ".join(lines)

    @staticmethod
    def _resolve_device_map(device_map: str):
        lowered = str(device_map).strip().lower()
        if lowered in {"auto", "balanced", "balanced_low_0", "sequential"}:
            return lowered
        if lowered == "cpu":
            return {"": "cpu"}
        if lowered.startswith("cuda:"):
            gpu_idx = int(lowered.split(":", 1)[1])
            return {"": gpu_idx}
        if lowered.isdigit():
            return {"": int(lowered)}
        return device_map

    @staticmethod
    def _resolve_torch_dtype(torch_dtype: str, torch):
        lowered = str(torch_dtype).strip().lower()
        if lowered == "auto":
            return "auto"
        if lowered in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if lowered in {"fp16", "float16", "half"}:
            return torch.float16
        if lowered in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

    def generate(self, prompt: str, max_new_tokens: int = 256) -> LLMResult:
        out = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
            top_p=self.top_p,
            return_full_text=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        raw = out[0]["generated_text"] if out else ""
        cleaned = self._clean_text(raw)
        if not cleaned:
            cleaned = "조건을 조금 더 맞춰보겠습니다."
        return LLMResult(text=cleaned, raw_text=raw, prompt=prompt, backend="hf-local")


def build_llm(
    backend: str,
    name: str,
    model_path: str | None,
    temperature: float,
    top_p: float,
    allow_dummy_fallback: bool,
    device_map: str,
    torch_dtype: str = "auto",
    low_cpu_mem_usage: bool = True,
    max_memory: dict[int | str, str] | None = None,
) -> LLMBackend:
    normalized = backend.lower().strip()
    if normalized == "dummy":
        return DummyLLM(name=name)

    if normalized != "hf":
        raise ValueError(f"Unsupported backend: {backend}")

    if not model_path:
        raise ValueError("HF backend requires --model-path (or agent-specific model paths)")

    try:
        return HFLocalLLM(
            model_path=model_path,
            name=name,
            temperature=temperature,
            top_p=top_p,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            max_memory=max_memory,
        )
    except Exception:
        if not allow_dummy_fallback:
            raise
        return DummyLLM(name=name)
