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
                "MESSAGE: I will prioritize candidate options that reduce current schedule conflicts."
            )
        elif "action: accept or reject" in normalized.lower():
            u_match = re.search(r"my utility:\\s*([0-9]*\\.?[0-9]+)", normalized, flags=re.IGNORECASE)
            t_match = re.search(
                r"current acceptance threshold:\\s*([0-9]*\\.?[0-9]+)",
                normalized,
                flags=re.IGNORECASE,
            )
            utility = float(u_match.group(1)) if u_match else 0.0
            threshold = float(t_match.group(1)) if t_match else 0.5
            action = "ACCEPT" if utility >= threshold else "REJECT"
            msg = (
                "I can accept under these terms."
                if action == "ACCEPT"
                else "These terms are below my current criteria and need adjustment."
            )
            text = f"ACTION: {action}\\nMESSAGE: {msg}"
        else:
            text = f"MESSAGE: {self.name}: I will make a balanced choice that reflects the constraints."
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
            cleaned = "I will adjust the terms a bit further."
        return LLMResult(text=cleaned, raw_text=raw, prompt=prompt, backend="hf-local")


class VLLMLocalLLM:
    def __init__(
        self,
        model_path: str,
        name: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        trust_remote_code: bool = True,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        enable_prefix_caching: bool = True,
        swap_space: int = 4,
    ):
        from vllm import LLM, SamplingParams

        self.name = name
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self._SamplingParams = SamplingParams

        kwargs = {
            "model": model_path,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": max(1, int(tensor_parallel_size)),
            "dtype": str(dtype),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "enable_prefix_caching": bool(enable_prefix_caching),
            "swap_space": int(swap_space),
        }
        if max_model_len is not None:
            kwargs["max_model_len"] = int(max_model_len)

        self._llm = LLM(**kwargs)

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""
        return " ".join(lines)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> LLMResult:
        sampling_params = self._SamplingParams(
            temperature=max(self.temperature, 1e-5),
            top_p=self.top_p,
            max_tokens=max(1, int(max_new_tokens)),
        )
        out = self._llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
        raw = ""
        if out and out[0].outputs:
            raw = out[0].outputs[0].text
        cleaned = self._clean_text(raw)
        if not cleaned:
            cleaned = "I need to revise this proposal to satisfy constraints."
        return LLMResult(text=cleaned, raw_text=raw, prompt=prompt, backend="vllm")


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
    trust_remote_code: bool = True,
    vllm_tensor_parallel_size: int = 1,
    vllm_dtype: str = "auto",
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_model_len: int | None = None,
    vllm_enable_prefix_caching: bool = True,
    vllm_swap_space: int = 4,
) -> LLMBackend:
    normalized = backend.lower().strip()
    if normalized == "dummy":
        return DummyLLM(name=name)

    if normalized not in {"hf", "vllm"}:
        raise ValueError(f"Unsupported backend: {backend}")

    if not model_path:
        raise ValueError(f"{normalized.upper()} backend requires --model-path (or agent-specific model paths)")

    try:
        if normalized == "hf":
            return HFLocalLLM(
                model_path=model_path,
                name=name,
                temperature=temperature,
                top_p=top_p,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                max_memory=max_memory,
            )

        return VLLMLocalLLM(
            model_path=model_path,
            name=name,
            temperature=temperature,
            top_p=top_p,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=vllm_tensor_parallel_size,
            dtype=vllm_dtype,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_model_len=vllm_max_model_len,
            enable_prefix_caching=vllm_enable_prefix_caching,
            swap_space=vllm_swap_space,
        )
    except Exception:
        if not allow_dummy_fallback:
            raise
        return DummyLLM(name=name)
