from __future__ import annotations

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

    def generate(self, prompt: str, max_new_tokens: int = 96) -> LLMResult:
        ...


class DummyLLM:
    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str, max_new_tokens: int = 96) -> LLMResult:
        style = "제안을 조정해보겠습니다." if "offer" in prompt.lower() else "조건을 재검토해볼게요."
        short = re.sub(r"\s+", " ", prompt.strip())
        short = short[-120:]
        text = f"{self.name}: {style} ({short})"
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
    ):
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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
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
        merged = " ".join(lines)
        if len(merged) > 360:
            merged = merged[:360].rstrip() + "..."
        return merged

    def generate(self, prompt: str, max_new_tokens: int = 96) -> LLMResult:
        out = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
            top_p=self.top_p,
            return_full_text=False,
            pad_token_id=self.tokenizer.pad_token_id,
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
        )
    except Exception:
        if not allow_dummy_fallback:
            raise
        return DummyLLM(name=name)
