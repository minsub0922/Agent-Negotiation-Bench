# Synthetic Negotiation Training Pipeline

This document explains the end-to-end workflow added to this repository:

1. Generate diverse teacher traces with vLLM (or HF/dummy backends).
2. Build synthetic SFT and preference datasets from traces.
3. Train a smaller target negotiator with TRL (`SFTTrainer` + DPO stage).
4. Evaluate and rank negotiator models on shared negotiation scenarios.

## 1) Install

Base dependencies (simulation + TRL training):

```bash
pip install -r requirements.txt
```

For vLLM teacher inference:

```bash
pip install -r requirements-vllm.txt
```

## 2) Generate Scenario Dataset

```bash
python -m src.main generate-dataset \
  --num-scenarios 200 \
  --num-agents 3 \
  --seed 42 \
  --output data/scenarios_train.jsonl
```

## 3) Generate Diverse Teacher Traces + Training Data

### 3.1 vLLM teachers

```bash
python -m src.main generate-traces \
  --dataset data/scenarios_train.jsonl \
  --llm-backend vllm \
  --teacher-model-paths teacher_a=/models/teacher_a,teacher_b=/models/teacher_b \
  --episodes-per-teacher 3 \
  --temperature 0.8 \
  --temperature-jitter 0.25 \
  --top-p 0.92 \
  --top-p-jitter 0.05 \
  --shuffle-agent-order \
  --shuffle-scenario-order \
  --decision-policy llm-hybrid \
  --require-explicit-accept \
  --output-dir outputs/synthetic \
  --trace-id teacher_mix_v1
```

Outputs:

- `outputs/synthetic/<trace_id>/runs/*`: per-episode negotiation runs.
- `outputs/synthetic/<trace_id>/training_data/sft_{train,eval}.jsonl`
- `outputs/synthetic/<trace_id>/training_data/dpo_{train,eval}.jsonl`
- `outputs/synthetic/<trace_id>/trace_generation_summary.{json,csv}`

### 3.2 Build training data from existing run dirs

```bash
python -m src.main build-training-data \
  --run-dirs outputs/synthetic/teacher_mix_v1/runs/trace-001-...,outputs/synthetic/teacher_mix_v1/runs/trace-002-... \
  --output-dir outputs/synthetic/teacher_mix_v1/training_data_rebuilt
```

## 4) Train Target Model (SFT)

```bash
python -m src.main train-sft \
  --model-path /models/target_base \
  --train-file outputs/synthetic/teacher_mix_v1/training_data/sft_train.jsonl \
  --eval-file outputs/synthetic/teacher_mix_v1/training_data/sft_eval.jsonl \
  --output-dir outputs/train/target_sft_v1 \
  --max-seq-length 2048 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 \
  --num-train-epochs 1.5 \
  --bf16 \
  --use-lora
```

## 5) RL-Style Preference Optimization (DPO)

```bash
python -m src.main train-rl \
  --model-path outputs/train/target_sft_v1/final_model \
  --ref-model-path /models/target_base \
  --train-file outputs/synthetic/teacher_mix_v1/training_data/dpo_train.jsonl \
  --eval-file outputs/synthetic/teacher_mix_v1/training_data/dpo_eval.jsonl \
  --output-dir outputs/train/target_dpo_v1 \
  --beta 0.1 \
  --max-seq-length 2048 \
  --max-prompt-length 1024 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-6 \
  --num-train-epochs 1.0 \
  --bf16 \
  --use-lora
```

## 6) Evaluate Negotiator Models

```bash
python -m src.main evaluate-models \
  --dataset data/scenarios_eval.jsonl \
  --llm-backend vllm \
  --model-paths base=/models/target_base,sft=outputs/train/target_sft_v1/final_model,rl=outputs/train/target_dpo_v1/final_model \
  --num-scenarios 100 \
  --decision-policy llm-hybrid \
  --require-explicit-accept \
  --output-dir outputs/evaluations \
  --eval-id negotiator_eval_v1
```

Outputs:

- `outputs/evaluations/<eval_id>/runs/*`: per-model run outputs and full negotiation traces.
- `outputs/evaluations/<eval_id>/leaderboard.{json,csv,md}`

## 7) Suggested Iteration Loop

1. Start with broad teacher diversity (`episodes-per-teacher`, jitter, multiple teachers).
2. Run SFT and verify baseline metrics improve over base model.
3. Run DPO to improve decision quality on threshold-sensitive responses.
4. Track regressions using the same eval dataset and leaderboard outputs.
