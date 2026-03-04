# negmas 기반 2-Agent 여행 일정 협상 프로젝트

이 프로젝트는 `negmas`를 사용해, 각 에이전트의 캘린더 일정과 선호를 바탕으로 여행 일정을 협상하는 실험 파이프라인입니다.

- 랜덤 시나리오 데이터셋 생성
- 로컬 LLM 2개(각 agent 별 모델 경로 가능)로 협상 발화 생성
- 협상 상세 로그(JSONL) + 사람이 읽기 쉬운 대화 JSON 저장
- 정량 평가 지표 저장 및 실험 요약(CSV/JSON) 생성

## 1) 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 데이터셋 생성

```bash
python -m src.main generate-dataset \
  --num-scenarios 30 \
  --seed 42 \
  --horizon-days 14 \
  --output data/scenarios.jsonl
```

## 3) 협상 실행 (기존 데이터셋 사용)

### 3-1. 로컬 HF 모델 2개 사용

```bash
python -m src.main run \
  --dataset data/scenarios.jsonl \
  --llm-backend hf \
  --agent-a-model-path /absolute/path/to/model_a \
  --agent-b-model-path /absolute/path/to/model_b \
  --max-steps 40 \
  --output-dir outputs
```

### 3-2. 하나의 모델 경로를 두 agent에 공통 적용

```bash
python -m src.main run \
  --dataset data/scenarios.jsonl \
  --llm-backend hf \
  --model-path /absolute/path/to/model \
  --max-steps 40 \
  --output-dir outputs
```

### 3-3. 빠른 점검용 더미 LLM

```bash
python -m src.main run \
  --dataset data/scenarios.jsonl \
  --llm-backend dummy \
  --max-steps 40 \
  --output-dir outputs
```

## 4) 전체 파이프라인 한 번에 실행

```bash
python -m src.main full \
  --num-scenarios 20 \
  --seed 7 \
  --dataset-output data/scenarios.jsonl \
  --llm-backend hf \
  --model-path /absolute/path/to/model \
  --output-dir outputs
```

## 출력 구조

실행 시 `outputs/run_YYYYMMDD_HHMMSS/` 아래에 저장됩니다.

- `experiment_summary.csv`: 시나리오별 정량평가 행 데이터
- `experiment_summary.json`: 전체 평균/합의율 등 집계
- `run_config.json`: 실행 파라미터(모델 경로, seed, backend 등)
- `scenario_xxxx/scenario.json`: 해당 시나리오 원본
- `scenario_xxxx/metrics.json`: 해당 협상 지표
- `scenario_xxxx/negmas_trace.jsonl`: `negmas` 레벨 협상 trace
- `scenario_xxxx/agent_event_log.jsonl`: agent propose/respond + LLM prompt/raw 응답 포함 상세 로그
- `scenario_xxxx/dialogue_human_readable.json`: 사람이 읽기 쉬운 대화 로그 + 각 턴 효용

## 주요 정량지표

- `agreement_reached`: 합의 여부
- `utility_agent_a`, `utility_agent_b`: 최종 합의안에서 각 agent 효용
- `social_welfare`: 두 agent 효용 합
- `nash_product`: `(u_a-r_a)*(u_b-r_b)` 기반 Nash product
- `pareto_optimal`: 최종 합의안의 파레토 최적 여부
- `welfare_ratio_vs_best`, `nash_ratio_vs_best`: 전체 outcome space 최적 대비 비율
- `calendar_conflict_ratio_agent_a/b`: 합의된 여행일과 각자 캘린더 충돌 비율

## 참고

- LLM 응답은 협상 발화 생성에 사용되며, 제안/수락 결정은 `negmas` 협상자 전략(`LLMCalendarNegotiator`)과 효용함수에 따라 이루어집니다.
- 로컬 모델 메모리 사용량이 큰 경우 `--allow-dummy-fallback` 옵션을 사용해 점검 가능합니다.
