# negmas 기반 N-Agent 여행 일정 협상 프로젝트

이 프로젝트는 `negmas`를 사용해, 각 에이전트의 캘린더 일정과 선호를 바탕으로 여행 일정을 협상하는 실험 파이프라인입니다.

- 랜덤 시나리오 데이터셋 생성
- 로컬 LLM N개(각 agent 별 모델 경로 가능)로 협상 발화 생성
- 협상 상세 로그(JSONL) + 사람이 읽기 쉬운 대화 JSON 저장
- 채팅 내역처럼 메시지만 보는 human-only transcript 저장
- 정량 평가 지표 저장 + 보기 쉬운 요약 MD/JSON 생성

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
  --num-agents 3 \
  --seed 42 \
  --horizon-days 14 \
  --dataset-id ds_paper_v1 \
  --output data/scenarios.jsonl
```

## 3) 협상 실행 (기존 데이터셋 사용)

### 3-1. 로컬 HF 모델 사용 (agent별 모델 경로 지정)

```bash
python -m src.main run \
  --dataset data/scenarios.jsonl \
  --dataset-id ds_paper_v1 \
  --run-id run_modelA_modelB_ds1 \
  --llm-backend hf \
  --num-gpus 2 \
  --torch-dtype bfloat16 \
  --decision-policy llm-hybrid \
  --require-explicit-accept \
  --agent-model-paths agent_a=/absolute/path/to/model_a,agent_b=/absolute/path/to/model_b,agent_c=/absolute/path/to/model_c \
  --llm-max-new-tokens 256 \
  --max-steps 40 \
  --output-dir outputs
```

### 3-2. 하나의 모델 경로를 모든 agent에 공통 적용

```bash
python -m src.main run \
  --dataset data/scenarios.jsonl \
  --dataset-id ds_paper_v1 \
  --llm-backend hf \
  --num-gpus 1 \
  --torch-dtype bfloat16 \
  --decision-policy llm-hybrid \
  --require-explicit-accept \
  --model-path /absolute/path/to/model \
  --llm-max-new-tokens 256 \
  --max-steps 40 \
  --output-dir outputs
```

### 3-3. 빠른 점검용 더미 LLM

```bash
python -m src.main run \
  --dataset data/scenarios.jsonl \
  --dataset-id ds_paper_v1 \
  --llm-backend dummy \
  --llm-max-new-tokens 256 \
  --max-steps 40 \
  --output-dir outputs
```

## 4) 전체 파이프라인 한 번에 실행

```bash
python -m src.main full \
  --num-scenarios 20 \
  --num-agents 3 \
  --seed 7 \
  --dataset-id ds_full_v1 \
  --run-id run_full_baseline \
  --dataset-output data/scenarios.jsonl \
  --llm-backend hf \
  --num-gpus 2 \
  --torch-dtype bfloat16 \
  --decision-policy llm-hybrid \
  --require-explicit-accept \
  --model-path /absolute/path/to/model \
  --llm-max-new-tokens 256 \
  --output-dir outputs
```

## 출력 구조

실행 시 `outputs/<run_id>/` 아래에 저장됩니다. (`run_id`는 기본적으로 모델 시그니처 + dataset id를 포함해 자동 생성)

- `experiment_summary.csv`: 시나리오별 정량평가 행 데이터
- `experiment_summary.json`: 전체 평균/합의율 등 집계
- `my_metrics.csv`: 새 `my_metrics` 블록의 시나리오별 flatten 결과
- `my_metrics_summary.json`: `my_metrics`의 per-run/per-scenario 평균·표준편차 요약
- `quant_summary.md`: 사람이 바로 읽을 수 있는 정량 요약 리포트
- `quant_summary_human_readable.json`: 문자열 포맷(퍼센트 등) 적용한 정량 요약
- `github_issue_collection.md`: 시나리오별 Issue 파일 목록 + 실험 개요
- `run_config.json`: 실행 파라미터(모델 경로, seed, backend 등)
- `dataset_config_snapshot.json`: 실행에 사용한 dataset 식별정보(id/hash/path) 스냅샷
- `scenario_xxxx/scenario.json`: 해당 시나리오 원본
- `scenario_xxxx/metrics.json`: 해당 협상 지표
  - 기존 metrics는 그대로 유지되고, 동일 레벨에 `my_metrics` 블록이 추가 저장됩니다.
- `scenario_xxxx/metrics_human_readable.md`: 시나리오 단위 정량 요약
- `scenario_xxxx/negmas_trace.jsonl`: `negmas` 레벨 협상 trace
- `scenario_xxxx/agent_event_log.jsonl`: agent propose/respond + LLM prompt/raw 응답 포함 상세 로그
  - `decision_source`, `llm_action`, `llm_choice`, `offer_candidates` 필드로 실제 의사결정 출처 추적 가능
- `scenario_xxxx/dialogue_human_readable.json`: 사람이 읽기 쉬운 대화 로그 + 각 턴 효용
- `scenario_xxxx/chat_transcript.json`: 메시지만 포함한 채팅형 로그(JSON)
- `scenario_xxxx/chat_transcript.txt`: 메시지만 포함한 채팅형 로그(TXT)
- `scenario_xxxx/issue_bundle.json`: 채팅+정량+상세 JSON 통합 구조
- `scenario_xxxx/issue_bundle.md`: GitHub Issue 본문으로 바로 사용할 수 있는 통합 마크다운

## 주요 정량지표

- `agreement_reached`: 합의 여부
- `utility_by_agent`: 최종 합의안에서 agent별 효용
- `reservation_by_agent`: agent별 reservation value
- `calendar_conflict_ratio_by_agent`: agent별 캘린더 충돌 비율
- `my_metrics`: 기존 metrics와 독립적으로 계산되는 추가 평가 블록
- `social_welfare`: 모든 agent 효용 합
- `nash_product`: 모든 agent에 대해 `Π max(u_i-r_i, 0)` 기반 Nash product
- `pareto_optimal`: 최종 합의안의 파레토 최적 여부
- `welfare_ratio_vs_best`, `nash_ratio_vs_best`: 전체 outcome space 최적 대비 비율
- 하위호환 키(`utility_agent_a`, `utility_agent_b` 등)도 함께 저장됨

정량 지표 해설은 `docs/QUANT_METRICS_GUIDE.md` 를 참고하세요.

## 참고

- LLM 응답은 협상 발화 생성에 사용되며, 제안/수락 결정은 `negmas` 협상자 전략(`LLMCalendarNegotiator`)과 효용함수에 따라 이루어집니다.
- 답변 잘림을 줄이기 위해 기본 `--llm-max-new-tokens` 값을 256으로 높였고, 출력 후처리에서 길이 하드컷을 제거했습니다.
- 로컬 모델 메모리 사용량이 큰 경우 `--allow-dummy-fallback` 옵션을 사용해 점검 가능합니다.
- `run`/`full` 실행 시 자동 생성되는 `run_id`에는 모델 시그니처 + dataset id가 포함됩니다(원하면 `--run-id`로 직접 지정 가능).
- 데이터셋 생성 시 `<dataset>.config.json` 파일이 함께 저장되며, `dataset_id`/hash/seed 등을 기록합니다.
- `--num-gpus`를 지정하면 agent 배치는 자동으로 결정됩니다.
  - `--num-gpus 0`: 모든 agent CPU
  - `--num-gpus 1`: 모든 agent `cuda:0`
  - `--num-gpus >= 2`: `agent_i -> cuda:(i % num_gpus)` 라운드로빈 배치
  - 모든 agent가 동일 모델 경로(`--model-path` 또는 동일한 `--agent-model-paths`)를 쓰면, 모델 인스턴스를 공유해 메모리를 절약합니다.
- 실제 배치 결과는 실행 로그의 `[PLACEMENT] ...`와 `run_config.json > agent_placement`에 저장됩니다.
- 14B 모델에서 속도가 느리면 `--llm-max-new-tokens 64` 또는 `96`으로 낮춰 먼저 점검하세요.
- OOM이 나면 `--torch-dtype bfloat16`(H100 권장), `--llm-max-new-tokens 64`, `--max-steps 20`으로 먼저 안정화하세요.
- `--decision-policy` 기본값은 `llm-hybrid`이며, 모델 출력이 제안 선택/수락 결정에 반영됩니다.
  - `heuristic`: 기존처럼 효용 기반 규칙만 사용
  - `llm-hybrid`: LLM 출력 + 최소 가드레일
  - `llm-only`: LLM 출력 중심으로 결정
- `--require-explicit-accept` 기본값은 `True`입니다. 모델이 명시적으로 `ACTION: ACCEPT`를 출력하지 않으면 합의로 처리하지 않습니다.

## my_metrics Smoke Test

```bash
python scripts/smoke_test_my_metrics.py
```

- 3 runs x 3 scenarios를 dummy backend로 실행해 `my_metrics` 저장/계산을 점검합니다.
- 기본적으로 테스트 산출물은 실행 후 자동 정리됩니다. 보관하려면 `--keep-artifacts`를 사용하세요.
