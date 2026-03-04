# Quant Metrics Guide

이 문서는 실행 결과의 정량 지표를 해석하기 위한 가이드입니다.

## 핵심 지표

| 지표 | 의미 | 해석 포인트 |
|---|---|---|
| `agreement_reached` | 해당 시나리오에서 합의 성립 여부 | `False`가 많으면 양보 전략/예약값 설정 점검 |
| `utility_agent_a`, `utility_agent_b` | 최종 합의안에서 각 agent 효용 | 두 값 차이가 크면 불공정 가능성 |
| `social_welfare` | `utility_agent_a + utility_agent_b` | 전체 효율성 지표 |
| `nash_product` | `max(u_a-r_a,0) * max(u_b-r_b,0)` | 효율과 공정성 절충 지표 |
| `pareto_optimal` | 합의안이 파레토 최적이면 `True` | `False`면 개선 가능한 합의안이 존재 |
| `welfare_ratio_vs_best` | 전체 가능한 결과 중 최대 welfare 대비 비율 | 1.0에 가까울수록 효율적 |
| `nash_ratio_vs_best` | 전체 가능한 결과 중 최대 Nash 대비 비율 | 1.0에 가까울수록 균형적 |
| `negotiation_steps` | 합의/종료까지 걸린 스텝 수 | 너무 크면 탐색/양보 속도 조절 필요 |
| `calendar_conflict_ratio_agent_a/b` | 합의된 여행일 중 busy slot 비율 | 낮을수록 캘린더 충돌이 적음 |

## 집계 파일별 용도

- `experiment_summary.csv`
시나리오 단위 raw 수치 분석용(통계, 시각화, 회귀 등).

- `experiment_summary.json`
실험 단위 평균값/비율의 원본 숫자.

- `quant_summary_human_readable.json`
퍼센트/반올림 포맷을 적용한 사람이 읽기 쉬운 JSON.

- `quant_summary.md`
보고서에 바로 붙일 수 있는 마크다운 요약.

- `scenario_xxxx/metrics_human_readable.md`
시나리오 단위 상세 요약.

## 빠른 분석 체크리스트

1. `agreement_rate`가 낮으면 `reservation_value`, `concession_exponent`, `max_steps`를 조정.
2. `welfare_ratio_vs_best`는 높은데 `nash_ratio_vs_best`가 낮으면 한쪽 편향 합의 가능성 점검.
3. `calendar_conflict_ratio_agent_a/b`가 높으면 window score 가중치 또는 캘린더 제약 반영 비중 상향.
4. `pareto_rate`가 낮으면 제안 생성 정책(탑밴드 샘플링, aspiration 곡선) 개선 검토.

## 채팅 로그와 함께 보는 방법

- `chat_transcript.txt`: 실제 대화 흐름 확인
- `metrics_human_readable.md`: 같은 시나리오의 결과 지표 확인

위 두 파일을 함께 보면 어떤 발화 패턴이 합의 효율/공정성에 영향을 주는지 빠르게 확인할 수 있습니다.
