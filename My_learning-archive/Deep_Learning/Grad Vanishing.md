# Gradient Vanishing의 보안책

현대 딥러닝 아키텍처는  
“gradient vanishing이 생길 수 있는 모든 지점마다  
각각의 ‘국소적(local) 해결책’을 체계적으로 배치한 구조

## 1. Residual Connection

- Self-attention, FFN은 모두 **비선형 + 선형 조합**
    
- Jacobian의 spectral norm < 1 → gradient 감소


## 2. LayerNorm: “분산 붕괴/폭발 방지기”

## 3. Self-Attention 자체의 안정화 설계

QK^T 내적하고 root(d_K)로 나누기

## 4. Position Embedding / Embedding Layer에서도 발생하는 문제

### ❌ 문제

- 입력 embedding 분산이 크거나 작으면
- 초반 layer부터 saturation

### ✅ 해결

- embedding scaling (`√d_model`)

## 5. Optimizer 차원의 해결책 (Block 외부)


✅ Adam 
- gradient norm 안정화
- 희귀 파라미터 업데이트 보정

✅ Learning rate warmup:
- 초반 gradient 폭발 방지

✅ Gradient clipping:
- local explosion 방지

## 6. 대충 정리 (이거 말고도 많은 Method들 있음.)

| 위험 지점       | 문제                 | 해결 장치          |     |
| ----------- | ------------------ | -------------- | --- |
| Linear 연쇄   | 분산 붕괴 / 폭발         | LayerNorm      |     |
| 비선형 변환      | Jacobian < 1       | Residual       |     |
| Dot-product | softmax saturation | Scaling (1/√d) |     |
| 긴 depth     | 곱셈 사슬              | Identity path  |     |
| 토큰 혼합       | gradient 쏠림        | Multi-head     |     |
| 전체 학습       | 불안정                | Adam, warmup   |     |
