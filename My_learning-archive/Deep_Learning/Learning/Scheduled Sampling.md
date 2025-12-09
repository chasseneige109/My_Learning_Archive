
### Scheduled Sampling (Bengio et al., 2015)

훈련 중 매 step마다:

- 확률 p: 정답 토큰 사용
- 확률 1−p: 모델 예측 토큰 사용
- 
### Scheduled Sampling의 본질적 문제 발견

문제의 핵심은 이거였음:

> “샘플링이라는 **이산적 결정**이 학습 그래프를 끊어버린다”

- argmax → gradient 0
- sampling decision → non-differentiable
- 그래서:
    
    - 이론적으로 불안정
    - 실험 결과도 일관적이지 않음

### 그래서 나온 연구 흐름

여기서 두 갈래가 갈라집니다.


## 갈래 A: Scheduled Sampling을 **버리자**

이 쪽이 **주류**가 됨 ✅

- Transformer LM:
    
    - Teacher Forcing + MLE
    - 대신:
        
        - 대규모 데이터
        - better architectures
        - better decoding strategies
            
- RL-based methods:
    - REINFORCE
    - SCST (Self-Critical)

👉 GPT, BERT, T5 등 **주력 모델은 이 경로**

---

## 3. 갈래 B: Scheduled Sampling을 “미분 가능하게 고쳐보자”

👉 **여기서 Gumbel trick이 등장**
