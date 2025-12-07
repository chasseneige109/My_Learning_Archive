
## 1. 핵심 질문 (Key Misconception)

> [!QUESTION] 의문점
> 
> "Dropout 확률 $p$를 적용해서 평균 내는 것이라면, 그냥 처음부터 가중치 $W$에 $p$를 곱해서(Scaling) 학습하는 거랑 똑같은 거 아닌가?"

**답변:** **학습(Training) 때는 완전히 다르고, 추론(Inference) 때는 수학적으로 같다.**

---

## 2. 학습 (Training) 단계: "Randomness & Ensemble"

단순 스케일링과 가장 큰 차이는 **확률적 노이즈(Stochasticity)**의 유무.

- **매커니즘:** 매 미니배치마다 실제로 베르누이 시행(동전 던지기)으로 뉴런을 랜덤하게 끈다.
    
- **효과 1 (Ensemble):** 뉴런이 $N$개일 때, 이론적으로 **$2^N$개의 서로 다른 서브 네트워크(Sub-network)**를 번갈아 가며 학습시키는 효과.
    
- **효과 2 (Co-adaptation 방지):** 특정 뉴런이 다른 뉴런에 의존하지 않고, 독립적인 특징(Feature)을 학습하도록 강제함.
    
- **비교:** 만약 단순히 가중치만 $p$배로 줄여서 학습한다면, 앙상블 효과 없이 **그냥 가중치가 작은 하나의 모델**일 뿐임.
    

## 3. 추론 (Test/Inference) 단계: "Approximation"

학습된 $2^N$개의 모델을 다 실행해서 평균 낼 수 없으므로, **가중치 스케일링(Weight Scaling)**으로 근사함.

- **매커니즘:** 모든 뉴런을 켜되, 출력값이 튀지 않도록 가중치를 조절.
    
- Weight Scaling Rule:
    
    $$E[y_{train}] \approx p \cdot W_{train} \cdot x$$
    
    따라서 테스트 때는 $W_{test} \leftarrow p \cdot W_{train}$ 으로 가중치를 줄여서 사용.
    
- **의미:** 학습 때의 '확률적 평균'을 '결정론적(Deterministic) 연산'으로 변환. 이때는 질문한 대로 **가중치 스케일링과 수학적으로 동일**해짐.
    

---

## 4. 실무 구현: Inverted Dropout (역 드롭아웃)

테스트 때마다 곱하기 연산을 하는 번거로움을 없애기 위해, **학습 때 미리 스케일링**을 수행함.

- 학습 시: 살아남은 뉴런의 출력에 $\frac{1}{p}$을 곱해서 값을 키워줌 (Scale-up).
    
    $$y = \frac{1}{p} \times Mask \times x$$
    
- **추론 시:** 아무 연산 없이 그대로 사용 (Identity).
    
- **결과:** 현대 딥러닝 프레임워크(PyTorch, TensorFlow)의 표준 구현 방식.