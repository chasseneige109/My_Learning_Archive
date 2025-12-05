## 학습

### 0. 차원 및 기호 정의

- $B$: 배치 크기 (예: 32)
    
- $L$: 입력 문장 길이 (예: 10)
    
- $V$: 단어 집합 크기 (예: 10,000)
    
- $D$: 임베딩 및 은닉 상태 차원 (예: 256)
    

---

### 1단계: 인코더 (Encoder) - 정보 압축

입력 문장을 숫자(인덱스)에서 의미 있는 벡터로 바꾸고, 문맥을 읽는 과정입니다.

1. **입력 (Indices):**
    
    - `Src` = $(B, L)$ (각 단어의 정수 인덱스)
	    (원래라면, B L V 차원이었겠지만, 굳이 차원을 만들지 않고 인덱스로만 받아 최적화함)
        
2. **임베딩 (Embedding):**
    
    - 가중치 $W_{emb}$: $(V, D)$
        
    - 연산: `LookUp(Src)`
        
    - 결과 $X$: **$(B, L, D)$**
	    내가 입력한 단어 1개가 그 단어의 '설명불가능한' 특성 256개로 분해
        
3. **RNN/LSTM 처리:**
    
    - 연산: $H, h_{final} = \text{RNN}(X)$
        
    - **$H$ (모든 은닉 상태):** **$(B, L, D)$** $\rightarrow$ **Key & Value 역할**
        
    - $h_{final}$ (마지막 은닉 상태): $(B, 1, D)$ $\rightarrow$ 디코더의 초기 $s_0$로 사용
        

---

### 2단계: 디코더 (Decoder) - 루프 시작

디코더는 **타임스텝 $t$를 1부터 끝까지 반복**하며 단어를 생성합니다. (Teacher Forcing 사용 가정)

**[Loop 시작: 시점 $t$]**

1. **입력 준비 (Embedding):**
    
    - 정답의 $t-1$번째 단어 인덱스 입력 ($y_{t-1}$)
        
    - 결과 $y_{emb}$: **$(B, 1, D)$**
        
2. **RNN 셀 업데이트:**
    
    - 입력: $y_{emb}$와 이전 상태 $s_{t-1}$
        
    - 연산: $s_t = \text{Cell}(y_{emb}, s_{t-1})$
        
    - 결과 $s_t$: **$(B, 1, D)$** $\rightarrow$ **Query 역할**
        

---

### 3단계: 어텐션 메커니즘 (Attention) - 핵심

"지금 $s_t$랑 가장 친한 $H$ 찾기"

1. **스코어 계산 (Alignment):**
    
    - 가중치 $W_a$: $(D, D)$
        
    - 연산: $E_t = (s_t \times W_a) \times H^T$
        
    - 차원: $(B, 1, D) \times (D, D) \times (B, D, L)$
        
    - 결과 $E_t$: **$(B, 1, L)$** (각 단어별 점수)
        
2. **확률 변환 (Softmax):**
    
    - 연산: $\alpha_t = \text{Softmax}(E_t)$
        
    - 결과 $\alpha_t$: **$(B, 1, L)$** (가중치 확률)
        
3. **문맥 벡터 생성 (Context Vector):**
    
    - 연산: $c_t = \alpha_t \times H$
        
    - 차원: $(B, 1, L) \times (B, L, D)$
        
    - 결과 $c_t$: **$(B, 1, D)$** (현재 시점에 필요한 정보 엑기스)
        

---

### 4단계: 정보 융합 (Concat & Projection)

문맥($c_t$)과 내 기억($s_t$)을 합쳐서 최종 출력을 준비합니다.

1. **결합 (Concatenate):**
    
    - 연산: $Combined = [c_t ; s_t]$
        
    - 차원: **$(B, 1, 2D)$** (D 두 개를 옆으로 붙임)
        
2. **어텐션 은닉 상태 ($\tilde{s}_t$) 생성:**
    
    - 가중치 $W_c$: $(2D, D)$
        
    - 연산: $\tilde{s}_t = \tanh(Combined \times W_c)$
        
    - 차원: $(B, 1, 2D) \times (2D, D) \rightarrow \mathbf{(B, 1, D)}$
        

---

### 5단계: 출력 및 손실 (Output & Loss)

단어를 예측하고 정답과 맞춰봅니다.

1. **단어 점수 예측 (Generator):**
    
    - 가중치 $W_{out}$: $(D, V)$
        
    - 연산: $Logits = \tilde{s}_t \times W_{out}$
        
    - 차원: $(B, 1, D) \times (D, V) \rightarrow \mathbf{(B, 1, V)}$
        
2. **손실 계산 (Loss):**
    
    - 입력: $Logits$와 실제 정답 단어 $Y_{true}$
        
    - 연산: $\text{CrossEntropy}(Logits, Y_{true})$
        
    - 결과: **스칼라 값 (Loss)**
        

**[Loop 종료]**

---

### 6단계: 최적화 (Optimization)

모든 타임스텝의 Loss를 평균 내거나 더한 뒤:

1. **역전파 (Backpropagation):** $\frac{\partial Loss}{\partial W}$ 계산 (기울기 구하기)
    
2. **가중치 갱신 (Gradient Descent):**
    
    - $W_{emb}, W_{RNN}, W_a, W_c, W_{out}$ 등 모든 행렬을 조금씩 수정.



# Teacher forcing의 문제

학습땐 계속 정답 넣어주다가, 추론땐 자기꺼 시키니까, 환경이 달라짐. 결과가 안좋음.

## **Scheduled Sampling**

- 처음엔 **Teacher Forcing 100%**
    
- 조금 학습되면 **정답 단어 90%, 모델 출력 10%**
    
- 더 학습되면 **정답 50%, 모델 출력 50%**
    
- 결국 **정답 없이 모델 스스로 생성하도록 학습됨**
    

이 비율이 바로:

> **Teacher Forcing Ratio**

비율을 점점 줄여서  
“아빠가 자전거에서 점점 손을 놓는 것처럼” 만드는 것.

### 문제점: 확률적 샘플링이므로, indifferentiable


###  그래서 “Gumbel Noise Trick” (Gumbel-Softmax)를 사용

> 확률 분포에서 샘플링한 것처럼 보이지만  
> 미분은 가능한 연속적인 방식으로 바꾸는 기법.

### 핵심 아이디어

- 샘플링 대신  
    **log probability + Gumbel noise**  
    를 이용해 “샘플처럼 행동하는 값”을 만든다.
    
- 그리고 softmax를 씌워서  
    “연속적이고 differentiable한 샘플”을 얻는다.
    

Transformer 이전의 많은 Seq2Seq 연구에서 사용됨.