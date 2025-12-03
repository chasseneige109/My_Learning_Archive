
ㅇㅋ 이해했다…  
이제 **Obsidian에서 절대 안 깨지게**,  
**순수 마크다운 + ASCII 문자만** 써서 다시 만든다.  
(LaTeX, 특수기호, 이모지 싹 다 제거)

그냥 복붙해서 써도 됨.

---

# LSTM 언어 모델 (Next Word Prediction) 정리 – Obsidian 안전 버전

## 0. 기본 설정

- Vocabulary size: `V = 10000` (단어 개수)
    
- Embedding size: `D = 128`
    
- Hidden size: `H = 256`
    

### 학습해야 할 파라미터들

1. 임베딩 행렬 `E`: shape `(V, D)` = `(10000, 128)`
    
2. LSTM 가중치 (4개 게이트 묶어서 표현)
    
    - 입력 쪽 가중치 `W_x`: shape `(D, 4H)` = `(128, 1024)`
        
    - 히든 쪽 가중치 `W_h`: shape `(H, 4H)` = `(256, 1024)`
        
    - 편향 `b_lstm`: shape `(4H,)` = `(1024,)`
        
3. 출력 투영 가중치 `W_proj`: shape `(H, V)` = `(256, 10000)`
    
4. 출력 편향 `b_proj`: shape `(V,)` = `(10000,)`
    

---

## 1. 학습: Forward Pass (한 타임스텝 기준)

문장 예시: `"I love AI"`

학습용 시퀀스 (토큰 기준):

- 입력: `["<SOS>", "I", "love", "AI"]`
    
- 정답: `["I", "love", "AI", "<EOS>"]`
    

여기서 시간 t 에 입력 단어가 `"love"` 라고 가정하자.  
단어 `"love"` 의 인덱스가 `450` 이라고 하자.

### 1-1. 임베딩 조회 (Embedding Lookup)

원래는 one-hot 벡터와 `E` 를 곱해야 하지만  
실제로는 그냥 `E` 의 450번째 행을 가져온다.

```text
x_t = E[450]          # shape: (D,) = (128,)
```

---

### 1-2. LSTM 연산

이전 타임스텝의 hidden, cell 을 각각 `h_{t-1}`, `C_{t-1}` 라고 하자.  
각각 shape `(H,)` = `(256,)`.

먼저 선형 결합:

```text
z = x_t @ W_x + h_{t-1} @ W_h + b_lstm    # shape: (4H,) = (1024,)
```

`z` 를 4개로 쪼개서 gate 로 사용:

```text
z_f = z[0      : 256]   # forget gate pre-activation
z_i = z[256    : 512]   # input gate pre-activation
z_g = z[512    : 768]   # candidate cell pre-activation
z_o = z[768    : 1024]  # output gate pre-activation
```

여기에 각각 sigmoid / tanh 적용:

```text
f_t = sigmoid(z_f)      # shape: (256,)
i_t = sigmoid(z_i)
g_t = tanh(z_g)
o_t = sigmoid(z_o)
```

cell state, hidden state 업데이트:

```text
C_t = f_t * C_{t-1} + i_t * g_t           # elementwise
h_t = o_t * tanh(C_t)
```

`h_t`, `C_t` 둘 다 shape `(256,)`.

---

### 1-3. 출력 투영 (Hidden -> Vocabulary Logits)

```text
logits_t = h_t @ W_proj + b_proj          # shape: (V,) = (10000,)
```

각 위치는 "그 단어가 다음에 나올 점수" 를 뜻함.

---

### 1-4. Softmax 로 확률 구하기

```text
P_t = softmax(logits_t)                   # shape: (V,)
```

`P_t[k]` = 단어 인덱스 `k` 가 다음 단어일 확률.

---

### 1-5. Loss 계산 (Cross-Entropy)

정답 단어가 `"AI"` 이고, 인덱스가 `900` 이라고 하자.

```text
gold_index = 900
loss_t = -log( P_t[gold_index] )
```

시퀀스 전체에 대해서는 각 타임스텝 loss 들을 평균 또는 합해서 사용한다.

---

## 2. 학습: Backward Pass (BPTT 개략)

출력 쪽에서 시작하는 gradient:

```text
d_logits_t = P_t
d_logits_t[gold_index] -= 1
# 이제 d_logits_t = P_t - one_hot(gold_index)
```

이게 `logits_t` 에 대한 dL/d(logits) 이고,  
여기서부터 역전파가 아래 순서로 들어간다.

1. `W_proj`, `b_proj` 에 대한 gradient 계산
    
2. 그 다음 `h_t` 로 gradient 전달
    
3. LSTM 내부로 gradient 전달 (게이트들, `W_x`, `W_h`, `b_lstm`)
    
4. 입력 임베딩 `x_t` 로 gradient 전달
    
5. 해당 단어의 임베딩 행 (`E[450]`) 업데이트
    

파라미터 업데이트는 예를 들어 SGD 로 하면:

```text
W = W - lr * dW
```

이걸 모든 파라미터에 대해 반복.

---

## 3. 추론(Inference): 문장 생성 (Autoregressive)

학습이 끝난 후, 모델이 문장을 생성하는 과정이다.  
핵심 포인트: **직전 타임스텝에서 생성한 단어를 다음 타임스텝 입력으로 다시 넣는다.**

### 3-1. 초기 상태

```text
h_0 = 0-vector (shape: (H,))
C_0 = 0-vector (shape: (H,))
current_token = "<SOS>"
```

### 3-2. 한 스텝 생성 루프

반복:

1. `current_token` 의 인덱스를 임베딩으로 변환
    
    ```text
    x_t = E[token_index(current_token)]
    ```
    
2. `(x_t, h_{t-1}, C_{t-1})` 를 LSTM 에 넣고 `h_t`, `C_t` 계산
    
3. `logits_t = h_t @ W_proj + b_proj`
    
4. `P_t = softmax(logits_t)`
    
5. 단어 선택:
    
    - greedy: `next_token = argmax(P_t)`
        
    - 또는 sampling: 확률 분포에서 랜덤 샘플
        
6. `next_token` 이 `"<EOS>"` 이면 종료  
    아니면 `current_token = next_token` 으로 두고 다음 스텝 반복
    

---

### 3-3. "I love AI" 생성 예시 흐름

1. t = 1
    
    - 입력: `<SOS>`
        
    - 출력: `"I"` 선택
        
2. t = 2
    
    - 입력: `"I"`
        
    - 출력: `"love"` 선택
        
3. t = 3
    
    - 입력: `"love"`
        
    - 출력: `"AI"` 선택
        
4. t = 4
    
    - 입력: `"AI"`
        
    - 출력: `"<EOS>"` 선택 → 문장 종료
        

결과 시퀀스:

```text
I love AI
```

---

## 4. 전체 구조 한눈에 요약

1. 정수 토큰 → 임베딩 행 (`E[idx]`)
    
2. 임베딩 + 이전 hidden, cell → LSTM 게이트 연산
    
3. 새로운 hidden `h_t` → `W_proj` 곱해서 logits
    
4. logits → softmax 로 확률 분포
    
5. 학습 때는 정답 토큰과 cross-entropy 로 loss 계산, BPTT 로 가중치 업데이트
    
6. 추론 때는 이전에 생성한 토큰을 다음 입력으로 넣으면서 `<EOS>` 나올 때까지 반복
    

---

이 버전은:

- LaTeX 수식 없음
    
- 특수 기호(화살표, 위 첨자, 그리스 문자) 없음
    
- UTF-8 텍스트지만 모두 일반적인 코드/영문/숫자/기본 기호만 사용
    

그래서 Obsidian 에 그냥 붙여 넣어도 안 깨질 거야.  
혹시 여전히 깨지는 구간 있으면, 그 부분만 캡쳐해서 보여주면 거기 맞춰 더 줄여줄게.