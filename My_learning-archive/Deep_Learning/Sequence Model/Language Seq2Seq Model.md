
## one-hot 기반 embedding 단점

1. vocabulary를 **미리 고정**해야 한다
    
2. 새 단어 추가가 불가능함 (차원을 늘려야 하므로)
    

이건 one-hot의 근본적 한계.
(이건 나중에 subword tokenization으로 해결된다.)



---

### 0. 셋팅 (The Setup) 🛠️

먼저 행렬들의 크기(Dimension)를 정의합시다.

- **$V$ (Vocabulary Size):** 10,000개 (단어장 크기)
    
- **$D$ (Embedding Size):** 128차원 (단어 벡터 크기)
    
- **$H$ (Hidden Size):** 256차원 (LSTM 내부 메모리 크기)
    

#### 1) 준비된 행렬들 (Parameters)

1. **임베딩 행렬 $\mathbf{E}$:** $(V \times D) = (10000 \times 128)$
    
2. **LSTM 가중치 $\mathbf{W}_{lstm}$:** 입력과 은닉 상태를 처리하는 4개의 게이트용 가중치 통합본.
    
    - 입력 처리용: $(D \times 4H) = (128 \times 1024)$
        
    - 은닉 처리용: $(H \times 4H) = (256 \times 1024)$
        
    - 편향 $\mathbf{b}_{lstm}$: $(1 \times 4H) = (1 \times 1024)$
        
3. **출력 투영 행렬 $\mathbf{W}_{proj}$:** $(H \times V) = (256 \times 10000)$
    
4. **출력 편향 $\mathbf{b}_{proj}$:** $(1 \times V) = (1 \times 10000)$
    

---

### Phase 1: 학습 단계 (Training) - Forward Pass 🏃

목표: "I love"를 보고 "AI"를 예측해라.

입력 데이터: ['<SOS>', 'I', 'love', 'AI'] (정수로 변환된 인덱스)

정답 데이터: ['I', 'love', 'AI', '<EOS>'] (한 칸씩 밀린 시퀀스)

우리는 시간 $t$에서의 동작 하나만 현미경으로 보겠습니다. (예: 입력이 'love'일 때)

#### Step 1. 임베딩 조회 (Embedding Lookup)

입력 단어 'love'의 인덱스가 450번이라고 합시다.

원래는 원-핫 벡터($1 \times 10000$)와 행렬 $\mathbf{E}$를 곱해야 하지만, 실제로는 그냥 **$\mathbf{E}$의 450번째 행을 가져옵니다 (Slicing).**

$$\mathbf{x}_t = \mathbf{E}[450] \quad (\text{크기}: 1 \times 128)$$

#### Step 2. LSTM 연산 (The Core)

입력 $\mathbf{x}_t$와 이전 상태 $\mathbf{h}_{t-1}$ ($1 \times 256$)이 들어옵니다.

1. 선형 결합 (Linear Combination):
    
    4개의 게이트(f, i, g, o)를 위해 한 번에 계산합니다.
    
    $$\mathbf{z} = \mathbf{x}_t \cdot \mathbf{W}_{x} + \mathbf{h}_{t-1} \cdot \mathbf{W}_{h} + \mathbf{b}$$
    
    - 행렬 크기 확인: $(1 \times 128) \times (128 \times 1024) + (1 \times 256) \times (256 \times 1024) = (1 \times 1024)$
        
    - 결과 $\mathbf{z}$는 1024차원 벡터입니다.
        
2. 게이트 분할 및 활성화 (Split & Activate):
    
    $\mathbf{z}$를 256개씩 4조각으로 쪼갭니다.
    
    - $\mathbf{f}_t = \sigma(\mathbf{z}_{0:256})$ (망각)
        
    - $\mathbf{i}_t = \sigma(\mathbf{z}_{256:512})$ (입력 스위치)
        
    - $\tilde{\mathbf{C}}_t = \tanh(\mathbf{z}_{512:768})$ (내용)
        
    - $\mathbf{o}_t = \sigma(\mathbf{z}_{768:1024})$ (출력)
        
3. **상태 업데이트 (Cell Update):** (원소별 곱셈 $\odot$과 덧셈)
    
    - $\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t \quad (\text{크기}: 1 \times 256)$
        
    - $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t) \quad (\text{크기}: 1 \times 256)$
        

#### Step 3. 출력 투영 (Projection to Logits)

이제 256차원의 은닉 상태 $\mathbf{h}_t$를 다시 10,000개의 단어 확률 공간으로 확장합니다.

$$\mathbf{logits}_t = \mathbf{h}_t \cdot \mathbf{W}_{proj} + \mathbf{b}_{proj}$$

- 크기: $(1 \times 256) \times (256 \times 10000) = (1 \times 10000)$
    
- 이 벡터는 각 단어에 대한 점수(Score)입니다. $-\infty$에서 $+\infty$ 사이의 값을 가집니다.
    

#### Step 4. 확률 변환 (Softmax)

점수를 확률로 바꿉니다.

$$\mathbf{P}_t = \text{Softmax}(\mathbf{logits}_t) = \frac{e^{\mathbf{logits}_t}}{\sum e^{\mathbf{logits}_t}}$$

- 크기: $(1 \times 10000)$
    
- 모든 원소의 합은 1입니다.
    

#### Step 5. 손실 계산 (Loss Calculation)

실제 정답은 'AI'이고, 인덱스는 900번이라고 합시다.

우리는 $\mathbf{P}_t$ 벡터의 900번째 값($P_{t, 900}$)만 봅니다.

$$\text{Loss}_t = -\log(P_{t, 900})$$

---

### Phase 2: 학습 단계 (Training) - Backward Pass 🔙

이제 계산된 Loss를 줄이기 위해 행렬들을 수정합니다.

1. Gradient 시작점:
    
    가장 끝단인 Logits의 미분값은 아주 깔끔하게 나옵니다.
    
    $$\frac{\partial \text{Loss}}{\partial \mathbf{logits}_t} = \mathbf{P}_t - \mathbf{y}_t$$
    
    - $\mathbf{y}_t$는 정답 인덱스만 1인 원-핫 벡터입니다.
        
    - 즉, `(예측확률) - (정답이면 1, 아니면 0)` 벡터가 오차 신호가 되어 역류합니다.
        
2. **역전파 진행 (BPTT):**
    
    - $\mathbf{W}_{proj}$ 수정: 오차 신호를 타고 내려와 투영 행렬의 기울기를 구합니다.
        
    - LSTM 내부: 오차가 $\mathbf{h}_t$를 타고 $\mathbf{C}_t$와 게이트들을 거쳐 $\mathbf{W}_{lstm}$까지 도달합니다.
        
    - $\mathbf{E}$ 수정: 마지막으로 임베딩 행렬 $\mathbf{E}$의 `450`번째 행(love)에 해당하는 부분도 미세하게 수정됩니다.
        
3. 파라미터 업데이트 (Optimizer):
    
    $$\mathbf{W} \leftarrow \mathbf{W} - \alpha \cdot \text{Gradient}$$
    

---

### Phase 3: 추론 단계 (Inference) - Generation 🔮

학습이 끝났습니다. 이제 모델에게 "I"라는 단어만 주고 문장을 이어 쓰게 해 봅시다. (Autoregression)

#### Time 0: 초기화

- $\mathbf{h}_{init}, \mathbf{C}_{init}$은 0 벡터로 시작하거나, 인코더가 있다면 인코더의 마지막 상태를 가져옵니다.
    
- 첫 입력: `<SOS>` (Start of Sentence) 토큰.
    

#### Time 1: 첫 생성

1. **입력:** `<SOS>` 인덱스 $\rightarrow$ 임베딩 $\mathbf{x}_1$.
    
2. **LSTM:** $\mathbf{h}_0, \mathbf{C}_0$와 $\mathbf{x}_1$ 연산 $\rightarrow$ $\mathbf{h}_1$ 생성.
    
3. **출력:** $\mathbf{h}_1 \cdot \mathbf{W}_{proj} \rightarrow$ Softmax $\rightarrow$ $\mathbf{P}_1$.
    
4. **단어 결정 (Decoding):**
    
    - $\mathbf{P}_1$에서 확률이 제일 높은 단어를 보니 **"I"**였습니다. (Argmax 또는 Sampling)
        
    - 출력 단어: **"I"**
        

#### Time 2: 피드백 (Autoregression) ⭐ 중요!

학습 때와 여기가 다릅니다. 학습 때는 정답 데이터('I')를 넣어줬지만(Teacher Forcing), 지금은 정답을 모릅니다.

방금 모델이 뱉은 "I"를 다시 입력으로 씁니다.

1. **입력:** 방금 뽑은 **"I"**의 인덱스 $\rightarrow$ 임베딩 $\mathbf{x}_2$.
    
2. **LSTM:** $\mathbf{h}_1, \mathbf{C}_1$ (Time 1의 기억)와 $\mathbf{x}_2$ 연산 $\rightarrow$ $\mathbf{h}_2$ 생성.
    
3. **출력:** $\mathbf{h}_2$로 다음 단어 예측 $\rightarrow$ **"love"**가 나옴.
    

#### Time 3: 반복

1. **입력:** 방금 뽑은 **"love"**를 입력으로 사용.
    
2. **LSTM:** 이전 기억과 연산.
    
3. **출력:** **"AI"** 예측.
    

#### Time 4: 종료

1. **입력:** **"AI"** 입력.
    
2. **출력:** **`<EOS>`** (End of Sentence) 토큰이 높은 확률로 나옴.
    
3. **판단:** `<EOS>`가 나왔으므로 생성을 멈춤.
    

최종 결과물: **"I love AI"**

---

### 요약: 행렬의 여행 ✈️

1. **입력 (정수)** $\xrightarrow{\text{Slicing}}$ **임베딩 (128차원)**
    
2. **임베딩 + 과거기억** $\xrightarrow{\text{Matrix Mul (4x)}}$ **게이트 계산 (1024차원)**
    
3. **게이트** $\xrightarrow{\text{Split & Tanh/Sigmoid}}$ **셀/은닉 상태 업데이트 (256차원)**
    
4. **은닉 상태** $\xrightarrow{\text{Matrix Mul}}$ **로짓 (10000차원)**
    
5. **로짓** $\xrightarrow{\text{Softmax}}$ **확률**
    
6. **(학습 시)** 정답과 비교해 Backprop.
    
7. **(추론 시)** 가장 높은 거 뽑아서 **다음 단계 입력으로 재사용 (Loop).**