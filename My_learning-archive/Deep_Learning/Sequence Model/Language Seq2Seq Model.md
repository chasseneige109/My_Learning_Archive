
## one-hot 기반 embedding 단점

1. vocabulary를 **미리 고정**해야 한다
    
2. 새 단어 추가가 불가능함 (차원을 늘려야 하므로)
    

이건 one-hot의 근본적 한계.

그래서 embedding 자체가 잘 작동해도  
one-hot이라는 기반 representation은 여전히 불편함.

(이건 나중에 subword tokenization으로 해결된다.)


네, **언어 모델(Language Model)**, 특히 **LSTM 기반의 다음 단어 예측(Next Word Prediction) 모델**이 어떻게 학습되고 추론하는지, **행렬 연산(Matrix Operation) 수준**에서 A부터 Z까지 해부해 드리겠습니다.

우리의 목표는 **"I love AI"**라는 문장을 학습하고, 다시 생성해내는 것입니다.

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
        (x_t)
    - 은닉 처리용: $(H \times 4H) = (256 \times 1024)$
        (h_t-1)
    - 편향 $\mathbf{b}_{lstm}$: $(1 \times 4H) = (1 \times 1024)$
    --> 4개의 게이트 각각이 각자의 W_x와 W_h를 가짐.
    --> 그걸 그냥 한 번에 옆으로 256 x 4개 나열해서 계산
1. **출력 투영 행렬 $\mathbf{W}_{proj}$:** $(H \times V) = (256 \times 10000)$
    
2. **출력 편향 $\mathbf{b}_{proj}$:** $(1 \times V) = (1 \times 10000)$

### Phase 1: 학습 단계 (Training) - Forward Pass 🏃

- **입력 데이터**: `['<SOS>', 'I', 'love', 'AI']`
    
- **정답 데이터**: `['I', 'love', 'AI', '<EOS>']` (한 칸씩 밀린 시퀀스)
    

**상황 설정**: 시간 $t$에서 입력 단어가 **'love'** (인덱스 450)인 경우를 살펴봅니다.

#### Step 1. 임베딩 조회 (Embedding Lookup)

입력 단어의 인덱스(450)를 사용해 임베딩 행렬 $\mathbf{E}$에서 해당 행을 가져옵니다 (Slicing).

$$\mathbf{x}_t = \mathbf{E}[450]$$

- **크기**: $(1 \times 128)$

#### Step 2. LSTM 연산 (The Core)

입력 $\mathbf{x}_t$와 이전 상태 $\mathbf{h}_{t-1}$ ($1 \times 256$)이 들어옵니다.

1. 선형 결합 (Linear Combination):

4개의 게이트($f, i, g, o$)를 위해 한 번에 계산합니다.

$$\mathbf{z} = \mathbf{x}_t \cdot \mathbf{W}_{x} + \mathbf{h}_{t-1} \cdot \mathbf{W}_{h} + \mathbf{b}_{lstm}$$

- **행렬 크기 확인**: $(1 \times 128) \times (128 \times 1024) + (1 \times 256) \times (256 \times 1024)$
    
- **결과**: $\mathbf{z}$는 $(1 \times 1024)$ 크기의 벡터입니다.
    

2. 게이트 분할 및 활성화 (Split & Activate):

$\mathbf{z}$를 256개씩 4조각으로 쪼갭니다.

- $\mathbf{f}_t = \sigma(\mathbf{z}_{0:256})$ (망각 게이트)
    
- $\mathbf{i}_t = \sigma(\mathbf{z}_{256:512})$ (입력 게이트)
    
- $\tilde{\mathbf{C}}_t = \tanh(\mathbf{z}_{512:768})$ (새로운 정보 후보)
    
- $\mathbf{o}_t = \sigma(\mathbf{z}_{768:1024})$ (출력 게이트)
    

3. 상태 업데이트 (Cell Update):

원소별 곱셈($\odot$)과 덧셈을 수행합니다.
(PyTorch LSTM은 cuDNN 커널을 씀. GPU 병렬화 지림)
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$

- **결과 크기**: 둘 다 $(1 \times 256)$
    

#### Step 3. 출력 투영 (Projection to Logits)

256차원의 은닉 상태를 다시 10,000개의 단어 확률 공간으로 확장합니다.

$$\mathbf{logits}_t = \mathbf{h}_t \cdot \mathbf{W}_{proj} + \mathbf{b}_{proj}$$

- **크기**: $(1 \times 10000)$
    
- 이 값은 아직 확률이 아닌 점수(Score)입니다.
    

#### Step 4. 확률 변환 (Softmax)

점수를 0~1 사이의 확률로 변환합니다.

$$\mathbf{P}_t = \text{Softmax}(\mathbf{logits}_t) = \frac{e^{\mathbf{logits}_t}}{\sum e^{\mathbf{logits}_t}}$$

- **크기**: $(1 \times 10000)$
    

#### Step 5. 손실 계산 (Loss Calculation)

실제 정답은 **'AI'** (인덱스 900)입니다. 예측 확률 벡터 $\mathbf{P}_t$에서 900번째 값만 확인합니다.

$$\text{Loss}_t = -\log(P_{t, 900})$$

---

### Phase 2: 학습 단계 (Training) - Backward Pass 🔙

Loss를 줄이기 위해 파라미터들을 수정합니다.

1. Gradient 시작점:

출력층의 미분값은 매우 직관적입니다.

$$\frac{\partial \text{Loss}}{\partial \mathbf{logits}_t} = \mathbf{P}_t - \mathbf{y}_t$$

- $\mathbf{y}_t$: 정답 인덱스(900번)만 1이고 나머지는 0인 원-핫 벡터.
    
- 의미: **(내 예측 확률) - (정답이면 1, 아니면 0)**
    

**2. 역전파 진행 (BPTT):**

- $\mathbf{W}_{proj}$ 수정: 오차 신호를 타고 내려와 투영 행렬의 기울기를 구합니다.
    
- **LSTM 내부**: 오차가 $\mathbf{h}_t$를 타고 $\mathbf{C}_t$와 게이트들을 거쳐 $\mathbf{W}_{lstm}$까지 도달합니다.
    
- $\mathbf{E}$ 수정: 마지막으로 임베딩 행렬 $\mathbf{E}$의 **450번째 행(love)** 부분만 미세하게 수정됩니다.
    

**3. 파라미터 업데이트 (Optimizer):**

$$\mathbf{W} \leftarrow \mathbf{W} - \eta \cdot \text{Gradient}$$

- ($\eta$: 학습률)
    

---

### Phase 3: 추론 단계 (Inference) - Generation 🔮

학습 완료 후, 모델에게 "I"만 주고 문장을 생성하게 합니다. (**Autoregression**)

**Time 0: 초기화**

- $\mathbf{h}_{init}, \mathbf{C}_{init}$은 0 벡터 혹은 인코더의 마지막 상태.
    
- 첫 입력: `<SOS>`
    

**Time 1: 첫 생성**

1. **입력**: `<SOS>` $\rightarrow$ 임베딩 $\mathbf{x}_1$
    
2. **LSTM**: $\mathbf{h}_0, \mathbf{C}_0$와 $\mathbf{x}_1$ 연산 $\rightarrow$ $\mathbf{h}_1$ 생성
    
3. **출력**: $\mathbf{h}_1 \cdot \mathbf{W}_{proj} \rightarrow$ Softmax $\rightarrow$ $\mathbf{P}_1$
    
4. **결정**: $\mathbf{P}_1$에서 확률이 가장 높은 단어 **"I"** 선택.
    

**Time 2: 피드백 (Autoregression) ⭐**

- 방금 모델이 뱉은 **"I"**를 다시 입력으로 씁니다. (학습 때와 다름)
    

1. **입력**: **"I"**의 인덱스 $\rightarrow$ 임베딩 $\mathbf{x}_2$
    
2. **LSTM**: $\mathbf{h}_1, \mathbf{C}_1$ (Time 1의 기억)와 $\mathbf{x}_2$ 연산 $\rightarrow$ $\mathbf{h}_2$ 생성
    
3. **출력**: $\mathbf{h}_2$로 다음 단어 예측 $\rightarrow$ **"love"** 선택.
    

**Time 3: 반복**

1. **입력**: **"love"**를 입력으로 사용.
    
2. **출력**: **"AI"** 예측.
    

**Time 4: 종료**

1. **입력**: **"AI"** 입력.
    
2. **출력**: **`<EOS>`**가 높은 확률로 나옴.
    
3. **판단**: 문장 생성 종료.
    

---

### 요약: 행렬의 여행 ✈️

1. **입력 (정수)** $\xrightarrow{\text{Slicing}}$ **임베딩 (128차원)**
    
2. **임베딩 + 과거기억** $\xrightarrow{\text{Matrix Mul}}$ **게이트 계산 (1024차원)**
    
3. **게이트** $\xrightarrow{\text{Split & Activation}}$ **셀/은닉 상태 (256차원)**
    
4. **은닉 상태** $\xrightarrow{\text{Matrix Mul}}$ **로짓 (10000차원)**
    
5. **로짓** $\xrightarrow{\text{Softmax}}$ **확률**
    
6. **(학습 시)** 정답과 비교해 **Backprop**.
    
7. **(추론 시)** 가장 높은 거 뽑아서 **다음 단계 입력으로 재사용 (Loop)**.