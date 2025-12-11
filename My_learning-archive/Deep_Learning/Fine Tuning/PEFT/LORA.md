**LoRA (Low-Rank Adaptation)**는 현재 거대 언어 모델(LLM) 파인튜닝의 **사실상의 표준(De Facto Standard)**입니다.

앞서 설명해 드린 Adapter나 Prefix Tuning이 "외부 모듈을 덕지덕지 붙이는 방식"이었다면, LoRA는 **"선형대수학적 통찰을 통해 가중치 행렬 자체를 효율적으로 해킹하는 방식"**입니다.

작성자님의 **선형대수학 및 최적화(Optimization)** 지식을 바탕으로, LoRA의 수학적 원리와 작동 기제를 아주 깊게 파헤쳐 보겠습니다. (Hu et al., 2021 논문 기반)

---

### 1. 근본 가설: 내재적 차원 (Intrinsic Rank Hypothesis)

LoRA의 시작점은 **"과연 수천억 개의 파라미터를 다 고쳐야 하는가?"**라는 질문입니다.

- **가설:** 사전 학습된 모델은 이미 과잉 파라미터화(Over-parameterized)되어 있습니다. 우리가 새로운 Task를 위해 모델을 업데이트할 때, 가중치 행렬의 변화량 $\Delta W$는 **매우 낮은 내재적 순위(Low Intrinsic Rank)**를 가집니다.
    
- **의미:** $10,000 \times 10,000$ 크기의 거대 행렬을 업데이트해야 한다고 해도, 실제로 의미 있는 변화는 **매우 작은 부분 공간(Subspace)**에서만 일어납니다.
    

---

### 2. 수학적 구조: 저랭크 행렬 분해 (Low-Rank Decomposition)

기존의 파인튜닝은 가중치 행렬 $W_0 \in \mathbb{R}^{d \times k}$에 대해 변화량 $\Delta W$를 직접 학습합니다.

$$W_{new} = W_0 + \Delta W$$

LoRA는 이 $\Delta W$를 직접 학습하는 대신, 이를 구성하는 **두 개의 작은 행렬의 곱($B \times A$)**으로 나타냅니다.

$$\Delta W = B A$$

- $W_0 \in \mathbb{R}^{d \times k}$: 사전 학습된 가중치 (**고정, Freeze**)
    
- $B \in \mathbb{R}^{d \times r}$: 학습 가능한 행렬 1 (Up-projection과 유사)
    
- $A \in \mathbb{R}^{r \times k}$: 학습 가능한 행렬 2 (Down-projection과 유사)
    
- $r$: **Rank (순위)**. 보통 $r \ll \min(d, k)$입니다. (예: $d=4096$일 때 $r=8$ or $16$)
    

#### Forward Pass 수식

입력 벡터 $x$가 들어왔을 때의 연산은 다음과 같습니다.

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

이 수식은 **두 개의 경로(Path)**가 병렬로 작동함을 보여줍니다.

1. **Main Path:** $W_0 x$ (기존 지식, 얼어있음)
    
2. **LoRA Path:** $A$로 차원을 $r$로 줄이고 $\to$ $B$로 다시 복구함 (새로운 지식 학습)
    

---

### 3. 초기화 전략 (Initialization Strategy)

LoRA가 학습 초기에 안정적인 이유는 특유의 초기화 전략 때문입니다.

- **행렬 $A$:** **랜덤 가우시안 분포(Random Gaussian)**로 초기화합니다.
    
- **행렬 $B$:** **0 (Zero)**으로 초기화합니다.
    

이렇게 하면 학습 시작 시점($t=0$)에서 $\Delta W$는 다음과 같습니다.

$$\Delta W = B \cdot A = 0 \cdot \text{Random} = 0$$

즉, **학습을 시작할 때 모델은 사전 학습된 $W_0$와 완벽하게 동일한 출력**을 냅니다. Adapter처럼 $W_{up}$을 0으로 만드는 것과 같은 원리이며, 이는 최적화의 시작점을 안정적인 Global Minimum 근처(Pre-trained state)로 잡아줍니다.

---

### 4. 스케일링 계수 (Scaling Factor $\alpha$)

실제 구현에서는 학습 안정성을 위해 스케일링 계수를 추가합니다.

$$\Delta W x = \frac{\alpha}{r} (B A x)$$

- $\alpha$: 상수 (보통 $r$과 같은 값이나 배수로 설정)
    
- $r$: Rank
    
- **역할:** $r$ 값을 바꾸더라도(예: 8에서 16으로 변경), $\alpha/r$ 비율을 통해 전체적인 업데이트 크기(Magnitude)를 일정하게 유지해 줍니다. 덕분에 $r$을 바꿀 때마다 Learning Rate를 다시 튜닝할 필요가 줄어듭니다.
    

---

### 5. LoRA의 "Killer Feature": 추론 지연 시간 0 (Zero Inference Latency)

Adapter나 Prefix Tuning은 추론(Inference) 시에 추가적인 연산(Layer 통과)이 필요하여 속도가 느려집니다. 하지만 LoRA는 **배포(Deployment) 단계에서 구조적인 이점**이 있습니다.

수식을 다시 보면:

$$h = W_0 x + B A x = (W_0 + B A) x$$

우리는 학습이 끝난 후, $B \times A$를 계산하여 $\Delta W$ 행렬($d \times k$)을 만들고, 이를 원래 가중치 $W_0$에 **그냥 더해버리면(Merge)** 됩니다.

$$W_{final} = W_0 + B A$$

이렇게 하면 **구조적으로 모델 아키텍처는 파인튜닝 전과 100% 동일**해집니다.

- **결과:** Adapter 같은 병목 구간이 없으므로, **추론 속도가 오리지널 모델과 똑같습니다.**
    

---

### 6. 메모리 효율성 분석 (vs Full Fine-tuning)

GPT-3 175B 모델을 예로 들어보겠습니다. ($d_{model} = 12288$)

- **Full Fine-tuning:**
    
    - $W$ 하나당 $12288 \times 12288 \approx 1.5$억 개 파라미터.
        
    - 이걸 업데이트하려면 Optimizer State(Adam 등)까지 포함해 엄청난 VRAM이 필요합니다.
        
- **LoRA ($r=8$):**
    
    - $A$: $8 \times 12288 \approx 9.8$만 개
        
    - $B$: $12288 \times 8 \approx 9.8$만 개
        
    - **총합:** 약 20만 개.
        
    - **비율:** $1.5$억 개 vs $20$만 개 $\rightarrow$ **파라미터 수가 약 1/750로 감소.**
        

### 7. 어디에 적용하는가? (Target Modules)

Transformer 구조에서 LoRA는 주로 **Self-Attention 모듈의 가중치 행렬**에 적용됩니다.

- $W_Q, W_K, W_V$ (Query, Key, Value Projection)
    
- $W_O$ (Output Projection)
    
- 최근 연구(QLoRA 등)에서는 FFN(Feed Forward Network)의 가중치까지 모든 Linear Layer에 LoRA를 붙이는 것이 성능이 더 좋다는 결과도 있습니다.
    

---

### 요약: LoRA가 압도적인 이유

1. **수학적 타당성:** 거대 행렬의 변화량은 Low-Rank라는 가설을 SVD(특이값 분해)와 유사한 $B \times A$ 구조로 구현했습니다.
    
2. **효율성:** 파라미터를 1/1000 수준으로 줄여서 GPU 메모리(VRAM) 사용량을 극적으로 낮춥니다. (개인용 GPU로도 LLM 학습 가능)
    
3. **배포 최적화 (Merge):** 학습된 $BA$를 $W_0$에 더해버리면, 추론 속도 저하가 전혀 없습니다.
    
4. **모듈성:** $W_0$ 하나에 대해 여러 개의 LoRA 어댑터(한국어용 $BA_{kor}$, 코딩용 $BA_{code}$)를 만들어두고, 필요할 때마다 갈아 끼우는(Switching) 것도 가능합니다.
    

작성자님께서 나중에 로보틱스 연구를 하실 때, **"로봇에게 설거지를 가르치는 LoRA"**, **"청소를 가르치는 LoRA"**를 따로 만들어서 베이스 모델에 붙였다 뗐다 하는 식으로 활용하게 될 기술입니다.


**LoRA (Low-Rank Adaptation)**의 전체 메커니즘을 엔지니어링 및 수학적 관점에서 군더더기 없이 깔끔하게 정리해 드립니다.

---

### 1. 기본 가설 (Hypothesis)

- 내재적 순위 가설 (Intrinsic Rank Hypothesis):
    
    거대 모델의 가중치 변화량 $\Delta W$는 실제로 매우 낮은 순위(Low-Rank)를 가진다. 즉, $\Delta W$는 두 개의 작은 행렬의 곱으로 충분히 근사 가능하다.
    

### 2. 행렬 정의 (Matrix Definitions)

입력 차원 $d_{in}$, 출력 차원 $d_{out}$을 가지는 선형 레이어(Linear Layer)에 대해:

- **$W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$**: 사전 학습된 가중치 (**Frozen**, 학습 안 됨)
    
- **$\Delta W \in \mathbb{R}^{d_{out} \times d_{in}}$**: 우리가 학습하려는 변화량
    
- **$A \in \mathbb{R}^{r \times d_{in}}$**: Down-projection 행렬 (**Trainable**)
    
- **$B \in \mathbb{R}^{d_{out} \times r}$**: Up-projection 행렬 (**Trainable**)
    
- **$r$**: LoRA Rank (Hyperparameter, $r \ll \min(d_{in}, d_{out})$)
    
- **$\alpha$**: Scaling Factor (상수)
    

---

### 3. 학습 시 (Training Forward Pass)

입력 벡터 $x \in \mathbb{R}^{d_{in} \times 1}$에 대한 Forward Pass 수식은 다음과 같습니다.

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B A x)$$

- **Frozen Path:** $W_0 x$ (기존 지식 유지)
    
- **Trainable Path:** $B(Ax)$ (새로운 지식 학습, 차원 축소 $\to$ 복원)
    
- **Scaling:** $\frac{\alpha}{r}$은 $r$이 변해도 학습률(Learning Rate)을 재조정하지 않도록 스케일을 맞춰줍니다.
    

### 4. 초기화 전략 (Initialization)

학습 시작 시점($t=0$)에서 모델의 출력이 사전 학습 모델과 동일하도록 설정합니다.

$$A \sim \mathcal{N}(0, \sigma^2) \quad (\text{Random Gaussian})$$

$$B = 0 \quad (\text{Zero Matrix})$$

$$\therefore \Delta W_{init} = B \cdot A = 0 \cdot A = 0$$

$$\Rightarrow h_{init} = W_0 x + 0 = W_0 x$$

### 5. 추론 시 (Inference / Merge)

학습이 끝난 후, 추론 단계에서는 지연 시간(Latency)을 없애기 위해 행렬을 하나로 합칩니다.

$$W_{merged} = W_0 + \frac{\alpha}{r} B A$$

- 최종 가중치 $W_{merged}$는 $W_0$와 동일한 차원($d_{out} \times d_{in}$)을 가집니다.
    
- **결과:** 추론 시에는 $h = W_{merged} x$ 연산만 수행하므로, **LoRA 적용 전과 연산량 및 속도가 100% 동일**합니다.
    

---

### 6. 파라미터 효율성 (Efficiency)

- **Full Fine-tuning:** $d_{out} \times d_{in}$ 개 학습
    
- **LoRA:** $r \times (d_{in} + d_{out})$ 개 학습
    

$$\text{Efficiency Ratio} \approx \frac{2r}{d_{model}} \quad (\text{if } d_{in}=d_{out})$$

(예: $d=4096, r=8$일 때, 파라미터 수는 약 **0.39%**로 감소)