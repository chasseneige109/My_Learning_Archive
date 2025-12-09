
### 1. 핵심 Objective: Next Token Prediction (NTP)

GPT의 모든 것은 **"다음에 올 단어 맞추기"**라는 단 하나의 목표로 귀결됩니다. 이를 수학적으로 **Log-Likelihood Maximization**이라고 합니다.

#### 수식적 의미

주어진 문장 $x = (x_1, x_2, \dots, x_T)$에 대하여, 모델은 다음 확률의 곱을 최대화하도록 학습합니다.

$$P(x) = \prod_{t=1}^{T} P(x_t | x_1, \dots, x_{t-1})$$

이를 로그(Log)를 취해 합(Sum)으로 바꾸면, 우리가 최대화해야 할 목적 함수(Objective Function)가 됩니다.

$$\mathcal{L}(\theta) = \sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$$

- **$x_{<t}$:** 현재 시점 $t$ 이전의 모든 토큰들 (Context)
    
- **$x_t$:** 현재 맞춰야 할 정답 토큰 (Target)
    
- **의미:** "지금까지의 문맥($x_{<t}$)을 다 봤을 때, 실제 정답($x_t$)이 나올 확률을 최대한 높여라."
    

---

### 2. 구조 (Structure): Decoder Stack & Causal Masking

GPT는 앞서 우리가 행렬 단위로 뜯어보았던 **Decoder Block**만 수십 개 쌓아 올린 구조입니다.

#### 핵심 메커니즘: Causal Self-Attention

- **Encoder와의 차이:** Encoder는 미래를 커닝할 수 있지만, GPT는 절대 미래를 보면 안 됩니다.
    
- **구현:** **Masked Self-Attention**을 사용합니다.
    
    - Attention Score 행렬의 상삼각(Upper Triangle) 부분을 $-\infty$로 채워, Softmax 후 **0**이 되게 만듭니다.
        
    - 이로 인해 $t$번째 토큰은 $1 \sim t$번째 토큰까지만 어텐션을 줄 수 있습니다.
        

#### 데이터 흐름

1. **입력:** `The robot`
    
2. **Decoder Stack:** 마스크된 어텐션을 통과하며 문맥을 파악합니다.
    
3. **출력:** `is` (확률이 가장 높은 다음 토큰)
    
4. **재입력 (Autoregressive):** `The robot is`를 다시 입력으로 넣습니다.
    

---

### 3. GPT-3의 발견: "Scale is All You Need"

GPT-1, GPT-2까지는 모델이 좋긴 했지만, 특정 태스크(번역, 요약 등)를 잘하려면 여전히 **파인 튜닝(Fine-tuning)**이 필요했습니다. 하지만 파라미터를 1,750억 개로 늘린 **GPT-3**에서 놀라운 현상이 발견됩니다.

#### In-context Learning (문맥 내 학습)

모델의 가중치(Weight)를 업데이트하지 않고도, **프롬프트(입력)에 예시를 몇 개 넣어주는 것만으로** 새로운 태스크를 수행합니다.

1. **Zero-shot:** 예시 없이 바로 물어봄.
    
    - 입력: `Translate to Korean: Cheese ->`
        
    - 출력: `치즈`
        
2. **One-shot:** 예시 1개 줌.
    
    - 입력: `Translate to Korean: Apple -> 사과 \n Cheese ->`
        
    - 출력: `치즈`
        
3. **Few-shot:** 예시 여러 개 줌. (성능 대폭 향상)
    

#### 왜 이게 가능한가? (Meta-learning)

모델이 방대한 데이터를 학습하면서 **"패턴을 파악하고 따르는 능력"** 자체를 학습했기 때문입니다. 입력된 예시들(Few-shot samples)이 가중치를 바꾸진 않지만, **Attention 메커니즘을 통해 '지금 내가 해야 할 작업이 번역이구나'라는 문맥(Context)을 형성**하게 합니다.

---

### 4. GPT-4: 멀티모달로의 확장 (Multimodality)

GPT-4는 텍스트뿐만 아니라 **이미지**도 입력으로 받습니다. 하지만 놀랍게도 **기본 학습 목표(Objective)는 변하지 않았습니다.**

#### 이미지도 토큰이다

1. **이미지 처리:** 이미지를 작은 패치(Patch)로 쪼갠 후, ViT(Vision Transformer) 같은 인코더를 통해 벡터로 변환합니다.
    
2. **통합:** 이 이미지 벡터들을 텍스트 임베딩 벡터와 같은 공간에 둡니다.
    
    - 입력 시퀀스: `[Image Token 1] ... [Image Token N] [Text Token: "Describe"] [Text Token: "this"]`
        
3. **Next Token Prediction:**
    
    - 모델 입장에서는 이미지 토큰도 그냥 **"앞에 있는 문맥(Context)"**일 뿐입니다.
        
    - "앞에 이런 이미지 정보가 있고, 'Describe this'라는 텍스트가 있으니, 다음에 올 텍스트 토큰은 'A cat...' 이겠구나"라고 예측합니다.
        

#### 즉, GPT-4의 본질

**"이미지를 보고 이해한다"**는 것은, **"이미지 정보를 조건부(Condition)로 받아서, 그에 맞는 텍스트 확률 분포를 생성한다"**는 것과 수학적으로 동일합니다.

---

### 요약

|**모델**|**특징**|**핵심 혁신**|
|---|---|---|
|**GPT-1/2**|Decoder-only|Next Token Prediction만으로도 언어 생성이 됨을 증명|
|**GPT-3**|**Huge Scale**|**Fine-tuning 없이** 예시만으로 학습하는 **In-context Learning** 발견|
|**GPT-4**|**Multimodal**|이미지를 텍스트와 동일한 **토큰 시퀀스**로 취급하여 통합 예측|