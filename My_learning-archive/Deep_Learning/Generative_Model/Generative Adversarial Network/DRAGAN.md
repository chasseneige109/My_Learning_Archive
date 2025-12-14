## DRAGAN (Discriminative Regularization for Adversarial Networks)
---

### 사전 요약
$$L_{Total} = \underbrace{L_{Original\_GAN}}_{\text{원래 하던 일}} + \underbrace{\lambda \cdot R_{DRAGAN}}_{\text{추가된 족쇄}}$$

1. **원래 하던 일 ($L_{Original\_GAN}$):**
    
    - 진짜($x_{real}$)는 1, 가짜($G(z)$)는 0으로 분류해라.
        ---> 진짜 'point' 에서 기울기 무한, 가짜는 0
    - (Standard GAN Loss 사용)
        
2. **추가된 족쇄 ($R_{DRAGAN}$):**
    
    - **"단, 진짜 데이터($x_{real}$)에 노이즈를 섞은 위치($\hat{x}$)에서는 기울기($\nabla$)가 0(또는 $k$)이 되어야 한다."**
        --> 1 , 2가 합쳐지면 스무딩됨
    - 이것이 바로 질문하신 **"진짜 데이터만 n개 샘플링하여 구한 정규화항"**입니다.




---
### 1. DRAGAN의 핵심 목표와 배경

DRAGAN이 해결하려는 핵심 문제는 다음과 같습니다:

1. **Sharp Peak (날카로운 봉우리) 문제:** 이상적인 Discriminator $D^*$는 실제 데이터 $p_{data}$가 있는 곳에서만 $D(x)=1$에 가깝고, 그 외의 모든 곳에서는 빠르게 $0$으로 떨어지는 날카로운 함수 형태를 갖게 됩니다.
    
2. **Gradient Vanishing:** 이러한 날카로운 $D$ 함수는 $p_{data}$ 근처를 벗어난 영역($p_g$가 있는 곳)에서 $\nabla_x D(x) \approx 0$ (기울기가 0)이 되어, Generator가 $p_{data}$로 이동할 방향 정보를 얻지 못하고 학습이 멈춥니다.
    
3. **해결책:** $p_{data}$ 근처의 영역에서 $D$의 Gradient Norm을 강제로 작게 만들어 $D$의 함수 형태를 부드럽게 (Smooth) 만들면, $p_g$ 샘플들이 $p_{data}$로 이동할 때도 의미 있는 기울기 신호를 받을 수 있습니다.
    

### 2. DRAGAN의 목적 함수

DRAGAN은 기존 GAN의 목적 함수 $\min_G \max_D V(G, D)$에 **Discriminative Regularization (DR)** 항을 추가하여 $D$를 훈련합니다.

#### DRAGAN Discriminator의 목적 함수 (최대화):

$D$는 다음의 목적 함수 $L_D$를 최대화합니다.

$$L_D(\theta_D) = V(G, D) - \lambda \cdot \mathbb{E}_{\tilde{x} \sim \mathcal{P}(\tilde{x})} \left[ \|\nabla_{\tilde{x}} D(\tilde{x})\| - k \right]^2$$

여기서 각 항의 의미는 다음과 같습니다.

1. $V(G, D)$: 기존 GAN의 Value Function.
    
    $$V(G, D) = \mathbb{E}_{x \sim p_{data}} [\log D(x)] + \mathbb{E}_{z \sim p_{z}} [\log (1 - D(G(z)))]$$
    
2. $\lambda$: Regularization Strength (페널티 강도)를 조절하는 하이퍼파라미터.
    
3. $k$: Target Gradient Norm (목표 기울기 크기). DRAGAN 논문에서는 $k=0$을 사용했습니다.
    
4. $\mathbb{E}_{\tilde{x} \sim \mathcal{P}(\tilde{x})} [\dots]$: 페널티를 적용할 **샘플링 분포**에 대한 기댓값.
    

#### 핵심: 샘플링 분포 $\mathcal{P}(\tilde{x})$

DRAGAN의 가장 중요한 특징은 Gradient Penalty를 적용하는 샘플 $\tilde{x}$가 어디에서 오는지입니다.

$$\tilde{x} = x + \delta$$

- $x$: 실제 데이터 분포 $p_{data}$에서 샘플링된 **진짜 이미지**.
    
- $\delta$: **랜덤 노이즈** ($\delta$는 $0$을 중심으로 하는 균일 분포 $U(-\epsilon, \epsilon)$에서 샘플링됨).
    

즉, $\tilde{x}$는 **$p_{data}$의 샘플 주변 영역**에서만 추출됩니다.

### 3. Gradient Penalty 항의 효과 분석

DRAGAN의 정규화 항을 다시 보면, ($k=0$ 가정):

$$R_{DR} = \lambda \cdot \mathbb{E}_{\tilde{x} \sim \mathcal{P}(\tilde{x})} \left[ \|\nabla_{\tilde{x}} D(\tilde{x})\|^2 \right]$$

$D$가 $L_D$를 **최대화**하려고 할 때, 이 $R_{DR}$ 항은 **감소**해야 합니다 (목적 함수에 마이너스 부호로 붙어있기 때문입니다).

$$\max_{D} L_D \iff \min_{D} \left( -V(G, D) + R_{DR} \right)$$

$\min_D R_{DR}$을 달성하기 위해, $D$는 **실제 데이터 $p_{data}$ 주변 영역 $\mathcal{P}(\tilde{x})$**에서 자신의 **기울기 크기 $\|\nabla_{\tilde{x}} D(\tilde{x})\|$ 를 0으로 만들려고 노력**합니다.

#### 결과: Real Data 주변의 Smoother Landscape

$D$의 기울기가 0에 가깝다는 것은 **함수의 값이 그 주변에서 거의 변하지 않는다**는 의미입니다.

- $p_{data}$ 주변의 $D$ 함수 형태가 **뾰족한 Peak** 대신 **완만한 언덕(Smooth Hill)**처럼 변합니다.
    
- 이 완만한 지형 덕분에, $G$가 생성한 샘플 $G(z)$가 $p_{data}$ 영역으로 조금만 가까워져도 **0이 아닌 의미 있는 기울기** $\nabla_x D(x)$를 받게 되어 학습이 멈추지 않고 지속적으로 $p_{data}$를 향해 이동할 수 있게 됩니다.
    

DRAGAN은 WGAN-GP(Wasserstein GAN with Gradient Penalty)와 유사하지만, WGAN-GP가 $p_{data}$와 $p_g$를 잇는 직선 위에서 페널티를 주는 반면, DRAGAN은 **오직 $p_{data}$ 주변 영역**에서만 페널티를 주어 $D$의 지역적(Local) 부드러움을 강제한다는 차이가 있습니다.