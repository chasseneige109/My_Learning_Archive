### 1. 🧙‍♂️ Primal $\rightarrow$ Dual Form 

WGAN이론의 시작은 [[Wasserstein Distance]]**의 원래 정의(Primal Form)에서 비롯됩니다.
수학적 유도는 여기를 참고! [[Kantorovich-Rubinstein Duality]]

$$W(P, Q) = \inf_{\gamma \in \Pi} \mathbb{E}_{(x, y) \sim \gamma} [\|x - y\|]$$

- **문제:** 이 식은 최적의 운송 계획 $\gamma$를 무한히 많은 경우의 수 중에서 찾아야 하므로 신경망으로 풀 수 없습니다.
    
- **해결:** **칸토로비치-루빈슈타인 쌍대성(Kantorovich-Rubinstein Duality)**이라는 수학적 정리를 이용해 문제를 뒤집습니다. (Dual Form)
    

$$W(P, Q) = \sup_{\|f\|_L \le 1} \left( \mathbb{E}_{x \sim P}[f(x)] - \mathbb{E}_{x \sim Q}[f(x)] \right)$$

- 이제 복잡한 **'운송 계획 $\gamma$'** 대신, 단순한 **'함수 $f$'** 하나만 찾으면 됩니다. 이 함수 $f$가 WGAN의 **Critic** 네트워크입니다.
    

### 2. 🚦 1-Lipschitz 제약: Critic의 "속도 제한"

Dual Form 수식에서 $W(P, Q)$가 되려면, $\sup$ (최댓값)을 찾는 함수 $f$가 **반드시** 1-Lipschitz 조건을 만족해야 합니다.

- **정의:** 입력 $x_1, x_2$에 대해 $\frac{|f(x_1) - f(x_2)|}{\|x_1 - x_2\|} \le 1$
    
    - 즉, **함수 $f$의 기울기(Gradient)가 어떤 지점에서도 1을 넘을 수 없습니다.**
        
- **필요성:**
    
    - 이 제약은 $f$가 점수 차이를 **무한대로 부풀리는 것을 막아**줍니다.
        
    - $f$의 출력값 차이가 입력 공간의 거리($\|x_1 - x_2\|$)와 비례하도록 묶어주어, **$f$가 두 분포 사이의 거리를 측정하는 '정확한 자(Ruler)' 역할을 하도록 강제**합니다.
        

### 3. 📈 Critic 학습 (= W 구하기)
[[WGAN_GP]] / [[Spectral Normalization (SN)]] <- 대표적, 혁신적인 방법들

#### 1. $f(x)$의 정체: "전체 네트워크 함수"

수학적으로 $f(x; \theta)$는 파라미터 $\theta$를 가진 딥러닝 모델 전체를 나타내는 함수 표기입니다.

- **입력 ($x$):** 고차원 이미지 (예: $64 \times 64 \times 3$ 픽셀)
    
- **중간층 (Hidden Layers):** 특징을 추출하는 여러 개의 Convolution Layer + LeakyReLU 등.
    
- **출력 ($y$):** 스칼라 값 하나 (실수 $s \in \mathbb{R}$)
    

즉, $f$는 **이미지 $\rightarrow$ 실수(점수)** 로 매핑하는 함수입니다.

---

#### 2. 가장 결정적인 차이: "마지막 레이어 (The Last Layer)"

기존 GAN의 Discriminator와 WGAN의 Critic $f(x)$는 99% 똑같이 생겼지만, **맨 마지막 레이어(Output Layer)**에서 결정적인 차이가 있습니다.

##### (1) 기존 GAN (Discriminator)

- **마지막 레이어:** Fully Connected Layer $\rightarrow$ **Sigmoid Activation**
    
- **수식:** $D(x) = \sigma(W \cdot h + b)$
    
- **출력 범위:** $[0, 1]$ (확률)
    
- **의미:** "이 이미지가 진짜일 확률은 70%다."
    

##### (2) WGAN (Critic $f$)

- **마지막 레이어:** Fully Connected Layer $\rightarrow$ **(아무것도 없음)**
    
- **수식:** $f(x) = W \cdot h + b$ (Linear Activation)
    
- **출력 범위:** $(-\infty, \infty)$ (모든 실수)
    
- **의미:** "이 이미지는 10.5점짜리 리얼함(Realness)을 가졌다."
    

> **핵심:** $f(x)$의 마지막은 활성화 함수가 없는 **선형(Linear) 레이어**입니다. 그래서 $f$의 출력값은 제한 없이 커지거나 작아질 수 있습니다.

---

#### 3. 내부 레이어의 제약 (Batch Norm 금지)

WGAN-GP(Gradient Penalty)를 사용할 때 $f(x)$ 내부 레이어 구성에도 중요한 제약이 생깁니다.

- **Batch Normalization 사용 불가:**
    
    - Batch Norm은 배치 내의 다른 샘플들의 통계량(평균, 분산)을 이용해 내 데이터를 정규화합니다.
        
    - 이렇게 되면 $f(x)$가 $x$ 하나에만 의존하는 게 아니라, **배치 내의 다른 데이터들에도 의존**하게 됩니다.
        
    - WGAN-GP는 **"각 데이터 포인트($\hat{x}$)마다 독립적으로"** Gradient Penalty를 계산해야 하는데, Batch Norm이 데이터들을 섞어버리면 이 수학적 가정이 깨집니다.
        
- **대안:** 대신 **Layer Normalization**이나 **Instance Normalization**을 사용합니다. (이들은 샘플끼리 섞이지 않으니까요.)


### 📈 