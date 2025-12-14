##  대전제: 가정(Manifold hypothesis):

- 현실 세계의 데이터(이미지, 음성 등)는
    
    - 거대한 고차원 공간 전체에 퍼져 있지 않고,
        
    - **어딘가의 저차원 manifold 위에 놓여 있다.**



## Loss 계산
![[Pasted image 20251213215457.png]]

![[Pasted image 20251213220152.png]]



$$\log P(x) = \underbrace{\mathbb{E}_{Q}[\log P(x|z)]}_{\text{1번 항}} - \underbrace{D_{KL}(Q(z|x) \| P(z))}_{\text{2번 항}} + \underbrace{D_{KL}(Q(z|x) \| P(z|x))}_{\text{3번 항}}$$

질문자님의 해석을 하나씩 짚어보겠습니다.

---

### **1번 항: Reconstruction Term 계산 (복원 오차)**

$$\mathcal{L}_{recon} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$$

이 항은 적분($\int$)이 포함되어 있어 직접 계산이 불가능합니다. 
따라서 **Monte Carlo Approximation**을 사용하여 계산합니다.

#### **단계 1: 몬테카를로 근사 (Monte Carlo Approximation)**

기댓값($\mathbb{E}$)을 샘플링 평균으로 대체합니다. 보통 $L=1$ (샘플 1개)을 사용합니다.

$$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] \approx \log p_\theta(x|z^{(l)})$$

여기서 $z^{(l)} = \mu + \sigma \odot \epsilon^{(l)}$ (Reparameterization Trick) 입니다.

#### **단계 2: 분포 가정에 따른 손실 함수 도출**

Decoder의 출력 $p_\theta(x|z)$가 어떤 분포를 따르냐에 따라 계산식이 달라집니다.

Case A: 데이터가 실수(Real-valued)인 경우 (예: 일반 이미지)

$p_\theta(x|z)$를 **정규분포(Gaussian)**로 가정합니다. (평균은 Decoder 출력 $\hat{x}$, 분산은 상수 1로 가정)

$$p_\theta(x|z) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{\|x - \hat{x}\|^2}{2} \right)$$

양변에 로그($\log$)를 취하면:

$$\log p_\theta(x|z) = -\frac{1}{2} \|x - \hat{x}\|^2 - \underbrace{\log\sqrt{2\pi}}_{\text{상수}}$$

상수를 무시하면, 우리가 흔히 쓰는 **MSE (Mean Squared Error)**가 됩니다.

$$\therefore \mathcal{L}_{recon} \approx - \text{MSE}(x, \hat{x})$$

Case B: 데이터가 0과 1 사이인 경우 (예: 흑백 MNIST)

$p_\theta(x|z)$를 **베르누이 분포(Bernoulli)**로 가정합니다.

$$\log p_\theta(x|z) = \sum_{i} \left( x_i \log \hat{x}_i + (1-x_i) \log (1-\hat{x}_i) \right)$$

이는 **BCE (Binary Cross Entropy)**의 음수 값과 같습니다.

$$\therefore \mathcal{L}_{recon} \approx - \text{BCE}(x, \hat{x})$$

---

### **2번 항: KL Divergence 계산 (정규화)**

$$\mathcal{L}_{KL} = D_{KL}(q_\phi(z|x) \| p(z))$$

두 분포를 모두 다변량 정규분포(Multivariate Gaussian)로 **가정** 하는 것이 국룰.
이 가정 하에 적분 없이 닫힌 해(Closed-form)로 계산 가능합니다.

- $q_\phi(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ (Encoder 출력)
    
- $p(z) = \mathcal{N}(0, I)$ (표준 정규분포)
    

계산의 편의를 위해 **1차원(스칼라) $z$에 대해 먼저 유도**하고, 나중에 차원 $J$만큼 더하겠습니다 (독립변수 가정).

#### **단계 1: 정의 대입**

$$D_{KL}(q \| p) = \int q(z) (\log q(z) - \log p(z)) \, dz = \mathbb{E}_{q(z)} [\log q(z) - \log p(z)]$$

#### **단계 2: 로그 확률밀도함수(Log-PDF) 전개**

정규분포 PDF: $f(z) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(z-\mu)^2}{2\sigma^2}}$

1. $\log q(z)$ 전개:
    
    $$\log q(z) = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma^2) - \frac{(z-\mu)^2}{2\sigma^2}$$
    
2. $\log p(z)$ 전개 ($\mu=0, \sigma=1$ 대입):
    
    $$\log p(z) = -\frac{1}{2}\log(2\pi) - \frac{z^2}{2}$$
    

#### **단계 3: 두 식의 차이 ($\log q - \log p$)**

$$\log q(z) - \log p(z) = -\frac{1}{2}\log(\sigma^2) - \frac{(z-\mu)^2}{2\sigma^2} + \frac{z^2}{2}$$

#### **단계 4: 기댓값($\mathbb{E}_q$) 취하기**

이제 위 식 전체에 $q(z)$에 대한 기댓값을 취합니다.

- $\mathbb{E}_q[(z-\mu)^2]$는 분산의 정의이므로 **$\sigma^2$** 입니다.
    
- $\mathbb{E}_q[z^2]$는 $\text{Var}(z) + (\mathbb{E}[z])^2$ 이므로 **$\sigma^2 + \mu^2$** 입니다.
    

대입하면:

$$\begin{aligned} \mathbb{E}[\dots] &= -\frac{1}{2}\log(\sigma^2) - \frac{\mathbb{E}[(z-\mu)^2]}{2\sigma^2} + \frac{\mathbb{E}[z^2]}{2} \\ &= -\frac{1}{2}\log(\sigma^2) - \frac{\sigma^2}{2\sigma^2} + \frac{\sigma^2 + \mu^2}{2} \\ &= -\frac{1}{2}\log(\sigma^2) - \frac{1}{2} + \frac{\sigma^2 + \mu^2}{2} \end{aligned}$$

#### **단계 5: 식 정리 (최종 공식)**

이를 $-\frac{1}{2}$로 묶어서 정리하고, 모든 차원 $J$에 대해 합(Sum)을 구합니다.

$$D_{KL} = -\frac{1}{2} \sum_{j=1}^{J} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

---

### **최종 요약: 코드로 구현되는 수식**

VAE의 최종 Loss Function $\mathcal{L} = -\text{ELBO} = -\text{(1번)} + \text{(2번)}$ 은 다음과 같습니다.

$$\text{Loss} = \text{BCE}(x, \hat{x}) + \beta \cdot \left[ -\frac{1}{2} \sum_{j=1}^{J} (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2) \right]$$






## 수학적 의의

#### **A. "점"이 아니라 "영역"으로 매핑한다**

- **일반 AE:** 입력 이미지를 잠재 공간의 좌표 $(2.5, -1.3)$ 같은 **점**으로 보냅니다.
    
- **VAE:** 입력 이미지를 $(2.5, -1.3)$을 중심으로 하고 반지름(분산)이 있는 **구름(분포)**으로 보냅니다.
    
    - Encoder는 "이 이미지는 대략 이 근처 쯤에 있어"라고 **불확실성(Uncertainty)**을 포함해서 알려주는 것입니다.
        

#### **B. 두 가지 힘의 줄다리기 (Trade-off)**

Loss Function의 두 항은 서로 상충되는 목적을 가집니다.

1. **Reconstruction Loss (복원):** "데이터를 완벽하게 복구하고 싶어!"
    
    - Encoder에게 분산($\sigma$)을 0으로 만들라고 압박합니다. 분산이 없어야(점으로 찍어야) Decoder가 복원하기 쉬우니까요. (AE처럼 되려고 함)
        
2. **KL Divergence (정규화):** "잠재 공간을 예쁘게 정리하고 싶어!"
    
    - 모든 잠재 변수의 분포가 표준정규분포 $\mathcal{N}(0, I)$를 따르도록 강제합니다.
        
    - 데이터들을 원점 주변으로 모으고(평균 $\to$ 0), 적당히 퍼뜨려서(분산 $\to$ 1) 빈 공간을 없앱니다.
        

#### **C. 결과: 매끄러운 잠재 공간 (Smooth Latent Manifold)**

이 줄다리기 덕분에 VAE의 잠재 공간은 다음과 같은 특징을 가집니다.

- **밀집성 (Dense):** KL 항 때문에 데이터들이 원점 근처에 옹기종기 모입니다.
    
- **연속성 (Continuous):** 비슷한 이미지들은 서로 겹치는 분포를 가집니다. 따라서 $z_1$(고양이)과 $z_2$(개) 사이의 중간 지점을 찍으면, "개냥이" 같은 자연스러운 중간 이미지가 생성됩니다.
    

결론적으로 VAE는:

데이터를 억지로 외우는 게 아니라(Overfitting), 데이터가 생성되는 **근본적인 확률 규칙(Manifold)**을 학습하여, 존재하지 않았던 새로운 데이터를 '그럴싸하게' 상상해 낼 수 있는 능력을 갖추게 됩니다.