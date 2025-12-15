### 1. 기본 개념: 마르코프 체인 (Markov Chain)

Forward Process는 이전 상태 $x_{t-1}$에 의해서만 현재 상태 $x_t$가 결정되는 **마르코프 체인**입니다.

- **$x_0$**: 원본 이미지 (데이터)
    
- **$x_T$**: 완전한 가우시안 노이즈 (Standard Gaussian Noise)
    
- **$T$**: 총 스텝 수 (보통 1000회 등)
    

---

### 2. 단일 스텝 전이 (Single Step Transition)

시점 $t-1$에서 $t$로 갈 때, 데이터에 노이즈를 조금 섞습니다. 이를 조건부 확률분포로 표현하면 다음과 같습니다.

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$$

이를 행렬/벡터 연산 식으로 풀어서 쓰면 **Reparameterization Trick**을 사용하여 다음과 같이 표현됩니다.

$$x_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \boldsymbol{\epsilon}_{t-1}$$

**[행렬/벡터 요소 분석]**

- $\mathbf{x}_{t-1}, \mathbf{x}_t$: $D$차원 벡터 (이미지를 1차원으로 펼친 것, $D = H \times W \times C$).
    
- $\beta_t$: $0$과 $1$ 사이의 매우 작은 스칼라 값 (Variance Schedule). 노이즈의 양을 조절합니다.
    
- $\boldsymbol{\epsilon}_{t-1}$: **노이즈 벡터**로, $\mathcal{N}(\mathbf{0}, \mathbf{I})$를 따릅니다.
    
- $\mathbf{I}$: $D \times D$ 크기의 **단위 행렬(Identity Matrix)**입니다. 이는 픽셀 간의 노이즈가 서로 독립적(independent)임을 의미합니다.
    

---

### 3. 임의 시점 전이 (Anytime Transition) - 핵심

매번 $t$번의 루프를 돌려 노이즈를 더하는 것은 비효율적입니다. 다행히 가우시안 분포의 성질 덕분에, **$x_0$에서 바로 $x_t$를 계산하는 닫힌 해(Closed-form)**를 행렬 식으로 유도할 수 있습니다.
[[Anytime Transition]] <--- 여기에서 증명!

- $\alpha_t = 1 - \beta_t$
    
- $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ (누적 곱)
    

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

- 가장 중요한 공식:

$$x_t = \underbrace{\sqrt{\bar{\alpha}_t} \mathbf{x}_0}_{\text{Signal}} + \underbrace{\sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}}_{\text{Noise}}$$

**[수식의 의미]**

1. **Signal Term ($\sqrt{\bar{\alpha}_t} \mathbf{x}_0$):** 시간이 지날수록($t$가 커질수록) $\bar{\alpha}_t$는 0에 수렴합니다. 즉, 원본 데이터의 정보량(Mean)이 점차 0으로 줄어듭니다.
    
2. **Noise Term ($\sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$):** 반대로 노이즈의 계수는 1에 가까워집니다. 즉, 공분산(Covariance)이 커지며 데이터가 노이즈에 덮이게 됩니다.
    
3. **행렬 관점:**
    
    - 평균 벡터: $\boldsymbol{\mu} = \sqrt{\bar{\alpha}_t} \mathbf{x}_0$
        
    - 공분산 행렬: $\boldsymbol{\Sigma} = (1 - \bar{\alpha}_t) \mathbf{I}$
        

---

### 4. 요약: 행렬 공간에서의 변화

이 과정을 고차원 벡터 공간에서 시각화하면 다음과 같습니다.

1. **초기 상태 ($t=0$):** 데이터 분포가 복잡한 매니폴드(Manifold) 위에 존재합니다.
    
2. **진행 과정:** 각 스텝마다 데이터 포인트(벡터)들이 원점 방향으로 조금씩 당겨지고(Scale Down), 구형(Spherical)으로 퍼지는 불확실성(Noise)이 더해집니다.
    
3. **최종 상태 ($t=T$):** $\bar{\alpha}_T \approx 0$이 되어, 데이터는 평균이 $\mathbf{0}$이고 공분산이 $\mathbf{I}$인 **Isotropic Gaussian Distribution(구형 가우시안)** 형태가 됩니다.
    

> **결론:** Forward Diffusion은 **원본 이미지 벡터($x_0$)를 스칼라로 축소**하고, 그 빈자리를 **단위 행렬 공분산을 가진 노이즈 벡터($\epsilon$)로 채워나가는 선형 변환** 과정입니다.