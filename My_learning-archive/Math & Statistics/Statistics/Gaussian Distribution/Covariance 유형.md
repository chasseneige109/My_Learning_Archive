**다변량 정규분포(Multivariate Normal Distribution)**의 일반적인 확률 밀도 함수(PDF) 식은 다음과 같습니다. 데이터의 차원을 $D$라고 할 때:

$$\mathcal{N}(x; \mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^D |\Sigma|}} \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)$$

여기서 $\Sigma$의 형태에 따라 이 복잡한 식이 어떻게 **단순화**되는지 보여드리겠습니다.

---

### 1. 구형 공분산 (Spherical Covariance)

> $\Sigma = \sigma^2 \mathbf{I}$ (단일 값 $\sigma$만 사용)

행렬식($|\Sigma|$)과 역행렬($\Sigma^{-1}$)이 아주 간단해지므로, 식이 **유클리드 거리(L2 Norm)** 형태로 바뀝니다.

$$\mathcal{N}(x; \mu, \sigma^2 \mathbf{I}) = \frac{1}{(2\pi \sigma^2)^{D/2}} \exp \left( -\frac{1}{2\sigma^2} \|x - \mu\|^2 \right)$$

- **해석:** 데이터 $x$와 평균 $\mu$ 사이의 **직선 거리($\|x - \mu\|^2$)**만 계산하면 됩니다. 방향(각도)은 무시하고 거리에 따라서만 확률이 줄어듭니다.
    

---

### 2. 대각 공분산 (Diagonal Covariance)

> $\Sigma = \text{diag}(\sigma_1^2, \dots, \sigma_D^2)$ (각 차원마다 분산이 다름)

행렬 연산이 사라지고, 단순히 **1차원 정규분포들의 곱(Product)**으로 표현됩니다.

$$\mathcal{N}(x; \mu, \Sigma_{\text{diag}}) = \prod_{d=1}^{D} \frac{1}{\sqrt{2\pi \sigma_d^2}} \exp \left( -\frac{(x_d - \mu_d)^2}{2\sigma_d^2} \right)$$

- **해석:** $x_1$은 $\sigma_1$으로 나누고, $x_2$는 $\sigma_2$로 나누어 계산한 뒤 싹 다 곱합니다. 차원별로 "중요도(퍼짐 정도)"를 다르게 취급합니다.
    

---

### 3. 완전 공분산 (Full Covariance)

> $\Sigma$는 일반적인 대칭 행렬 (모든 원소가 살아있음)

이 경우는 **식의 단순화가 불가능**합니다. 원래의 행렬 연산 식을 그대로 써야 합니다.

$$\mathcal{N}(x; \mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^D \det(\Sigma)}} \exp \left( -\frac{1}{2} \sum_{i=1}^D \sum_{j=1}^D (x_i - \mu_i) (\Sigma^{-1})_{ij} (x_j - \mu_j) \right)$$

- **해석:** 지수(exp) 안에 **이중 합(Double Summation)**이 생깁니다. $x_i$와 $x_j$가 곱해지는 항이 존재하며, 이것이 바로 **"두 변수 간의 상호작용(상관관계)"**을 계산하는 부분입니다.
    
- **비용:** $\Sigma^{-1}$ (역행렬)을 구하는 데 연산량이 $O(D^3)$만큼 듭니다. 차원이 1000만 넘어도 계산이 거의 불가능해집니다.
    

---

### 요약: 계산 복잡도 비교

|**형태**|**식의 핵심 부분**|**연산 비용 (Log-Likelihood 계산 시)**|
|---|---|---|
|**Spherical**|상수 $\times$ 거리의 제곱|$O(D)$ (매우 빠름)|
|**Diagonal**|각 차원별 거리의 합|$O(D)$ (빠름, VAE 표준)|
|**Full**|행렬 $\times$ 벡터 연산|$O(D^2)$ ~ $O(D^3)$ (너무 느림)|

이러한 이유로 딥러닝(VAE, Diffusion)에서는 **Spherical**이나 **Diagonal** 형태만 주로 사용하는 것입니다.