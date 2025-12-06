
### 1. 📐 핵심 계산 수식

입력 행렬 $X \in \mathbb{R}^{N \times D}$에서, Layer Norm은 '각 샘플 $i$'에 대해 독립적으로 통계량을 계산합니다.

#### 1) 평균 $\mu_i$ 및 분산 $\sigma_i^2$ (샘플 $i$에 대해)

$$\mu_i = \frac{1}{D} \sum_{j=1}^{D} x_{i j}$$

$$\sigma_i^2 = \frac{1}{D} \sum_{j=1}^{D} (x_{i j} - \mu_i)^2$$

#### 2) 정규화된 출력 $\hat{x}_{i j}$

$$\hat{x}_{i j} = \frac{x_{i j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}$$

---

### 2. 🎚️ 어파인 변환 (Affine Transformation)

정규화된 $\hat{X}$에 **피처별로 학습 가능한** 스케일 $\boldsymbol{\gamma}$와 이동 $\boldsymbol{\beta}$를 적용합니다. ($\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^{D}$)

$$y_{i j} = \gamma_j \hat{x}_{i j} + \beta_j$$

- $\boldsymbol{\gamma}$와 $\boldsymbol{\beta}$는 **피처 차원**에 대해서만 학습됩니다.
    

---

### 3. ⚖️ Batch Norm과의 비교

| **구분**       | **Layer Normalization (Layer Norm)** | **Batch Normalization (Batch Norm)** |
| ------------ | ------------------------------------ | ------------------------------------ |
| **통계량 계산 축** | **피처 차원 ($D$)** $\rightarrow$ (샘플 내) | **배치 차원 ($N$)** $\downarrow$ (피처 간)  |
| **의존성**      | **배치 크기에 독립적**                       | 배치 크기에 의존적 (배치가 작으면 성능 저하)           |
| **주요 사용처**   | **RNN, Transformer** (가변 길이 시퀀스)     | CNN, Fully Connected Layer           |