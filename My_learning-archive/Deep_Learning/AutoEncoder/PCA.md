**PCA (Principal Component Analysis, 주성분 분석)**는 고차원 데이터의 **정보(분산, Variance)를 가장 잘 보존하는 새로운 '직교 좌표계(Orthogonal Basis)'를 찾는 선형 변환** 기법입니다.

---
### 1. 데이터 세팅 및 전처리 (Preprocessing)

가장 먼저 데이터를 행렬로 정의하고, **중심화(Centering)**를 수행해야 합니다.

- **데이터 행렬:** $X \in \mathbb{R}^{N \times D}$
    
    - $N$: 샘플 개수
        
    - $D$: 피처(차원) 개수
        
- 중심화 (Mean Centering):
    
    각 차원의 평균을 0으로 만듭니다. PCA는 '분산'을 다루는데, 원점이 0이어야 공분산 계산이 $X^T X$ 꼴로 깔끔하게 떨어지기 때문입니다.
    
    $$X \leftarrow X - \mu$$
    

---

### 2. 핵심 도구: 공분산 행렬 (Covariance Matrix)

데이터의 퍼짐 정도와 변수 간의 상관관계를 담고 있는 정사각 행렬 $\Sigma$를 정의합니다.

$$\Sigma = \frac{1}{N-1} X^T X \in \mathbb{R}^{D \times D}$$

- **차원 분석:** $(D \times N) \times (N \times D) \rightarrow D \times D$
    
- **성질:**
    
    1. **대칭 행렬 (Symmetric):** $(X^T X)^T = X^T X$ 이므로 대칭입니다.
        
    2. **준양의 부호 (Positive Semi-Definite):** 모든 벡터 $v$에 대해 $v^T \Sigma v \ge 0$ 입니다.
        
- **의미:**
    
    - 대각 성분($\Sigma_{ii}$): $i$번째 차원의 **분산** (정보량).
        
    - 비대각 성분($\Sigma_{ij}$): $i$번째와 $j$번째 차원의 **공분산** (상관관계).
        

---

### 3. 목표 설정: 분산 최대화 (Optimization Problem)

우리는 데이터를 어떤 방향 벡터 $w$ ($\|w\|=1$)로 투영(Projection)하고 싶습니다.

이때, 투영된 데이터들이 겹치지 않고 최대한 넓게 퍼져 있어야 정보 손실이 적습니다.

- **투영된 데이터:** $y = X w$ ($\in \mathbb{R}^{N \times 1}$)
    
- 투영된 데이터의 분산:
    
    $$\text{Var}(y) = \frac{1}{N-1} y^T y = \frac{1}{N-1} (X w)^T (X w) = w^T \left( \frac{1}{N-1} X^T X \right) w$$
    
    $$\therefore \text{Var}(y) = w^T \Sigma w$$
    

[최적화 문제 정의]

$$\text{Maximize } \quad w^T \Sigma w \quad \text{subject to } \quad w^T w = 1$$

---

### 4. 문제 해결: 라그랑주 승수법 (Lagrange Multiplier)

제약 조건이 있는 최적화 문제를 풀기 위해 라그랑주 함수 $\mathcal{L}$을 도입합니다. ($\lambda$는 라그랑주 승수)

$$\mathcal{L}(w, \lambda) = w^T \Sigma w - \lambda (w^T w - 1)$$

$w$에 대해 편미분하여 0이 되는 지점을 찾습니다.

$$\frac{\partial \mathcal{L}}{\partial w} = 2 \Sigma w - 2 \lambda w = 0$$

$$\Sigma w = \lambda w$$

[유레카!]

이 식은 선형대수학의 고유값 문제(Eigenvalue Problem) 정의 그 자체입니다!

- **결론:**
    
    1. 우리가 찾는 최적의 투영 축 $w$는 공분산 행렬 $\Sigma$의 **고유벡터(Eigenvector)**입니다.
        
    2. 그때의 최대 분산 값 $w^T \Sigma w = w^T (\lambda w) = \lambda (w^T w) = \lambda$는 **고유값(Eigenvalue)**입니다.
        

즉, **"데이터의 분산이 가장 큰 축은 공분산 행렬의 가장 큰 고유값을 가진 고유벡터이다."**

---

### 5. 알고리즘: 고유값 분해 (Eigendecomposition)

$\Sigma$는 대칭 행렬이므로, **스펙트럼 정리(Spectral Theorem)**에 의해 항상 직교 행렬로 대각화 가능합니다.

$$\Sigma = Q \Lambda Q^T$$

- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_D)$: 고유값 행렬 ($\lambda_1 \ge \lambda_2 \ge \dots$)
    
- $Q = [q_1, q_2, \dots, q_D]$: 고유벡터 행렬 (서로 **직교(Orthogonal)**함)
    

**PCA 수행 과정:**

1. 고유값 $\lambda$를 크기순으로 정렬합니다.
    
2. 상위 $k$개의 고유값에 해당하는 고유벡터 $q_1, \dots, q_k$를 선택합니다.
    
3. 이들을 열 벡터로 하는 변환 행렬 $W_{pca} \in \mathbb{R}^{D \times k}$를 만듭니다.
    

---

### 6. 차원 축소 및 복원 (Linear Autoencoder와의 연결)

이제 $W_{pca}$를 이용해 데이터를 변환해 봅시다.

#### (1) 차원 축소 (Encoding)

데이터 $X$를 주성분 축으로 회전 및 투영합니다.

$$Z = X W_{pca} \in \mathbb{R}^{N \times k}$$

- 이것이 바로 **Principal Components (주성분 점수)**입니다.
    
- 데이터 간의 상관관계가 제거되고(Decorrelation), 정보가 압축되었습니다.
    

#### (2) 차원 복원 (Decoding)

압축된 $Z$를 다시 원래 차원으로 돌립니다. $W_{pca}$는 직교 행렬의 일부이므로 역행렬은 전치행렬입니다.

$$\hat{X} = Z W_{pca}^T \in \mathbb{R}^{N \times D}$$

이 $\hat{X}$는 원본 $X$를 $k$차원 부분 공간(Subspace)으로 투영한 결과이며, 이 과정에서의 손실 $\|X - \hat{X}\|_F^2$는 최소화됩니다.

---

### 7. 실전 테크닉: SVD (특이값 분해)

실제 컴퓨터(NumPy, PyTorch 등)에서는 $X^T X$를 계산하는 과정에서 데이터 정밀도 손실이 발생할 수 있어, 공분산 행렬을 만들지 않고 **데이터 행렬 $X$를 직접 SVD** 합니다.

$$X = U S V^T$$

여기서 $V$의 열 벡터들이 바로 $\Sigma$의 고유벡터(주성분 $W$)와 수학적으로 동일합니다.

- $\Sigma = X^T X = (V S U^T)(U S V^T) = V S^2 V^T$
    
- 즉, $X$의 특이값($S$)의 제곱이 공분산 행렬의 고유값($\Lambda$)이 됩니다. ($\lambda_i = s_i^2 / (N-1)$)
    

### 요약

**PCA는...**

1. 데이터의 **공분산 행렬($X^T X$)을 고유값 분해**하는 것이다.
    
2. 가장 큰 고유값에 대응하는 고유벡터(주성분)들은 데이터가 **가장 넓게 퍼져있는(분산이 큰) 축**이다.
    
3. 이 축들은 서로 **직교(Orthogonal)**한다.
    
4. 이 축으로 데이터를 투영하면, **정보 손실을 최소화**하면서 차원을 줄일 수 있다.