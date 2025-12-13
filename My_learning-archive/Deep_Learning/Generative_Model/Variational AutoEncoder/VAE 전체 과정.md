VAE의 Forward Pass 과정을 행렬 차원(Dimension)과 연산 위주로 간결하게 나열하겠습니다.

**기호 정의:**

- $B$: Batch Size (데이터 개수)
    
- $D$: Input Dimension (입력 차원, 예: 784)
    
- $H$: Hidden Dimension (은닉층 차원, 예: 400)
    
- $Z$: Latent Dimension (잠재 공간 차원, 예: 20)
    
- $W, b$: 가중치 행렬 및 편향 벡터
    
- $\odot$: Element-wise multiplication (아다마르 곱)
    

---

### **1. Encoder (입력 $\to$ 잠재 변수 파라미터)**

입력 데이터를 압축하여 분포의 평균($\mu$)과 분산($\log\sigma^2$)을 계산합니다.

1. Input:
    
    $$X \in \mathbb{R}^{B \times D}$$
    
2. Hidden Layer:
    
    $$H_{enc} = \text{ReLU}(X W_1 + b_1) \in \mathbb{R}^{B \times H}$$
    
    (여기서 $W_1 \in \mathbb{R}^{D \times H}$)
    
3. Mean ($\mu$) & Log-Variance ($\log\sigma^2$):
    
    $$\mu = H_{enc} W_\mu + b_\mu \in \mathbb{R}^{B \times Z}$$
    
    $$\log(\sigma^2) = H_{enc} W_\sigma + b_\sigma \in \mathbb{R}^{B \times Z}$$
    
    (여기서 $W_\mu, W_\sigma \in \mathbb{R}^{H \times Z}$)
    

---

### **2. Reparameterization (샘플링)**

미분 가능하도록 노이즈를 주입하여 잠재 벡터 $Z$를 생성합니다.

4. Standard Deviation:
    
    $$\sigma = \exp\left(\frac{1}{2} \log(\sigma^2)\right) \in \mathbb{R}^{B \times Z}$$
    
    (Element-wise 연산)
    
5. Noise Sampling:
    **몬테카를로 근사
    $$\epsilon \sim \mathcal{N}(0, I) \in \mathbb{R}^{B \times Z}$$
    
    (표준정규분포에서 랜덤 추출)
    
6. Latent Vector $Z$:
    
    $$Z = \mu + (\sigma \odot \epsilon) \in \mathbb{R}^{B \times Z}$$
    

---

### **3. Decoder (잠재 변수 $\to$ 복원 출력)**

샘플링된 $Z$를 다시 원래 차원으로 확장합니다.

7. Hidden Layer:
    
    $$H_{dec} = \text{ReLU}(Z W_3 + b_3) \in \mathbb{R}^{B \times H}$$
    
    (여기서 $W_3 \in \mathbb{R}^{Z \times H}$)
    
8. Output Reconstruction:
    
    $$\hat{X} = \text{Sigmoid}(H_{dec} W_4 + b_4) \in \mathbb{R}^{B \times D}$$
    
    (여기서 $W_4 \in \mathbb{R}^{H \times D}$)
    
    (데이터가 실수 범위라면 Sigmoid 대신 Identity나 데이터 전처리에 따라 tanh사용 가능)
    

---

### **4. Loss Calculation (스칼라 값 도출)**

행렬 연산 결과를 하나의 스칼라 값(Loss)으로 합칩니다.

9. Reconstruction Loss (BCE or MSE):
    
    $$\mathcal{L}_{recon} = \sum_{d=1}^{D} \text{BCE}(X_{:,d}, \hat{X}_{:,d}) \quad (\text{Scalar})$$
    
    (보통 Batch 평균, Dimension 합)
    
10. KL Divergence:
    
    $$\mathcal{L}_{KL} = -\frac{1}{2} \sum_{j=1}^{Z} (1 + \log(\sigma^2)_{:,j} - \mu_{:,j}^2 - \exp(\log(\sigma^2))_{:,j}) \quad (\text{Scalar})$$
    
11. Total Loss:
    
    $$\mathcal{L} = \mathcal{L}_{recon} + \mathcal{L}_{KL}$$