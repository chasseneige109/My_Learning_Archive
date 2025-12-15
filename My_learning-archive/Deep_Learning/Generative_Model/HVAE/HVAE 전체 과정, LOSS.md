가장 기본적인 **2계층 VAE ($x \leftrightarrow z_1 \leftrightarrow z_2$)**를 예로 들어, 데이터가 행렬 형태로 어떻게 변환되는지 단계별로 설명해 드리겠습니다.

설명의 편의를 위해 다음과 같이 설정을 잡겠습니다.

- **배치 크기 ($B$):** 64
    
- **입력 데이터 ($x$):** 784차원 (MNIST 같은 이미지)
    
- **하위 잠재변수 ($z_1$):** 64차원 (이미지의 디테일 담당)
    
- **상위 잠재변수 ($z_2$):** 32차원 (이미지의 전체적 구조 담당)
    

---

### 1. 학습 과정 (Training Loop)

학습은 **Encoder(Bottom-up)** 로 올라갔다가, **Decoder(Top-down)** 로 내려오면서 손실(Loss)을 계산하는 과정입니다.

#### Step 1: Encoder (Bottom-Up) - 추론

데이터 $x$를 보고 $z_1$을 뽑고, 그 $z_1$을 보고 $z_2$를 뽑습니다.

1. **입력:** $x \in \mathbb{R}^{64 \times 784}$
    
2. **Layer 1 ($x \rightarrow z_1$):**
    
    - 가중치 행렬 $W_{e1} \in \mathbb{R}^{784 \times 128}$ (Hidden)
        
    - $h_1 = \text{ReLU}(x \cdot W_{e1})$
        
    - **통계량 추출:** $h_1$에서 $\mu_{q1}, \log\sigma^2_{q1}$ (각각 $64 \times 64$ 차원) 계산.
        
    - 샘플링 ($z_1$): Reparameterization Trick 사용 ($\epsilon_1 \sim \mathcal{N}(0, I)$)
        
        $$z_1 = \mu_{q1} + \sigma_{q1} \odot \epsilon_1 \quad (\in \mathbb{R}^{64 \times 64})$$
        
3. **Layer 2 ($z_1 \rightarrow z_2$):**
    
    - 이번엔 방금 뽑은 **$z_1$이 입력**이 됩니다.
        
    - 가중치 행렬 $W_{e2} \in \mathbb{R}^{64 \times 32 \times 2}$ (평균/분산용)
        
    - **통계량 추출:** $z_1$을 통과시켜 $\mu_{q2}, \log\sigma^2_{q2}$ (각각 $64 \times 32$ 차원) 계산.
        
    - 샘플링 ($z_2$): ($\epsilon_2 \sim \mathcal{N}(0, I)$)
        
        $$z_2 = \mu_{q2} + \sigma_{q2} \odot \epsilon_2 \quad (\in \mathbb{R}^{64 \times 32})$$
        

> **결과:** 현재 우리 손에는 샘플링된 $z_1, z_2$와, Encoder가 생각한 분포 파라미터들($\mu_q, \sigma_q$)이 있습니다.

---

#### Step 2: Decoder (Top-Down) - 생성 및 조건부 Prior 계산

이제 위에서부터 다시 내려오면서 **"복원"**과 **"Prior 분포 계산"**을 수행합니다.

1. **최상위 Prior ($P(z_2)$):**
    
    - 가장 꼭대기 $z_2$는 부모가 없으므로 **표준 정규분포 $\mathcal{N}(0, I)$** 라고 가정합니다.
        
    - $\mu_{p2} = 0, \sigma_{p2} = 1$ (고정값)
        
2. **Conditional Prior ($z_2 \rightarrow z_1$):** **(여기가 핵심)**
    
    - 샘플링된 $z_2$($64 \times 32$)를 Decoder 신경망에 넣습니다.
        
    - $h_{d1} = \text{ReLU}(z_2 \cdot W_{d1})$
        
    - 이 신경망은 $z_1$의 **Prior 분포 파라미터($\mu_{p1}, \sigma_{p1}$)** 를 출력합니다.
        
        - Encoder가 뱉은 $\mu_{q1}$과는 다른 값입니다. **"Top-down 관점에서 $z_2$를 보니 $z_1$은 대충 이럴 거야"**라고 예측하는 것입니다.
            
3. **복원 ($z_1 \rightarrow \hat{x}$):**
    
    - 샘플링된 $z_1$($64 \times 64$)을 마지막 Decoder 신경망에 넣습니다.
        
    - $\hat{x} = \text{Sigmoid}(z_1 \cdot W_{d2}) \quad (\in \mathbb{R}^{64 \times 784})$
        

---

#### Step 3: 손실 함수 계산 (Loss)

HVAE의 손실 함수는 세 부분으로 나뉩니다.

1. **Reconstruction Loss (복원 오차):**
    
    - 입력 $x$와 복원된 $\hat{x}$의 차이.
        
    - $\text{Loss}_{recon} = \| x - \hat{x} \|^2$
        
2. **KL Divergence (Top Layer):**
    
    - Encoder가 뽑은 $z_2$ 분포($\mu_{q2}, \sigma_{q2}$)와 표준 정규분포($0, 1$)의 차이.
        
    - $D_{KL}(Q(z_2|z_1) || \mathcal{N}(0, I))$
        
3. **Conditional KL Divergence (Middle Layer):** **(HVAE의 진짜 이유)**
    
    - Encoder가 데이터($x$)를 보고 생각한 $z_1$의 분포($\mu_{q1}, \sigma_{q1}$)와
        
    - Decoder가 상위개념($z_2$)만 보고 예측한 $z_1$의 분포($\mu_{p1}, \sigma_{p1}$) 사이의 거리.
        
    - **의미:** "Bottom-up 정보와 Top-down 정보가 일치하도록 맞춰라."
        
    - $D_{KL}(\mathcal{N}(\mu_{q1}, \sigma_{q1}) || \mathcal{N}(\mu_{p1}, \sigma_{p1}))$
        

Total Loss = Recon + KL(Top) + KL(Middle)

이 Loss를 미분하여 모든 가중치($W_e, W_d$)를 업데이트합니다.

---

### 2. 추론/생성 과정 (Inference / Generation)

학습이 끝난 후, 새로운 이미지를 생성할 때는 **Encoder는 필요 없고 Decoder(Top-down)만 사용**합니다.

1. **Step 1: 최상위 $z_2$ 샘플링**
    
    - 아무 정보도 없으므로 표준 정규분포에서 랜덤 노이즈를 뽑습니다.
        
    - $z_2 \sim \mathcal{N}(0, I) \quad (\in \mathbb{R}^{1 \times 32})$
        
2. **Step 2: $z_1$ 생성 (Not Random, but Probabilistic)**
    
    - $z_2$를 Decoder Layer 1에 통과시켜 $\mu_{p1}, \sigma_{p1}$를 얻습니다.
        
    - 이 분포에서 $z_1$을 샘플링합니다.
        
    - $z_1 \leftarrow \mu_{p1} + \sigma_{p1} \odot \epsilon$
        
    - _(참고: 일반 VAE는 여기서 샘플링을 안 하고 그냥 고정된 Prior를 쓰지만, HVAE는 상위 계층이 하위 계층의 '분포'를 지정해주므로 여기서도 샘플링이 일어납니다.)_
        
3. **Step 3: 이미지 생성**
    
    - $z_1$을 Decoder Layer 2에 통과시켜 최종 이미지 $\hat{x}$를 만듭니다.
        

### 3. 요약: 행렬 관점에서의 차이

- **일반 VAE:**
    
    - Loss 계산 시: $D_{KL}(\mathcal{N}(\mu_{enc}, \sigma_{enc}) || \mathbf{\mathcal{N}(0, 1)})$
        
    - 무조건 **고정된 표준 정규분포(0, 1)**와 비교합니다.
        
- **계층적 VAE:**
    
    - 중간 계층 Loss: $D_{KL}(\mathcal{N}(\mu_{q}, \sigma_{q}) || \mathbf{\mathcal{N}(\mu_{p}, \sigma_{p})})$
        
    - 비교 대상이 **상위 계층 연산 결과로 나온 행렬($\mu_p, \sigma_p$)**입니다.
        
    - 즉, "0으로 가라"고 강제하는 게 아니라 **"상위 계층이 시키는 대로 가라"**고 유연하게 학습합니다.