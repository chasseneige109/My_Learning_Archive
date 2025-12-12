
Autoencoder는 **"입력을 그대로 출력으로 복원하되, 그 과정에서 데이터의 핵심 특징(Latent Representation)을 압축적으로 학습하는 것"**을 목표로 하는 비지도 학습 신경망입니다.

---

### 1. 표기법 및 차원 정의 (Notation & Dimensions)

가장 단순한 형태인 1개의 은닉층(Hidden Layer)을 가진 **Undercomplete Autoencoder**를 기준으로 설명합니다. (입력보다 은닉층 차원이 작은 경우)

우리는 벡터 하나가 아닌, **배치(Batch)** 단위의 행렬 데이터를 처리한다고 가정합니다.

- **$N$**: 배치 크기 (Number of samples)
    
- **$D_{in}$**: 입력 데이터의 원본 차원 (Input dimension)
    
- **$D_{latent}$**: 압축될 잠재 공간의 차원 (Latent dimension). 여기서 **$D_{latent} < D_{in}$** (병목 구간)입니다.
    

#### 데이터 행렬

- **입력 행렬 $X \in \mathbb{R}^{N \times D_{in}}$**: 원본 데이터 배치입니다.
    

#### 모델 파라미터 (가중치 및 편향)

Autoencoder는 크게 **인코더(Encoder)**와 **디코더(Decoder)** 두 부분으로 나뉩니다.

- **인코더 (압축)**:
    
    - 가중치 행렬 **$W_E \in \mathbb{R}^{D_{in} \times D_{latent}}$**
        
    - 편향 벡터 **$b_E \in \mathbb{R}^{1 \times D_{latent}}$** (브로드캐스팅됨)
        
- **디코더 (복원)**:
    
    - 가중치 행렬 **$W_D \in \mathbb{R}^{D_{latent} \times D_{in}}$**
        
    - 편향 벡터 **$b_D \in \mathbb{R}^{1 \times D_{in}}$** (브로드캐스팅됨)
        

#### 활성화 함수

- $\sigma(\cdot)$: 비선형 활성화 함수 (예: ReLU, Sigmoid, Tanh). 행렬의 각 원소에 독립적으로 적용(Element-wise)됩니다.
    

---

### 2. 순전파 과정 (Forward Pass: The Architecture)

데이터가 모델을 통과하여 복원되는 과정입니다.

#### Step 1: 인코더 (Encoder) - 차원 축소

입력 $X$를 저차원의 잠재 표현(Latent Representation) $Z$로 매핑합니다. 이는 **아핀 변환(Affine Transformation)** 후 비선형 함수를 적용하는 과정입니다.

$$Z = \sigma_{\text{enc}}(X W_E + b_E)$$

- **행렬 연산 분석**:
    
    - $X W_E$: $(N \times D_{in}) \times (D_{in} \times D_{latent}) \rightarrow (N \times D_{latent})$
        
    - $+ b_E$: $(N \times D_{latent})$ 행렬의 각 행에 $b_E$가 더해짐 (Broadcasting)
        
- **결과**: **잠재 행렬 $Z \in \mathbb{R}^{N \times D_{latent}}$** 가 생성됩니다. 이것이 데이터의 압축된 특징입니다.
    

#### Step 2: 디코더 (Decoder) - 차원 복원

압축된 $Z$를 다시 원래 차원의 데이터 $\hat{X}$(Reconstruction)으로 복원합니다. 인코더의 역과정입니다.

$$\hat{X} = \sigma_{\text{dec}}(Z W_D + b_D)$$

- **행렬 연산 분석**:
    
    - $Z W_D$: $(N \times D_{latent}) \times (D_{latent} \times D_{in}) \rightarrow (N \times D_{in})$
        
    - $+ b_D$: $(N \times D_{in})$ 크기로 브로드캐스팅되어 더해짐.
        
- **결과**: **복원된 행렬 $\hat{X} \in \mathbb{R}^{N \times D_{in}}$** 가 생성됩니다. $X$와 동일한 차원입니다.
    

> **Note:** 디코더의 활성화 함수 $\sigma_{\text{dec}}$는 입력 데이터의 범위에 따라 달라집니다.
> 
> - 입력이 $[0, 1]$로 정규화된 이미지라면: **Sigmoid**
>     
> - 입력이 일반적인 실수 범위라면: **Identity (항등 함수)**를 주로 사용합니다.
>     

---

### 3. 목적 함수 (Objective Function: Loss)

Autoencoder의 목표는 입력 $X$와 복원된 $\hat{X}$의 차이를 최소화하는 것입니다. 이를 위해 손실 함수(Loss Function)를 정의합니다.

가장 일반적으로 **MSE (Mean Squared Error)**를 사용하며, 이는 행렬 관점에서 **프로베니우스 노름(Frobenius Norm)**의 제곱과 관련이 깊습니다.

$$L(\theta) = \frac{1}{2N} \| X - \hat{X} \|_F^2$$

- $\theta = \{W_E, b_E, W_D, b_D\}$: 학습할 모든 파라미터
    
- $\| A \|_F^2 = \sum_{i}\sum_{j} |a_{ij}|^2$: 행렬의 모든 원소의 제곱합 (행렬 내적 $\text{Tr}(A^T A)$와 동일)
    

이를 원소 단위 합으로 풀어서 쓰면 다음과 같습니다.

$$L = \frac{1}{2N} \sum_{i=1}^{N} \sum_{j=1}^{D_{in}} (x_{ij} - \hat{x}_{ij})^2$$

---

### 4. 역전파 및 파라미터 업데이트 (Backpropagation)

손실 함수 $L$을 최소화하기 위해 경사 하강법(Gradient Descent)을 사용합니다. 핵심은 각 가중치 행렬에 대한 손실 함수의 기울기(Gradient)를 계산하는 것입니다.

- 업데이트 규칙 (Learning Rate $\eta$):
    
    $W \leftarrow W - \eta \frac{\partial L}{\partial W}$
    

행렬 미분(Matrix Calculus)과 연쇄 법칙(Chain Rule)을 사용하여 기울기를 유도합니다. (편의상 $\sigma_{\text{dec}}$를 항등 함수로 가정하여 수식을 간소화합니다. $\hat{X} = Z W_D + b_D$)

#### (1) 출력단 오차 행렬 ($\delta_{out}$) 정의

먼저 출력단에서의 오차(Error Signal)를 정의합니다. 이는 손실 함수를 $\hat{X}$로 미분한 것과 같습니다.

$$\delta_{out} = \frac{\partial L}{\partial \hat{X}} = \frac{1}{N}(\hat{X} - X) \in \mathbb{R}^{N \times D_{in}}$$

#### (2) 디코더 가중치 기울기 ($\frac{\partial L}{\partial W_D}$)

연쇄 법칙에 의해 $\frac{\partial L}{\partial W_D} = \frac{\partial L}{\partial \hat{X}} \cdot \frac{\partial \hat{X}}{\partial W_D}$ 형태가 됩니다. 행렬 미분 결과는 다음과 같습니다.

$$\frac{\partial L}{\partial W_D} = Z^T \delta_{out}$$

- 차원 확인: $(D_{latent} \times N) \times (N \times D_{in}) \rightarrow (D_{latent} \times D_{in})$. ($W_D$와 동일 차원, OK)
    

#### (3) 잠재 공간 오차 행렬 ($\delta_{latent}$) 로의 역전파

오차 신호를 디코더를 통해 거꾸로 흘려보내 잠재 공간 $Z$에서의 오차를 계산합니다.

$$\delta_{latent} = \frac{\partial L}{\partial Z} = \delta_{out} W_D^T \in \mathbb{R}^{N \times D_{latent}}$$

여기에 인코더 활성화 함수의 미분값 $\sigma'{\text{enc}}$을 원소별 곱(Hadamard Product, $\odot$)으로 적용해야 최종적인 은닉층 오차 신호가 됩니다.

$$\delta{E} = \delta_{latent} \odot \sigma'_{\text{enc}}(X W_E + b_E)$$

#### (4) 인코더 가중치 기울기 ($\frac{\partial L}{\partial W_E}$)

마지막으로 인코더 입력단까지 연쇄 법칙을 적용합니다.

$$\frac{\partial L}{\partial W_E} = X^T \delta_{E}$$

- 차원 확인: $(D_{in} \times N) \times (N \times D_{latent}) \rightarrow (D_{in} \times D_{latent})$. ($W_E$와 동일 차원, OK)
    

---

### 5. 요약: 전체 흐름도 (Matrix View)

1. **Input:** $X \in \mathbb{R}^{N \times D_{in}}$
    
2. **Encoder (Forward):** $Z = \sigma(X W_E + b_E)$
    
    - $\to$ $D_{in}$ 차원에서 $D_{latent}$ 차원으로 압축 (병목 통과)
        
3. **Decoder (Forward):** $\hat{X} = \sigma'(Z W_D + b_D)$
    
    - $\to$ $D_{latent}$ 차원에서 다시 $D_{in}$ 차원으로 복원
        
4. **Loss Calculation:** $L = \frac{1}{2N} \| X - \hat{X} \|_F^2$
    
    - $\to$ 원본과 복원본의 차이를 스칼라 값으로 계산
        
5. **Backpropagation:** $\frac{\partial L}{\partial W_D}, \frac{\partial L}{\partial W_E}$ 계산
    
    - $\to$ 연쇄 법칙을 통해 오차 신호를 역방향으로 전파하며 기울기 행렬 계산
        
6. **Update:** $W_E, W_D$ 등 파라미터 갱신
    

이 과정을 반복하면 $W_E$와 $W_D$는 데이터 $X$의 핵심적인 특징을 $Z$에 효율적으로 압축하고, 다시 복원하는 최적의 선형/비선형 변환 행렬로 수렴하게 됩니다.