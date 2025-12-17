**Diffusion Model (Stable Diffusion, LDM 기준)**의 **학습(Training)**과 **추론(Inference)** 과정을 아주 미시적인 레이어 단위까지 파고들어 설명해 드리겠습니다.

거시적인 흐름에서 시작해, 픽셀(Latent) 하나가 겪는 수난과 재탄생의 과정을 **End-to-End**로 추적해 봅시다.

---

### Phase 1: 학습 과정 (Training Loop)

**목표:** "노이즈가 얼마나 끼었는지 맞춰라 ($\epsilon \approx \epsilon_\theta$)"

#### 1. 데이터 준비 (Data Preparation)

- **이미지($x$):** $512 \times 512 \times 3$ (RGB) 이미지를 준비합니다.
    
- **VAE Encoder (Freeze):** 이미지를 잠재 공간으로 압축합니다.
    
    - $x \xrightarrow{\text{VAE}} z_0$ (크기: $64 \times 64 \times 4$)
        
    - 이 $z_0$가 이제부터 우리가 다룰 '진짜 데이터'입니다.
        
- **텍스트($y$):** "A cute cat" 같은 프롬프트.
    
- **Text Encoder (CLIP/T5, Freeze):** 텍스트를 벡터로 변환합니다.
    
    - $y \xrightarrow{\text{CLIP}} c$ (크기: $77 \times 768$ 혹은 $77 \times 1024$)
        
    - 77은 토큰 길이, 768/1024는 임베딩 차원입니다.
        

#### 2. 노이즈 주입 (Forward Diffusion)

- **Time ($t$):** $1$에서 $1000$ 사이의 정수를 랜덤하게 하나 뽑습니다. (예: $t=500$)
    
- **Noise ($\epsilon$):** $z_0$와 같은 크기($64 \times 64 \times 4$)의 **가우시안 노이즈($\mathcal{N}(0, I)$)**를 생성합니다.
    
- **Mixing:** $z_0$와 $\epsilon$을 $t$에 맞는 비율($\bar{\alpha}_t$)로 섞습니다.
    
    - $$z_t = \sqrt{\bar{\alpha}_t} \cdot z_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$
        
    - 결과물 $z_t$는 "형체를 알아보기 힘든 노이즈 낀 잠재 벡터"입니다.
        

#### 3. Reverse Process U-Net

이제 $z_t$ (노이즈 낀 이미지), $t$ (시점), $c$ (텍스트) 세 가지가 **U-Net**으로 들어갑니다.

##### **[3-1] 시간 임베딩 (Time Embedding)**

- 정수 $t$를 신경망이 이해할 수 있는 벡터로 바꿉니다.
    
- **Sinusoidal Positional Encoding:** $t \to$ `[320]` 차원 벡터.
    
- **MLP:** `Linear(320->1280)` $\to$ `SiLU` $\to$ `Linear(1280->1280)`.
    
- 결과: $t_{emb}$ 벡터 생성. (모든 ResBlock에 공급됨)
    

##### **[3-2] 입력 컨볼루션**

- $z_t (4 \text{ch})$ $\to$ `Conv3x3` $\to$ Feature Map ($320 \text{ch}$).
    

##### **[3-3] DownBlocks (압축 구간)**

여기서 **ResBlock**과 **Transformer**가 반복됩니다. 미시적으로 들어갑니다.

- **(A) ResNet Block (Spatial Feature + Time):**
    
    1. **Input:** $h$ (Feature Map)
        
    2. **GN:** `GroupNorm(32, h)` $\to$ 채널을 32개 그룹으로 나눠 정규화.
        
    3. **SiLU:** $x \cdot \sigma(x)$ 활성화 함수.
        
    4. **Conv1:** `Conv3x3(h)` $\to$ 특징 추출.
        
    5. **Time Injection:** 아까 만든 $t_{emb}$를 `Linear`로 변환하여 $h$의 채널 수에 맞게 더해줌 ($h + t_{proj}$). **(시간 정보 주입!)**
        
    6. **GN & SiLU:** 다시 정규화 및 활성화.
        
    7. **Conv2:** `Conv3x3`.
        
    8. **Skip:** 입력($h$)과 결과물을 더함. (차원이 다르면 $1 \times 1$ Conv로 맞춰서 더함).
        
- **(B) Spatial Transformer (Text Condition):**
    
    1. **Reshape:** $(B, C, H, W) \to (B, H \cdot W, C)$ (이미지를 시퀀스로 폄).
        
    2. **LN:** `LayerNorm`.
        
    3. **Self-Attention:** `MultiHead(Q=x, K=x, V=x)` $\to$ 이미지 내부 픽셀끼리 관계 파악.
        
    4. **Add:** 잔차 연결.
        
    5. **LN:** `LayerNorm`.
        
    6. **Cross-Attention:**
        
        - Query($Q$) = 이미지($x$).
            
        - Key($K$), Value($V$) = **텍스트 임베딩($c$)**.
            
        - `Softmax(Q @ K.T) @ V` 연산으로 텍스트 정보가 이미지에 스며듦.
            
    7. **Add:** 잔차 연결.
        
    8. **FFN (Feed Forward):** `LN` $\to$ `Linear` $\to$ `GEGLU` $\to Linear`.
        
    9. **Reshape:** 다시 $(B, C, H, W)$ 이미지 형태로 복구.
        
- **(C) Downsample:**
    
    - `Conv3x3(stride=2)`를 통해 가로세로 크기를 절반으로 줄임 ($64 \to 32$).
        

##### **[3-4] Middle Block (가장 깊은 곳)**

- 가장 압축된 상태 ($8 \times 8$ 혹은 그 부근).
    
- `ResBlock` $\to$ `Transformer` $\to$ `ResBlock` 순서로 수행.
    
- 여기서 텍스트와 이미지의 전역적(Global)인 의미가 가장 강하게 결합됨.
    

##### **[3-5] UpBlocks (복원 구간)**

- DownBlock의 역순.
    
- **Concat:** 인코더의 같은 층에서 온 Feature Map을 채널 방향으로 붙임. (Skip Connection).
    
- **Upsample:** `Nearest Interpolation` (2배 확대) $\to$ `Conv3x3`.
    

##### **[3-6] 출력층**

- `GroupNorm` $\to$ `SiLU` $\to$ `Conv3x3(출력 채널=4)`.
    
- 최종 결과물: **$\epsilon_\theta$ (모델이 예측한 노이즈)**.
    

#### 4. Loss 계산 및 역전파

- **정답:** 2번 단계에서 넣었던 실제 노이즈 $\epsilon$.
    
- **예측:** U-Net이 뱉어낸 $\epsilon_\theta$.
    
- **Loss:** MSE Loss $= ||\epsilon - \epsilon_\theta||^2$.
    
- **Backprop:** 오차를 줄이는 방향으로 U-Net 내부의 모든 파라미터(Conv 가중치, Attention 가중치 등)를 업데이트.
    

---

### Phase 2: 추론 과정 (Inference / Sampling)

**목표:** "순수 노이즈에서 시작해, 텍스트에 맞는 이미지를 조각해내기"

#### 1. 초기화

- **Latent:** $z_T$를 완전한 랜덤 노이즈 $\mathcal{N}(0, I)$로 생성 ($64 \times 64 \times 4$).
    
- **Prompt:** 사용자가 입력한 텍스트를 CLIP으로 변환 ($c$).
    

#### 2. Denoising Loop (예: 50 Step)

$t = T(1000)$에서 시작해서 $t=0$까지 줄여나갑니다.

- **Step 1: 입력 복제 (Classifier-Free Guidance)**
    
    - $z_t$를 두 개로 복사합니다.
        
    - 하나는 텍스트 조건($c$)을 달고 들어갈 놈, 하나는 빈 조건($\phi$, Unconditional)을 달고 들어갈 놈.
        
- **Step 2: U-Net 예측**
    
    - $z_t$를 U-Net에 통과시킵니다.
        
    - $\epsilon_{cond} = \text{UNet}(z_t, t, c)$ (텍스트 있는 버전)
        
    - $\epsilon_{uncond} = \text{UNet}(z_t, t, \phi)$ (텍스트 없는 버전 - 창의성/형태 담당)
        
- **Step 3: 가이던스 적용 (CFG)**
    
    - $\epsilon_{final} = \epsilon_{uncond} + w \times (\epsilon_{cond} - \epsilon_{uncond})$
        
    - $w$ (Guidance Scale): 이 값이 클수록 텍스트를 더 강하게 따릅니다.
        
- **Step 4: 노이즈 제거 (Scheduler Step)**
    
    - 모델이 예측한 노이즈($\epsilon_{final}$)를 $z_t$에서 뺍니다.
        
    - 하지만 확 빼버리면 그림이 망가지므로, 수학적 공식(DDIM, Euler, DPM++ 등)에 따라 **"이번 스텝에서 뺄 만큼만"** 뺍니다.
        
    - 그리고 다음 스텝의 적절한 노이즈 양을 맞춘 $z_{t-1}$을 만듭니다.
        
- **반복:** 위 과정을 $t=1$이 될 때까지 반복합니다. $z_0$ (깨끗한 Latent)가 나옵니다.
    

#### 3. 디코딩 (Decoding)

- 최종적으로 얻은 $z_0$ ($64 \times 64 \times 4$)를 **VAE Decoder**에 넣습니다.
    
- **Upsampling:** $64 \times 64 \to 512 \times 512$.
    
- **Output:** 사람이 볼 수 있는 RGB 이미지가 탄생합니다.
    

---

### 요약: 미시적 관점의 데이터 흐름

1. **ResBlock:** 시간($t$)을 보고 "아 지금 노이즈가 많구나/적구나"를 파악해서 Feature Map의 **값의 분포(Scale & Shift)**를 조절합니다.
    
2. **Self-Attention:** "왼쪽 위의 귀"와 "오른쪽 아래의 꼬리"가 서로 통신하며 **형태적 일관성**을 맞춥니다.
    
3. **Cross-Attention:** "갈색 털"이라는 텍스트($K, V$)가 이미지의 털 부분($Q$)에 **색칠(값 주입)**을 합니다.
    
4. **LayerNorm/GroupNorm:** 이 모든 과정에서 숫자가 너무 커지거나 작아지지 않게 **계속 정규화**를 해줘서 학습을 안정시킵니다.
    
5. **SiLU/GEGLU:** 비선형성을 줘서 단순한 선형 변환이 아니라 **복잡한 표현**을 가능하게 합니다.