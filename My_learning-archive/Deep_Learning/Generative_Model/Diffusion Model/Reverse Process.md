**Denoising Process (Sampling)**는 학습이 완료된 AI 모델($\epsilon_\theta$)을 사용하여, 완전한 무질서($x_T$)에서 질서($x_0$)를 찾아내는 과정입니다.

이 과정은 $t=T$에서 $t=0$까지 시간을 거꾸로 돌리는 **Reverse Markov Chain**이며, 수학적으로 엄밀하게 유도된 **단 하나의 점화식(Recurrence Formula)**을 반복하는 행렬 연산입니다.

이 과정을 단계별 행렬/벡터 수식으로 완벽하게 해부해 드리겠습니다.

---

### 1. 초기화 (Initialization)

모든 것은 완전한 가우시안 노이즈에서 시작합니다.

- **상태:** $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
    
- **차원:** $\mathbf{x}_T \in \mathbb{R}^D$ (이미지를 1차원 벡터로 가정, $D = H \times W \times C$)
    
- **의미:** 모든 픽셀값이 평균 0, 분산 1로 독립적인 랜덤 상태입니다.
    

---

### 2. Denoising 루프 (Iterative Refinement)

$t = T, T-1, \dots, 1$ 순서로 다음 과정을 반복합니다.

목표는 $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}t)$의 분포에서 샘플링하여 $\mathbf{x}_{t-1}$을 구하는 것입니다.

이 분포는 가우시안으로 근사됩니다:

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

이때, **평균 벡터 $\boldsymbol{\mu}_\theta$**와 **공분산 행렬 $\boldsymbol{\Sigma}_\theta$**를 결정하는 엄밀한 과정을 서술합니다.

#### Step 2-1. AI 모델의 노이즈 예측 (Prediction)

현재 상태 $\mathbf{x}_t$를 입력받아, 포함된 노이즈를 예측합니다.

$$\hat{\boldsymbol{\epsilon}} = \epsilon_\theta(\mathbf{x}_t, t)$$

- $\epsilon_\theta$: 학습된 Neural Network (U-Net)
    
- $\hat{\boldsymbol{\epsilon}} \in \mathbb{R}^D$: 예측된 노이즈 벡터
    

#### Step 2-2. 원본 데이터 추정 (Estimation of $x_0$)

앞서 Forward Process 식을 역이용하여, 현재 노이즈 예측값을 바탕으로 **"잠정적인 원본 $\hat{\mathbf{x}}_0$"**를 추정합니다. (이 과정은 수식 내부에 내재되어 있지만, 유도를 위해 명시합니다.)

$$\hat{\mathbf{x}}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\hat{\boldsymbol{\epsilon}})$$

#### Step 2-3. 후위 평균 벡터 결정 (Posterior Mean Calculation)

가장 중요한 단계입니다. 
베이즈 정리로 유도했던 **이상적인 평균 $\tilde{\boldsymbol{\mu}}_t$** 공식에,  [[Proof) Mean 구하기|Proof) Mean 구하기]]
방금 구한 **추정된 원본 $\hat{\mathbf{x}}_0$**를 대입합니다.

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \hat{\mathbf{x}}_0)$$

이 식을 $\epsilon_\theta$에 대한 식으로 정리하면 **실제 사용되는 최종 업데이트 공식**이 나옵니다:

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right)$$

- **$\frac{1}{\sqrt{\alpha_t}}$:** 전체 스케일 보정 (Scaling Factor)
    
- **$\mathbf{x}_t$:** 현재 이미지
    
- **$\frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}$:** 예측된 노이즈를 얼마나 뺄지 결정하는 계수
    

#### Step 2-4. 공분산 행렬과 노이즈 주입 (Langevin Step)

샘플링의 다양성을 위해 확률적인 요소를 추가합니다.

- **공분산 행렬:** $\boldsymbol{\Sigma}_\theta(x_t, t) = \sigma_t^2 \mathbf{I}$
    
    - 보통 $\sigma_t^2 = \beta_t$ (Forward process variance) 혹은 $\tilde{\beta}_t$ (Posterior variance)를 사용합니다. 두 값은 거의 비슷합니다.
        
- **랜덤 노이즈 샘플링:** $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ (단, $t=1$일 때는 $\mathbf{z}=\mathbf{0}$)
    

---

### 3. 최종 업데이트 수식 (The Single Equation)

위의 모든 과정을 합치면, 컴퓨터가 $t$ 시점에서 수행하는 **단 한 줄의 행렬 연산**은 다음과 같습니다.

$$\mathbf{x}_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right)}_{\text{Deterministic Term (Denoising)}} + \underbrace{\sigma_t \mathbf{z}}_{\text{Stochastic Term (Noise Injection)}}$$

이 식을 행렬 관점에서 해석하면:

1. **방향 수정 (Direction):** 예측된 노이즈 벡터 $\epsilon_\theta$ 방향의 반대 방향으로 벡터를 이동시킵니다. (노이즈 제거)
    
2. **스케일 조정 (Scaling):** $\frac{1}{\sqrt{\alpha_t}}$ 스칼라를 곱해 데이터의 분산 크기를 맞춥니다.
    
3. **요동 (Fluctuation):** $\sigma_t \mathbf{z}$ 벡터를 더해, 결정론적인 궤적이 아닌 확률 분포를 따르도록 미세하게 흔들어줍니다. 이것이 생성 결과의 다양성을 보장합니다.
    

---

### 4. 최종 출력 (Final Output)

$t=1$까지 루프가 끝나면 $\mathbf{x}_0$를 얻게 됩니다.

이 벡터는 실수(Real number) 값이므로, 이미지 포맷에 맞게 처리합니다.

- **Clipping:** 픽셀 값 범위를 $[-1, 1]$ 또는 $[0, 255]$로 자릅니다.
    
- **Reshaping:** $D$차원 벡터를 $(H, W, C)$ 텐서로 변환하여 이미지를 완성합니다.
    

### 요약: 수학적 알고리즘

1. $\mathbf{x}_T \leftarrow \text{Sample from } \mathcal{N}(\mathbf{0}, \mathbf{I})$
    
2. **for** $t = T, \dots, 1$ **do**
    
    - $\mathbf{z} \leftarrow \text{Sample from } \mathcal{N}(\mathbf{0}, \mathbf{I})$ if $t > 1$ else $\mathbf{0}$
        
    - $\mathbf{x}_{t-1} \leftarrow \frac{1}{\sqrt{\alpha_t}} (\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t, t)) + \sigma_t \mathbf{z}$
        
3. **return** $\mathbf{x}_0$
    

이 과정이 바로 Diffusion Model이 노이즈에서 의미 있는 이미지를 "조각"해내는 수학적으로 엄밀한 과정입니다.