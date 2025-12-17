
---
### SDE (확률 미분방정식): "운명 + 우연"

SDE는 ODE의 세계에 **'무작위성(Randomness)'**을 한 스푼 넣은 것입니다. 세상의 많은 현상(주식 시장, 분자의 확산, 열의 이동)은 매끄럽게 움직이지 않고 파르르 떨리며 움직입니다.

$$dX_t = \underbrace{f(X_t, t)dt}_{\text{Drift (흐름)}} + \underbrace{g(X_t, t)dW_t}_{\text{Diffusion (떨림)}}$$

#### 1. Drift Term ($f(X_t, t)dt$)

- **경향성(Trend)**
- **의미:** 노이즈가 없다면 갔을 "원래의 방향"입니다. ODE에서의 $f$

#### 2. Diffusion Term ($g(X_t, t)$)

- **변동성(Volatility)의 크기**
- **의미:** 노이즈가 얼마나 세게 작용할지를 결정하는 계수(Coefficient)입니다.

#### 3. Brownian Motion ($dW_t$)

- **예측 불가능한 우연(Noise Source)**입니다.
- **정의:** 위너 프로세스(Wiener Process, $W_t$)의 미소 변화량입니다.
- **성질:**
    - **독립 증분:** 지금의 떨림은 1초 전의 떨림과 아무 상관이 없습니다.
        
    - 가우시안: 아주 짧은 시간 $\Delta t$ 동안의 변화량 $W_{t+\Delta t} - W_t$는 평균이 0이고 분산이 $\Delta t$인 정규분포를 따릅니다.
    $$W_{t+\Delta t} - W_t \sim \mathcal{N}(0, \Delta t)$$
        
    - 이 성질 때문에 컴퓨터로 시뮬레이션할 때는 보통 $\sqrt{\Delta t} \cdot \epsilon$ ($\epsilon$은 표준정규분포)으로 구현합니다. 분산이 $\Delta t$가 되려면 표준편차는 $\sqrt{\Delta t}$여야 하니까요.

---

Discrete한 DDPM($x_t$)을 Continuous한 SDE($X_t$)로 확장하면, Diffusion 모델을 **"데이터 분포를 가우시안 분포로 녹여버리는 유체의 흐름"**으로 해석할 수 있습니다.

이 관점은 **Score-based Generative Modeling (Yang Song et al.)**의 핵심 이론이기도 합니다.

우리가 앞서 본 DDPM의 Forward 식은 다음과 같습니다.

$$x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t$$

여기서 전체 단계 $T$를 무한대로 늘리고, 한 스텝의 크기를 아주 작은 시간 단위 $\Delta t$로 쪼갬.

이때 $\beta_t$는 아주 작은 값인 $\beta(t)\Delta t$로 치환할 수 있습니다.

이제 위 식을 $\Delta t$에 대해 테일러 급수 근사($\sqrt{1-x} \approx 1 - \frac{x}{2}$)를 적용해 변형하면 다음과 같은 형태가 됩니다.

$$x_{t} - x_{t-1} \approx \underbrace{-\frac{1}{2}\beta(t)x_{t-1}\Delta t}_{\text{Drift (흐름)}} + \underbrace{\sqrt{\beta(t)}\sqrt{\Delta t}\epsilon_t}_{\text{Diffusion (확산)}}$$

이 식에서 $\Delta t \to 0$인 극한을 취하면, **확률 미분 방정식(SDE)**이 짠 하고 나타납니다.

$$d\mathbf{x} = \underbrace{-\frac{1}{2}\beta(t)\mathbf{x} \, dt}_{\mathbf{f}(\mathbf{x}, t)dt} + \underbrace{\sqrt{\beta(t)} \, d\mathbf{w}}_{\mathbf{g}(t)d\mathbf{w}}$$

- $d\mathbf{w}$: 브라운 운동(Wiener Process)의 미소 변화량 (연속적인 노이즈)
    
---

### 2. $f$와 $g$의 물리적 의미

위에서 유도된 식 $d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x} dt + \sqrt{\beta(t)} d\mathbf{w}$ 를 뜯어보면 두 가지 힘이 싸우고 있는 것을 볼 수 있습니다.

#### ① Drift 항: $f(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\mathbf{x}$

- **역할:** "데이터를 0으로 끌어당김"
    
- **이미지:** 잉크가 물에 퍼지면서 색이 옅어지는 것처럼, 원래 데이터의 진한 정보(Signal)를 지워버리는 역할을 합니다.
    

#### ② Diffusion 항: $g(t) = \sqrt{\beta(t)}$

- **역할:** "무작위 노이즈 주입"
- 랜덤한 움직임($d\mathbf{w}$)을 계속 더해줍니다. $\beta(t)$의 크기만큼 데이터가 제멋대로 흔들리게 만듭니다.
    
- **이미지:** 물 분자가 끊임없이 잉크 입자를 때려서 사방으로 흩어지게 만드는 힘입니다.
    

---

### 3. "데이터 분포 → 가우시안 분포"의 흐름

이 SDE를 $t=0$에서 $t=T$까지 쭉 실행하면 어떤 일이 벌어질까요?

1. **시작 ($t=0$):** 데이터 분포 $p_{data}(\mathbf{x})$에서 시작합니다. (복잡한 형태, 예: 강아지 사진들의 분포)
    
2. **과정 ($0 < t < T$):**
    
    - **Drift**에 의해 원본 이미지는 점점 흐려지고(0으로 수렴),
        
    - **Diffusion**에 의해 노이즈가 계속 덮입혀집니다.
        
3. **끝 ($t=T$):** 원래 데이터의 형태는 온데간데없고, 완전한 무질서 상태인 **표준 정규 분포(Standard Gaussian, $\mathcal{N}(\mathbf{0}, \mathbf{I})$)**가 됩니다.
    

즉, **Forward SDE**는 **"복잡한 데이터 구조를 단순한 가우시안 노이즈로 붕괴시키는 과정"**입니다.

### 4. 핵심: 왜 굳이 SDE로 해석하나요?

이것이 진짜 중요한 이유입니다. 수학자 **Anderson(1982)**의 이론에 따르면,

**"어떤 Forward SDE가 존재하면, 시간을 거꾸로 돌리는 Reverse SDE도 반드시 존재한다"**는 것이 증명되어 있습니다. [[M) Anderson's Reverse SDE]] <---- 수식 유도

$$d\mathbf{x} = \left[ \mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x}) \right] dt + g(t) d\bar{\mathbf{w}}$$

이 무시무시해 보이는 식에서 중요한 건 딱 하나, $\nabla_\mathbf{x} \log p_t(\mathbf{x})$ 입니다.

이것이 바로 **Score Function(점수 함수)**입니다.

- **결론:** 우리가 **Score Function($\nabla \log p_t$)만 알아낼 수 있다면(학습한다면)**, SDE 공식을 이용해 가우시안 노이즈를 다시 데이터 분포로 되돌리는 **Reverse SDE**를 풀 수 있다는 뜻입니다.
[[Score_Matching]] **<--- 이후 과정

### 5. **요약:**

- Forward Diffusion은 데이터를 가우시안 노이즈로 만드는 SDE 흐름이다.
    
- 이 흐름의 반대(Reverse SDE)를 타면 노이즈에서 데이터를 생성할 수 있다.
    
- 그 반대 흐름을 타기 위한 '나침반'이 바로 **Score Function**이고, U-Net은 이것을 학습한다.
    