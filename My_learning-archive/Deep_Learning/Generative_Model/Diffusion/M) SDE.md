
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