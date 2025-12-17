이 증명의 핵심은 **Reparameterization Trick**을 재귀적(recursive)으로 적용하고, **"두 독립적인 가우시안 분포의 선형 결합은 여전히 가우시안 분포이다"**라는 성질을 이용하는 것입니다.

---

### 1. 전제 조건 및 정의

- 단일 스텝 전이 (Markov Chain):
    
    $$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)I)$$
    
    이를 Reparameterization Trick을 써서 표현하면:
    
    $$x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1}, \quad \text{where } \epsilon_{t-1} \sim \mathcal{N}(0, I)$$
    
- 표기법:
    
    $\alpha_t = 1 - \beta_t$
    
    $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$
    
- 가우시안 합의 성질 (핵심):
    
    서로 독립인 두 정규분포 $X \sim \mathcal{N}(0, \sigma_1^2 I)$와 $Y \sim \mathcal{N}(0, \sigma_2^2 I)$가 있을 때, 이들의 합은 다음과 같습니다.
    
    $$X + Y \sim \mathcal{N}(0, (\sigma_1^2 + \sigma_2^2)I)$$
    
    즉, 표준 정규분포 $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, I)$에 대해 다음이 성립합니다.
    
    $$\sqrt{a}\epsilon_1 + \sqrt{b}\epsilon_2 = \sqrt{a+b}\epsilon^*, \quad \text{where } \epsilon^* \sim \mathcal{N}(0, I)$$
    

---

### 2. 단계별 증명 (Step-by-Step Derivation)

#### Step 1: $x_t$를 $x_{t-1}$로 표현

$$x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1}$$

#### Step 2: $x_{t-1}$을 $x_{t-2}$로 표현하여 대입

$x_{t-1} = \sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2}$ 이므로, 이를 위 식에 대입합니다.

$$\begin{aligned} x_t &= \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2}) + \sqrt{1-\alpha_t}\epsilon_{t-1} \\ &= \sqrt{\alpha_t \alpha_{t-1}}x_{t-2} + \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1}}_{\text{두 노이즈 항의 결합}} \end{aligned}$$

#### Step 3: 노이즈 항 병합 (가우시안 합의 성질 적용)

위 식의 뒷부분(노이즈 항들)은 서로 독립인 두 가우시안 분포의 합입니다. 새로운 분산을 계산해 봅시다.

$$\begin{aligned} \text{Variance} &= (\sqrt{\alpha_t(1-\alpha_{t-1})})^2 + (\sqrt{1-\alpha_t})^2 \\ &= \alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) \\ &= \alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t \\ &= 1 - \alpha_t\alpha_{t-1} \end{aligned}$$

따라서, 두 노이즈 항을 하나의 새로운 노이즈 $\bar{\epsilon}_{t-2} \sim \mathcal{N}(0, I)$로 합칠 수 있습니다.

$$x_t = \sqrt{\alpha_t \alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}}\bar{\epsilon}_{t-2}$$

#### Step 4: 일반화 (귀납적 적용)

위 과정을 $x_0$에 도달할 때까지 반복하면 패턴이 보입니다.

$x_t$를 $x_0$까지 확장하면, 계수는 모든 $\alpha$들의 곱이 되고, 노이즈의 분산은 $1 - (\text{모든 } \alpha \text{들의 곱})$이 됩니다.

$$\begin{aligned} x_t &= \sqrt{\alpha_t \alpha_{t-1} \dots \alpha_1}x_0 + \sqrt{1 - (\alpha_t \alpha_{t-1} \dots \alpha_1)}\epsilon \\ &= \sqrt{\prod_{s=1}^t \alpha_s} x_0 + \sqrt{1 - \prod_{s=1}^t \alpha_s} \epsilon \end{aligned}$$

### 3. 최종 결론

$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ 정의를 대입하면:

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

이것은 평균이 $\sqrt{\bar{\alpha}_t}x_0$이고 분산이 $(1-\bar{\alpha}_t)I$인 가우시안 분포와 같습니다.

$$\therefore q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$

**증명 완료.**

이 폐형식(Closed form) 덕분에 Diffusion Model 학습 시 $t-1, t-2, \dots$를 순차적으로 계산할 필요 없이, **임의의 시점 $t$의 노이즈 낀 이미지를 $x_0$로부터 즉시 생성**할 수 있어 학습 효율이 매우 높아집니다.