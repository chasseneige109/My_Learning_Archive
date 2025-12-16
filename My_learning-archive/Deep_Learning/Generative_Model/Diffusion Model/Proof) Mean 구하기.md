이 공식은 **두 개의 가우시안 확률 밀도 함수(PDF)의 곱**을 정리하여 **완전제곱식(Square completion)**을 만드는 과정을 통해 유도됩니다.

수학적으로 **베이즈 정리**를 전개하고, 지수(Exponent) 부분만 떼어내어 계수 비교를 하면 정확히 해당 식을 얻을 수 있습니다.

상세 유도 과정은 다음과 같습니다.

---

### 1. 목표 설정: 조건부 확률의 정의

우리가 구하려는 분포는 $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$입니다.

베이즈 정리에 의해 다음 비례식이 성립합니다.

$$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \propto q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) \cdot q(\mathbf{x}_{t-1} | \mathbf{x}_0)$$

여기서 마르코프 성질에 의해 $q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) = q(\mathbf{x}_t | \mathbf{x}_{t-1})$ 이므로:

$$\propto \underbrace{\mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})}_{\text{(A) 전이 확률}} \cdot \underbrace{\mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0, (1-\bar{\alpha}_{t-1})\mathbf{I})}_{\text{(B) 과거 시점 분포}}$$

(여기서 $\beta_t = 1 - \alpha_t$ 입니다.)

### 2. 가우시안 지수(Exponent) 전개

가우시안 분포의 핵심은 **지수 부분(Exponent)**에 있습니다.

$e^{-\frac{1}{2}(\dots)}$ 꼴이므로, 로그를 취해 지수 내부의 $\mathbf{x}_{t-1}$에 대한 2차 함수만 봅니다. (나머지 상수는 무시)

$$L = -\frac{1}{2\beta_t}||\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1}||^2 - \frac{1}{2(1-\bar{\alpha}_{t-1})}||\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0||^2$$

이 식을 전개하여 $\mathbf{x}_{t-1}$에 대해 내림차순 정리합니다.

#### (1) 2차항 ($\mathbf{x}_{t-1}^T \mathbf{x}_{t-1}$) 정리 → 분산 유도

$$-\frac{1}{2} \mathbf{x}_{t-1}^T \mathbf{x}_{t-1} \left( \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}} \right)$$

괄호 안의 통분 과정이 핵심입니다:

$$\frac{\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1-\bar{\alpha}_{t-1})} = \frac{\alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t}{\beta_t(1-\bar{\alpha}_{t-1})} = \frac{1 - \bar{\alpha}_t}{\beta_t(1-\bar{\alpha}_{t-1})}$$

(참고: $\alpha_t \bar{\alpha}_{t-1} = \bar{\alpha}_t$)

이것은 **역분산(Precision)**입니다. 따라서 분산 $\tilde{\beta}_t$는 이것의 역수입니다:

$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$

#### (2) 1차항 ($\mathbf{x}_{t-1}$) 정리 → 평균 유도

$\mathbf{x}_{t-1}$이 포함된 항만 모읍니다:

$$\mathbf{x}_{t-1}^T \left( \frac{\sqrt{\alpha_t}}{\beta_t}\mathbf{x}_{t} + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\mathbf{x}_0 \right)$$

가우시안 분포에서 1차항의 계수는 **$\frac{\text{평균}}{\text{분산}}$**과 같습니다.

따라서, 평균 $\tilde{\boldsymbol{\mu}}_t$ = (분산 $\tilde{\beta}_t$) $\times$ (1차항 계수) 입니다.

### 3. 최종 평균 공식 유도

위에서 구한 분산($\tilde{\beta}_t$)과 1차항 계수를 곱합니다.

$$\tilde{\boldsymbol{\mu}}_t = \left( \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t \right) \times \left( \frac{\sqrt{\alpha_t}}{\beta_t}\mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\mathbf{x}_0 \right)$$

분배법칙으로 항을 각각 계산해 봅시다.

1. $\mathbf{x}_t$ 항:
    
    $$\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t \cdot \frac{\sqrt{\alpha_t}}{\beta_t} = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$$
    
    ($\beta_t$ 약분됨)
    
2. $\mathbf{x}_0$ 항:
    
    $$\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t \cdot \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}$$
    
    ($1-\bar{\alpha}_{t-1}$ 약분됨)
    

### ∴ 결론

두 결과를 합치면 질문하신 공식이 완성됩니다.

$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\mathbf{x}_t$$

이 공식은 **"현재 정보($\mathbf{x}_t$)와 원본 정보($\mathbf{x}_0$)를 가중 평균(Weighted Average)하여, 바로 직전 단계($\mathbf{x}_{t-1}$)를 추정하는 최적의 지점"**을 수학적으로 도출한 결과입니다.