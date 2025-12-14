
본질적으로 **무한 차원에서의 선형 계획법(Infinite-dimensional LP)** 문제입니다.

---

### 1. Primal Problem: LP 형태로 정의하기

우선, Wasserstein 거리를 구하는 원래 문제(Primal)를 LP 표준형으로 다시 써봅시다.

목표: 비용(Cost) 최소화

$$\min_{\gamma} \int_{X \times Y} \|x - y\| \, d\gamma(x, y)$$

**제약 조건 (Constraints):**

1. **비음성 조건:** 모든 $(x, y)$에 대해 $\gamma(x, y) \ge 0$
    
2. **주변 분포(Marginal) 보존:**
    
    - $y$에 대해 적분하면 $P_r$이 나와야 함: $\int_Y d\gamma(x, y) = dP_r(x)$
        
    - $x$에 대해 적분하면 $P_g$가 나와야 함: $\int_X d\gamma(x, y) = dP_g(y)$
        

이것은 전형적인 **수송 문제(Transportation Problem)**이며, 변수 $\gamma$에 대한 선형 함수이므로 LP입니다.

---

### 2. 라그랑주(Lagrangian) 도입

제약 조건이 있는 최적화 문제를 풀기 위해 **라그랑주 승수(Lagrange Multiplier)**를 도입합니다. 여기서는 제약 조건이 함수 형태(모든 $x, y$에 대해 성립)이므로, 승수도 함수 형태인 $f(x)$와 $g(y)$가 됩니다. (원랜 내적이지만, 무한히 많은 제약이므로 적분)

- 제약 조건 1에 대한 승수: $f(x)$
    
- 제약 조건 2에 대한 승수: $g(y)$
    

라그랑주 함수 $\mathcal{L}(\gamma, f, g)$는 다음과 같이 정의됩니다.

$$\begin{aligned} \mathcal{L}(\gamma, f, g) &= \underbrace{\int \|x - y\| d\gamma(x, y)}_{\text{목적 함수}} \\ &- \underbrace{\int f(x) \left( \int_Y d\gamma(x, y) - dP_r(x) \right)}_{\text{제약 1}} - \underbrace{\int g(y) \left( \int_X d\gamma(x, y) - dP_g(y) \right)}_{\text{제약 2}} \end{aligned}$$

이 식을 $\gamma$와 관련된 항과 그렇지 않은 항으로 묶어서 정리하면:

$$\mathcal{L}(\gamma, f, g) = \int_{X \times Y} \underbrace{\left( \|x - y\| - f(x) - g(y) \right)}_{(*)} \, d\gamma(x, y) + \int_X f(x) dP_r(x) + \int_Y g(y) dP_g(y)$$

---

### 3. Dual Function 유도

Primal Problem은 다음과 같습니다:

$$\inf_{\gamma \ge 0} \sup_{f, g} \mathcal{L}(\gamma, f, g)$$

Dual Problem은 min-max 순서를 바꾼 것입니다 (Strong Duality 가정):

$$\sup_{f, g} \inf_{\gamma \ge 0} \mathcal{L}(\gamma, f, g)$$

이제 내부의 $\inf_{\gamma \ge 0}$ 부분을 봅시다.

식 $(*)$에 해당하는 $\left( \|x - y\| - f(x) - g(y) \right)$ 부분이 만약 음수인 지점 $(x, y)$가 존재한다면?

- 우리는 $\gamma(x, y)$를 무한대로 키워서 전체 값을 $-\infty$로 만들 수 있습니다. (최소화 문제이므로)
    

따라서, 유의미한 최솟값(infimum)이 존재하려면(Dual Feasibility), **$\gamma$의 계수가 모든 곳에서 0 이상**이어야 합니다.

$$\|x - y\| - f(x) - g(y) \ge 0 \quad \iff \quad f(x) + g(y) \le \|x - y\|$$

이 조건이 만족되면 $\gamma$ 항의 적분값의 최솟값은 0이 됩니다 ($\gamma=0$일 때). 남는 것은 뒤의 두 항뿐입니다.

Dual Problem 정의:

$$\sup_{f, g} \left\{ \mathbb{E}_{x \sim P_r}[f(x)] + \mathbb{E}_{y \sim P_g}[g(y)] \right\}$$

Subject to:

$$f(x) + g(y) \le \|x - y\| \quad (\forall x, y)$$

---

### 4. 1-Lipschitz 조건으로의 변환 (핵심 트릭)

**아래 내용은 [[c_transform]] <-- 여기서 수학적 원리를 설명
위의 Dual 식에서 우리는 두 함수 $f$와 $g$를 다뤄야 합니다. 이를 하나로 합쳐봅시다.

제약 조건 $f(x) + g(y) \le |x - y|$를 $g(y)$에 대해 정리하면:

$$g(y) \le \|x - y\| - f(x)$$

우리는 목적 함수(Maximize)를 최대화하고 싶으므로, $g(y)$를 가능한 한 크게 설정해야 합니다. 주어진 $f$에 대해 가장 큰 $g(y)$는 상한(infimum)을 취한 값입니다. (이를 $c$-transform이라고 합니다.)


여기서 아주 교묘한 대칭성을 이용합니다. 만약 우리가 $g(y) = -f(y)$라고 둔다면 어떻게 될까요?

제약 조건은 다음과 같이 변합니다.

$$f(x) - f(y) \le \|x - y\| \quad \iff \quad f(x) - f(y) \le |x - y|$$

이 식은 정확히 Lipschitz 연속성(Lipschitz Continuity)의 정의 ($K=1$)입니다.

$$g(y) = \inf_x (\|x - y\| - f(x))$$
$$\frac{|f(x) - f(y)|}{|x - y|} \le 1$$

즉, $f$가 1-Lipschitz 함수이면 $g = -f$로 두었을 때 위 제약 조건을 만족하며, 수학적으로 $c(x, y) = |x-y|$인 거리 공간(Metric Space)에서는 $g = -f$인 경우가 최적해를 포함한다는 것이 증명되어 있습니다.

---

### 5. 최종 결론

따라서 Dual Problem 식의 $g(y)$ 자리에 $-f(y)$를 대입하고, 제약 조건을 $f$가 1-Lipschitz 함수인 조건($\|f\|_L \le 1$)으로 바꾸면 최종 식이 완성됩니다.

$$\sup_{\|f\|_L \le 1} \left\{ \mathbb{E}_{x \sim P_r}[f(x)] + \mathbb{E}_{y \sim P_g}[-f(y)] \right\}$$

$$= \sup_{\|f\|_L \le 1} \left( \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{x \sim P_g}[f(x)] \right)$$

### 요약: 유도 과정의 논리적 흐름

1. **LP 설정:** Wasserstein 거리를 무한 차원 선형 계획법으로 정의.
    
2. **쌍대성(Duality):** 라그랑주 승수법을 써서 Dual Problem으로 변환.
    
3. **제약 조건 유도:** $\gamma$가 발산하지 않기 위해 $f(x) + g(y) \le |x - y|$라는 조건 도출.
    
4. **함수 단일화:** 거리 공간의 성질을 이용해 $g = -f$ 관계를 적용.
    
5. **Lipschitz:** $f(x) - f(y) \le |x - y|$가 1-Lipschitz 조건과 동일함을 확인.