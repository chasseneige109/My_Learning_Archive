Diffusion Model(DDPM)의 Loss 함수가 `단순한 MSE`($\|\epsilon - \epsilon_\theta\|^2$)로 귀결되는 과정을 **ELBO의 탄생부터 최종 수식까지** 엄밀하게 유도해 드리겠습니다.

이 증명 과정은 크게 **4단계**로 나뉩니다.

1. **ELBO 유도:** 왜 Log Likelihood 대신 ELBO를 쓰는가?
    
2. **항의 전개와 분해:** 복잡한 수식을 $L_T, L_{t-1}, L_0$로 쪼개기.
    
3. **Posterior($q$) 분석:** 정답지 분포의 평균($\tilde{\mu}_t$) 구하기.
    
4. **Reparameterization:** 평균 차이를 노이즈 차이로 치환하기.
    

---

### Step 1. ELBO (Evidence Lower Bound)의 유도

우리의 목표는 모델 $p_\theta$가 실제 데이터 $x_0$를 생성할 확률(Log Likelihood)을 최대화하는 것입니다.

$$\text{Maximize } \log p_\theta(x_0)$$

하지만 Diffusion 과정에서 $x_0$는 $x_1, x_2, \dots, x_T$라는 잠재 변수(Latent Variable)들을 거쳐서 나옵니다. 따라서 $p_\theta(x_0)$는 모든 잠재 변수에 대해 적분(Marginalize)해야 하는데, 이건 계산이 불가능합니다(Intractable).

$$p_\theta(x_0) = \int p_\theta(x_{0:T}) dx_{1:T}$$

그래서 **Jensen's Inequality(젠슨 부등식)**을 사용하여 계산 가능한 **하한선(Lower Bound)**을 만듭니다.

$$\begin{aligned} \log p_\theta(x_0) &= \log \int p_\theta(x_{0:T}) dx_{1:T} \\ &= \log \int \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} q(x_{1:T}|x_0) dx_{1:T} \quad (\text{분자 분모에 } q \text{ 곱함}) \\ &= \log \mathbb{E}_{q(x_{1:T}|x_0)} \left[ \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \right] \\ &\ge \mathbb{E}_{q(x_{1:T}|x_0)} \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \right] \quad (\because \log(\mathbb{E}[x]) \ge \mathbb{E}[\log(x)]) \end{aligned}$$

이 마지막 식이 바로 **ELBO(Evidence Lower Bound)**입니다. 우리는 이 식에 마이너스를 붙인 **Negative ELBO를 최소화**할 것입니다.

$$\text{Loss} = \mathbb{E}_q \left[ -\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \right]$$

---

### Step 2. Markov Property를 이용한 항의 분해

이제 Forward($q$)와 Reverse($p$) 과정을 정의합니다. Diffusion은 **Markov Chain**이므로 다음과 같이 조건부 확률의 곱으로 표현됩니다.

- **Forward (Encoder):** $q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})$
    
- **Reverse (Decoder):** $p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t)$
    

이 정의를 Loss 식에 대입하여 전개합니다.

$$\begin{aligned} L &= \mathbb{E}_q \left[ -\log \frac{p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^T q(x_t|x_{t-1})} \right] \\ &= \mathbb{E}_q \left[ -\log p(x_T) - \sum_{t=1}^T \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})} \right] \end{aligned}$$

이 식을 $t$에 따라 묶어서 정리하면(DDPM 논문의 Eq. 5), 아래와 같이 세 개의 항으로 깔끔하게 나뉩니다. **(핵심 과정)**

$$L = \underbrace{D_{KL}(q(x_T|x_0) || p(x_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t))}_{L_{t-1}} \underbrace{- \log p_\theta(x_0|x_1)}_{L_0}$$

1. **$L_T$:** $x_T$는 순수 노이즈(Gaussian)로 고정되므로 **상수(0에 근사)**입니다. 무시합니다.
	$$L_T = D_{KL}(\underbrace{q(x_T | x_0)}_{\text{사실상 }\mathcal{N}(0,I)} \ || \ \underbrace{p(x_T)}_{\text{정의상 }\mathcal{N}(0,I)})$$
    
2. **$L_0$:** 마지막 복원 단계입니다.
    
3. **$L_{t-1}$:** **우리가 풀어야 할 핵심 항**입니다.
    

---

### Step 3. Posterior $q(x_{t-1}|x_t, x_0)$의 분석

$L_{t-1}$은 두 분포 사이의 KL Divergence입니다.

$$D_{KL}(\color{blue}{q(x_{t-1}|x_t, x_0)} \ || \ \color{red}{p_\theta(x_{t-1}|x_t)})$$

여기서 **파란색 $q$ (Posterior)**는 베이즈 정리를 통해 **Closed Form(수식)**으로 구할 수 있습니다.

$$q(x_{t-1}|x_t, x_0) = \frac{q(x_t|x_{t-1}) q(x_{t-1}|x_0)}{q(x_t|x_0)}$$

위 식의 $q$들은 모두 가우시안 분포입니다. 가우시안끼리 곱하고 나누면 결과도 가우시안입니다.

가우시안 지수 항(exponential term)을 정리하여 완전제곱꼴로 만들면 다음 결과가 나옵니다.

	$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I})$



$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t$$

이 식은 $x_0$를 포함하고 있습니다. 하지만 우리는 $x_t$와 노이즈 $\epsilon$ 만으로 이 식을 표현하고 싶습니다.

Forward Process의 정의인 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ 식을 변형하여 $x_0$를 소거합니다.

$$x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}$$

이를 위 $\tilde{\mu}_t$ 식에 대입하고 정리하면, **아주 중요한 공식**이 탄생합니다.

$$\mathbf{\tilde{\mu}_t(x_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon \right)} \quad (\text{★ 정답 평균})$$

---

### Step 4. Reparameterization (KL to MSE)

이제 다시 KL Divergence로 돌아갑니다.

우리는 모델 $p_\theta$가 위에서 구한 정답 분포 $q$를 닮기를 원합니다.

모델 $p_\theta(x_{t-1}|x_t)$의 평균을 $\mu_\theta$라고 합시다.

두 가우시안 분포의 KL Divergence는 **평균의 차이의 제곱($\|\mu_1 - \mu_2\|^2$)**에 비례합니다.

$$L_{t-1} \propto \mathbb{E}_{x_0, \epsilon} \left[ \| \tilde{\mu}_t(x_t, \epsilon) - \mu_\theta(x_t, t) \|^2 \right]$$

모델 $\mu_\theta$가 정답 $\tilde{\mu}_t$를 잘 예측하게 하려면, 모델의 수식 구조를 정답과 똑같이 맞춰주는 것이 유리합니다.

따라서 **모델 $\mu_\theta$를 다음과 같이 정의(Parameterization)**합니다.

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \color{red}{\epsilon_\theta(x_t, t)} \right)$$

(여기서 $\epsilon_\theta$는 모델이 예측해야 할 노이즈 값입니다.)

이제 이 모델 식과 정답 식을 Loss 함수에 대입합니다.

$$\begin{aligned} Loss &= \| \tilde{\mu}_t - \mu_\theta \|^2 \\ &= \left\| \left( \frac{1}{\sqrt{\alpha_t}} ( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \color{blue}{\epsilon} ) \right) - \left( \frac{1}{\sqrt{\alpha_t}} ( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \color{red}{\epsilon_\theta} ) \right) \right\|^2 \\ &= \left\| \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}} (\color{blue}{\epsilon} - \color{red}{\epsilon_\theta}) \right\|^2 \end{aligned}$$

상수항들을 정리하면 다음과 같습니다.

$$L_{t-1} = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)} \| \epsilon - \epsilon_\theta(x_t, t) \|^2$$

DDPM 논문 저자들은 앞에 붙은 복잡한 가중치 계수($\lambda_t$)를 **1로 단순화(Simplified)**하는 것이 학습 성능이 더 좋다는 것을 발견했습니다.

### 최종 결론 (Final Loss)

$$L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) \|^2 \right]$$

**요약:**

1. **ELBO**에서 출발하여,
    
2. **KL Divergence** 문제로 변환했고,
    
3. **Bayes Rule**로 정답 평균을 구해보니 "노이즈($\epsilon$)" 항이 들어있었고,
    
4. 모델도 똑같이 "노이즈($\epsilon_\theta$)"를 예측하게 만들어서 뺐더니,
    
5. 결국 **MSE Loss**만 남게 되었습니다.