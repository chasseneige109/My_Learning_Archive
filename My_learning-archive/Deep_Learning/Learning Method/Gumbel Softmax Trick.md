
## Gumbel 분포

### 정의

표준 Gumbel(0, 1) 분포:

G=−log⁡(−log⁡(U)), U∼Uniform(0,1)

이 분포는 중요한 성질을 가짐.

---

## Gumbel-Max Trick (이산 샘플링)

다음은 **정확한 categorical sampling**입니다:

y = arg⁡max⁡i (log⁡pi + gi) , gi ∼ Gumbel (0,1)

✅ 매우매우 놀랍게도:

Pr⁡(y=i) = pi

즉,

> **Gumbel noise + argmax = categorical sampling**

하지만…

❌ 여전히 `argmax` → 미분 불가

---

## 
$$y_i = \frac{\exp((\log(\pi_i) + g_i) / \tau)}{\sum_{j=1}^{K} \exp((\log(\pi_j) + g_j) / \tau)}$$
- $\pi_i$: 기존 Logit의 확률값
    
- $g_i$: Gumbel Noise (샘플링의 무작위성을 담당)
    
- **$\tau$ (Temperature): 핵심 파라미터**
    

#### $\tau$ (온도)의 역할

- **$\tau \rightarrow \infty$:** 출력이 Uniform Distribution(모든 확률이 비슷)에 가까워짐.
    
- **$\tau \rightarrow 0$:** 출력이 **One-hot Vector (Argmax)** 에 매우 가까워짐.
    

핵심 트릭:

학습 초기에는 $\tau$를 높게 잡아서 부드럽게(Soft) 학습하다가, 학습이 진행될수록 $\tau$를 0에 가깝게 줄여서 실제 Argmax(Discrete)와 거의 똑같이 동작하게 만듭니다. 이렇게 하면 **미분 가능성(Differentiability)**을 유지하면서도 **이산적인 선택(Discrete Choice)**을 흉내 낼 수 있습니다.