
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
