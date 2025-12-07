
## 2. Gumbel 분포

### 2.1 정의

표준 Gumbel(0, 1) 분포:

G=−log⁡(−log⁡(U)), U∼Uniform(0,1)

이 분포는 중요한 성질을 가짐.

---

### 2.2 Gumbel-Max Trick (이산 샘플링)

다음은 **정확한 categorical sampling**입니다:

y = arg⁡max⁡i (log⁡pi + gi) , gi ∼ Gumbel (0,1)

✅ 놀랍게도:

Pr⁡(y=i)=pi\Pr(y = i) = p_iPr(y=i)=pi​

즉,

> **Gumbel noise + argmax = categorical sampling**

하지만…

❌ 여전히 `argmax` → 미분 불가

---

## 3. Gumbel-Softmax 핵심 아이디어

### 아이디어

> `argmax`를 `softmax`로 치환하자  
> 그리고 temperature τ\tauτ로 sharpness를 조절하자

---

## 4. Gumbel-Softmax 수식 (핵심)

### 4.1 입력

- logits: αi=log⁡pi\alpha_i = \log p_iαi​=logpi​
    
- Gumbel noise: gi∼Gumbel(0,1)g_i \sim \text{Gumbel}(0,1)gi​∼Gumbel(0,1)
    

---

### 4.2 Gumbel-Softmax 샘플

yi=exp⁡((αi+gi)/τ)∑j=1Kexp⁡((αj+gj)/τ)y_i = \frac{ \exp\left((\alpha_i + g_i)/\tau\right) }{ \sum_{j=1}^K \exp\left((\alpha_j + g_j)/\tau\right) }yi​=∑j=1K​exp((αj​+gj​)/τ)exp((αi​+gi​)/τ)​

✅ 결과:

y∈ΔK−1y \in \Delta^{K-1}y∈ΔK−1

- yi∈(0,1)y_i \in (0,1)yi​∈(0,1)
    
- ∑iyi=1\sum_i y_i = 1∑i​yi​=1
    

즉, **연속적인 확률 벡터**