### 1. 최적의 Discriminator $D^*$ 유도

GAN의 Value Function $V(G, D)$는 다음과 같이 정의됩니다.

$$V(G, D) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log(1 - D(G(z)))]$$

Generator $G$가 고정되어 있다고 가정하고, $D$를 최대화하는 최적의 $D^*$를 찾습니다. 두 번째 항의 기대값을 $x$에 대한 적분으로 변환하기 위해 $x = G(z)$로 치환하면, 생성된 데이터의 분포는 $p_g$를 따르게 되므로 식은 다음과 같이 변형됩니다. (확률론: Change of Variables Formula)

$$V(G, D) = \int_x \left[ p_{data}(x) \log D(x) + p_g(x) \log(1 - D(x)) \right] dx$$

이 적분식이 최대가 되려면, 피적분 함수(integrand)가 각 $x$ 지점에서 최대가 되어야 합니다. 피적분 함수를 $y = D(x)$에 대한 함수 $f(y)$로 둡니다. ($a = p_{data}(x)$, $b = p_g(x)$라 가정)

$$f(y) = a \log y + b \log(1 - y)$$

이 함수를 $y$에 대해 미분하여 극대값을 찾습니다.

$$\frac{d}{dy} f(y) = \frac{a}{y} - \frac{b}{1 - y} = 0$$

$$a(1 - y) = by \quad \Rightarrow \quad a = (a + b)y \quad \Rightarrow \quad y = \frac{a}{a + b}$$

따라서, 최적의 Discriminator $D^*(x)$는 다음과 같습니다.

$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

---

### 2. GAN 목적 함수 = JSD 최소화 증명

이제 구한 최적의 Discriminator $D^*$를 원래의 목적 함수 $V(G, D)$에 대입하여, Generator 입장에서 최소화해야 하는 함수 $C(G)$를 구합니다.

$$\begin{aligned} C(G) &= V(G, D^*) \\ &= \mathbb{E}_{x \sim p_{data}} \left[ \log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} \right] + \mathbb{E}_{x \sim p_g} \left[ \log \frac{p_g(x)}{p_{data}(x) + p_g(x)} \right] \end{aligned}$$

이 식을 JSD 형태($\frac{P+Q}{2}$와의 비교)로 맞추기 위해, 분모와 분자에 조작을 가합니다. ($\log 4 = 2\log 2$를 더하고 뺍니다)

$$\begin{aligned} C(G) &= \int_x p_{data}(x) \log \left( \frac{p_{data}(x)}{(p_{data}(x) + p_g(x))/2} \cdot \frac{1}{2} \right) dx \\ &\quad + \int_x p_g(x) \log \left( \frac{p_g(x)}{(p_{data}(x) + p_g(x))/2} \cdot \frac{1}{2} \right) dx \end{aligned}$$

로그의 성질($\log AB = \log A + \log B$)을 이용하여 분리합니다.

$$\begin{aligned} C(G) &= \int_x p_{data}(x) \log \left( \frac{p_{data}(x)}{\frac{p_{data}(x) + p_g(x)}{2}} \right) dx - \log 2 \int_x p_{data}(x) dx \\ &\quad + \int_x p_g(x) \log \left( \frac{p_g(x)}{\frac{p_{data}(x) + p_g(x)}{2}} \right) dx - \log 2 \int_x p_g(x) dx \end{aligned}$$

여기서 $\int p(x)dx = 1$이고, KL Divergence의 정의($KL(P||Q) = \int P \log \frac{P}{Q}$)를 적용하면:

$$C(G) = KL\left(p_{data} \left\| \frac{p_{data} + p_g}{2}\right.\right) + KL\left(p_g \left\| \frac{p_{data} + p_g}{2}\right.\right) - 2\log 2$$

Jensen-Shannon Divergence의 정의는 $JSD(P||Q) = \frac{1}{2}KL(P||M) + \frac{1}{2}KL(Q||M)$ (단, $M=\frac{P+Q}{2}$)이므로, 최종적으로 다음을 얻습니다.

$$C(G) = 2 \cdot JSD(p_{data} \| p_g) - \log 4$$

**결론:** Generator의 학습 목표 $\min_G V(G, D^*)$는 상수($-\log 4$)를 제외하면 **JSD를 최소화하는 것과 수학적으로 동치**입니다.

- 완벽한 학습 시 ($p_g = p_{data}$): $JSD = 0$, 최솟값은 $-\log 4$.
    
- 이때 $D^*(x) = \frac{p_{data}}{2p_{data}} = 0.5$.
    

---

### 3. 문제점: 서로 겹치지 않는 분포 (Disjoint Supports)

실제 고차원 공간에서 $p_{data}$와 $p_g$의 **지지(Support)**가 겹치지 않는 경우($Supp(p_{data}) \cap Supp(p_g) \approx \emptyset$), JSD는 심각한 학습 불안정성을 초래합니다.

#### 수학적 상황

두 분포가 완전히 분리되어 있다면, 어떤 $x$에 대해서도 다음 두 가지 중 하나입니다.

1. $x \in Supp(p_{data}) \implies p_g(x) = 0 \implies D^*(x) = \frac{p_{data}}{p_{data} + 0} = 1$
    
2. $x \in Supp(p_g) \implies p_{data}(x) = 0 \implies D^*(x) = \frac{0}{0 + p_g} = 0$
    

#### JSD의 값

이 경우 $JSD$ 값을 계산해보면:

$$\begin{aligned} JSD(p_{data} \| p_g) &= \frac{1}{2} \int_{Supp(p_{data})} p_{data} \log \frac{p_{data}}{p_{data}/2} dx + \frac{1}{2} \int_{Supp(p_g)} p_g \log \frac{p_g}{p_g/2} dx \\ &= \frac{1}{2} \int p_{data} \log 2 \, dx + \frac{1}{2} \int p_g \log 2 \, dx \\ &= \frac{1}{2} \log 2 + \frac{1}{2} \log 2 = \log 2 \end{aligned}$$

#### Gradient Vanishing

- 두 분포가 겹치지 않는 한, 거리가 얼마나 떨어져 있든 상관없이 **JSD 값은 항상 상수 $\log 2$로 고정**됩니다.
    
- 상수 함수의 기울기(Gradient)는 **0**입니다 ($\nabla_G JSD \approx 0$).
    
- 즉, Generator는 데이터를 진짜 분포 쪽으로 가깝게 만들기 위해 **어느 방향으로 파라미터를 수정해야 할지(Backprop) 정보를 전혀 얻지 못하게 됩니다.**
    

이것이 초기 GAN 학습이 어렵고 불안정한 주된 이론적 이유이며, 분포 간의 '거리'를 측정할 때 겹치지 않아도 유의미한 값을 주는 **Earth Mover's Distance (Wasserstein Distance)**가 필요한 이유입니다.