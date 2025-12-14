### 1. 🧙‍♂️ Primal $\rightarrow$ Dual Form 

WGAN이론의 시작은 [[Wasserstein Distance]]**의 원래 정의(Primal Form)에서 비롯됩니다.
수학적 유도는 여기를 참고! [[Kantorovich-Rubinstein Duality]]

$$W(P, Q) = \inf_{\gamma \in \Pi} \mathbb{E}_{(x, y) \sim \gamma} [\|x - y\|]$$

- **문제:** 이 식은 최적의 운송 계획 $\gamma$를 무한히 많은 경우의 수 중에서 찾아야 하므로 신경망으로 풀 수 없습니다.
    
- **해결:** **칸토로비치-루빈슈타인 쌍대성(Kantorovich-Rubinstein Duality)**이라는 수학적 정리를 이용해 문제를 뒤집습니다. (Dual Form)
    

$$W(P, Q) = \sup_{\|f\|_L \le 1} \left( \mathbb{E}_{x \sim P}[f(x)] - \mathbb{E}_{x \sim Q}[f(x)] \right)$$

- 이제 복잡한 **'운송 계획 $\gamma$'** 대신, 단순한 **'함수 $f$'** 하나만 찾으면 됩니다. 이 함수 $f$가 WGAN의 **Critic** 네트워크입니다.
    

### 2. 🚦 1-Lipschitz 제약: Critic의 "속도 제한"

Dual Form 수식에서 $W(P, Q)$가 되려면, $\sup$ (최댓값)을 찾는 함수 $f$가 **반드시** 1-Lipschitz 조건을 만족해야 합니다.

- **정의:** 입력 $x_1, x_2$에 대해 $\frac{|f(x_1) - f(x_2)|}{\|x_1 - x_2\|} \le 1$
    
    - 즉, **함수 $f$의 기울기(Gradient)가 어떤 지점에서도 1을 넘을 수 없습니다.**
        
- **필요성:**
    
    - 이 제약은 $f$가 점수 차이를 **무한대로 부풀리는 것을 막아**줍니다.
        
    - $f$의 출력값 차이가 입력 공간의 거리($\|x_1 - x_2\|$)와 비례하도록 묶어주어, **$f$가 두 분포 사이의 거리를 측정하는 '정확한 자(Ruler)' 역할을 하도록 강제**합니다.
        

### 4. 📈 WGAN의 손실 함수

WGAN의 최종 $W(P_r, P_g)$ 최소화 과정은 다음과 같은 두 단계의 반복입니다.

1. **Critic($f$) 학습:** Critic을 $L_D$를 이용해 여러 번 업데이트하여 $W$를 정확히 **측정(Maximize)**합니다.
    
2. **Generator($G$) 학습:** Critic이 측정한 $W$를 $L_G$를 이용해 한 번 업데이트하여 $W$를 **최소화(Minimize)**합니다.

**직관:** Critic이 1-Lipschitz 제약을 지키며 열심히 점수 차이를 벌리면, 그 최대 점수 차이가 바로 **두 분포를 일치시키기 위해 필요한 최소한의 작업량(Wasserstein Distance)**이 됩니다. Generator는 이 거리를 좁히는 방향으로 움직여 학습이 안정적으로 진행됩니다.

추가적인 궁금증이 있으신가요?