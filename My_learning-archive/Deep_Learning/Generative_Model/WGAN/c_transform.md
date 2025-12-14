### **1. 배경: Dual Problem의 제약 조건**

최적 운송의 Dual Problem은 두 함수 $\phi(x)$와 $\psi(y)$를 찾는 것입니다. 이때 운송 비용 $c(x, y)$에 대해 다음 **제약 조건**을 만족해야 합니다.

$$\phi(x) + \psi(y) \le c(x, y)$$

(직관적 의미: 운송업자가 떼어가는 수수료($\phi + \psi$)는, 고객이 직접 운반하는 비용($c$)보다 비싸면 안 된다.)

우리는 이익을 최대화하고 싶으므로, $\phi(x)$와 $\psi(y)$를 **가능한 한 크게** 만들고 싶습니다.

---

### **2. $c$-transform의 정의**

만약 우리가 $\psi(y)$를 이미 정했다고 칩시다. 그렇다면, 제약 조건을 깨지 않는 선에서 $\phi(x)$를 얼마나 크게 설정할 수 있을까요?

부등식을 $\phi(x)$에 대해 정리하면:

$$\phi(x) \le c(x, y) - \psi(y)$$

이 식은 **모든 $y$에 대해** 성립해야 합니다. 따라서 $\phi(x)$가 가질 수 있는 최댓값은, $c(x, y) - \psi(y)$ 값들 중 **가장 작은 값(infimum)**이 됩니다.

이것이 바로 **$c$-transform**의 정의입니다.

$$\psi^c(x) = \inf_{y} \left[ c(x, y) - \psi(y) \right]$$

- **의미:** 함수 $\psi$가 주어졌을 때, 비용 함수 $c$를 고려하여 만들 수 있는 **가장 최적의 짝꿍 함수 $\phi$**.
    
- 이것은 Convex Analysis(볼록 해석학)에서 말하는 **Legendre-Fenchel Transform(르장드르 변환)**의 일반화된 형태입니다.
    

---

### **3. 왜 이게 중요한가? (변수 줄이기)**

$c$-transform 덕분에 우리는 두 함수 $\phi, \psi$를 따로따로 최적화할 필요가 없어집니다.

1. **어차피 최적의 해에서는 $\phi(x)$가 $\psi^c(x)$와 같아집니다.**
    
2. 그러므로 식에서 $\phi$를 지워버리고, $\psi$ 하나만 남기거나 (또는 그 반대), 혹은 $f$ 하나로 통일할 수 있게 됩니다.
    

$$\text{Maximize } (\dots \phi \dots \psi \dots) \quad \rightarrow \quad \text{Maximize } (\dots f^c \dots f \dots)$$