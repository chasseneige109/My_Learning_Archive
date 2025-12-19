**Change of Variables Formula (변수 변환 공식)**은 확률론과 미적분학에서 매우 깊이 있고 중요한 개념입니다. 특히 딥러닝의 생성 모델(GAN, Normalizing Flows 등)을 이해하는 데 필수적인 수학적 기초입니다.

이 공식의 핵심은 **"공간이 왜곡되면, 부피도 변하고, 그에 따라 밀도(Density)도 변해야 한다"**는 것입니다.

****"공간의 부피가 변하면 밀도도 그 역수만큼 변해야 총량이 보존된다"**는 물리적/기하학적 보존 법칙을 수식화한 것입니다.

- 1차원: 기울기(미분)의 역수를 곱함.
    
- 다차원: 자코비안 행렬식(부피 변화율)의 역수를 곱함.
---

### 1. 직관: "확률 질량 보존"과 "부피의 변화"

확률 밀도 함수(PDF)에서 가장 중요한 대원칙은 **총 확률의 합(질량)은 보존된다**는 것입니다.

어떤 변환 $x = G(z)$가 있을 때, $z$ 공간의 아주 작은 영역 $\Delta z$에 있는 확률 질량은 변환된 $x$ 공간의 영역 $\Delta x$에 그대로 옮겨져야 합니다.

$$P(z \in \Delta z) = P(x \in \Delta x)$$

이것을 밀도 함수($p$)와 부피($\text{Volume}$)의 곱으로 표현하면 다음과 같습니다.

$$p_z(z) \cdot |\text{Vol}(\Delta z)| = p_x(x) \cdot |\text{Vol}(\Delta x)|$$

여기서 $G$라는 함수가 공간을 **늘리거나 줄이면**, 부피 $\text{Vol}(\Delta x)$가 변하게 되고, 확률 질량을 보존하기 위해 **밀도 $p_x(x)$는 반대로 줄어들거나 늘어나야 합니다**.

> **비유:** 잼(확률 질량)을 빵(공간)에 바를 때, 빵을 넓게 펴면(부피 증가) 잼의 두께(밀도)는 얇아집니다.

---

### 2. 1차원(Univariate) 경우

$z$와 $x$가 스칼라일 때, 변환 함수 $x = G(z)$가 **단조 증가하거나 단조 감소(일대일 대응, Invertible)**한다고 가정합니다.

미소 구간 $dz$와 $dx$ 사이의 관계는 미분으로 정의됩니다.

$$dx = G'(z) dz \quad \Rightarrow \quad dz = \frac{1}{G'(z)} dx$$

확률 보존 법칙에 따라:

$$p_x(x) |dx| = p_z(z) |dz|$$

이를 정리하면 1차원 변수 변환 공식이 나옵니다:

$$p_x(x) = p_z(z) \left| \frac{dz}{dx} \right| = p_z(G^{-1}(x)) \left| \frac{1}{G'(G^{-1}(x))} \right|$$

- **$\left| \frac{dz}{dx} \right|$ 항:** 이 항이 바로 **공간의 왜곡을 보정하는 스케일링 팩터**입니다.
    

---

### 3. 다차원(Multivariate) 경우: 자코비안(Jacobian)의 등장

이제 $z$와 $x$가 $n$차원 벡터인 경우($\mathbf{z}, \mathbf{x} \in \mathbb{R}^n$)를 보겠습니다. 함수 $\mathbf{x} = G(\mathbf{z})$는 벡터를 입력받아 벡터를 출력합니다.

이때 1차원의 미분계수($G'(z)$)에 해당하는 것이 바로 **자코비안 행렬(Jacobian Matrix, $J$)**입니다.

#### 자코비안 행렬 정의

$$J_G(\mathbf{z}) = \frac{\partial \mathbf{x}}{\partial \mathbf{z}} = \begin{bmatrix} \frac{\partial x_1}{\partial z_1} & \cdots & \frac{\partial x_1}{\partial z_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial x_n}{\partial z_1} & \cdots & \frac{\partial x_n}{\partial z_n} \end{bmatrix}$$

이 행렬은 국소적으로(locally) **선형 변환**을 나타냅니다.

#### 자코비안 행렬식 (Determinant)의 기하학적 의미

자코비안 행렬의 **행렬식(Determinant)**, 즉 $\det(J)$는 **부피 확대율**을 의미합니다.

- $|\det(J)| > 1$: 부피가 팽창함 (밀도는 감소)
    
- $|\det(J)| < 1$: 부피가 수축함 (밀도는 증가)
    

#### 최종 공식

다차원 변수 변환 공식은 다음과 같습니다.

$$p_x(\mathbf{x}) = p_z(\mathbf{z}) \left| \det \left( \frac{\partial \mathbf{z}}{\partial \mathbf{x}} \right) \right|$$

또는 역함수 관계를 이용하면:

$$p_x(\mathbf{x}) = p_z(\mathbf{z}) \frac{1}{\left| \det \left( \frac{\partial \mathbf{x}}{\partial \mathbf{z}} \right) \right|}$$

이 수식은 **Normalizing Flows** (Glow, RealNVP 등) 모델의 핵심 원리입니다. 단순한 분포($p_z$)를 복잡한 함수 $G$로 변환하여 복잡한 데이터 분포($p_x$)를 모델링하고, 이 공식을 통해 $p_x$의 값을 정확히 계산(Likelihood 계산)할 수 있기 때문입니다.

---

### 4. GAN 적분 식에서의 적용 (Deep Dive)

앞서 질문하신 GAN의 적분 변환으로 돌아가 보겠습니다.

$$\int_z p_z(z) \log(1 - D(G(z))) \, dz \quad \rightarrow \quad \int_x p_g(x) \log(1 - D(x)) \, dx$$

이 과정은 엄밀히 말하면 **LOTUS (Law of the Unconscious Statistician, 무의식적인 통계학자의 법칙)**와 **변수 변환**이 결합된 형태입니다.

1. 적분 변수 변경: $z \to x$로 적분 영역을 바꿉니다. 이때 $dx$와 $dz$의 관계는 자코비안 행렬식에 의해 결정됩니다.
    
    $$dz = \left| \det \left( \frac{\partial \mathbf{z}}{\partial \mathbf{x}} \right) \right| dx$$
    
2. 대입: 적분 식에 대입합니다.
    
    $$\int_x p_z(G^{-1}(x)) \cdot \log(1 - D(x)) \cdot \underbrace{\left| \det \left( \frac{\partial \mathbf{z}}{\partial \mathbf{x}} \right) \right| dx}_{dz}$$
    
3. 밀도 함수 정의 ($p_g(x)$): 여기서 변수 변환 공식에 의해 $p_g(x)$가 정의됩니다.
    
    $$p_g(x) = p_z(G^{-1}(x)) \left| \det \left( \frac{\partial \mathbf{z}}{\partial \mathbf{x}} \right) \right|$$
    
4. 최종 형태: 위 $p_g(x)$ 정의를 식에 대입하면 깔끔하게 정리됩니다.
    
    $$\int_x p_g(x) \log(1 - D(x)) \, dx$$
    

### 요약

**Change of Variables Formula**는 단순히 변수를 바꾸는 테크닉이 아니라, **"공간의 부피가 변하면 밀도도 그 역수만큼 변해야 총량이 보존된다"**는 물리적/기하학적 보존 법칙을 수식화한 것입니다.

- 1차원: 기울기(미분)의 역수를 곱함.
    
- 다차원: 자코비안 행렬식(부피 변화율)의 역수를 곱함.