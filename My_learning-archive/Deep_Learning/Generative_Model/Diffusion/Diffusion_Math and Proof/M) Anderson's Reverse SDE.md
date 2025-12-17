**Anderson's Reverse SDE**라고 불리는, Diffusion Model(Score-based Generative Model)의 **'성배(Holy Grail)'**와도 같은 공식입니다.

질문하신 **"이 식의 정체"**와 **"유도 과정(직관적 설명)"**을 나누어 설명해 드리겠습니다.

---

### 1. 식의 정체: $\nabla_\mathbf{x} \log p_t(\mathbf{x})$가 대체 뭐야?

$$d\mathbf{x} = \left[ \mathbf{f}(\mathbf{x}, t) - g(t)^2 \color{red}{\nabla_\mathbf{x} \log p_t(\mathbf{x})} \right] dt + g(t) d\bar{\mathbf{w}}$$

이 식에서 가장 중요한 빨간색 부분, $\nabla_\mathbf{x} \log p_t(\mathbf{x})$를 **스코어 함수(Score Function)**라고 부릅니다.

- **의미:** 현재 데이터 분포($p_t$)의 **"확률 밀도가 가장 가파르게 증가하는 방향"**을 가리키는 벡터입니다.
    
- **직관:** 산을 오르는 등산객을 상상해 보세요.
    
    - 안개가 자욱해서(노이즈) 아무것도 안 보입니다.
        
    - 하지만 발밑을 보니 경사가 가장 가파른 쪽이 보입니다.
        
    - **"이쪽으로 가면 산 정상(데이터가 모여 있는 곳, 원본 이미지)이 나오겠구나!"** 라고 알려주는 나침반이 바로 Score Function입니다.
        
- **역할:** 시간을 거꾸로 돌릴 때, 노이즈($g(t)d\bar{w}$)에 의해 흩어지려는 입자들을 강제로 **"데이터가 있는 쪽으로 끌어당기는 힘"** 역할을 합니다.
    

---

### 2. 유도 과정: 어떻게 저런 식이 튀어나왔나?

이 식을 엄밀하게 유도하려면 확률 미분방정식 이론이 필요하지만, **Fokker-Planck Equation (FPE)**을 사용하면 논리의 흐름을 명확하게 이해할 수 있습니다.

#### Step 1. Forward 과정의 확률 분포 변화 (FPE)

Forward SDE ($d\mathbf{x} = \mathbf{f} dt + g d\mathbf{w}$)를 따르는 입자들이 있다고 칩시다.

이 입자들의 확률 분포 $p_t(\mathbf{x})$가 시간에 따라 어떻게 변하는지는 물리학의 Fokker-Planck 방정식으로 기술됩니다.

$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (\mathbf{f} p_t) + \frac{1}{2} g^2 \nabla^2 p_t$$

- 첫 번째 항($-\nabla \cdot (\mathbf{f} p_t)$): Drift($\mathbf{f}$)에 의해 확률분포가 이동함.
    
- 두 번째 항($\frac{1}{2} g^2 \nabla^2 p_t$): Diffusion($g$)에 의해 확률분포가 넓게 퍼짐(확산).
    

#### Step 2. 시간을 거꾸로 뒤집기

이제 시간의 흐름을 반대로 봅니다. 즉, $t$가 줄어드는 방향입니다.

확률 분포의 변화식(PDE)에서 부호를 반대로 뒤집으면 됩니다. 하지만 단순히 부호만 바꾸면 되는 게 아니라, 확률 보존 법칙을 만족하며 뒤집어야 합니다.

수학적 기교를 부려 두 번째 항(확산 항)을 변형해 봅시다.

(항등식 $\nabla^2 p = \nabla \cdot (\nabla p) = \nabla \cdot (p \nabla \log p)$ 이용)

$$\frac{1}{2} g^2 \nabla^2 p_t = \frac{1}{2} g^2 \nabla \cdot (p_t \nabla \log p_t)$$

이걸 원래 FPE 식에 대입해서, **전체 식을 $-\nabla \cdot ((\dots)p_t)$ 형태(Drift 형태)**로 묶어내면 다음과 같은 모양을 유도해낼 수 있습니다.

$$\frac{\partial p_t}{\partial t} = -\nabla \cdot \left[ \left( \mathbf{f} - \frac{1}{2}g^2 \nabla \log p_t \right) p_t \right] + (\text{나머지 항})$$

#### Step 3. Reverse SDE의 형태 발견

Anderson(1982)은 위와 같은 PDE 분석을 통해, **"시간을 거꾸로 흐르게 할 때, 확률 분포 $p_t$가 Forward 때와 정확히 같은 경로를 밟아 돌아오게 하려면"** Drift 항이 바뀌어야 한다는 것을 발견했습니다.

- **Forward Drift:** $\mathbf{f}(\mathbf{x}, t)$
    
- **Reverse Drift:** $\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})$
    

즉, **"원래 흐름($\mathbf{f}$)에서 '확산되려는 힘'을 뺀 만큼($-g^2 \nabla \log p$) 반대로 당겨줘야 한다"**는 결론이 나옵니다.

---

### 3. 이게 왜 대단한가? (Deep Learning과의 연결)

여기서 소름 돋는 연결고리가 등장합니다.

Reverse SDE 식을 쓰려면 $\nabla_\mathbf{x} \log p_t(\mathbf{x})$를 알아야 합니다.

그런데 우리는 $p_t(\mathbf{x})$(노이즈 낀 데이터의 실제 확률분포)를 모릅니다. 식은 찾았는데 값을 못 구하는 상황이죠.

여기서 **Deep Learning**이 등장합니다.

**"우리가 $p_t$는 모르지만, 딥러닝 모델(U-Net)한테 $\nabla_\mathbf{x} \log p_t(\mathbf{x})$ 값을 예측하도록 시키면 어떨까?"**

이것이 Score Matching 기법입니다.

그리고 놀랍게도, 우리가 앞서 열심히 배웠던 **"노이즈 예측 모델($\epsilon_\theta$)"**이 사실은 이 Score Function과 비례 관계임이 밝혀졌습니다.

$$\text{Score Function}: \quad \nabla_\mathbf{x} \log p_t(\mathbf{x}) \approx -\frac{\epsilon_\theta(\mathbf{x}_t, t)}{\sigma_t}$$

- **직관적 해석:**
    
    - **노이즈($\epsilon_\theta$):** 원점에서 멀어진 방향.
        
    - **스코어($\nabla \log p$):** 데이터 밀도가 높은 곳(원점)으로 가려는 방향.
        
    - 둘은 정확히 **반대 방향(Minus sign)**입니다.
        

### 요약

1. **식의 정체:** 노이즈 때문에 흩어진 데이터를 다시 뭉치게 만드는 **Reverse SDE 공식**입니다.
    
2. **핵심 항:** **$\nabla \log p$ (Score Function)**은 "데이터가 어디에 많이 모여있는지" 알려주는 나침반입니다.
    
3. **유도 원리:** Fokker-Planck 방정식(확률분포의 시간 변화)을 역시간으로 뒤집어서 풀어낸 결과입니다.
    
4. **결론:** U-Net이 예측하는 **"노이즈($\epsilon$)"**가 사실은 이 **"Score Function($\nabla \log p$)"**의 다른 이름이었습니다. 이로써 DDPM(노이즈 예측)과 SDE(스코어 기반 생성)가 수학적으로 **대통합**을 이루게 됩니다.