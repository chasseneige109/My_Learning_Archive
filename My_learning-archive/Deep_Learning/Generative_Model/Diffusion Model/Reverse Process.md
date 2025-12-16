**Reverse Diffusion (역확산)** 과정은 "완전한 노이즈($x_T$)"에서 시작하여 한 단계씩 "노이즈를 걷어내며" 원본 데이터($x_0$)로 돌아가는 과정입니다.

이를 행렬(벡터) 연산 단위로 아주 정밀하게 해부해 보겠습니다. 전체 과정은 크게 **1. 목표 설정 (이상적인 분포)**, **2. 현실적 타협 (신경망의 개입)**, **3. 최종 연산 (샘플링)**으로 나뉩니다.

---

### 1. 목표 설정: "이상적인 되감기" ($q$)

우리가 하고 싶은 것은 $q(x_{t-1} | x_t)$를 알아내는 것입니다. 즉, 현재의 노이즈 상태 $x_t$를 보고 바로 전 단계 $x_{t-1}$이 무엇이었을지 맞추는 것입니다.

베이즈 정리를 쓰면, 만약 **정답 $x_0$를 알고 있다는 가정하에** 이상적인 역방향 분포를 구할 수 있습니다.

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; {\tilde{\mu}_t(x_t, x_0)}, {\tilde{\beta}_t \mathbf{I}})$$

여기서 가장 중요한 **평균 벡터 $\tilde{\mu}_t$**는 행렬 연산으로 다음과 같이 정의됩니다. 

$$\tilde{\mu}_t(x_t, x_0) = \underbrace{\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \mathbf{x}_t}_{\text{현재 상태 반영}} + \underbrace{\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0}_{\text{원본 정답 반영}}$$

- **행렬 관점:** $t$ 시점의 이미지 행렬 $\mathbf{x}_t$와 원본 이미지 행렬 $\mathbf{x}_0$를 스칼라 값(복잡한 계수들)으로 **선형 결합(Linear Combination/Mixing)** 한 것입니다.
    
- **문제점:** 우리는 생성(Generation) 할 때 **정답 $\mathbf{x}_0$를 모릅니다.** (알면 생성할 필요가 없으니까요.)
    

---

### 2. 현실적 타협: "정답 대신 예측값 쓰기"

$x_0$를 모르니까, **신경망(U-Net)을 시켜서 $x_0$를 예측하거나, 관련된 무언가를 예측하게** 해야 합니다.

#### (1) $x_0$를 $x_t$와 $\epsilon$으로 치환하기

우리는 Forward process 공식에서 다음 관계를 알고 있습니다.

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$

이를 $x_0$에 대해 정리하면:

$$x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}$$

#### (2) 평균 식 $\tilde{\mu}_t$에 대입하기

위의 $x_0$ 식을 아까 본 복잡한 평균 식 $\tilde{\mu}_t(x_t, x_0)$에 쑤겨 넣고 정리하면, 놀랍게도 $x_0$는 사라지고 **노이즈 $\epsilon$만 남습니다.**

$$\mathbf{\mu}_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \mathbf{\epsilon}_\theta(x_t, t) \right)$$

- **$\mathbf{x}_t$:** 현재 우리가 가지고 있는 노이즈 낀 이미지 (Input)
    
- **$\mathbf{\epsilon}_\theta(x_t, t)$:** **신경망이 예측해야 할 유일한 미지수.** (신경망아, 지금 $x_t$에 껴있는 노이즈가 뭐니?)
    

---

### 3. 최종 연산: "샘플링 (Denoising Step)"

이제 실제로 이미지를 생성하는 한 스텝($x_t \to x_{t-1}$)을 행렬 연산으로 수행합니다.

$$\mathbf{x}_{t-1} = \color{blue}{\frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \right)} + \color{red}{\sigma_t \mathbf{z}}$$

이 식을 **행렬(텐서) 연산 순서**대로 뜯어보겠습니다. (이미지 크기 $64 \times 64$ 가정)

1. **입력 ($\mathbf{x}_t$):** $64 \times 64$ 행렬이 준비됩니다.
    
2. **노이즈 예측 ($\mathbf{\epsilon}_\theta$):**
    
    - U-Net에 $\mathbf{x}_t$와 시간 $t$를 넣습니다.
        
    - U-Net은 입력과 똑같은 크기($64 \times 64$)의 **노이즈 맵**을 출력합니다.
        
3. **뺄셈 (Denoising):**
    
    - 원본 $\mathbf{x}_t$에서 예측된 노이즈 $\mathbf{\epsilon}_\theta$에 적절한 계수($\frac{1-\alpha}{\dots}$)를 곱한 값을 뺍니다.
        
    - **의미:** "노이즈라고 생각되는 부분을 지우개로 지운다."
        
4. **스케일링:**
    
    - 전체에 $\frac{1}{\sqrt{\alpha_t}}$를 곱해줍니다. (노이즈를 뺐으니 전체적인 픽셀 값의 크기가 작아진 것을 다시 원래대로 복구하는 역할)
        
5. **랜덤 노이즈 추가 ($\mathbf{z}$):** [중요!]
    
    - 새로운 랜덤 가우시안 노이즈 $\mathbf{z} \sim \mathcal{N}(0, I)$ ($64 \times 64$)를 아주 조금($\sigma_t$) 더해줍니다.
        
    - **이유:** $x_{t-1}$은 확정된 값 하나가 아니라 **확률 분포**입니다. 분포에서 샘플링을 하려면 랜덤성이 필요합니다. (이게 없으면 이미지가 밋밋해집니다.)
        

---

### 요약: 행렬 단위 프로세스

Reverse Diffusion의 한 스텝은 결국 다음과 같은 **벡터 연산(Vector Arithmetic)**입니다.

$$\text{Next Image} = \text{Scale} \times (\text{Current Image} - \text{Scale}' \times \text{Predicted Noise}) + \text{Small Random Noise}$$

- **입력:** 노이즈 낀 덩어리
    
- **핵심 연산:** 신경망이 "이게 노이즈야"라고 찾아낸 부분을 **행렬 뺄셈**으로 제거.
    
- **출력:** 조금 더 선명해진 덩어리
    

이 과정을 $T=1000$번 반복하면, 처음의 완전한 노이즈 행렬이 의미 있는 이미지 행렬로 바뀌게 됩니다.

Would you like me to ...

혹시 이 과정에서 사용되는 **U-Net 구조(Attention 등)**가 노이즈를 예측하기 위해 행렬 내부에서 구체적으로 어떤 연산을 하는지 설명해 드릴까요?