우리의 목표는 신경망 $s_\theta(x)$가 실제 데이터 분포의 Score $\nabla_x \log p_{\text{data}}(x)$를 흉내 내도록 만드는 것입니다.

말씀하신 대로 가장 직관적인 목표 함수(Loss Function)는 둘 사이의 차이를 줄이는 것입니다 (**Explicit Score Matching**).

$$L(\theta) = \mathbb{E}_{x \sim p_{\text{data}}} \left[ \| s_\theta(x) - \nabla_x \log p_{\text{data}}(x) \|^2 \right]$$

**하지만 역설적인 문제가 있습니다:**

- 우리는 실제 데이터 분포 $p_{\text{data}}(x)$를 모릅니다. (모르니까 모델을 만들어서 배우려고 하는 것이죠.)
    
- 따라서 $\nabla_x \log p_{\text{data}}(x)$ 값을 알 수가 없어서 위 식으로는 학습을 못 합니다.
    

현실적인 해결책 (Denoising Score Matching):

수학자들은 기막힌 우회로를 찾아냈습니다.

**"데이터에 일부러 노이즈를 더한 뒤, 그 노이즈를 예측하게 시키면, 수학적으로 Score를 학습하는 것과 똑같다"**는 사실이 증명되었습니다 (Vincent, 2011).
[[M) Anderson's Reverse SDE]] <---- 설명

이것이 바로 **Diffusion Model**의 학습 방식과 정확히 일치합니다.

> **Diffusion의 노이즈 예측($\epsilon_\theta$) $\approx$ Score 추정($s_\theta$)**
> 
> $$s_\theta(x_t, t) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$$
> 
> 즉, 우리가 앞에서 열심히 U-Net에게 **"노이즈($\epsilon$) 맞춰봐"**라고 시킨 것이, 사실은 수학적으로 **"데이터 분포의 기울기(Score)를 구해봐"**라고 시킨 것과 동치입니다.