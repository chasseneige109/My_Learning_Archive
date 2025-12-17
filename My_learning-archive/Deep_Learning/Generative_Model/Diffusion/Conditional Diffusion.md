**'조건부 생성(Conditional Generation)'**

우리가 알고 있는 DALL-E 3나 Midjourney, Stable Diffusion이 바로 이 원리로 작동합니다. 단순히 "아무거나 그려봐"가 아니라, **사용자의 의도($y$)를 모델의 등산 경로(Score)에 반영**하는 것이 핵심입니다.

---

### 1. 조건부 생성의 수학적 의미: 나침반의 수정

기존의 Unconditional 모델은 데이터 분포의 산을 오르는 나침반인 $s_\theta(x_t, t) = \nabla_{x_t} \log p(x_t)$를 가졌습니다.

반면, Conditional 모델은 **조건 $y$가 주어졌을 때의 점수(Score)**를 계산합니다.

$$\nabla_{x_t} \log p(x_t \mid y)$$

이것을 베이즈 정리(Bayes' Rule)로 풀어보면 아주 흥미로운 구조가 나옵니다.

$$\log p(x_t \mid y) \propto \log p(x_t) + \log p(y \mid x_t)$$

이 식을 미분하면 다음과 같은 두 가지 힘의 합이 됩니다.

$$\underbrace{\nabla_{x_t} \log p(x_t \mid y)}_{\text{조건부 나침반}} = \underbrace{\nabla_{x_t} \log p(x_t)}_{\text{데이터 자체의 품질(선명도)}} + \underbrace{\nabla_{x_t} \log p(y \mid x_t)}_{\text{조건 } y\text{와의 일치도}}$$

- **첫 번째 항:** "이미지가 깨끗하고 자연스러워야 해."
    
- **두 번째 항:** "그 이미지가 내가 말한 $y$(예: 강아지)와 닮아야 해."
    

---

### 2. 조건을 주입하는 두 가지 주요 방법

#### ① Classifier Guidance (분류기 가이드)

초기 모델에서 사용하던 방식입니다.

- **방법:** 별도로 학습된 이미지 분류기(Classifier)를 준비합니다.
    
- **작동:** Diffusion 모델이 이미지를 복원할 때, 분류기가 "음, 지금 그림은 '강아지' 확률이 낮아. 이쪽으로 수정해!"라고 가이드를 줍니다.
    
- **단점:** 노이즈 낀 이미지($x_t$)를 인식할 수 있는 별도의 분류기가 필요하며, 학습이 번거롭습니다.
    

#### ② Classifier-Free Guidance (CFG, 핵심 기술)

현재 대부분의 모델이 사용하는 **가장 중요한 기술**입니다.

- **방법:** **하나의 U-Net**이 조건이 있을 때($y$)와 없을 때($\emptyset$)를 동시에 학습합니다.
    
- 작동: 생성할 때 두 점수의 차이를 이용해 "조건 $y$의 특징을 더 강하게 부각"시킵니다.
    
    $$\hat{\epsilon} = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset))$$
    
    - 여기서 $w$(Guidance Scale)를 높이면, 사용자의 프롬프트를 아주 엄격하게 따르는 이미지가 나옵니다.
        

---

### 3. 다양한 조건($y$)의 형태와 주입 방식

|**조건의 종류**|**주입 방식 (Architecture)**|**설명**|
|---|---|---|
|**클래스 라벨**|**Time Embedding에 추가**|$t$ 정보 옆에 라벨 숫자를 붙여서 넣어줌.|
|**텍스트 프롬프트**|**Cross-Attention**|텍스트를 벡터로 만든 뒤, U-Net의 중간 층에서 이미지 특징과 결합.|
|**스케치 / 포즈**|**ControlNet / 채널 결합**|이미지와 똑같은 크기의 마스크를 입력 채널에 추가하여 형태를 강제함.|