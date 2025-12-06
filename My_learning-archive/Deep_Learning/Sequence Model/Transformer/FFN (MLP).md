
# Point - wise Feed forward Network (MLP)

**각 토큰(단어)별로 독립적으로(Position-wise)** 적용됩니다. 즉, $i$번째 단어를 계산할 때 $i+1$번째 단어를 쳐다보지 않습니다.

FFN(x)=W2​σ(W1​x+b1​)+b2​

- x: attention을 끝낸 각 토큰의 벡터
    
- W1​: **projection up** (차원 up, 학습가능)
    
- σ: ReLU 또는 GELU
    
- W2​: **projection down** (차원 down, 학습가능능)
    

📌 **중요**

> “down → up”이 아니라  
> **up → nonlinearity → down** 이 순서예요.


GPT류 모델에서:

- 모델 차원: `d_model = 768`
    
- MLP 내부 hidden 차원: `d_ff = 3072` (보통 4배)
    

### 🔹 Step 1: Projection UP

768  →  3072

`x ∈ ℝ⁷⁶⁸ W1 ∈ ℝ³⁰⁷²ˣ⁷⁶⁸ h = W1 x + b1`

👉 정보를 **넓은 공간**으로 펼침  
👉 feature끼리의 상호작용을 만들 공간 확보
저차원($d_{model}$)에서는 얽혀 있어서 구분이 안 되는 특징들을, 고차원($d_{ff}$)으로 넓게 펼쳐놓으면 선형적으로 분리하거나 비선형 변환을 주기가 훨씬 쉬워지기 때문입니다. (SVM의 커널 트릭과 유사한 직관)

---

### 🔹 Step 2: Activation (ReLU / GELU)

`h' = GELU(h)`

- 선형 변환만 두 번 하면 결국 하나의 선형 함수
    
- activation이 들어가서 **비선형 표현력** 생김
    
- Transformer 성능의 핵심 요소 중 하나
    

---

### 🔹 Step 3: Projection DOWN

3072  →  768

`y = W2 h' + b2 W2 ∈ ℝ⁷⁶⁸ˣ³⁰⁷²`

👉 다시 모델 차원으로 압축  
👉 다음 layer / residual connection에 넣을 수 있게 만듦

---

## 3️⃣ 이걸 직관적으로 말하면

### 🧠 Projection UP

> “이 단어 표현을 다양한 관점으로 확장해보자”

### 🔥 Activation

> “중요한 조합만 살리고, 필요 없는 건 억제하자”

### 🗜 Projection DOWN

> “필요한 정보만 요약해서 다시 원래 크기로 돌려놓자”

---

## 4️⃣ 왜 이게 ‘기억 저장소’처럼 느껴질까?

Attention은:

- **어디를 볼지** 정함 (routing)
    

MLP는:

- **봤을 때 무엇으로 해석할지** 정함 (computation)
    

Attention이 문맥을 통해 "그것(it)"이 "사과(apple)"라는 것을 알아냈다면(Context mixing), FFN은 "사과"라는 벡터를 보고 내부 파라미터($W_1, W_2$)에 저장된 지식인 "빨갛다", "과일이다", "맛있다"라는 속성을 꺼내서 벡터에 추가해주는 역할을 합니다.

그래서 LLM이 학습한 **"사실적 지식(Fact)"의 대부분은 Attention이 아니라 FFN의 가중치($W_1, W_2$) 안에 저장**되어 있다고 봅니다.

## 수학적 역할: 랭크 붕괴 방지 (Rank Collapse Prevention)

선형대수학 관점에서의 매우 중요한 역할입니다. (질문자님이 좋아하실 부분입니다)

- Attention의 한계:
    
    Attention 연산 $Z = \sum A_{ij} V_j$는 결국 다른 벡터들의 **선형 결합(Linear Combination)**입니다.
    
    수학적으로, 여러 벡터의 선형 결합(볼록 결합)을 반복하면 결과 행렬의 랭크(Rank)가 줄어드는 경향이 있습니다. 즉, 모든 토큰의 벡터가 서로 비슷비슷해지는(smoothing) 현상이 발생합니다. 이를 Rank Collapse라고 합니다.
    
- FFN의 해결책:
    
    FFN은 **비선형 활성화 함수(ReLU)**를 포함하고 있으며, 차원을 확장했다가 줄입니다.
    
    이 과정은 벡터 공간을 비선형적으로 비틀어버리기 때문에, 줄어들 뻔한 **Full Rank를 회복(Restoration)**시켜 줍니다.
    
    즉, Attention이 "평균"을 만들어 뭉뚱그린 정보를, FFN이 다시 "개성"을 부여해 분리해 줍니다.