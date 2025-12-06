
# Feed forward Network (MLP)

FFN(x)=W2​σ(W1​x+b1​)+b2​

- x: attention을 끝낸 각 토큰의 벡터
    
- W1​: **projection up** (차원 up)
    
- σ: ReLU 또는 GELU
    
- W2​: **projection down** (차원 down)
    

📌 **중요**

> “down → up”이 아니라  
> **up → nonlinearity → down** 이 순서예요.


GPT류 모델에서:

- 모델 차원: `d_model = 768`
    
- MLP 내부 hidden 차원: `d_ff = 3072` (보통 4배)
    

### 🔹 Step 1: Projection UP

768  →  3072768 \;\rightarrow\; 3072768→3072

`x ∈ ℝ⁷⁶⁸ W1 ∈ ℝ³⁰⁷²ˣ⁷⁶⁸ h = W1 x + b1`

👉 정보를 **넓은 공간**으로 펼침  
👉 feature끼리의 상호작용을 만들 공간 확보

---

### 🔹 Step 2: Activation (ReLU / GELU)

`h' = GELU(h)`

- 선형 변환만 두 번 하면 결국 하나의 선형 함수
    
- activation이 들어가서 **비선형 표현력** 생김
    
- Transformer 성능의 핵심 요소 중 하나
    

---

### 🔹 Step 3: Projection DOWN

3072  →  7683072 \;\rightarrow\; 7683072→768

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
    

그래서:

- 특정 패턴 → 특정 의미 변환
    
- 문법 규칙
    
- 개념 결합
    

🧠 이게 전부 **W₁, W₂ 안에 분산 저장**됨

---

## 5️⃣ 코드로 보면 (Transformer 표준)

`def transformer_ffn(x):     h = W1 @ x + b1      # projection up     h = gelu(h)         # activation     y = W2 @ h + b2     # projection down     return y`

✅ 이 연산은 **각 토큰마다 독립적으로** 수행됨  
✅ Self-attention만이 토큰 간 정보 교환을 담당